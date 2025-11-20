"""
SQLite-based long-term persistence for STUDY workflow memory system.

This module implements durable storage for completed investigations and
important findings using SQLite. Provides searchable archives of historical
investigation data with support for complex queries and cross-investigation
knowledge retrieval.

Architecture:
- SQLite database with normalized schema
- Three tables: investigations, memory_entries, memory_references
- Connection pooling and transaction management
- Full-text search support (future enhancement)
"""

import logging
import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .models import MemoryEntry, MemoryMetadata, MemoryQuery, MemoryType

logger = logging.getLogger(__name__)


class LongTermStorage:
    """
    SQLite-based persistent storage for memory entries.

    Provides durable storage for investigation memory entries with support
    for complex queries, historical analysis, and cross-investigation retrieval.

    Database Schema:
        investigations: Track investigation metadata
            - investigation_id (PK): Unique investigation identifier
            - created_at: Investigation start timestamp
            - completed_at: Investigation completion timestamp
            - persona_count: Number of personas involved
            - entry_count: Number of memory entries
            - metadata_json: Additional investigation metadata

        memory_entries: Store individual memory entries
            - id (PK): Auto-incrementing entry ID
            - entry_id: Unique entry identifier (for external reference)
            - investigation_id (FK): Link to investigations table
            - session_id: Session identifier
            - timestamp: Entry creation timestamp
            - persona: Persona identifier
            - findings: Main content/findings
            - evidence: Supporting evidence
            - confidence_before: Confidence before this step
            - confidence_after: Confidence after this step
            - memory_type: Type of memory entry
            - metadata_json: Additional entry metadata

        memory_references: Track relationships between entries
            - id (PK): Auto-incrementing reference ID
            - source_entry_id: Entry making the reference
            - target_entry_id: Entry being referenced
            - created_at: Reference creation timestamp

    Thread Safety:
        SQLite connections are not shared across threads. Each operation
        creates a new connection which is automatically closed.

    Example:
        >>> storage = LongTermStorage("investigations.db")
        >>> storage.initialize()
        >>> storage.save(entry_id, memory_entry)
        >>> results = storage.query(MemoryQuery(persona="researcher"))
        >>> storage.close()
    """

    def __init__(self, db_path: str = "study_memory.db"):
        """
        Initialize long-term storage with SQLite backend.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        logger.info(f"LongTermStorage initialized with db_path={db_path}")

    def initialize(self) -> None:
        """
        Initialize database schema.

        Creates tables if they don't exist. Safe to call multiple times.

        Raises:
            sqlite3.Error: If database initialization fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create investigations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    investigation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    persona_count INTEGER DEFAULT 0,
                    entry_count INTEGER DEFAULT 0,
                    metadata_json TEXT
                )
            """
            )

            # Create memory_entries table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT UNIQUE NOT NULL,
                    investigation_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    persona TEXT NOT NULL,
                    findings TEXT NOT NULL,
                    evidence TEXT,
                    confidence_before TEXT,
                    confidence_after TEXT,
                    memory_type TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
                )
            """
            )

            # Create indexes for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_investigation
                ON memory_entries(investigation_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_persona
                ON memory_entries(persona)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_timestamp
                ON memory_entries(timestamp)
            """
            )

            # Create memory_references table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entry_id TEXT NOT NULL,
                    target_entry_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_entry_id) REFERENCES memory_entries(entry_id),
                    FOREIGN KEY (target_entry_id) REFERENCES memory_entries(entry_id)
                )
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_references_source
                ON memory_references(source_entry_id)
            """
            )

            conn.commit()
            logger.info("Database schema initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def save(self, entry_id: str, entry: MemoryEntry) -> None:
        """
        Save a memory entry to persistent storage.

        Creates or updates the entry in the database. If the investigation
        doesn't exist, it's automatically created.

        Args:
            entry_id: Unique identifier for the entry
            entry: MemoryEntry instance to save

        Raises:
            sqlite3.Error: If save operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Ensure investigation exists
            cursor.execute(
                "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
                (entry.investigation_id,),
            )
            if not cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO investigations (investigation_id, created_at, metadata_json)
                    VALUES (?, ?, ?)
                """,
                    (
                        entry.investigation_id,
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps({}),
                    ),
                )

            # Insert or replace memory entry
            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_entries (
                    entry_id, investigation_id, session_id, timestamp,
                    persona, findings, evidence, confidence_before,
                    confidence_after, memory_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    entry.investigation_id,
                    entry.session_id,
                    entry.timestamp,
                    entry.persona,
                    entry.findings,
                    entry.evidence,
                    entry.confidence_before,
                    entry.confidence_after,
                    entry.memory_type.value,
                    json.dumps(entry.metadata),
                ),
            )

            # Save memory references
            for ref_id in entry.memory_references:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO memory_references (
                        source_entry_id, target_entry_id, created_at
                    ) VALUES (?, ?, ?)
                """,
                    (entry_id, ref_id, datetime.now(timezone.utc).isoformat()),
                )

            # Update investigation entry count
            cursor.execute(
                """
                UPDATE investigations
                SET entry_count = (
                    SELECT COUNT(*) FROM memory_entries
                    WHERE investigation_id = ?
                )
                WHERE investigation_id = ?
            """,
                (entry.investigation_id, entry.investigation_id),
            )

            conn.commit()
            logger.debug(f"Saved entry {entry_id} to persistent storage")

        except sqlite3.Error as e:
            logger.error(f"Failed to save entry {entry_id}: {e}")
            raise

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry from persistent storage.

        Args:
            entry_id: Unique identifier for the entry

        Returns:
            MemoryEntry if found, None otherwise

        Raises:
            sqlite3.Error: If retrieval operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT entry_id, investigation_id, session_id, timestamp,
                       persona, findings, evidence, confidence_before,
                       confidence_after, memory_type, metadata_json
                FROM memory_entries
                WHERE entry_id = ?
            """,
                (entry_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Get memory references
            cursor.execute(
                """
                SELECT target_entry_id
                FROM memory_references
                WHERE source_entry_id = ?
            """,
                (entry_id,),
            )
            references = [r[0] for r in cursor.fetchall()]

            # Reconstruct MemoryEntry
            return MemoryEntry(
                investigation_id=row[1],
                session_id=row[2],
                timestamp=row[3],
                persona=row[4],
                findings=row[5],
                evidence=row[6] or "",
                confidence_before=row[7] or "exploring",
                confidence_after=row[8] or "exploring",
                memory_type=MemoryType(row[9]) if row[9] else MemoryType.FINDING,
                memory_references=references,
                metadata=json.loads(row[10]) if row[10] else {},
            )

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve entry {entry_id}: {e}")
            raise

    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Query persistent storage for entries matching filter criteria.

        Builds a SQL query based on the MemoryQuery filters and returns
        matching entries with pagination and sorting.

        Args:
            query: MemoryQuery with filter criteria

        Returns:
            List of matching MemoryEntry instances (may be empty)

        Raises:
            sqlite3.Error: If query operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build WHERE clauses
            where_clauses = []
            params = []

            if query.investigation_id:
                where_clauses.append("investigation_id = ?")
                params.append(query.investigation_id)

            if query.persona:
                where_clauses.append("persona = ?")
                params.append(query.persona)

            if query.memory_type:
                where_clauses.append("memory_type = ?")
                params.append(query.memory_type.value)

            if query.confidence_level:
                where_clauses.append("confidence_after >= ?")
                params.append(query.confidence_level)

            if query.time_range_start:
                where_clauses.append("timestamp >= ?")
                params.append(query.time_range_start)

            if query.time_range_end:
                where_clauses.append("timestamp <= ?")
                params.append(query.time_range_end)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Build ORDER BY clause
            order_by = query.sort_by
            if order_by not in ["timestamp", "confidence_after", "persona"]:
                order_by = "timestamp"
            sort_order = "DESC" if query.sort_order == "desc" else "ASC"

            # Execute query
            sql = f"""
                SELECT entry_id, investigation_id, session_id, timestamp,
                       persona, findings, evidence, confidence_before,
                       confidence_after, memory_type, metadata_json
                FROM memory_entries
                WHERE {where_sql}
                ORDER BY {order_by} {sort_order}
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit, query.offset])

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            # Convert rows to MemoryEntry instances
            results = []
            for row in rows:
                entry_id = row[0]

                # Get memory references for this entry
                cursor.execute(
                    """
                    SELECT target_entry_id
                    FROM memory_references
                    WHERE source_entry_id = ?
                """,
                    (entry_id,),
                )
                references = [r[0] for r in cursor.fetchall()]

                entry = MemoryEntry(
                    investigation_id=row[1],
                    session_id=row[2],
                    timestamp=row[3],
                    persona=row[4],
                    findings=row[5],
                    evidence=row[6] or "",
                    confidence_before=row[7] or "exploring",
                    confidence_after=row[8] or "exploring",
                    memory_type=MemoryType(row[9]) if row[9] else MemoryType.FINDING,
                    memory_references=references,
                    metadata=json.loads(row[10]) if row[10] else {},
                )
                results.append(entry)

            logger.debug(f"Query returned {len(results)} entries")
            return results

        except sqlite3.Error as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry from persistent storage.

        Args:
            entry_id: Unique identifier for the entry to delete

        Returns:
            True if entry was deleted, False if not found

        Raises:
            sqlite3.Error: If delete operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete references first (foreign key constraint)
            cursor.execute(
                "DELETE FROM memory_references WHERE source_entry_id = ? OR target_entry_id = ?",
                (entry_id, entry_id),
            )

            # Delete entry
            cursor.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.debug(f"Deleted entry {entry_id} from persistent storage")
            else:
                logger.debug(f"Entry {entry_id} not found for deletion")

            return deleted

        except sqlite3.Error as e:
            logger.error(f"Failed to delete entry {entry_id}: {e}")
            raise

    def get_metadata(self) -> MemoryMetadata:
        """
        Get storage statistics and metadata.

        Returns:
            MemoryMetadata with current storage statistics

        Raises:
            sqlite3.Error: If metadata retrieval fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get total entries
            cursor.execute("SELECT COUNT(*) FROM memory_entries")
            total_entries = cursor.fetchone()[0]

            # Get investigation count
            cursor.execute("SELECT COUNT(*) FROM investigations")
            investigation_count = cursor.fetchone()[0]

            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return MemoryMetadata(
                total_entries=total_entries,
                cache_entries=0,  # Persistence doesn't track cache
                persisted_entries=total_entries,
                investigation_count=investigation_count,
                storage_size_bytes=db_size,
            )

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            raise

    def close(self) -> None:
        """
        Close database connection.

        Safe to call multiple times.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection.

        Returns:
            sqlite3.Connection instance

        Note:
            Each call returns a new connection for thread safety.
            Caller is responsible for committing and closing.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
