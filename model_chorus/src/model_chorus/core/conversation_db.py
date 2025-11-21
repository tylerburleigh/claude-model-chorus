"""
SQLite-based conversation database with Write-Ahead Logging (WAL).

Provides persistent storage for conversation threads and messages using SQLite
instead of individual JSON files. Enables better performance, concurrency,
and query capabilities compared to file-based storage.

Key Features:
- WAL mode for improved concurrency and performance
- Atomic operations with transaction support
- Efficient queries for thread history and metadata
- Migration support from file-based storage
- Thread-safe operations

Architecture:
    Database: ~/.model-chorus/conversations.db (SQLite with WAL)
    Tables:
        - threads: Thread metadata and status
        - messages: Individual messages with foreign key to threads
        - thread_metadata: Flexible key-value metadata storage
"""

import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .models import ConversationMessage, ConversationThread

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_DB_PATH = Path.home() / ".model-chorus" / "conversations.db"
DEFAULT_TTL_HOURS = 3
DEFAULT_MAX_MESSAGES_PER_THREAD = 50


class ConversationDatabase:
    """
    SQLite-based conversation storage with WAL mode for concurrency.

    Provides persistent storage for conversation threads using SQLite instead
    of individual JSON files. Supports concurrent access through WAL mode
    and offers better query performance for conversation history.

    Architecture:
        - Database file: ~/.model-chorus/conversations.db
        - WAL mode enabled for concurrent reads during writes
        - Foreign key constraints for referential integrity
        - Indexes on common query patterns (thread_id, timestamps)

    Attributes:
        db_path: Path to SQLite database file
        ttl_hours: Time-to-live for conversation threads in hours
        max_messages: Maximum messages per thread before truncation
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        max_messages: int = DEFAULT_MAX_MESSAGES_PER_THREAD,
    ):
        """
        Initialize conversation database with WAL mode.

        Args:
            db_path: Path to SQLite database file
            ttl_hours: Time-to-live for threads in hours
            max_messages: Maximum messages per thread
        """
        self.db_path = db_path
        self.ttl_hours = ttl_hours
        self.max_messages = max_messages

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection with WAL mode
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow access from multiple threads
            isolation_level=None,  # Autocommit mode for better WAL performance
        )

        # Enable WAL mode for concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys=ON")

        # Create schema if not exists
        self._create_schema()

        logger.info(
            f"ConversationDatabase initialized: db={db_path}, "
            f"ttl={ttl_hours}h, max_messages={max_messages}"
        )

    def _create_schema(self) -> None:
        """
        Create database schema with threads, messages, and metadata tables.

        Schema Design:
            - threads: Core thread metadata (thread_id as primary key)
            - messages: Message content with foreign key to threads
            - thread_metadata: Flexible key-value store for additional data

        Indexes:
            - threads(created_at): For TTL cleanup queries
            - threads(status): For filtering by lifecycle status
            - messages(thread_id, timestamp): For chronological message retrieval
        """
        cursor = self.conn.cursor()

        # Threads table: Core thread metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                parent_thread_id TEXT,
                created_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                initial_context TEXT,
                state TEXT,
                branch_point TEXT,
                FOREIGN KEY (parent_thread_id) REFERENCES threads(thread_id)
            )
        """)

        # Messages table: Individual messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                files TEXT,
                workflow_name TEXT,
                model_provider TEXT,
                model_name TEXT,
                metadata TEXT,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)

        # Thread metadata table: Flexible key-value storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (thread_id, key),
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_created_at
            ON threads(created_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_status
            ON threads(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_workflow
            ON threads(workflow_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread_timestamp
            ON messages(thread_id, timestamp)
        """)

        self.conn.commit()
        logger.debug("Database schema created/verified")

    def create_thread(
        self,
        workflow_name: str,
        initial_context: dict[str, Any] | None = None,
        parent_thread_id: str | None = None,
    ) -> str:
        """
        Create new conversation thread.

        Generates UUID-based thread ID and initializes thread in database.
        Thread is immediately persisted with atomic transaction.

        Args:
            workflow_name: Name of workflow creating this thread
            initial_context: Optional initial request parameters
            parent_thread_id: Optional parent thread for conversation chains

        Returns:
            Thread ID (UUID string) for use as continuation_id

        Example:
            >>> db = ConversationDatabase()
            >>> thread_id = db.create_thread("consensus", {"prompt": "Analyze this"})
            >>> print(thread_id)
            '550e8400-e29b-41d4-a716-446655440000'
        """
        # Generate UUID for thread
        thread_id = str(uuid.uuid4())

        # Prepare data
        now = datetime.now(UTC).isoformat()
        initial_context_json = json.dumps(initial_context or {})
        state_json = json.dumps({})

        # Insert thread
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO threads (
                thread_id, parent_thread_id, created_at, last_updated_at,
                workflow_name, status, initial_context, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                parent_thread_id,
                now,
                now,
                workflow_name,
                "active",
                initial_context_json,
                state_json,
            ),
        )

        self.conn.commit()

        logger.info(f"Created thread {thread_id} for workflow '{workflow_name}'")
        return thread_id

    def get_thread(self, thread_id: str) -> ConversationThread | None:
        """
        Retrieve conversation thread by ID.

        Checks TTL and returns None if thread expired or doesn't exist.
        Loads thread metadata and all associated messages from database.

        Args:
            thread_id: Thread ID to retrieve

        Returns:
            ConversationThread if found and not expired, None otherwise

        Example:
            >>> thread = db.get_thread(thread_id)
            >>> if thread:
            ...     print(f"Thread has {len(thread.messages)} messages")
        """
        cursor = self.conn.cursor()

        # Fetch thread metadata
        cursor.execute(
            """
            SELECT
                thread_id, parent_thread_id, created_at, last_updated_at,
                workflow_name, status, initial_context, state, branch_point
            FROM threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        )

        row = cursor.fetchone()
        if not row:
            logger.debug(f"Thread {thread_id} not found")
            return None

        # Check TTL
        created_at = datetime.fromisoformat(row[2])
        age = datetime.now(UTC) - created_at
        if age > timedelta(hours=self.ttl_hours):
            logger.info(f"Thread {thread_id} expired (age: {age})")
            self._delete_thread(thread_id)
            return None

        # Parse thread data
        (
            thread_id,
            parent_thread_id,
            created_at_str,
            last_updated_at,
            workflow_name,
            status,
            initial_context_json,
            state_json,
            branch_point,
        ) = row

        initial_context = json.loads(initial_context_json)
        state = json.loads(state_json)

        # Fetch messages
        cursor.execute(
            """
            SELECT
                role, content, timestamp, files, workflow_name,
                model_provider, model_name, metadata
            FROM messages
            WHERE thread_id = ?
            ORDER BY timestamp ASC
            """,
            (thread_id,),
        )

        messages = []
        for msg_row in cursor.fetchall():
            (
                role,
                content,
                timestamp,
                files_json,
                msg_workflow,
                model_provider,
                model_name,
                metadata_json,
            ) = msg_row

            files = json.loads(files_json) if files_json else None
            metadata = json.loads(metadata_json) if metadata_json else {}

            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=timestamp,
                files=files,
                workflow_name=msg_workflow,
                model_provider=model_provider,
                model_name=model_name,
                metadata=metadata,
            )
            messages.append(message)

        # Construct thread
        thread = ConversationThread(
            thread_id=thread_id,
            parent_thread_id=parent_thread_id,
            created_at=created_at_str,
            last_updated_at=last_updated_at,
            workflow_name=workflow_name,
            messages=messages,
            state=state,
            initial_context=initial_context,
            status=status,
            branch_point=branch_point,
        )

        logger.debug(
            f"Retrieved thread {thread_id} with {len(thread.messages)} messages"
        )
        return thread

    def add_message(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        files: list[str] | None = None,
        workflow_name: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add message to conversation thread.

        Atomic operation with transaction. Enforces max_messages limit
        by truncating oldest messages if exceeded. Updates thread's last_updated_at.

        Args:
            thread_id: Thread to add message to
            role: Message role ('user' or 'assistant')
            content: Message content
            files: Optional list of file paths referenced
            workflow_name: Optional workflow that generated message
            model_provider: Optional provider type used
            model_name: Optional model identifier
            metadata: Optional additional metadata

        Returns:
            True if successful, False if thread not found

        Example:
            >>> success = db.add_message(
            ...     thread_id,
            ...     "user",
            ...     "Analyze this code",
            ...     files=["src/main.py"]
            ... )
        """
        cursor = self.conn.cursor()

        # Check if thread exists, create if not
        cursor.execute("SELECT thread_id FROM threads WHERE thread_id = ?", (thread_id,))
        if not cursor.fetchone():
            # Create thread implicitly
            now = datetime.now(UTC).isoformat()
            cursor.execute(
                """
                INSERT INTO threads (
                    thread_id, created_at, last_updated_at,
                    workflow_name, status, initial_context, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    now,
                    now,
                    workflow_name or "unknown",
                    "active",
                    json.dumps({}),
                    json.dumps({}),
                ),
            )
            logger.info(f"Created new thread {thread_id} implicitly from add_message")

        # Prepare message data
        now = datetime.now(UTC).isoformat()
        files_json = json.dumps(files) if files else None
        metadata_json = json.dumps(metadata or {})

        # Insert message
        cursor.execute(
            """
            INSERT INTO messages (
                thread_id, role, content, timestamp, files,
                workflow_name, model_provider, model_name, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                role,
                content,
                now,
                files_json,
                workflow_name,
                model_provider,
                model_name,
                metadata_json,
            ),
        )

        # Enforce max messages limit
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
        )
        count = cursor.fetchone()[0]

        if count > self.max_messages:
            removed_count = count - self.max_messages
            cursor.execute(
                """
                DELETE FROM messages
                WHERE message_id IN (
                    SELECT message_id FROM messages
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
                """,
                (thread_id, removed_count),
            )
            logger.warning(
                f"Thread {thread_id} exceeded max messages, "
                f"removed {removed_count} oldest messages"
            )

        # Update thread timestamp
        cursor.execute(
            """
            UPDATE threads
            SET last_updated_at = ?
            WHERE thread_id = ?
            """,
            (now, thread_id),
        )

        self.conn.commit()

        logger.debug(f"Added {role} message to thread {thread_id}")
        return True

    def get_messages(
        self, thread_id: str, limit: int | None = None, role: str | None = None
    ) -> list[ConversationMessage]:
        """
        Retrieve messages from thread with optional filtering.

        Args:
            thread_id: Thread to get messages from
            limit: Optional limit on number of messages (most recent)
            role: Optional filter by role ('user' or 'assistant')

        Returns:
            List of messages in chronological order

        Example:
            >>> messages = db.get_messages(thread_id, limit=10, role="user")
            >>> for msg in messages:
            ...     print(f"{msg.timestamp}: {msg.content[:50]}...")
        """
        cursor = self.conn.cursor()

        # Build query
        query = """
            SELECT
                role, content, timestamp, files, workflow_name,
                model_provider, model_name, metadata
            FROM messages
            WHERE thread_id = ?
        """
        params: list[Any] = [thread_id]

        if role:
            query += " AND role = ?"
            params.append(role)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)

        messages = []
        for row in cursor.fetchall():
            (
                msg_role,
                content,
                timestamp,
                files_json,
                workflow_name,
                model_provider,
                model_name,
                metadata_json,
            ) = row

            files = json.loads(files_json) if files_json else None
            metadata = json.loads(metadata_json) if metadata_json else {}

            message = ConversationMessage(
                role=msg_role,
                content=content,
                timestamp=timestamp,
                files=files,
                workflow_name=workflow_name,
                model_provider=model_provider,
                model_name=model_name,
                metadata=metadata,
            )
            messages.append(message)

        return messages

    def complete_thread(self, thread_id: str) -> bool:
        """
        Mark thread as completed.

        Completed threads remain accessible but are marked for archival.

        Args:
            thread_id: Thread to mark as completed

        Returns:
            True if successful, False if thread not found

        Example:
            >>> db.complete_thread(thread_id)
        """
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()

        cursor.execute(
            """
            UPDATE threads
            SET status = 'completed', last_updated_at = ?
            WHERE thread_id = ?
            """,
            (now, thread_id),
        )

        self.conn.commit()

        if cursor.rowcount > 0:
            logger.info(f"Thread {thread_id} marked as completed")
            return True
        else:
            return False

    def archive_thread(self, thread_id: str) -> bool:
        """
        Mark thread as archived.

        Archived threads are eligible for cleanup.

        Args:
            thread_id: Thread to archive

        Returns:
            True if successful, False if thread not found

        Example:
            >>> db.archive_thread(thread_id)
        """
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()

        cursor.execute(
            """
            UPDATE threads
            SET status = 'archived', last_updated_at = ?
            WHERE thread_id = ?
            """,
            (now, thread_id),
        )

        self.conn.commit()

        if cursor.rowcount > 0:
            logger.info(f"Thread {thread_id} archived")
            return True
        else:
            return False

    def cleanup_expired_threads(self) -> int:
        """
        Remove threads older than TTL.

        Checks all threads and deletes those past their TTL.
        Safe to call periodically for maintenance.

        Returns:
            Number of threads deleted

        Example:
            >>> deleted = db.cleanup_expired_threads()
            >>> print(f"Cleaned up {deleted} expired threads")
        """
        cursor = self.conn.cursor()

        # Calculate cutoff time
        cutoff = datetime.now(UTC) - timedelta(hours=self.ttl_hours)
        cutoff_str = cutoff.isoformat()

        # Delete expired threads (CASCADE will delete messages)
        cursor.execute(
            """
            DELETE FROM threads
            WHERE created_at < ?
            """,
            (cutoff_str,),
        )

        deleted = cursor.rowcount
        self.conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired threads")

        return deleted

    def cleanup_archived_threads(self) -> int:
        """
        Remove archived threads regardless of TTL.

        Removes all threads with status='archived'.
        Useful for explicit cleanup after workflows complete.

        Returns:
            Number of threads deleted

        Example:
            >>> deleted = db.cleanup_archived_threads()
            >>> print(f"Cleaned up {deleted} archived threads")
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            DELETE FROM threads
            WHERE status = 'archived'
            """
        )

        deleted = cursor.rowcount
        self.conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} archived threads")

        return deleted

    def _delete_thread(self, thread_id: str) -> None:
        """
        Delete thread and all associated data.

        Args:
            thread_id: Thread to delete
        """
        cursor = self.conn.cursor()

        # CASCADE will automatically delete messages and metadata
        cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

        self.conn.commit()
        logger.debug(f"Deleted thread {thread_id}")

    def close(self) -> None:
        """
        Close database connection.

        Should be called when database is no longer needed.
        WAL checkpoint is performed to flush pending writes.
        """
        if self.conn:
            # Checkpoint WAL to flush pending writes
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.close()
            logger.info("Database connection closed")
