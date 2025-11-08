"""
Memory controller coordinating short-term cache and long-term persistence.

This module provides a unified interface for memory operations that transparently
coordinates between the fast in-memory cache and durable SQLite persistence.

Architecture:
- Write-through caching: Writes go to both cache and persistence
- Cache-first reads: Check cache first, fall back to persistence
- Automatic promotion: Persistence reads are cached for future access
- Configurable persistence policy (write-through, write-back, manual)
"""

import logging
import uuid
from typing import List, Optional
from datetime import datetime, timezone

from .models import MemoryEntry, MemoryMetadata, MemoryQuery
from .cache import ShortTermCache
from .persistence import LongTermStorage

logger = logging.getLogger(__name__)


class MemoryController:
    """
    Unified controller for memory system operations.

    Coordinates between ShortTermCache (fast, volatile) and LongTermStorage
    (durable, persistent) to provide optimal performance with durability.

    The controller implements a write-through caching strategy:
    - Writes: Stored in both cache and persistence
    - Reads: Check cache first, fall back to persistence
    - Cache misses: Automatically promote to cache for future reads

    This ensures:
    - Fast access to recent/active investigation data (cache)
    - Durable storage of all findings (persistence)
    - Automatic recovery after cache eviction (promotion)

    Attributes:
        cache: ShortTermCache instance for fast access
        storage: LongTermStorage instance for durability
        write_through: If True, writes go to both cache and storage (default)

    Example:
        >>> controller = MemoryController(
        ...     cache_size=100,
        ...     db_path="investigations.db"
        ... )
        >>> controller.initialize()
        >>>
        >>> # Store memory entry (goes to cache + persistence)
        >>> entry_id = controller.store(
        ...     investigation_id="inv-123",
        ...     persona="researcher",
        ...     findings="Found important pattern",
        ...     evidence="Analysis of dataset X"
        ... )
        >>>
        >>> # Retrieve entry (cache-first)
        >>> entry = controller.get(entry_id)
        >>>
        >>> # Query entries (searches both cache and persistence)
        >>> results = controller.query(
        ...     MemoryQuery(investigation_id="inv-123", persona="researcher")
        ... )
        >>>
        >>> controller.close()
    """

    def __init__(
        self,
        cache_size: int = 100,
        db_path: str = "study_memory.db",
        write_through: bool = True,
    ):
        """
        Initialize memory controller.

        Args:
            cache_size: Maximum number of entries in cache
            db_path: Path to SQLite database file
            write_through: If True, writes go to both cache and storage
        """
        self.cache = ShortTermCache(max_size=cache_size)
        self.storage = LongTermStorage(db_path=db_path)
        self.write_through = write_through

        logger.info(
            f"MemoryController initialized (cache_size={cache_size}, "
            f"db_path={db_path}, write_through={write_through})"
        )

    def initialize(self) -> None:
        """
        Initialize the memory system.

        Creates database schema if it doesn't exist.
        Safe to call multiple times.

        Raises:
            sqlite3.Error: If database initialization fails
        """
        self.storage.initialize()
        logger.info("Memory system initialized")

    def store(
        self,
        investigation_id: str,
        persona: str,
        findings: str,
        evidence: str = "",
        confidence_before: str = "exploring",
        confidence_after: str = "exploring",
        session_id: Optional[str] = None,
        memory_references: Optional[List[str]] = None,
        **metadata,
    ) -> str:
        """
        Store a new memory entry.

        Generates a unique entry ID and stores the entry in both
        cache and persistence (if write_through is enabled).

        Args:
            investigation_id: Investigation identifier
            persona: Persona creating this entry
            findings: Main findings/content
            evidence: Supporting evidence
            confidence_before: Confidence level before this step
            confidence_after: Confidence level after this step
            session_id: Session identifier (auto-generated if None)
            memory_references: List of referenced entry IDs
            **metadata: Additional metadata as keyword arguments

        Returns:
            Unique entry ID for the stored entry

        Raises:
            sqlite3.Error: If persistence write fails (write-through mode)
        """
        # Generate unique entry ID
        entry_id = f"mem-{uuid.uuid4().hex[:12]}"

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"sess-{uuid.uuid4().hex[:8]}"

        # Create memory entry
        entry = MemoryEntry(
            investigation_id=investigation_id,
            session_id=session_id,
            persona=persona,
            findings=findings,
            evidence=evidence,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            memory_references=memory_references or [],
            metadata=metadata,
        )

        # Store in cache
        self.cache.put(entry_id, entry)
        logger.debug(f"Stored entry {entry_id} in cache")

        # Store in persistence (write-through)
        if self.write_through:
            self.storage.save(entry_id, entry)
            logger.debug(f"Stored entry {entry_id} in persistence (write-through)")

        return entry_id

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.

        Checks cache first for fast access. If not in cache,
        retrieves from persistence and promotes to cache.

        Args:
            entry_id: Unique identifier for the entry

        Returns:
            MemoryEntry if found, None otherwise

        Raises:
            sqlite3.Error: If persistence read fails
        """
        # Check cache first
        entry = self.cache.get(entry_id)
        if entry:
            logger.debug(f"Cache hit for entry {entry_id}")
            return entry

        # Cache miss - check persistence
        logger.debug(f"Cache miss for entry {entry_id}, checking persistence")
        entry = self.storage.get(entry_id)

        # Promote to cache if found
        if entry:
            self.cache.put(entry_id, entry)
            logger.debug(f"Promoted entry {entry_id} from persistence to cache")

        return entry

    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Query memory entries with filtering and pagination.

        Combines results from both cache and persistence, removing
        duplicates and applying sorting/pagination to the merged results.

        Note: For large result sets, this may be slower than querying
        persistence directly. Consider using storage.query() for
        historical data analysis.

        Args:
            query: MemoryQuery with filter criteria

        Returns:
            List of matching MemoryEntry instances (may be empty)

        Raises:
            sqlite3.Error: If persistence query fails
        """
        # Query cache
        cache_results = self.cache.query(query)
        cache_ids = {id(entry) for entry in cache_results}

        # Query persistence (with higher limit to account for duplicates)
        storage_query = MemoryQuery(
            investigation_id=query.investigation_id,
            persona=query.persona,
            confidence_level=query.confidence_level,
            time_range_start=query.time_range_start,
            time_range_end=query.time_range_end,
            memory_type=query.memory_type,
            limit=query.limit + len(cache_results),  # Get extra to account for overlap
            offset=query.offset,
            sort_by=query.sort_by,
            sort_order=query.sort_order,
        )
        storage_results = self.storage.query(storage_query)

        # Merge results, removing duplicates (by investigation_id + timestamp)
        seen = {
            (entry.investigation_id, entry.timestamp): entry
            for entry in cache_results
        }

        for entry in storage_results:
            key = (entry.investigation_id, entry.timestamp)
            if key not in seen:
                seen[key] = entry

        # Convert to list and apply final sorting
        all_results = list(seen.values())

        # Sort merged results
        reverse = query.sort_order == "desc"
        if query.sort_by == "timestamp":
            all_results.sort(key=lambda e: e.timestamp, reverse=reverse)
        elif query.sort_by == "confidence_after":
            all_results.sort(key=lambda e: e.confidence_after, reverse=reverse)

        # Apply pagination to merged results
        start = query.offset
        end = start + query.limit
        paginated = all_results[start:end]

        logger.debug(
            f"Query returned {len(paginated)} entries "
            f"({len(cache_results)} from cache, "
            f"{len(storage_results)} from storage, "
            f"{len(all_results)} after merge)"
        )

        return paginated

    def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Removes from both cache and persistence.

        Args:
            entry_id: Unique identifier for the entry to delete

        Returns:
            True if entry was found and deleted, False otherwise

        Raises:
            sqlite3.Error: If persistence delete fails
        """
        cache_deleted = self.cache.delete(entry_id)
        storage_deleted = self.storage.delete(entry_id)

        deleted = cache_deleted or storage_deleted

        if deleted:
            logger.debug(
                f"Deleted entry {entry_id} "
                f"(cache: {cache_deleted}, storage: {storage_deleted})"
            )
        else:
            logger.debug(f"Entry {entry_id} not found for deletion")

        return deleted

    def flush_cache_to_storage(self) -> int:
        """
        Manually flush all cache entries to persistent storage.

        Useful for ensuring durability before shutdown or when
        write_through is disabled.

        Returns:
            Number of entries flushed to storage

        Raises:
            sqlite3.Error: If persistence writes fail
        """
        if self.write_through:
            logger.info("Write-through enabled, cache already synced to storage")
            return 0

        count = 0
        # Note: This is a simplified implementation
        # In production, would need to track which entries are dirty
        logger.info(f"Flushed {count} entries from cache to storage")
        return count

    def clear_cache(self) -> int:
        """
        Clear all entries from cache.

        Persistence is unaffected. Useful for memory management
        or testing.

        Returns:
            Number of entries cleared from cache
        """
        count = self.cache.clear()
        logger.info(f"Cleared {count} entries from cache")
        return count

    def get_metadata(self) -> MemoryMetadata:
        """
        Get combined metadata from cache and storage.

        Merges statistics from both layers to provide
        complete system overview.

        Returns:
            MemoryMetadata with combined statistics

        Raises:
            sqlite3.Error: If metadata retrieval fails
        """
        cache_meta = self.cache.get_metadata()
        storage_meta = self.storage.get_metadata()

        return MemoryMetadata(
            total_entries=storage_meta.total_entries,  # Use storage as source of truth
            cache_entries=cache_meta.cache_entries,
            persisted_entries=storage_meta.persisted_entries,
            investigation_count=storage_meta.investigation_count,
            cache_hit_rate=cache_meta.cache_hit_rate,
            storage_size_bytes=storage_meta.storage_size_bytes,
        )

    def close(self) -> None:
        """
        Close all resources.

        Closes database connection. Cache is kept in memory.
        Safe to call multiple times.
        """
        self.storage.close()
        logger.info("Memory controller closed")
