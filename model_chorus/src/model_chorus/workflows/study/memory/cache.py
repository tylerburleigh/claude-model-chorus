"""
In-memory short-term cache for STUDY workflow memory system.

This module implements a fast, LRU-based cache for active investigation memory entries.
The cache provides quick access to recent findings and context without persistence overhead.

Architecture:
- LRU (Least Recently Used) eviction policy using OrderedDict
- Configurable maximum size (default 100 entries)
- Thread-safe operations for concurrent access
- Statistics tracking (hits, misses, evictions)
"""

import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional
from datetime import datetime, timezone

from .models import MemoryEntry, MemoryMetadata, MemoryQuery

logger = logging.getLogger(__name__)


class ShortTermCache:
    """
    LRU-based in-memory cache for memory entries.

    Provides fast access to recent memory entries with automatic eviction
    of least recently used entries when the cache reaches capacity.

    The cache uses OrderedDict to maintain insertion/access order and
    implements LRU eviction by moving accessed items to the end and
    removing items from the beginning when full.

    Thread Safety:
        All operations are protected by a threading lock to ensure
        safe concurrent access from multiple personas/threads.

    Attributes:
        max_size: Maximum number of entries to cache (default 100)
        cache: OrderedDict storing entry_id -> MemoryEntry mappings
        stats: Statistics tracking cache performance
        lock: Threading lock for thread-safe operations

    Example:
        >>> cache = ShortTermCache(max_size=50)
        >>> cache.put(entry_id, memory_entry)
        >>> entry = cache.get(entry_id)
        >>> results = cache.query(MemoryQuery(persona="researcher"))
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize the short-term memory cache.

        Args:
            max_size: Maximum number of entries to cache (must be > 0)

        Raises:
            ValueError: If max_size <= 0
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.max_size = max_size
        self.cache: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.lock = threading.Lock()

        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "puts": 0,
            "deletes": 0,
        }

        logger.info(f"ShortTermCache initialized with max_size={max_size}")

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry from cache.

        Accessing an entry moves it to the end of the OrderedDict,
        marking it as recently used for LRU eviction.

        Args:
            entry_id: Unique identifier for the memory entry

        Returns:
            MemoryEntry if found, None otherwise

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            if entry_id in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(entry_id)
                self.stats["hits"] += 1
                logger.debug(f"Cache hit for entry_id={entry_id}")
                return self.cache[entry_id]
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for entry_id={entry_id}")
                return None

    def put(self, entry_id: str, entry: MemoryEntry) -> None:
        """
        Store a memory entry in cache.

        If the cache is at capacity, the least recently used entry
        (first item in OrderedDict) is evicted before adding the new entry.

        Args:
            entry_id: Unique identifier for the memory entry
            entry: MemoryEntry instance to cache

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            # If entry already exists, update it and move to end
            if entry_id in self.cache:
                self.cache.move_to_end(entry_id)
                self.cache[entry_id] = entry
                logger.debug(f"Updated existing cache entry: {entry_id}")
            else:
                # Check if we need to evict
                if len(self.cache) >= self.max_size:
                    # Remove least recently used (first item)
                    evicted_id, _ = self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
                    logger.debug(
                        f"Evicted LRU entry {evicted_id} (cache full at {self.max_size})"
                    )

                # Add new entry at end
                self.cache[entry_id] = entry
                self.stats["puts"] += 1
                logger.debug(f"Added new cache entry: {entry_id}")

    def delete(self, entry_id: str) -> bool:
        """
        Remove a memory entry from cache.

        Args:
            entry_id: Unique identifier for the memory entry to remove

        Returns:
            True if entry was found and removed, False otherwise

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            if entry_id in self.cache:
                del self.cache[entry_id]
                self.stats["deletes"] += 1
                logger.debug(f"Deleted cache entry: {entry_id}")
                return True
            else:
                logger.debug(f"Delete failed, entry not found: {entry_id}")
                return False

    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Query cache for entries matching filter criteria.

        Iterates through all cached entries and returns those matching
        the query filters. Results are sorted according to query.sort_by
        and query.sort_order, then paginated with query.limit and query.offset.

        Args:
            query: MemoryQuery with filter criteria

        Returns:
            List of matching MemoryEntry instances (may be empty)

        Thread Safety:
            This operation is thread-safe.

        Note:
            This performs a full scan of the cache. For large result sets
            or complex queries, consider using the persistence layer instead.
        """
        with self.lock:
            results = []

            # Filter entries
            for entry in self.cache.values():
                if self._matches_query(entry, query):
                    results.append(entry)

            # Sort results
            reverse = query.sort_order == "desc"
            if query.sort_by == "timestamp":
                results.sort(key=lambda e: e.timestamp, reverse=reverse)
            elif query.sort_by == "confidence_after":
                results.sort(key=lambda e: e.confidence_after, reverse=reverse)
            # Add more sort fields as needed

            # Paginate
            start = query.offset
            end = start + query.limit
            paginated = results[start:end]

            logger.debug(
                f"Query matched {len(results)} entries, returning {len(paginated)} after pagination"
            )
            return paginated

    def _matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        """
        Check if a memory entry matches query filter criteria.

        Args:
            entry: MemoryEntry to check
            query: MemoryQuery with filter criteria

        Returns:
            True if entry matches all non-None query filters, False otherwise
        """
        # Filter by investigation_id
        if query.investigation_id and entry.investigation_id != query.investigation_id:
            return False

        # Filter by persona
        if query.persona and entry.persona != query.persona:
            return False

        # Filter by memory_type
        if query.memory_type and entry.memory_type != query.memory_type:
            return False

        # Filter by confidence_level (minimum)
        if query.confidence_level:
            # Note: This assumes confidence levels can be compared
            # May need more sophisticated comparison logic
            if entry.confidence_after < query.confidence_level:
                return False

        # Filter by time range
        if query.time_range_start and entry.timestamp < query.time_range_start:
            return False
        if query.time_range_end and entry.timestamp > query.time_range_end:
            return False

        return True

    def clear(self) -> int:
        """
        Remove all entries from cache.

        Returns:
            Number of entries that were cleared

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {count} entries from cache")
            return count

    def size(self) -> int:
        """
        Get current number of entries in cache.

        Returns:
            Number of entries currently cached

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            return len(self.cache)

    def get_metadata(self) -> MemoryMetadata:
        """
        Get cache statistics and metadata.

        Returns:
            MemoryMetadata with current cache statistics

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return MemoryMetadata(
                total_entries=len(self.cache),
                cache_entries=len(self.cache),
                persisted_entries=0,  # Cache doesn't track persistence
                cache_hit_rate=hit_rate,
            )

    def get_stats(self) -> Dict[str, int]:
        """
        Get detailed cache statistics.

        Returns:
            Dictionary with cache operation counts

        Thread Safety:
            This operation is thread-safe.
        """
        with self.lock:
            return self.stats.copy()
