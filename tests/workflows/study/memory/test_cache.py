"""
Tests for memory cache layer (ShortTermCache).

Verifies LRU eviction, cache promotion, metrics tracking,
and overall cache behavior.
"""

import pytest

from model_chorus.workflows.study.memory import (
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    ShortTermCache,
)


@pytest.fixture
def cache():
    """Create a ShortTermCache instance with small size for testing."""
    return ShortTermCache(max_size=3)


def test_lru_eviction(cache):
    """Test that LRU eviction works correctly when cache is full."""
    # Fill cache to capacity (3 entries)
    for i in range(3):
        entry = MemoryEntry(
            investigation_id=f"inv-{i}",
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        cache.put(f"entry-{i}", entry)

    assert cache.size() == 3

    # Access entry-0 to make it recently used
    cache.get("entry-0")

    # Add a 4th entry, should evict entry-1 (least recently used)
    entry = MemoryEntry(
        investigation_id="inv-3",
        session_id="sess-3",
        persona="researcher",
        findings="Finding 3",
    )
    cache.put("entry-3", entry)

    # Cache should still be size 3
    assert cache.size() == 3

    # entry-1 should be evicted
    assert cache.get("entry-1") is None

    # entry-0, entry-2, entry-3 should still be present
    assert cache.get("entry-0") is not None
    assert cache.get("entry-2") is not None
    assert cache.get("entry-3") is not None


def test_cache_metrics_hits_and_misses(cache):
    """Test that cache hit/miss metrics are tracked correctly."""
    entry = MemoryEntry(
        investigation_id="test-inv",
        session_id="test-sess",
        persona="researcher",
        findings="Test finding",
    )
    cache.put("entry-1", entry)

    # Get existing entry (should be a hit)
    cache.get("entry-1")

    # Get non-existing entry (should be a miss)
    cache.get("non-existent")

    # Check stats
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["puts"] == 1


def test_cache_eviction_metric(cache):
    """Test that eviction count is tracked correctly."""
    # Fill cache beyond capacity to trigger evictions
    for i in range(5):
        entry = MemoryEntry(
            investigation_id=f"inv-{i}",
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        cache.put(f"entry-{i}", entry)

    # Should have evicted 2 entries (5 - 3 = 2)
    stats = cache.get_stats()
    assert stats["evictions"] == 2


def test_cache_update_existing(cache):
    """Test updating an existing cache entry."""
    # Add entry
    entry1 = MemoryEntry(
        investigation_id="test-inv",
        session_id="test-sess",
        persona="researcher",
        findings="Original finding",
    )
    cache.put("entry-1", entry1)

    # Update with new content
    entry2 = MemoryEntry(
        investigation_id="test-inv",
        session_id="test-sess",
        persona="researcher",
        findings="Updated finding",
    )
    cache.put("entry-1", entry2)

    # Should still be size 1
    assert cache.size() == 1

    # Retrieved entry should have updated content
    retrieved = cache.get("entry-1")
    assert retrieved.findings == "Updated finding"


def test_cache_delete(cache):
    """Test deleting entries from cache."""
    entry = MemoryEntry(
        investigation_id="test-inv",
        session_id="test-sess",
        persona="researcher",
        findings="To be deleted",
    )
    cache.put("entry-1", entry)

    assert cache.size() == 1

    # Delete
    deleted = cache.delete("entry-1")
    assert deleted is True
    assert cache.size() == 0

    # Delete non-existent
    deleted = cache.delete("entry-1")
    assert deleted is False


def test_cache_query(cache):
    """Test querying cache with filters."""
    # Add entries with different attributes
    entries = [
        ("entry-1", "inv-1", "researcher"),
        ("entry-2", "inv-1", "critic"),
        ("entry-3", "inv-2", "researcher"),
    ]

    for entry_id, inv_id, persona in entries:
        entry = MemoryEntry(
            investigation_id=inv_id,
            session_id="test-sess",
            persona=persona,
            findings=f"Finding from {persona}",
        )
        cache.put(entry_id, entry)

    # Query by investigation
    results = cache.query(MemoryQuery(investigation_id="inv-1"))
    assert len(results) == 2

    # Query by persona
    results = cache.query(MemoryQuery(persona="researcher"))
    assert len(results) == 2

    # Query with multiple filters
    results = cache.query(
        MemoryQuery(investigation_id="inv-1", persona="researcher")
    )
    assert len(results) == 1
    assert results[0].persona == "researcher"


def test_cache_clear(cache):
    """Test clearing all entries from cache."""
    # Add multiple entries
    for i in range(3):
        entry = MemoryEntry(
            investigation_id=f"inv-{i}",
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        cache.put(f"entry-{i}", entry)

    assert cache.size() == 3

    # Clear cache
    count = cache.clear()
    assert count == 3
    assert cache.size() == 0


def test_cache_metadata(cache):
    """Test retrieving cache metadata."""
    # Add some entries
    for i in range(2):
        entry = MemoryEntry(
            investigation_id=f"inv-{i}",
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        cache.put(f"entry-{i}", entry)

    # Generate some hits and misses
    cache.get("entry-0")  # hit
    cache.get("entry-1")  # hit
    cache.get("missing")  # miss

    metadata = cache.get_metadata()
    assert metadata.cache_entries == 2
    assert metadata.cache_hit_rate == 2 / 3  # 2 hits out of 3 total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
