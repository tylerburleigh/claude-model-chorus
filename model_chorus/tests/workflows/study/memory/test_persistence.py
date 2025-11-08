"""
Tests for memory persistence layer (LongTermStorage).

Verifies that memory entries are correctly persisted to SQLite,
retrieved accurately, and survive across sessions.
"""

import pytest
import tempfile
import os
from pathlib import Path

from model_chorus.workflows.study.memory import (
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    LongTermStorage,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def storage(temp_db):
    """Create and initialize a LongTermStorage instance."""
    store = LongTermStorage(db_path=temp_db)
    store.initialize()
    yield store
    store.close()


def test_save_and_retrieve(storage):
    """Test that entries can be saved and retrieved correctly."""
    # Create test entry
    entry = MemoryEntry(
        investigation_id="test-inv-001",
        session_id="test-sess-001",
        persona="researcher",
        findings="Test finding",
        evidence="Test evidence",
        confidence_before="low",
        confidence_after="high",
        memory_type=MemoryType.FINDING,
    )

    # Save entry
    entry_id = "test-entry-001"
    storage.save(entry_id, entry)

    # Retrieve entry
    retrieved = storage.get(entry_id)

    # Verify
    assert retrieved is not None
    assert retrieved.investigation_id == "test-inv-001"
    assert retrieved.persona == "researcher"
    assert retrieved.findings == "Test finding"
    assert retrieved.confidence_after == "high"


def test_persistence_across_sessions(temp_db):
    """Test that data persists across storage instances (sessions)."""
    entry_id = "persist-test-001"
    entry = MemoryEntry(
        investigation_id="persist-inv-001",
        session_id="persist-sess-001",
        persona="critic",
        findings="Persistent finding",
    )

    # First session: save
    storage1 = LongTermStorage(db_path=temp_db)
    storage1.initialize()
    storage1.save(entry_id, entry)
    storage1.close()

    # Second session: retrieve
    storage2 = LongTermStorage(db_path=temp_db)
    storage2.initialize()
    retrieved = storage2.get(entry_id)
    storage2.close()

    # Verify data persisted
    assert retrieved is not None
    assert retrieved.findings == "Persistent finding"
    assert retrieved.persona == "critic"


def test_query_by_investigation(storage):
    """Test querying entries by investigation ID."""
    # Create multiple entries for different investigations
    for i in range(3):
        entry = MemoryEntry(
            investigation_id=f"inv-{i % 2}",  # Alternate between inv-0 and inv-1
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        storage.save(f"entry-{i}", entry)

    # Query for inv-0
    query = MemoryQuery(investigation_id="inv-0")
    results = storage.query(query)

    # Should get 2 entries (0 and 2)
    assert len(results) == 2
    assert all(r.investigation_id == "inv-0" for r in results)


def test_query_by_persona(storage):
    """Test querying entries by persona."""
    # Create entries with different personas
    personas = ["researcher", "critic", "planner"]
    for i, persona in enumerate(personas):
        entry = MemoryEntry(
            investigation_id="test-inv",
            session_id=f"sess-{i}",
            persona=persona,
            findings=f"Finding from {persona}",
        )
        storage.save(f"entry-{i}", entry)

    # Query for researcher
    query = MemoryQuery(persona="researcher")
    results = storage.query(query)

    assert len(results) == 1
    assert results[0].persona == "researcher"


def test_delete(storage):
    """Test deleting entries."""
    entry = MemoryEntry(
        investigation_id="delete-test",
        session_id="delete-sess",
        persona="researcher",
        findings="To be deleted",
    )

    entry_id = "delete-me"
    storage.save(entry_id, entry)

    # Verify it exists
    assert storage.get(entry_id) is not None

    # Delete it
    deleted = storage.delete(entry_id)
    assert deleted is True

    # Verify it's gone
    assert storage.get(entry_id) is None

    # Delete again should return False
    assert storage.delete(entry_id) is False


def test_memory_references(storage):
    """Test that memory references are saved and retrieved."""
    # Create entry with references
    entry = MemoryEntry(
        investigation_id="ref-test",
        session_id="ref-sess",
        persona="researcher",
        findings="Finding with references",
        memory_references=["ref-1", "ref-2", "ref-3"],
    )

    entry_id = "entry-with-refs"
    storage.save(entry_id, entry)

    # Retrieve and verify references
    retrieved = storage.get(entry_id)
    assert retrieved is not None
    assert len(retrieved.memory_references) == 3
    assert "ref-1" in retrieved.memory_references
    assert "ref-2" in retrieved.memory_references


def test_get_metadata(storage):
    """Test retrieving storage metadata."""
    # Add some entries
    for i in range(5):
        entry = MemoryEntry(
            investigation_id=f"inv-{i % 2}",
            session_id=f"sess-{i}",
            persona="researcher",
            findings=f"Finding {i}",
        )
        storage.save(f"entry-{i}", entry)

    # Get metadata
    metadata = storage.get_metadata()

    assert metadata.total_entries == 5
    assert metadata.persisted_entries == 5
    assert metadata.investigation_count == 2  # inv-0 and inv-1
    assert metadata.storage_size_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
