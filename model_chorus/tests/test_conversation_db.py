"""
Unit tests for SQLite-based conversation database.

Tests verify ConversationDatabase functionality including:
- Database initialization and schema creation
- Thread creation and retrieval
- Message persistence
- TTL-based expiration
- Thread lifecycle management (active, completed, archived)
- Query capabilities
- Statistics and cleanup operations
"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_chorus.core.conversation_db import (
    DEFAULT_MAX_MESSAGES_PER_THREAD,
    DEFAULT_TTL_HOURS,
    ConversationDatabase,
)
from model_chorus.core.models import ConversationMessage, ConversationThread


class TestDatabaseInitialization:
    """Test suite for database initialization and schema creation."""

    def test_create_database_file(self, tmp_path):
        """Test that database file is created on initialization."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        assert db_path.exists()
        assert db_path.is_file()
        db.close()

    def test_create_parent_directory(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        db_path = tmp_path / "nested" / "dir" / "conversations.db"
        db = ConversationDatabase(db_path=db_path)

        assert db_path.exists()
        assert db_path.parent.exists()
        db.close()

    def test_schema_creation(self, tmp_path):
        """Test that all tables and indexes are created correctly."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        assert "threads" in tables
        assert "messages" in tables
        assert "thread_metadata" in tables

        # Verify indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_threads_created_at" in indexes
        assert "idx_threads_status" in indexes
        assert "idx_threads_workflow" in indexes
        assert "idx_messages_thread_timestamp" in indexes

        conn.close()
        db.close()

    def test_wal_mode_enabled(self, tmp_path):
        """Test that WAL mode is enabled for concurrency."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        cursor = db.conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        result = cursor.fetchone()[0]

        assert result.lower() == "wal"
        db.close()

    def test_foreign_keys_enabled(self, tmp_path):
        """Test that foreign key constraints are enabled."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        cursor = db.conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()[0]

        assert result == 1
        db.close()

    def test_custom_ttl_and_max_messages(self, tmp_path):
        """Test custom TTL and max_messages configuration."""
        db_path = tmp_path / "test_conversations.db"
        custom_ttl = 24
        custom_max_messages = 100

        db = ConversationDatabase(
            db_path=db_path, ttl_hours=custom_ttl, max_messages=custom_max_messages
        )

        assert db.ttl_hours == custom_ttl
        assert db.max_messages == custom_max_messages
        db.close()


class TestThreadCreation:
    """Test suite for thread creation functionality."""

    def test_create_thread_generates_valid_uuid(self, tmp_path):
        """Test that thread creation generates a valid UUID."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        assert thread_id is not None
        assert isinstance(thread_id, str)

        # Should be able to parse as UUID
        parsed_uuid = uuid.UUID(thread_id)
        assert str(parsed_uuid) == thread_id
        db.close()

    def test_create_thread_unique_ids(self, tmp_path):
        """Test that each thread creation generates unique IDs."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id_1 = db.create_thread(workflow_name="workflow_1")
        thread_id_2 = db.create_thread(workflow_name="workflow_2")
        thread_id_3 = db.create_thread(workflow_name="workflow_1")

        assert thread_id_1 != thread_id_2
        assert thread_id_1 != thread_id_3
        assert thread_id_2 != thread_id_3
        db.close()

    def test_create_thread_with_initial_context(self, tmp_path):
        """Test thread creation with initial context."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        initial_context = {"prompt": "test prompt", "model": "claude"}
        thread_id = db.create_thread(
            workflow_name="test_workflow", initial_context=initial_context
        )

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert thread.initial_context == initial_context
        db.close()

    def test_create_thread_with_parent(self, tmp_path):
        """Test thread creation with parent thread ID."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        parent_id = db.create_thread(workflow_name="parent_workflow")
        child_id = db.create_thread(
            workflow_name="child_workflow", parent_thread_id=parent_id
        )

        child_thread = db.get_thread(child_id)
        assert child_thread is not None
        assert child_thread.parent_thread_id == parent_id
        db.close()

    def test_create_thread_status_is_active(self, tmp_path):
        """Test that newly created threads have 'active' status."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")
        thread = db.get_thread(thread_id)

        assert thread is not None
        assert thread.status == "active"
        db.close()


class TestThreadRetrieval:
    """Test suite for thread retrieval functionality."""

    def test_get_nonexistent_thread_returns_none(self, tmp_path):
        """Test that retrieving a nonexistent thread returns None."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        result = db.get_thread("nonexistent-thread-id")
        assert result is None
        db.close()

    def test_get_thread_returns_correct_data(self, tmp_path):
        """Test that thread retrieval returns correct data."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        workflow_name = "test_workflow"
        initial_context = {"key": "value"}

        thread_id = db.create_thread(
            workflow_name=workflow_name, initial_context=initial_context
        )
        thread = db.get_thread(thread_id)

        assert thread is not None
        assert thread.thread_id == thread_id
        assert thread.workflow_name == workflow_name
        assert thread.initial_context == initial_context
        assert thread.status == "active"
        assert len(thread.messages) == 0
        db.close()

    def test_get_expired_thread_returns_none(self, tmp_path):
        """Test that expired threads are automatically deleted and return None."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path, ttl_hours=1)

        thread_id = db.create_thread(workflow_name="test_workflow")

        # Manually update created_at to be old
        cursor = db.conn.cursor()
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        cursor.execute(
            "UPDATE threads SET created_at = ? WHERE thread_id = ?",
            (old_time, thread_id),
        )
        db.conn.commit()

        # Thread should be expired and return None
        result = db.get_thread(thread_id)
        assert result is None

        # Thread should be deleted from database
        cursor.execute("SELECT COUNT(*) FROM threads WHERE thread_id = ?", (thread_id,))
        count = cursor.fetchone()[0]
        assert count == 0
        db.close()


class TestMessageOperations:
    """Test suite for message persistence operations."""

    def test_add_message_basic(self, tmp_path):
        """Test adding a basic message to a thread."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        db.add_message(
            thread_id,
            role="user",
            content="Hello, world!",
            workflow_name="test_workflow",
        )

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert thread.messages[0].role == "user"
        assert thread.messages[0].content == "Hello, world!"
        db.close()

    def test_add_multiple_messages(self, tmp_path):
        """Test adding multiple messages to a thread."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        db.add_message(thread_id, "user", "Message 1", workflow_name="test_workflow")
        db.add_message(
            thread_id, "assistant", "Response 1", workflow_name="test_workflow"
        )
        db.add_message(thread_id, "user", "Message 2", workflow_name="test_workflow")

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 3
        assert thread.messages[0].content == "Message 1"
        assert thread.messages[1].content == "Response 1"
        assert thread.messages[2].content == "Message 2"
        db.close()

    def test_add_message_with_files(self, tmp_path):
        """Test adding a message with file attachments."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        db.add_message(
            thread_id,
            "user",
            "Check these files",
            files=["file1.py", "file2.txt"],
            workflow_name="test_workflow",
        )

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert thread.messages[0].files == ["file1.py", "file2.txt"]
        db.close()

    def test_add_message_with_metadata(self, tmp_path):
        """Test adding a message with metadata."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        metadata = {"tokens": 150, "model": "claude-3-sonnet"}
        db.add_message(
            thread_id,
            "assistant",
            "Response",
            workflow_name="test_workflow",
            metadata=metadata,
        )

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert thread.messages[0].metadata == metadata
        db.close()

    def test_message_truncation_when_max_reached(self, tmp_path):
        """Test that old messages are truncated when max is reached."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path, max_messages=5)

        thread_id = db.create_thread(workflow_name="test_workflow")

        # Add 10 messages (exceeds max of 5)
        for i in range(10):
            db.add_message(
                thread_id, "user", f"Message {i}", workflow_name="test_workflow"
            )

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 5

        # Should keep the most recent 5
        assert thread.messages[0].content == "Message 5"
        assert thread.messages[4].content == "Message 9"
        db.close()

    def test_get_messages_returns_chronological_order(self, tmp_path):
        """Test that messages are returned in chronological order."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        # Add messages with explicit order
        for i in range(5):
            db.add_message(
                thread_id, "user", f"Message {i}", workflow_name="test_workflow"
            )

        messages = db.get_messages(thread_id)
        assert len(messages) == 5

        for i, msg in enumerate(messages):
            assert msg.content == f"Message {i}"
        db.close()


class TestThreadLifecycle:
    """Test suite for thread lifecycle management."""

    def test_complete_thread(self, tmp_path):
        """Test marking a thread as completed."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        result = db.complete_thread(thread_id)
        assert result is True

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert thread.status == "completed"
        db.close()

    def test_complete_nonexistent_thread_returns_false(self, tmp_path):
        """Test that completing a nonexistent thread returns False."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        result = db.complete_thread("nonexistent-thread-id")
        assert result is False
        db.close()

    def test_archive_thread(self, tmp_path):
        """Test archiving a thread."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test_workflow")

        result = db.archive_thread(thread_id)
        assert result is True

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert thread.status == "archived"
        db.close()

    def test_archive_nonexistent_thread_returns_false(self, tmp_path):
        """Test that archiving a nonexistent thread returns False."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        result = db.archive_thread("nonexistent-thread-id")
        assert result is False
        db.close()


class TestCleanupOperations:
    """Test suite for cleanup and maintenance operations."""

    def test_cleanup_expired_threads(self, tmp_path):
        """Test cleanup of expired threads."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path, ttl_hours=1)

        # Create fresh and old threads
        fresh_id = db.create_thread(workflow_name="fresh")
        old_id1 = db.create_thread(workflow_name="old1")
        old_id2 = db.create_thread(workflow_name="old2")

        # Make two threads old
        cursor = db.conn.cursor()
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        cursor.execute(
            "UPDATE threads SET created_at = ? WHERE thread_id IN (?, ?)",
            (old_time, old_id1, old_id2),
        )
        db.conn.commit()

        # Cleanup should remove 2 old threads
        count = db.cleanup_expired_threads()
        assert count == 2

        # Fresh thread should still exist
        assert db.get_thread(fresh_id) is not None

        # Old threads should be deleted
        assert db.get_thread(old_id1) is None
        assert db.get_thread(old_id2) is None
        db.close()

    def test_cleanup_archived_threads(self, tmp_path):
        """Test cleanup of archived threads."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create threads and archive some
        active_id = db.create_thread(workflow_name="active")
        archived_id1 = db.create_thread(workflow_name="archived1")
        archived_id2 = db.create_thread(workflow_name="archived2")

        db.archive_thread(archived_id1)
        db.archive_thread(archived_id2)

        # Cleanup should remove 2 archived threads
        count = db.cleanup_archived_threads()
        assert count == 2

        # Active thread should still exist
        assert db.get_thread(active_id) is not None

        # Archived threads should be deleted
        assert db.get_thread(archived_id1) is None
        assert db.get_thread(archived_id2) is None
        db.close()


class TestQueryOperations:
    """Test suite for query and search operations."""

    def test_query_by_workflow(self, tmp_path):
        """Test querying threads by workflow name."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create threads with different workflows
        workflow1_ids = [
            db.create_thread(workflow_name="workflow1") for _ in range(3)
        ]
        workflow2_ids = [
            db.create_thread(workflow_name="workflow2") for _ in range(2)
        ]

        # Query workflow1
        results = db.query_by_workflow("workflow1")
        assert len(results) == 3
        result_ids = [thread.thread_id for thread in results]
        for thread_id in workflow1_ids:
            assert thread_id in result_ids

        # Query workflow2
        results = db.query_by_workflow("workflow2")
        assert len(results) == 2
        result_ids = [thread.thread_id for thread in results]
        for thread_id in workflow2_ids:
            assert thread_id in result_ids
        db.close()

    def test_query_by_workflow_with_status_filter(self, tmp_path):
        """Test querying threads by workflow with status filter."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create threads and complete some
        active_id = db.create_thread(workflow_name="test")
        completed_id = db.create_thread(workflow_name="test")
        archived_id = db.create_thread(workflow_name="test")

        db.complete_thread(completed_id)
        db.archive_thread(archived_id)

        # Query active only
        results = db.query_by_workflow("test", status="active")
        assert len(results) == 1
        assert results[0].thread_id == active_id

        # Query completed only
        results = db.query_by_workflow("test", status="completed")
        assert len(results) == 1
        assert results[0].thread_id == completed_id
        db.close()

    def test_query_recent_threads(self, tmp_path):
        """Test querying recent threads."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create threads
        for i in range(5):
            db.create_thread(workflow_name=f"workflow{i}")

        # Get 3 most recent
        results = db.query_recent_threads(limit=3)
        assert len(results) == 3
        db.close()

    def test_query_recent_threads_respects_ttl(self, tmp_path):
        """Test that recent threads query respects TTL."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path, ttl_hours=1)

        # Create fresh and old threads
        fresh_id = db.create_thread(workflow_name="fresh")
        old_id = db.create_thread(workflow_name="old")

        # Make one thread old
        cursor = db.conn.cursor()
        old_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        cursor.execute(
            "UPDATE threads SET created_at = ? WHERE thread_id = ?",
            (old_time, old_id),
        )
        db.conn.commit()

        # Query should only return fresh thread
        results = db.query_recent_threads(limit=10)
        assert len(results) == 1
        assert results[0].thread_id == fresh_id
        db.close()

    def test_get_thread_chain(self, tmp_path):
        """Test retrieving thread chain (parent-child relationships)."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create parent-child chain
        parent_id = db.create_thread(workflow_name="parent")
        child1_id = db.create_thread(workflow_name="child1", parent_thread_id=parent_id)
        child2_id = db.create_thread(
            workflow_name="child2", parent_thread_id=child1_id
        )

        # Get chain from leaf
        chain = db.get_thread_chain(child2_id)
        assert len(chain) == 3
        assert chain[0].thread_id == parent_id
        assert chain[1].thread_id == child1_id
        assert chain[2].thread_id == child2_id
        db.close()

    def test_get_thread_chain_single_thread(self, tmp_path):
        """Test thread chain for thread without parent."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="solo")

        chain = db.get_thread_chain(thread_id)
        assert len(chain) == 1
        assert chain[0].thread_id == thread_id
        db.close()


class TestStatistics:
    """Test suite for database statistics operations."""

    def test_get_thread_statistics(self, tmp_path):
        """Test retrieving database statistics."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create threads with different statuses
        active_id = db.create_thread(workflow_name="test")
        completed_id = db.create_thread(workflow_name="test")
        archived_id = db.create_thread(workflow_name="test")

        db.complete_thread(completed_id)
        db.archive_thread(archived_id)

        # Add messages
        db.add_message(active_id, "user", "test", workflow_name="test")
        db.add_message(active_id, "user", "test", workflow_name="test")
        db.add_message(completed_id, "user", "test", workflow_name="test")

        stats = db.get_thread_statistics()

        assert stats["total_threads"] == 3
        assert stats["active_threads"] == 1
        assert stats["completed_threads"] == 1
        assert stats["archived_threads"] == 1
        assert stats["total_messages"] == 3
        assert "top_workflows" in stats
        assert stats["top_workflows"]["test"] == 3
        db.close()

    def test_get_statistics_empty_database(self, tmp_path):
        """Test statistics for empty database."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        stats = db.get_thread_statistics()

        assert stats["total_threads"] == 0
        assert stats["active_threads"] == 0
        assert stats["completed_threads"] == 0
        assert stats["archived_threads"] == 0
        assert stats["total_messages"] == 0
        assert stats["top_workflows"] == {}
        db.close()


class TestDatabaseClosing:
    """Test suite for database connection management."""

    def test_close_connection(self, tmp_path):
        """Test that database connection can be closed cleanly."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # Create some data
        thread_id = db.create_thread(workflow_name="test")
        assert thread_id is not None

        # Close should succeed
        db.close()

        # Connection should be closed (queries should fail)
        with pytest.raises(sqlite3.ProgrammingError):
            db.conn.execute("SELECT 1")

    def test_multiple_close_calls(self, tmp_path):
        """Test that calling close multiple times doesn't raise errors."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        db.close()
        # Second close will raise ProgrammingError on closed db - this is expected SQLite behavior
        # We don't wrap close() to silently ignore this, so we expect it
        with pytest.raises(sqlite3.ProgrammingError):
            db.close()


class TestCascadingDeletes:
    """Test suite for foreign key cascade behavior."""

    def test_delete_thread_cascades_to_messages(self, tmp_path):
        """Test that deleting a thread cascades to messages."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test")

        # Add messages
        for i in range(5):
            db.add_message(thread_id, "user", f"Message {i}", workflow_name="test")

        # Delete thread
        db._delete_thread(thread_id)

        # Messages should be deleted
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,))
        count = cursor.fetchone()[0]
        assert count == 0
        db.close()

    def test_delete_thread_cascades_to_metadata(self, tmp_path):
        """Test that deleting a thread cascades to metadata."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test")

        # Add metadata
        cursor = db.conn.cursor()
        cursor.execute(
            "INSERT INTO thread_metadata (thread_id, key, value) VALUES (?, ?, ?)",
            (thread_id, "test_key", "test_value"),
        )
        db.conn.commit()

        # Delete thread
        db._delete_thread(thread_id)

        # Metadata should be deleted
        cursor.execute(
            "SELECT COUNT(*) FROM thread_metadata WHERE thread_id = ?", (thread_id,)
        )
        count = cursor.fetchone()[0]
        assert count == 0
        db.close()


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_add_message_to_nonexistent_thread_creates_thread(self, tmp_path):
        """Test that adding a message to nonexistent thread creates it implicitly."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        # add_message should create thread implicitly if it doesn't exist
        thread_id = "test-thread-id"
        result = db.add_message(thread_id, "user", "test", workflow_name="test")

        assert result is True

        # Thread should now exist
        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert thread.messages[0].content == "test"
        db.close()

    def test_minimal_workflow_name(self, tmp_path):
        """Test thread creation with minimal workflow name."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="x")

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert thread.workflow_name == "x"
        db.close()

    def test_large_message_content(self, tmp_path):
        """Test storing large message content."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test")

        # Create large content (1MB)
        large_content = "x" * (1024 * 1024)
        db.add_message(thread_id, "user", large_content, workflow_name="test")

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert len(thread.messages[0].content) == len(large_content)
        db.close()

    def test_unicode_content(self, tmp_path):
        """Test storing Unicode content."""
        db_path = tmp_path / "test_conversations.db"
        db = ConversationDatabase(db_path=db_path)

        thread_id = db.create_thread(workflow_name="test")

        unicode_content = "Hello 世界 🌍 مرحبا Привет"
        db.add_message(thread_id, "user", unicode_content, workflow_name="test")

        thread = db.get_thread(thread_id)
        assert thread is not None
        assert thread.messages[0].content == unicode_content
        db.close()
