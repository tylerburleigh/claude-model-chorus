"""
Unit tests for conversation infrastructure.

Tests verify ConversationMemory functionality including:
- Thread creation with UUID generation
- Continuation ID handling
- Message persistence
- Thread lifecycle management
- File-based storage
"""

import json
import pytest
import uuid
from pathlib import Path
from datetime import datetime, timezone

from modelchorus.core.conversation import ConversationMemory
from modelchorus.core.models import ConversationMessage, ConversationThread


class TestConversationMemory:
    """Test suite for ConversationMemory class."""

    # ========================================================================
    # Thread Creation and Continuation ID Tests
    # ========================================================================

    def test_create_thread_generates_valid_uuid(self, tmp_path):
        """Test that thread creation generates a valid UUID as continuation_id."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Verify it's a valid UUID string
        assert thread_id is not None
        assert isinstance(thread_id, str)

        # Should be able to parse as UUID
        parsed_uuid = uuid.UUID(thread_id)
        assert str(parsed_uuid) == thread_id

    def test_create_thread_unique_ids(self, tmp_path):
        """Test that each thread creation generates unique continuation_id."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        thread_id_1 = memory.create_thread(workflow_name="workflow_1")
        thread_id_2 = memory.create_thread(workflow_name="workflow_2")
        thread_id_3 = memory.create_thread(workflow_name="workflow_1")

        # All IDs should be unique
        assert thread_id_1 != thread_id_2
        assert thread_id_1 != thread_id_3
        assert thread_id_2 != thread_id_3

    def test_create_thread_with_initial_context(self, tmp_path):
        """Test thread creation with initial context parameters."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        initial_context = {
            "prompt": "Test prompt",
            "models": ["claude", "gpt-5"],
            "temperature": 0.7
        }

        thread_id = memory.create_thread(
            workflow_name="consensus",
            initial_context=initial_context
        )

        # Retrieve and verify context was stored
        thread = memory.get_thread(thread_id)
        assert thread is not None
        assert thread.initial_context == initial_context

    def test_create_thread_persists_to_file(self, tmp_path):
        """Test that created threads are persisted to JSON files."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Check file was created
        thread_file = tmp_path / f"{thread_id}.json"
        assert thread_file.exists()

        # Verify file contains valid JSON
        with open(thread_file, "r") as f:
            data = json.load(f)
            assert data["thread_id"] == thread_id
            assert data["workflow_name"] == "test_workflow"

    def test_create_thread_with_parent(self, tmp_path):
        """Test creating thread chains with parent_thread_id."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        # Create parent thread
        parent_id = memory.create_thread(workflow_name="parent_workflow")

        # Create child thread
        child_id = memory.create_thread(
            workflow_name="child_workflow",
            parent_thread_id=parent_id
        )

        # Verify relationship
        child_thread = memory.get_thread(child_id)
        assert child_thread.parent_thread_id == parent_id
        assert child_id != parent_id
