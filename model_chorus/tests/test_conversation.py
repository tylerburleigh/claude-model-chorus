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

from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import ConversationMessage, ConversationThread


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
            "temperature": 0.7,
        }

        thread_id = memory.create_thread(workflow_name="consensus", initial_context=initial_context)

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
        child_id = memory.create_thread(workflow_name="child_workflow", parent_thread_id=parent_id)

        # Verify relationship
        child_thread = memory.get_thread(child_id)
        assert child_thread.parent_thread_id == parent_id
        assert child_id != parent_id

    # ========================================================================
    # Message Addition and Retrieval Tests
    # ========================================================================

    def test_add_message_to_thread(self, tmp_path):
        """Test adding a message to a conversation thread."""
        memory = ConversationMemory(conversations_dir=tmp_path)
        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Add user message
        memory.add_message(thread_id=thread_id, role="user", content="Test question")

        # Retrieve thread and verify message
        thread = memory.get_thread(thread_id)
        assert len(thread.messages) == 1
        assert thread.messages[0].role == "user"
        assert thread.messages[0].content == "Test question"

    def test_add_multiple_messages(self, tmp_path):
        """Test adding multiple messages maintains order."""
        memory = ConversationMemory(conversations_dir=tmp_path)
        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Add conversation
        memory.add_message(thread_id, "user", "Question 1")
        memory.add_message(thread_id, "assistant", "Answer 1")
        memory.add_message(thread_id, "user", "Question 2")
        memory.add_message(thread_id, "assistant", "Answer 2")

        # Verify order preserved
        thread = memory.get_thread(thread_id)
        assert len(thread.messages) == 4
        assert thread.messages[0].content == "Question 1"
        assert thread.messages[1].content == "Answer 1"
        assert thread.messages[2].content == "Question 2"
        assert thread.messages[3].content == "Answer 2"

    def test_add_message_with_metadata(self, tmp_path):
        """Test adding messages with metadata like model info."""
        memory = ConversationMemory(conversations_dir=tmp_path)
        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Add assistant message with metadata
        memory.add_message(
            thread_id=thread_id,
            role="assistant",
            content="Analysis complete",
            workflow_name="consensus",
            model_provider="cli",
            model_name="claude-3-opus",
            metadata={"tokens": 450, "latency_ms": 1200},
        )

        # Verify metadata preserved
        thread = memory.get_thread(thread_id)
        msg = thread.messages[0]
        assert msg.workflow_name == "consensus"
        assert msg.model_provider == "cli"
        assert msg.model_name == "claude-3-opus"
        assert msg.metadata["tokens"] == 450

    def test_get_messages_returns_chronological_order(self, tmp_path):
        """Test that get_messages returns messages in chronological order."""
        memory = ConversationMemory(conversations_dir=tmp_path)
        thread_id = memory.create_thread(workflow_name="test_workflow")

        # Add messages
        memory.add_message(thread_id, "user", "First")
        memory.add_message(thread_id, "assistant", "Second")
        memory.add_message(thread_id, "user", "Third")

        # Retrieve messages
        messages = memory.get_messages(thread_id)
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    def test_message_persistence_across_instances(self, tmp_path):
        """Test that messages persist across ConversationMemory instances."""
        thread_id = None

        # Create thread and add message in first instance
        memory1 = ConversationMemory(conversations_dir=tmp_path)
        thread_id = memory1.create_thread(workflow_name="test")
        memory1.add_message(thread_id, "user", "Persistent message")

        # Create new instance and retrieve
        memory2 = ConversationMemory(conversations_dir=tmp_path)
        messages = memory2.get_messages(thread_id)
        assert len(messages) == 1
        assert messages[0].content == "Persistent message"

    # ========================================================================
    # Context Management Tests
    # ========================================================================

    def test_thread_context_window_management(self, tmp_path):
        """Test that thread respects max_messages limit."""
        memory = ConversationMemory(
            conversations_dir=tmp_path, max_messages=3  # Low limit for testing
        )
        thread_id = memory.create_thread(workflow_name="test")

        # Add messages beyond limit
        for i in range(5):
            memory.add_message(thread_id, "user", f"Message {i}")

        # Should only keep most recent max_messages
        thread = memory.get_thread(thread_id)
        assert len(thread.messages) <= 3

    def test_get_thread_context_includes_state(self, tmp_path):
        """Test that thread context includes workflow state."""
        memory = ConversationMemory(conversations_dir=tmp_path)

        # Create thread with initial state
        thread_id = memory.create_thread(
            workflow_name="test",
            initial_context={
                "prompt": "Test",
                "state": {"current_step": 2, "models_consulted": ["claude", "gpt-5"]},
            },
        )

        # Retrieve and verify context
        thread = memory.get_thread(thread_id)
        assert thread.initial_context["state"]["current_step"] == 2
        assert thread.initial_context["state"]["models_consulted"] == ["claude", "gpt-5"]
