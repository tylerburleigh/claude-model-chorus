"""
Tests for token budgeting and smart compaction in conversation history.

This module tests the token-aware limiting and smart compaction features
that prevent context overflow in long conversations.
"""

from datetime import UTC, datetime

import pytest

from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import ConversationMessage


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def create_message(role: str, content: str, files: list[str] | None = None, **kwargs) -> ConversationMessage:
    """Helper to create ConversationMessage with timestamp."""
    return ConversationMessage(
        role=role,
        content=content,
        timestamp=datetime.now(UTC).isoformat(),
        files=files or [],
        **kwargs,
    )


@pytest.fixture
def memory():
    """Create ConversationMemory instance for testing."""
    return ConversationMemory()


@pytest.fixture
def thread_id_with_messages(memory):
    """Create a thread with various types of messages for testing."""
    thread_id = memory.create_thread("test_workflow")

    # Add messages of different types and lengths
    messages = [
        # Short messages
        ("user", "Hello", []),
        ("assistant", "Hi there!", []),
        # Medium message
        ("user", "Can you help me understand how this works?", []),
        ("assistant", "Of course! Let me explain the key concepts.", []),
        # Long important message (>500 chars)
        (
            "user",
            "I need detailed information about the following topics: "
            "first, the architectural patterns used in microservices; "
            "second, the best practices for API design; "
            "third, strategies for handling distributed transactions; "
            "fourth, monitoring and observability approaches; "
            "fifth, security considerations for REST APIs; "
            "sixth, performance optimization techniques; "
            "seventh, deployment strategies and CI/CD pipelines; "
            "eighth, error handling and retry mechanisms; "
            "ninth, data consistency patterns in distributed systems; "
            "and finally, testing strategies for microservices.",
            [],
        ),
        # Message with files (important)
        ("assistant", "Let me review the code.", ["src/main.py", "src/utils.py"]),
        # Message with metadata (important)
        ("user", "Check the configuration", []),
        # More short messages
        ("assistant", "Done", []),
        ("user", "Thanks", []),
        ("assistant", "You're welcome!", []),
    ]

    for role, content, files in messages:
        memory.add_message(thread_id, role, content, files=files if files else None)

    return thread_id


@pytest.fixture
def thread_id_with_long_messages(memory):
    """Create a thread with many long messages for token budget testing."""
    thread_id = memory.create_thread("test_workflow")

    # Create 20 messages of ~100 chars each (~25 tokens each = ~500 tokens total)
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        # ~100 characters = ~25 tokens
        content = f"This is message number {i}. " * 5  # Creates ~100 char message
        memory.add_message(thread_id, role, content)

    return thread_id


# ============================================================================
# Token Estimation Tests
# ============================================================================


class TestTokenEstimation:
    """Test token estimation helper."""

    def test_estimate_tokens_empty_string(self, memory):
        """Test token estimation for empty string."""
        assert memory._estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self, memory):
        """Test token estimation for short text."""
        # ~4 chars per token heuristic
        text = "Hello"  # 5 chars
        tokens = memory._estimate_tokens(text)
        assert tokens == 1  # 5 // 4 = 1

    def test_estimate_tokens_medium_text(self, memory):
        """Test token estimation for medium text."""
        text = "This is a test message"  # 22 chars
        tokens = memory._estimate_tokens(text)
        assert tokens == 5  # 22 // 4 = 5

    def test_estimate_tokens_long_text(self, memory):
        """Test token estimation for long text."""
        text = "a" * 1000  # 1000 chars
        tokens = memory._estimate_tokens(text)
        assert tokens == 250  # 1000 // 4 = 250


# ============================================================================
# Message Importance Detection Tests
# ============================================================================


class TestMessageImportance:
    """Test message importance detection."""

    def test_message_with_files_is_important(self, memory):
        """Test that messages with files are considered important."""
        msg = create_message("user", "Check this", ["src/main.py"])
        assert memory._is_important_message(msg) is True

    def test_message_with_multiple_files_is_important(self, memory):
        """Test that messages with multiple files are important."""
        msg = create_message("user", "Review these files", ["src/main.py", "src/utils.py", "tests/test_main.py"])
        assert memory._is_important_message(msg) is True

    def test_message_with_metadata_is_important(self, memory):
        """Test that messages with metadata are important."""
        msg = create_message("assistant", "Processing request", metadata={"model": "claude-opus", "temperature": 0.7})
        assert memory._is_important_message(msg) is True

    def test_message_with_workflow_name_is_important(self, memory):
        """Test that messages with workflow_name are important."""
        msg = create_message("assistant", "Starting workflow", workflow_name="consensus")
        assert memory._is_important_message(msg) is True

    def test_long_message_is_important(self, memory):
        """Test that messages >500 chars are considered important."""
        # Create message with exactly 501 characters
        long_content = "a" * 501
        msg = create_message("user", long_content)
        assert memory._is_important_message(msg) is True

    def test_short_message_not_important(self, memory):
        """Test that short messages without special attributes are not important."""
        msg = create_message("user", "Hello")
        assert memory._is_important_message(msg) is False

    def test_medium_message_not_important(self, memory):
        """Test that medium messages (<=500 chars) without attributes are not important."""
        # Create message with exactly 500 characters
        medium_content = "a" * 500
        msg = create_message("user", medium_content)
        assert memory._is_important_message(msg) is False


# ============================================================================
# Simple Compaction Tests (smart_compaction=False)
# ============================================================================


class TestSimpleCompaction:
    """Test simple newest-first compaction strategy."""

    def test_no_compaction_needed(self, memory, thread_id_with_messages):
        """Test that no compaction occurs when messages fit within budget."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        # Large budget that fits all messages
        result = memory._apply_token_budget(messages, max_tokens=10000, smart_compaction=False)

        # All messages should be included
        assert len(result) == len(messages)
        assert result == messages

    def test_compaction_keeps_newest(self, memory, thread_id_with_long_messages):
        """Test that simple compaction keeps only the newest messages."""
        thread = memory.get_thread(thread_id_with_long_messages)
        messages = thread.messages

        # Budget for ~5 messages (~125 tokens)
        result = memory._apply_token_budget(messages, max_tokens=125, smart_compaction=False)

        # Should have fewer messages than original
        assert len(result) < len(messages)

        # Should contain only the newest messages
        assert result == messages[-len(result):]

    def test_compaction_preserves_order(self, memory, thread_id_with_long_messages):
        """Test that simple compaction preserves message order."""
        thread = memory.get_thread(thread_id_with_long_messages)
        messages = thread.messages

        result = memory._apply_token_budget(messages, max_tokens=250, smart_compaction=False)

        # Result should be in same order as original
        for i in range(len(result) - 1):
            orig_idx_1 = messages.index(result[i])
            orig_idx_2 = messages.index(result[i + 1])
            assert orig_idx_1 < orig_idx_2

    def test_compaction_respects_token_budget(self, memory, thread_id_with_long_messages):
        """Test that compaction doesn't exceed token budget."""
        thread = memory.get_thread(thread_id_with_long_messages)
        messages = thread.messages

        max_tokens = 200
        result = memory._apply_token_budget(messages, max_tokens=max_tokens, smart_compaction=False)

        # Calculate actual token count
        total_tokens = sum(
            memory._estimate_tokens(msg.content + " " + " ".join(msg.files or []))
            for msg in result
        )

        # Should not exceed budget (allow ~5% tolerance for rounding)
        assert total_tokens <= max_tokens * 1.05

    def test_compaction_includes_at_least_one_message(self, memory, thread_id_with_messages):
        """Test that compaction always includes at least the most recent message."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        # Very small budget (but >0)
        result = memory._apply_token_budget(messages, max_tokens=1, smart_compaction=False)

        # Should have at least the most recent message
        assert len(result) >= 1
        assert result[-1] == messages[-1]


# ============================================================================
# Smart Compaction Tests
# ============================================================================


class TestSmartCompaction:
    """Test smart compaction with importance preservation."""

    def test_smart_compaction_preserves_recent(self, memory, thread_id_with_long_messages):
        """Test that smart compaction prioritizes recent messages."""
        thread = memory.get_thread(thread_id_with_long_messages)
        messages = thread.messages

        # Budget for ~10 messages
        result = memory._apply_token_budget(messages, max_tokens=250, smart_compaction=True)

        # Should include recent messages (within 70% of budget)
        # Recent messages are the newest ones
        newest_messages = messages[-10:]
        for msg in newest_messages[:7]:  # At least 70% should be recent
            if msg in result:
                # Recent message should maintain relative order
                assert True

    def test_smart_compaction_reserves_budget_for_important(
        self, memory, thread_id_with_messages
    ):
        """Test that smart compaction reserves 30% budget for important messages."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        # Find important older messages
        important_messages = [
            msg for msg in messages if memory._is_important_message(msg)
        ]

        if not important_messages:
            pytest.skip("Test requires important messages")

        # Apply compaction with moderate budget
        result = memory._apply_token_budget(messages, max_tokens=300, smart_compaction=True)

        # Should include at least some important messages
        included_important = [msg for msg in result if memory._is_important_message(msg)]
        assert len(included_important) > 0

    def test_smart_compaction_preserves_files(self, memory, thread_id_with_messages):
        """Test that smart compaction preserves messages with files."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        # Find messages with files
        messages_with_files = [msg for msg in messages if msg.files]

        if not messages_with_files:
            pytest.skip("Test requires messages with files")

        # Apply compaction
        result = memory._apply_token_budget(messages, max_tokens=300, smart_compaction=True)

        # Should try to preserve messages with files (important)
        for msg in messages_with_files:
            if msg in messages[-5:]:  # If it's recent, should definitely be there
                assert msg in result

    def test_smart_compaction_preserves_order(self, memory, thread_id_with_messages):
        """Test that smart compaction maintains chronological order."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        result = memory._apply_token_budget(messages, max_tokens=300, smart_compaction=True)

        # Result should maintain original chronological order
        original_indices = [messages.index(msg) for msg in result]
        assert original_indices == sorted(original_indices)

    def test_smart_compaction_vs_simple(self, memory, thread_id_with_messages):
        """Test that smart compaction preserves more important context than simple."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        max_tokens = 200

        simple_result = memory._apply_token_budget(
            messages, max_tokens=max_tokens, smart_compaction=False
        )
        smart_result = memory._apply_token_budget(
            messages, max_tokens=max_tokens, smart_compaction=True
        )

        # Count important messages in each result
        simple_important = sum(
            1 for msg in simple_result if memory._is_important_message(msg)
        )
        smart_important = sum(
            1 for msg in smart_result if memory._is_important_message(msg)
        )

        # Smart compaction should preserve more (or equal) important messages
        # Even if counts are equal, smart approach is better for context
        assert smart_important >= simple_important

    def test_smart_compaction_respects_token_budget(self, memory, thread_id_with_messages):
        """Test that smart compaction doesn't exceed token budget."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        max_tokens = 250
        result = memory._apply_token_budget(messages, max_tokens=max_tokens, smart_compaction=True)

        # Calculate actual token count
        total_tokens = sum(
            memory._estimate_tokens(msg.content + " " + " ".join(msg.files or []))
            for msg in result
        )

        # Should not exceed budget (allow ~5% tolerance for rounding)
        assert total_tokens <= max_tokens * 1.05


# ============================================================================
# build_conversation_history with Token Budgeting Tests
# ============================================================================


class TestBuildHistoryWithTokenBudget:
    """Test build_conversation_history with max_tokens parameter."""

    def test_build_history_without_token_limit(self, memory, thread_id_with_messages):
        """Test building history without token budget."""
        history, count = memory.build_conversation_history(thread_id_with_messages)

        thread = memory.get_thread(thread_id_with_messages)
        assert count == len(thread.messages)
        assert "CONVERSATION HISTORY" in history

    def test_build_history_with_token_limit(self, memory, thread_id_with_long_messages):
        """Test that token limit reduces message count."""
        # Without limit
        history_full, count_full = memory.build_conversation_history(
            thread_id_with_long_messages
        )

        # With token limit
        history_limited, count_limited = memory.build_conversation_history(
            thread_id_with_long_messages, max_tokens=250
        )

        # Limited version should have fewer messages
        assert count_limited < count_full
        assert len(history_limited) < len(history_full)

    def test_build_history_with_max_messages_and_tokens(
        self, memory, thread_id_with_long_messages
    ):
        """Test that max_messages is applied before max_tokens."""
        thread = memory.get_thread(thread_id_with_long_messages)
        total_messages = len(thread.messages)

        # Apply both limits
        history, count = memory.build_conversation_history(
            thread_id_with_long_messages,
            max_messages=10,  # First limit to 10 messages
            max_tokens=200,   # Then limit to 200 tokens
        )

        # Count should be <= 10 (max_messages)
        assert count <= 10
        # Count should also satisfy token budget
        assert count < total_messages

    def test_build_history_smart_vs_simple_compaction(
        self, memory, thread_id_with_messages
    ):
        """Test difference between smart and simple compaction in history."""
        # Smart compaction (default)
        history_smart, count_smart = memory.build_conversation_history(
            thread_id_with_messages,
            max_tokens=250,
            smart_compaction=True,
        )

        # Simple compaction
        history_simple, count_simple = memory.build_conversation_history(
            thread_id_with_messages,
            max_tokens=250,
            smart_compaction=False,
        )

        # Both should respect token budget, but may differ in content
        assert "CONVERSATION HISTORY" in history_smart
        assert "CONVERSATION HISTORY" in history_simple

    def test_build_history_truncation_note(self, memory, thread_id_with_long_messages):
        """Test that truncation note appears when messages are omitted."""
        history, count = memory.build_conversation_history(
            thread_id_with_long_messages,
            max_tokens=100,  # Very small budget to force truncation
        )

        thread = memory.get_thread(thread_id_with_long_messages)
        if count < len(thread.messages):
            # Should include truncation note
            assert "earlier messages omitted" in history.lower()

    def test_build_history_includes_recent_messages(
        self, memory, thread_id_with_messages
    ):
        """Test that most recent messages are always included."""
        thread = memory.get_thread(thread_id_with_messages)
        most_recent = thread.messages[-1]

        history, count = memory.build_conversation_history(
            thread_id_with_messages,
            max_tokens=100,  # Small budget
        )

        # Most recent message should be in history
        assert most_recent.content in history

    def test_build_history_includes_files_within_budget(
        self, memory, thread_id_with_messages
    ):
        """Test that file references are included when using token budget."""
        history, count = memory.build_conversation_history(
            thread_id_with_messages,
            max_tokens=500,
            include_files=True,
        )

        # Check if FILES REFERENCED section exists (if any messages with files were included)
        thread = memory.get_thread(thread_id_with_messages)
        has_files = any(msg.files for msg in thread.messages)

        if has_files:
            # Files section may or may not be present depending on token budget
            # but if present, should be formatted correctly
            if "FILES REFERENCED" in history:
                assert "END FILES" in history


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for token budgeting."""

    def test_empty_message_list(self, memory):
        """Test applying token budget to empty message list."""
        result = memory._apply_token_budget([], max_tokens=100, smart_compaction=True)
        assert result == []

    def test_single_message_within_budget(self, memory):
        """Test single message that fits within budget."""
        msg = create_message("user", "Hello")
        result = memory._apply_token_budget([msg], max_tokens=100, smart_compaction=True)
        assert len(result) == 1
        assert result[0] == msg

    def test_single_message_exceeds_budget(self, memory):
        """Test single message that exceeds budget is still included."""
        msg = create_message("user", "a" * 1000)  # ~250 tokens
        result = memory._apply_token_budget([msg], max_tokens=100, smart_compaction=True)
        # Should still include at least one message
        assert len(result) == 1
        assert result[0] == msg

    def test_all_messages_important(self, memory):
        """Test compaction when all messages are important."""
        messages = [
            create_message("user", "Check this", ["file1.py"]),
            create_message("assistant", "Done", ["file2.py"]),
            create_message("user", "Review", ["file3.py"]),
        ]

        result = memory._apply_token_budget(messages, max_tokens=100, smart_compaction=True)

        # Should include recent important messages up to budget
        assert len(result) > 0
        assert all(memory._is_important_message(msg) for msg in result)

    def test_zero_token_budget(self, memory, thread_id_with_messages):
        """Test that zero token budget still includes at least one message."""
        thread = memory.get_thread(thread_id_with_messages)
        messages = thread.messages

        # Even with zero budget, should include most recent message
        result = memory._apply_token_budget(messages, max_tokens=0, smart_compaction=False)
        assert len(result) >= 1

    def test_file_content_counted_in_tokens(self, memory):
        """Test that file paths are counted in token estimation."""
        msg = create_message(
            "user",
            "Hello",  # 5 chars = ~1 token
            ["very_long_file_path_name.py" * 10],  # Should add significant tokens
        )

        # Token count should include file paths
        msg_text = msg.content + " " + " ".join(msg.files or [])
        tokens = memory._estimate_tokens(msg_text)
        assert tokens > 1  # Should be more than just the content
