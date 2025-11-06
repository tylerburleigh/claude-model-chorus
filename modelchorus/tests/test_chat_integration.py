"""
Integration tests for ChatWorkflow with multiple providers.

Tests chat functionality with Claude, Gemini, and Codex providers to ensure
multi-provider compatibility and conversation continuity.
"""

import pytest
import os
from pathlib import Path

from modelchorus.workflows import ChatWorkflow
from modelchorus.providers import ClaudeProvider, GeminiProvider, CodexProvider
from modelchorus.core.conversation import ConversationMemory


# Skip all tests if providers are not configured
def is_provider_available(provider_class):
    """Check if a provider is available (API key configured)."""
    try:
        provider = provider_class()
        return provider.validate_api_key()
    except Exception:
        return False


# Provider availability flags
CLAUDE_AVAILABLE = is_provider_available(ClaudeProvider)
GEMINI_AVAILABLE = is_provider_available(GeminiProvider)
CODEX_AVAILABLE = is_provider_available(CodexProvider)
ANY_PROVIDER_AVAILABLE = CLAUDE_AVAILABLE or GEMINI_AVAILABLE or CODEX_AVAILABLE


@pytest.fixture
def conversation_memory():
    """Create ConversationMemory instance for testing."""
    return ConversationMemory()


@pytest.fixture(params=[
    pytest.param("claude", marks=pytest.mark.skipif(not CLAUDE_AVAILABLE, reason="Claude API not configured")),
    pytest.param("gemini", marks=pytest.mark.skipif(not GEMINI_AVAILABLE, reason="Gemini API not configured")),
    pytest.param("codex", marks=pytest.mark.skipif(not CODEX_AVAILABLE, reason="Codex API not configured")),
])
def provider_name(request):
    """Parameterized fixture for provider names."""
    return request.param


@pytest.fixture
def provider(provider_name):
    """Create provider instance based on provider name."""
    providers = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "codex": CodexProvider,
    }
    return providers[provider_name]()


@pytest.fixture
def chat_workflow(provider, conversation_memory):
    """Create ChatWorkflow instance with real provider."""
    return ChatWorkflow(
        provider=provider,
        conversation_memory=conversation_memory,
    )


@pytest.mark.skipif(not ANY_PROVIDER_AVAILABLE, reason="No providers configured")
class TestMultiProviderChat:
    """Test chat functionality across multiple providers."""

    @pytest.mark.asyncio
    async def test_basic_conversation(self, chat_workflow, provider_name):
        """Test basic single-turn conversation works with each provider."""
        result = await chat_workflow.run(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.0,
        )

        assert result.success is True, f"Chat failed with {provider_name}: {result.error}"
        assert result.synthesis is not None
        assert len(result.synthesis) > 0
        assert "thread_id" in result.metadata
        assert result.metadata["is_continuation"] is False

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, chat_workflow, provider_name):
        """Test multi-turn conversation continuity with each provider."""
        # First turn
        result1 = await chat_workflow.run(
            prompt="I'm thinking of a number between 1 and 10. Guess what it is.",
            temperature=0.7,
        )

        assert result1.success is True, f"First turn failed with {provider_name}"
        thread_id = result1.metadata["thread_id"]
        assert result1.metadata["conversation_length"] == 2  # user + assistant

        # Second turn - continue conversation
        result2 = await chat_workflow.run(
            prompt="No, try again with a different guess.",
            continuation_id=thread_id,
            temperature=0.7,
        )

        assert result2.success is True, f"Second turn failed with {provider_name}"
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["is_continuation"] is True
        assert result2.metadata["conversation_length"] == 4  # 2 previous + user + assistant

        # Third turn - verify conversation history is maintained
        result3 = await chat_workflow.run(
            prompt="What was my original question?",
            continuation_id=thread_id,
            temperature=0.7,
        )

        assert result3.success is True, f"Third turn failed with {provider_name}"
        assert result3.metadata["thread_id"] == thread_id
        assert result3.metadata["conversation_length"] == 6
        # Response should reference the original question about thinking of a number
        assert any(word in result3.synthesis.lower() for word in ["number", "guess", "thinking"])

    @pytest.mark.asyncio
    async def test_conversation_with_file_context(self, chat_workflow, provider_name, tmp_path):
        """Test chat with file context included."""
        # Create a temporary file
        test_file = tmp_path / "sample.py"
        test_file.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")

        result = await chat_workflow.run(
            prompt="What functions are defined in this file?",
            files=[str(test_file)],
            temperature=0.0,
        )

        assert result.success is True, f"File context chat failed with {provider_name}"
        assert result.synthesis is not None
        # Response should mention the functions
        assert "add" in result.synthesis or "multiply" in result.synthesis

    @pytest.mark.asyncio
    async def test_conversation_persistence(self, chat_workflow, provider_name):
        """Test that conversation state persists correctly."""
        # Start conversation
        result1 = await chat_workflow.run(
            prompt="Remember this: my favorite color is blue.",
            temperature=0.0,
        )

        assert result1.success is True
        thread_id = result1.metadata["thread_id"]

        # Verify conversation can be retrieved
        thread = chat_workflow.get_thread(thread_id)
        assert thread is not None
        assert thread.thread_id == thread_id
        assert len(thread.messages) == 2  # user + assistant

        # Continue and ask about remembered info
        result2 = await chat_workflow.run(
            prompt="What is my favorite color?",
            continuation_id=thread_id,
            temperature=0.0,
        )

        assert result2.success is True
        assert "blue" in result2.synthesis.lower()


@pytest.mark.skipif(not ANY_PROVIDER_AVAILABLE, reason="No providers configured")
class TestChatErrorHandling:
    """Test error handling in chat workflow."""

    @pytest.mark.asyncio
    async def test_invalid_continuation_id(self, chat_workflow, provider_name):
        """Test handling of invalid continuation ID."""
        # Try to continue with non-existent thread ID
        result = await chat_workflow.run(
            prompt="Continue this conversation.",
            continuation_id="non-existent-thread-id",
            temperature=0.0,
        )

        # Should create new conversation instead of failing
        assert result.success is True
        assert result.metadata["is_continuation"] is False

    @pytest.mark.asyncio
    async def test_empty_prompt(self, chat_workflow, provider_name):
        """Test handling of empty prompt."""
        result = await chat_workflow.run(
            prompt="",
            temperature=0.0,
        )

        # Should handle gracefully (may succeed with empty response or fail gracefully)
        # Either way, should not crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_conversation(self, chat_workflow, provider_name):
        """Test handling of very long multi-turn conversations."""
        thread_id = None

        # Create a conversation with 5 turns
        for i in range(5):
            result = await chat_workflow.run(
                prompt=f"This is message number {i+1}. Acknowledge it.",
                continuation_id=thread_id,
                temperature=0.0,
            )

            assert result.success is True, f"Turn {i+1} failed"

            if i == 0:
                thread_id = result.metadata["thread_id"]
            else:
                assert result.metadata["thread_id"] == thread_id

            expected_length = (i + 1) * 2  # Each turn adds user + assistant messages
            assert result.metadata["conversation_length"] == expected_length


@pytest.mark.skipif(not ANY_PROVIDER_AVAILABLE, reason="No providers configured")
class TestChatThreadManagement:
    """Test conversation thread management."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_threads(self, provider, conversation_memory):
        """Test managing multiple concurrent conversation threads."""
        workflow = ChatWorkflow(provider=provider, conversation_memory=conversation_memory)

        # Create two separate conversations
        result1 = await workflow.run(prompt="My name is Alice.")
        result2 = await workflow.run(prompt="My name is Bob.")

        assert result1.success and result2.success
        thread_id1 = result1.metadata["thread_id"]
        thread_id2 = result2.metadata["thread_id"]

        # Verify threads are different
        assert thread_id1 != thread_id2

        # Continue both conversations and verify separation
        result1_cont = await workflow.run(
            prompt="What is my name?",
            continuation_id=thread_id1,
        )
        result2_cont = await workflow.run(
            prompt="What is my name?",
            continuation_id=thread_id2,
        )

        assert result1_cont.success and result2_cont.success
        assert "alice" in result1_cont.synthesis.lower()
        assert "bob" in result2_cont.synthesis.lower()

    @pytest.mark.asyncio
    async def test_thread_retrieval(self, chat_workflow, provider_name):
        """Test retrieving conversation thread details."""
        # Create conversation
        result = await chat_workflow.run(prompt="Hello!")

        assert result.success
        thread_id = result.metadata["thread_id"]

        # Retrieve thread
        thread = chat_workflow.get_thread(thread_id)

        assert thread is not None
        assert thread.thread_id == thread_id
        assert len(thread.messages) == 2
        assert thread.messages[0].role == "user"
        assert thread.messages[0].content == "Hello!"
        assert thread.messages[1].role == "assistant"
