"""
Integration tests for ChatWorkflow with multiple providers.

Tests chat functionality with Claude, Gemini, and Codex providers to ensure
multi-provider compatibility and conversation continuity.
"""

import os

import pytest

# Import provider availability flags from shared test helpers
from .test_helpers import ANY_PROVIDER_AVAILABLE

from model_chorus.core.conversation import ConversationMemory
from model_chorus.providers import (
    ClaudeProvider,
    CodexProvider,
    CursorAgentProvider,
    GeminiProvider,
)
from model_chorus.workflows import ChatWorkflow


def get_run_kwargs(provider_name: str, prompt: str, **kwargs):
    """
    Get kwargs for chat_workflow.run() that are compatible with the provider.

    Filters out unsupported parameters (e.g., temperature for Gemini).
    Note: Fast models are automatically injected by the provider fixture.
    """
    run_kwargs = {"prompt": prompt, **kwargs}

    # Gemini doesn't support temperature, max_tokens, or other generation params
    if provider_name == "gemini":
        unsupported_params = ["temperature", "max_tokens"]
        for param in unsupported_params:
            run_kwargs.pop(param, None)

    return run_kwargs


@pytest.fixture
def conversation_memory(tmp_path):
    """Create ConversationMemory instance for testing.

    Uses a temporary directory for each test to ensure isolation.
    Uses a high max_messages limit (100) to allow long conversation tests
    to run without hitting the truncation limit.
    """
    return ConversationMemory(
        conversations_dir=tmp_path / "conversations", max_messages=100
    )


@pytest.fixture
def provider(
    provider_name,
    mock_claude_provider_full,
    mock_gemini_provider_full,
    mock_codex_provider_full,
    mock_cursor_agent_provider_full,
):
    """Create provider instance based on provider name.

    Uses mock providers if USE_MOCK_PROVIDERS=true, otherwise uses real CLI providers.
    Automatically configures the fastest model for each provider to minimize test time and cost.
    """

    USE_MOCK_PROVIDERS = os.getenv("USE_MOCK_PROVIDERS", "false").lower() == "true"

    # Fast models for each provider (mini/flash/haiku variants)
    fast_models = {
        "claude": "haiku",
        "gemini": "gemini-2.5-flash",
        "codex": "gpt-5-codex-mini",
        "cursor-agent": "composer-1",
    }

    if USE_MOCK_PROVIDERS:
        mock_providers = {
            "claude": mock_claude_provider_full,
            "gemini": mock_gemini_provider_full,
            "codex": mock_codex_provider_full,
            "cursor-agent": mock_cursor_agent_provider_full,
        }
        return mock_providers[provider_name]
    else:
        real_providers = {
            "claude": ClaudeProvider,
            "gemini": GeminiProvider,
            "codex": CodexProvider,
            "cursor-agent": CursorAgentProvider,
        }
        provider_instance = real_providers[provider_name]()

        # Wrap the generate method to inject the fast model into all requests
        original_generate = provider_instance.generate
        fast_model = fast_models[provider_name]

        async def generate_with_fast_model(request):
            # Inject fast model into request metadata if not already specified
            if "model" not in request.metadata:
                request.metadata["model"] = fast_model
            return await original_generate(request)

        provider_instance.generate = generate_with_fast_model
        return provider_instance


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
            **get_run_kwargs(
                provider_name,
                "What is 2+2? Answer with just the number.",
                temperature=0.0,
            )
        )

        assert (
            result.success is True
        ), f"Chat failed with {provider_name}: {result.error}"
        assert result.synthesis is not None
        assert len(result.synthesis) > 0
        assert "thread_id" in result.metadata
        assert result.metadata["is_continuation"] is False

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, chat_workflow, provider_name):
        """Test multi-turn conversation continuity with each provider."""
        # First turn
        result1 = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "I'm thinking of a number between 1 and 10. Guess what it is.",
                temperature=0.7,
            )
        )

        assert result1.success is True, f"First turn failed with {provider_name}"
        thread_id = result1.metadata["thread_id"]
        assert result1.metadata["conversation_length"] == 2  # user + assistant

        # Second turn - continue conversation
        result2 = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "No, try again with a different guess.",
                continuation_id=thread_id,
                temperature=0.7,
            )
        )

        assert result2.success is True, f"Second turn failed with {provider_name}"
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["is_continuation"] is True
        assert (
            result2.metadata["conversation_length"] == 4
        )  # 2 previous + user + assistant

        # Third turn - verify conversation history is maintained
        result3 = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "What was my original question?",
                continuation_id=thread_id,
                temperature=0.7,
            )
        )

        assert result3.success is True, f"Third turn failed with {provider_name}"
        assert result3.metadata["thread_id"] == thread_id
        assert result3.metadata["conversation_length"] == 6
        # Response should reference the original question about thinking of a number
        assert any(
            word in result3.synthesis.lower()
            for word in ["number", "guess", "thinking"]
        )

    @pytest.mark.asyncio
    async def test_conversation_with_file_context(
        self, chat_workflow, provider_name, tmp_path
    ):
        """Test chat with file context included."""
        # Create a temporary file
        test_file = tmp_path / "sample.py"
        test_file.write_text(
            """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        )

        result = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "What functions are defined in this file?",
                files=[str(test_file)],
                temperature=0.0,
            )
        )

        assert result.success is True, f"File context chat failed with {provider_name}"
        assert result.synthesis is not None
        # Response should mention the functions
        assert "add" in result.synthesis or "multiply" in result.synthesis

    @pytest.mark.asyncio
    async def test_conversation_persistence(self, chat_workflow, provider_name):
        """Test that conversation state persists correctly."""
        # Start conversation with a simple fact
        result1 = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "My name is Alice. Please just acknowledge this.",
                temperature=0.0,
            )
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
            **get_run_kwargs(
                provider_name,
                "What is my name?",
                continuation_id=thread_id,
                temperature=0.0,
            )
        )

        assert result2.success is True
        # The response should mention Alice (be flexible about the exact format)
        assert (
            "alice" in result2.synthesis.lower()
        ), f"Expected 'alice' in response, got: {result2.synthesis}"


@pytest.mark.skipif(not ANY_PROVIDER_AVAILABLE, reason="No providers configured")
class TestChatErrorHandling:
    """Test error handling in chat workflow."""

    @pytest.mark.asyncio
    async def test_invalid_continuation_id(self, chat_workflow, provider_name):
        """Test handling of invalid continuation ID."""
        # Try to continue with non-existent thread ID
        result = await chat_workflow.run(
            **get_run_kwargs(
                provider_name,
                "Continue this conversation.",
                continuation_id="non-existent-thread-id",
                temperature=0.0,
            )
        )

        # Should create new conversation instead of failing
        assert result.success is True
        assert result.metadata["is_continuation"] is False

    @pytest.mark.asyncio
    async def test_empty_prompt(self, chat_workflow, provider_name):
        """Test handling of empty prompt."""
        result = await chat_workflow.run(
            **get_run_kwargs(provider_name, "", temperature=0.0)
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
                **get_run_kwargs(
                    provider_name,
                    f"This is message number {i+1}. Acknowledge it.",
                    continuation_id=thread_id,
                    temperature=0.0,
                )
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
        workflow = ChatWorkflow(
            provider=provider, conversation_memory=conversation_memory
        )

        # Gemini CLI has a bug with "My name is X" prompts. Use third-person instead.
        # See GEMINI_FAILURE_ANALYSIS.md for details.
        is_gemini = isinstance(provider, GeminiProvider)
        prompt1 = "The user is Alice." if is_gemini else "My name is Alice."
        prompt2 = "The user is Bob." if is_gemini else "My name is Bob."

        # Create two separate conversations
        result1 = await workflow.run(prompt=prompt1)
        result2 = await workflow.run(prompt=prompt2)

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
        assert result1_cont.synthesis is not None
        assert result2_cont.synthesis is not None
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
