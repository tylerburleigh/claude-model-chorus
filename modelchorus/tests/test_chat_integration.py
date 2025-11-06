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


@pytest.mark.skipif(not ANY_PROVIDER_AVAILABLE, reason="No providers configured")
class TestLongConversations:
    """Test handling of long multi-turn conversations (20+ turns)."""

    @pytest.mark.asyncio
    async def test_20_turn_conversation(self, chat_workflow, provider_name):
        """Test conversation with 20 turns maintains context and stability."""
        thread_id = None
        conversation_topics = []

        # Create 20-turn conversation with varied topics
        for turn in range(20):
            # Vary the prompts to test different conversation patterns
            if turn == 0:
                prompt = "Let's count together. I'll say 1."
                conversation_topics.append("counting")
            elif turn < 5:
                prompt = f"Now you say {turn + 1}."
            elif turn == 5:
                prompt = "Good! Now let's switch to colors. My favorite is red."
                conversation_topics.append("colors")
            elif turn < 10:
                colors = ["blue", "green", "yellow", "purple"]
                prompt = f"What about {colors[turn - 6]}?"
            elif turn == 10:
                prompt = "Great! Let's talk about animals now. I like cats."
                conversation_topics.append("animals")
            elif turn < 15:
                animals = ["dogs", "birds", "fish", "rabbits"]
                prompt = f"What do you think about {animals[turn - 11]}?"
            else:
                prompt = f"This is turn {turn + 1}. Are you still following our conversation?"

            result = await chat_workflow.run(
                prompt=prompt,
                continuation_id=thread_id,
                temperature=0.7,
            )

            # Verify each turn succeeds
            assert result.success is True, f"Turn {turn + 1} failed with {provider_name}"

            if turn == 0:
                thread_id = result.metadata["thread_id"]
                assert result.metadata["is_continuation"] is False
            else:
                assert result.metadata["thread_id"] == thread_id
                assert result.metadata["is_continuation"] is True

            # Verify conversation length grows correctly
            expected_length = (turn + 1) * 2
            assert result.metadata["conversation_length"] == expected_length

        # After 20 turns, verify we can still reference early conversation
        result_final = await chat_workflow.run(
            prompt="What number did we start counting with at the very beginning?",
            continuation_id=thread_id,
            temperature=0.0,
        )

        assert result_final.success
        assert "1" in result_final.synthesis or "one" in result_final.synthesis.lower()

    @pytest.mark.asyncio
    async def test_25_turn_conversation_with_context_retention(self, chat_workflow, provider_name):
        """Test 25-turn conversation maintains context across topic changes."""
        thread_id = None

        # Initial context establishment
        result_0 = await chat_workflow.run(
            prompt="Remember this important fact: my birthday is July 15th.",
            temperature=0.0,
        )

        assert result_0.success
        thread_id = result_0.metadata["thread_id"]

        # Have a long conversation about other topics (23 turns)
        for turn in range(1, 24):
            if turn % 3 == 0:
                prompt = f"Tell me about topic number {turn}."
            elif turn % 3 == 1:
                prompt = f"Interesting! What else about that?"
            else:
                prompt = f"Got it. Let's move to something different."

            result = await chat_workflow.run(
                prompt=prompt,
                continuation_id=thread_id,
                temperature=0.7,
            )

            assert result.success, f"Turn {turn + 1} failed"

        # On turn 25, ask about the initial fact
        result_final = await chat_workflow.run(
            prompt="What is my birthday that I told you at the very start?",
            continuation_id=thread_id,
            temperature=0.0,
        )

        assert result_final.success
        # Should remember the birthday despite 23 intervening turns
        assert "july" in result_final.synthesis.lower() or "7" in result_final.synthesis
        assert "15" in result_final.synthesis

    @pytest.mark.asyncio
    async def test_conversation_length_tracking(self, chat_workflow, provider_name):
        """Test that conversation length is accurately tracked over many turns."""
        thread_id = None
        expected_lengths = []

        # Create 22 turns
        for turn in range(22):
            result = await chat_workflow.run(
                prompt=f"Message {turn + 1}",
                continuation_id=thread_id,
                temperature=0.0,
            )

            assert result.success

            if turn == 0:
                thread_id = result.metadata["thread_id"]

            # Each turn adds 2 messages (user + assistant)
            expected_length = (turn + 1) * 2
            expected_lengths.append(expected_length)

            assert result.metadata["conversation_length"] == expected_length, \
                f"Turn {turn + 1}: expected {expected_length}, got {result.metadata['conversation_length']}"

        # Verify thread state
        thread = chat_workflow.get_thread(thread_id)
        assert len(thread.messages) == 44  # 22 turns * 2 messages

    @pytest.mark.asyncio
    async def test_long_conversation_with_file_references(self, chat_workflow, provider_name, tmp_path):
        """Test long conversation with file context referenced across turns."""
        # Create test file
        test_file = tmp_path / "config.txt"
        test_file.write_text("SECRET_KEY=abc123\nDEBUG=true\n")

        # Start with file context
        result_0 = await chat_workflow.run(
            prompt="What is the SECRET_KEY in this file?",
            files=[str(test_file)],
            temperature=0.0,
        )

        assert result_0.success
        thread_id = result_0.metadata["thread_id"]
        assert "abc123" in result_0.synthesis

        # Continue for 20 more turns without the file
        for turn in range(1, 21):
            result = await chat_workflow.run(
                prompt=f"Conversation turn {turn + 1}. Just acknowledge this.",
                continuation_id=thread_id,
                temperature=0.7,
            )

            assert result.success

        # On turn 21, ask about the file content again
        result_final = await chat_workflow.run(
            prompt="What was that SECRET_KEY from the file I showed you earlier?",
            continuation_id=thread_id,
            temperature=0.0,
        )

        assert result_final.success
        # Should still remember the file content
        assert "abc123" in result_final.synthesis.lower()

    @pytest.mark.asyncio
    async def test_conversation_stability_under_load(self, chat_workflow, provider_name):
        """Test that conversation remains stable over 30 rapid turns."""
        thread_id = None

        # Rapid-fire 30 turns
        for turn in range(30):
            result = await chat_workflow.run(
                prompt=f"Turn {turn + 1}",
                continuation_id=thread_id,
                temperature=0.0,
            )

            # Every turn should succeed
            assert result.success, f"Failed at turn {turn + 1}"

            if turn == 0:
                thread_id = result.metadata["thread_id"]

            # Metadata should be consistent
            assert "thread_id" in result.metadata
            assert "conversation_length" in result.metadata
            assert result.metadata["conversation_length"] == (turn + 1) * 2

        # Verify final state
        thread = chat_workflow.get_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 60  # 30 turns * 2 messages
