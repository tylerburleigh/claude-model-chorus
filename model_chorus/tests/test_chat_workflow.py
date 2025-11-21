"""
Tests for ChatWorkflow functionality.

Tests conversation initiation, continuation, file context, and conversation tracking.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_chorus.workflows import ChatWorkflow

# Note: mock_provider and conversation_memory fixtures are now in conftest.py


@pytest.fixture
def chat_workflow(mock_provider, conversation_memory):
    """Create ChatWorkflow instance for testing."""
    return ChatWorkflow(
        provider=mock_provider,
        conversation_memory=conversation_memory,
    )


class TestChatWorkflowInitialization:
    """Test ChatWorkflow initialization."""

    def test_initialization_with_provider(self, mock_provider, conversation_memory):
        """Test basic initialization with provider."""
        workflow = ChatWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory,
        )

        assert workflow.name == "Chat"
        assert (
            workflow.description
            == "Single-model peer consultation with conversation threading"
        )
        assert workflow.provider == mock_provider
        assert workflow.conversation_memory == conversation_memory

    def test_initialization_without_memory(self, mock_provider):
        """Test initialization without conversation memory."""
        workflow = ChatWorkflow(provider=mock_provider)

        assert workflow.provider == mock_provider
        assert workflow.conversation_memory is None

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            ChatWorkflow(provider=None)

    def test_validate_config(self, chat_workflow):
        """Test config validation."""
        assert chat_workflow.validate_config() is True

    def test_get_provider(self, chat_workflow, mock_provider):
        """Test get_provider method."""
        assert chat_workflow.get_provider() == mock_provider


class TestConversationInitiation:
    """Test conversation creation and initiation."""

    @pytest.mark.asyncio
    async def test_new_conversation_creates_thread_id(self, chat_workflow):
        """Test that a new conversation creates a unique thread ID."""
        result = await chat_workflow.run(prompt="Hello, world!")

        assert result.success is True
        assert "thread_id" in result.metadata
        assert result.metadata["is_continuation"] is False
        assert (
            result.metadata["conversation_length"] == 2
        )  # Current turn: user + assistant

    @pytest.mark.asyncio
    async def test_new_conversation_has_response(self, chat_workflow):
        """Test that a new conversation gets a response."""
        result = await chat_workflow.run(prompt="What is 2+2?")

        assert result.success is True
        assert result.synthesis is not None
        assert len(result.synthesis) > 0
        assert "Response to: What is 2+2?" in result.synthesis

    @pytest.mark.asyncio
    async def test_new_conversation_without_memory(self, mock_provider):
        """Test conversation without memory still works."""
        workflow = ChatWorkflow(provider=mock_provider, conversation_memory=None)

        result = await workflow.run(prompt="Hello!")

        assert result.success is True
        assert result.synthesis is not None
        assert result.metadata["conversation_length"] == 0

    @pytest.mark.asyncio
    async def test_provider_generate_called_with_correct_params(
        self, chat_workflow, mock_provider
    ):
        """Test that provider.generate is called with correct parameters."""
        await chat_workflow.run(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=200,
        )

        # Verify provider.generate was called
        assert mock_provider.generate.called
        call_args = mock_provider.generate.call_args[0][0]  # Get GenerationRequest

        assert "Test prompt" in call_args.prompt
        assert call_args.temperature == 0.5
        assert call_args.max_tokens == 200


class TestConversationContinuation:
    """Test conversation continuation functionality."""

    @pytest.mark.asyncio
    async def test_continuation_uses_same_thread_id(self, chat_workflow):
        """Test that continuation uses the same thread ID."""
        # First message
        result1 = await chat_workflow.run(prompt="First message")
        thread_id = result1.metadata["thread_id"]

        # Continue conversation
        result2 = await chat_workflow.run(
            prompt="Second message",
            continuation_id=thread_id,
        )

        assert result2.success is True
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["is_continuation"] is True

    @pytest.mark.asyncio
    async def test_continuation_includes_history(self, chat_workflow, mock_provider):
        """Test that continuation includes conversation history in prompt."""
        # First message
        result1 = await chat_workflow.run(prompt="What is machine learning?")
        thread_id = result1.metadata["thread_id"]

        # Reset mock to track second call
        mock_provider.generate.reset_mock()

        # Continue conversation
        result2 = await chat_workflow.run(
            prompt="Give me an example",
            continuation_id=thread_id,
        )

        # Check that the prompt includes previous conversation
        call_args = mock_provider.generate.call_args[0][0]
        prompt_sent = call_args.prompt

        assert "Previous conversation:" in prompt_sent
        assert "USER:" in prompt_sent
        assert "ASSISTANT:" in prompt_sent
        assert "What is machine learning?" in prompt_sent

    @pytest.mark.asyncio
    async def test_continuation_tracks_message_count(self, chat_workflow):
        """Test that conversation length is tracked correctly."""
        # First message
        result1 = await chat_workflow.run(prompt="First")
        thread_id = result1.metadata["thread_id"]
        assert result1.metadata["conversation_length"] == 2  # user + assistant

        # Second message
        result2 = await chat_workflow.run("Second", continuation_id=thread_id)
        assert result2.metadata["conversation_length"] == 4  # 2 turns * 2 messages

        # Third message
        result3 = await chat_workflow.run("Third", continuation_id=thread_id)
        assert result3.metadata["conversation_length"] == 6  # 3 turns * 2 messages

    @pytest.mark.asyncio
    async def test_multiple_continuations(self, chat_workflow):
        """Test multiple conversation continuations."""
        thread_id = None

        for i in range(5):
            result = await chat_workflow.run(
                prompt=f"Message {i+1}",
                continuation_id=thread_id,
            )

            assert result.success is True

            if i == 0:
                thread_id = result.metadata["thread_id"]
                assert result.metadata["is_continuation"] is False
            else:
                assert result.metadata["thread_id"] == thread_id
                assert result.metadata["is_continuation"] is True


class TestFileContext:
    """Test file context handling."""

    @pytest.mark.asyncio
    async def test_file_context_included_in_prompt(
        self, chat_workflow, mock_provider, tmp_path
    ):
        """Test that file contents are included in the prompt."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test file content.")

        await chat_workflow.run(
            prompt="Review this file",
            files=[str(test_file)],
        )

        # Check that file content was included in prompt
        call_args = mock_provider.generate.call_args[0][0]
        prompt_sent = call_args.prompt

        assert "File context:" in prompt_sent
        assert str(test_file) in prompt_sent
        assert "This is test file content." in prompt_sent

    @pytest.mark.asyncio
    async def test_multiple_files_included(
        self, chat_workflow, mock_provider, tmp_path
    ):
        """Test that multiple files are included in context."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content of file 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content of file 2")

        await chat_workflow.run(
            prompt="Review these files",
            files=[str(file1), str(file2)],
        )

        call_args = mock_provider.generate.call_args[0][0]
        prompt_sent = call_args.prompt

        assert "Content of file 1" in prompt_sent
        assert "Content of file 2" in prompt_sent

    @pytest.mark.asyncio
    async def test_file_not_found_handled_gracefully(
        self, chat_workflow, mock_provider
    ):
        """Test that missing files are handled gracefully."""
        result = await chat_workflow.run(
            prompt="Review this file",
            files=["/nonexistent/file.txt"],
        )

        # Should still succeed, just without file content
        assert result.success is True

        # Check that error is noted in prompt
        call_args = mock_provider.generate.call_args[0][0]
        prompt_sent = call_args.prompt

        assert "Failed to read" in prompt_sent or "file.txt" in prompt_sent

    @pytest.mark.asyncio
    async def test_file_references_stored_in_conversation(
        self, chat_workflow, tmp_path
    ):
        """Test that file references are stored in conversation memory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        result = await chat_workflow.run(
            prompt="Review this",
            files=[str(test_file)],
        )

        thread_id = result.metadata["thread_id"]
        thread = chat_workflow.get_thread(thread_id)

        # Check that user message has files
        user_messages = [msg for msg in thread.messages if msg.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].files is not None
        assert str(test_file) in user_messages[0].files


class TestConversationTracking:
    """Test conversation history and tracking."""

    @pytest.mark.asyncio
    async def test_get_thread_retrieves_conversation(self, chat_workflow):
        """Test that get_thread retrieves conversation history."""
        result = await chat_workflow.run(prompt="Test message")
        thread_id = result.metadata["thread_id"]

        thread = chat_workflow.get_thread(thread_id)

        assert thread is not None
        assert thread.thread_id == thread_id
        assert len(thread.messages) == 2  # User message + Assistant response

    @pytest.mark.asyncio
    async def test_get_thread_returns_none_for_invalid_id(self, chat_workflow):
        """Test that get_thread returns None for invalid thread ID."""
        thread = chat_workflow.get_thread("invalid-thread-id")
        assert thread is None

    @pytest.mark.asyncio
    async def test_get_thread_without_memory_returns_none(self, mock_provider):
        """Test that get_thread returns None when no memory is available."""
        workflow = ChatWorkflow(provider=mock_provider, conversation_memory=None)

        thread = workflow.get_thread("any-id")
        assert thread is None

    @pytest.mark.asyncio
    async def test_get_message_count(self, chat_workflow):
        """Test get_message_count method."""
        result = await chat_workflow.run(prompt="First message")
        thread_id = result.metadata["thread_id"]

        # Initial count
        count = chat_workflow.get_message_count(thread_id)
        assert count == 2  # user + assistant

        # After continuation
        await chat_workflow.run("Second message", continuation_id=thread_id)
        count = chat_workflow.get_message_count(thread_id)
        assert count == 4  # 2 turns * 2 messages each


class TestErrorHandling:
    """Test error handling in ChatWorkflow."""

    @pytest.mark.asyncio
    async def test_provider_error_handled(self, conversation_memory):
        """Test that provider errors are handled gracefully."""
        # Create provider that raises an error
        error_provider = MagicMock()
        error_provider.provider_name = "error-provider"
        error_provider.validate_api_key = MagicMock(return_value=True)

        async def mock_generate_error(request):
            raise Exception("Provider error")

        error_provider.generate = AsyncMock(side_effect=mock_generate_error)

        # Mock async check_availability method
        async def mock_check_availability():
            return (True, None)  # is_available=True, error=None

        error_provider.check_availability = AsyncMock(
            side_effect=mock_check_availability
        )

        workflow = ChatWorkflow(
            provider=error_provider,
            conversation_memory=conversation_memory,
        )

        result = await workflow.run(prompt="Test")

        assert result.success is False
        assert result.error is not None
        assert "Provider error" in result.error

    @pytest.mark.asyncio
    async def test_get_result_returns_last_result(self, chat_workflow):
        """Test that get_result returns the most recent result."""
        result1 = await chat_workflow.run(prompt="First")
        result2 = await chat_workflow.run(prompt="Second")

        last_result = chat_workflow.get_result()

        assert last_result == result2
        assert last_result != result1
