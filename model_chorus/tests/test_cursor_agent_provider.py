"""
Tests for CursorAgentProvider CLI integration.
"""

import json
from unittest.mock import patch

import pytest

from model_chorus.providers.base_provider import GenerationRequest
from model_chorus.providers.cursor_agent_provider import CursorAgentProvider


class TestCursorAgentProvider:
    """Test suite for CursorAgentProvider."""

    def test_initialization(self):
        """Test CursorAgentProvider initialization."""
        provider = CursorAgentProvider()
        assert provider.provider_name == "cursor-agent"
        assert provider.cli_command == "cursor-agent"
        assert provider.timeout == 120
        assert provider.retry_limit == 3

    def test_build_command_basic(self, sample_generation_request):
        """Test building basic Cursor Agent CLI command."""
        provider = CursorAgentProvider()
        command = provider.build_command(sample_generation_request)

        assert command[0] == "cursor-agent"
        assert "-p" in command
        assert "--output-format" in command
        assert "json" in command
        # Prompt should be passed as positional argument (combined with system prompt if present)
        # sample_generation_request has both system_prompt and prompt
        assert (
            sample_generation_request.prompt in command[-1]
        )  # Check it's in the combined prompt

    def test_build_command_with_model(self):
        """Test building command with specific model."""
        provider = CursorAgentProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            metadata={"model": "composer-1"},
        )
        command = provider.build_command(request)

        assert "--model" in command
        assert "composer-1" in command

    def test_build_command_with_system_prompt(self):
        """Test building command with system prompt."""
        provider = CursorAgentProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            system_prompt="You are a helpful coding assistant.",
        )
        command = provider.build_command(request)

        # System prompt should be combined with user prompt
        full_prompt = command[-1]
        assert "You are a helpful coding assistant." in full_prompt
        assert "Test prompt" in full_prompt

    def test_build_command_without_system_prompt(self):
        """Test building command without system prompt."""
        provider = CursorAgentProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            system_prompt=None,
        )
        command = provider.build_command(request)

        # Only user prompt should be present
        assert command[-1] == "Test prompt"

    def test_parse_response_success(self, mock_cursor_agent_response):
        """Test parsing successful Cursor Agent CLI response."""
        provider = CursorAgentProvider()
        stdout = json.dumps(mock_cursor_agent_response)
        stderr = ""
        returncode = 0

        response = provider.parse_response(stdout, stderr, returncode)

        assert response.content == "This is a test response from Cursor Agent."
        assert response.model == "cursor-agent"
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 48
        assert response.usage.total_tokens == 60
        assert response.thread_id == "test-session-456"

    def test_parse_response_failure(self):
        """Test parsing failed CLI command."""
        provider = CursorAgentProvider()
        stdout = ""
        stderr = "Error: Invalid model specified"
        returncode = 1

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "failed with return code 1" in str(exc_info.value)
        assert "Invalid model specified" in str(exc_info.value)

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        provider = CursorAgentProvider()
        stdout = "Not valid JSON"
        stderr = ""
        returncode = 0

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "Failed to parse" in str(exc_info.value)

    def test_parse_response_error_result(self):
        """Test parsing error response from Cursor Agent."""
        provider = CursorAgentProvider()
        error_response = {
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "Model not found",
            "session_id": "test-session-123",
        }
        stdout = json.dumps(error_response)
        stderr = ""
        returncode = 0

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "returned error" in str(exc_info.value)
        assert "Model not found" in str(exc_info.value)

    def test_parse_response_with_session_id(self):
        """Test that session_id is extracted as thread_id."""
        provider = CursorAgentProvider()
        response_data = {
            "type": "result",
            "subtype": "success",
            "result": "Test response",
            "session_id": "session-abc-123",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        stdout = json.dumps(response_data)
        stderr = ""
        returncode = 0

        response = provider.parse_response(stdout, stderr, returncode)

        assert response.thread_id == "session-abc-123"

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_cursor_agent_response):
        """Test successful generation with mocked subprocess."""
        provider = CursorAgentProvider()
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.return_value = (
                json.dumps(mock_cursor_agent_response),
                "",
                0,
            )

            response = await provider.generate(request)

            assert response.content == "This is a test response from Cursor Agent."
            assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry(self, mock_cursor_agent_response):
        """Test generation with retry on failure."""
        provider = CursorAgentProvider(retry_limit=3)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            # Fail twice, succeed on third attempt
            mock_exec.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                (json.dumps(mock_cursor_agent_response), "", 0),
            ]

            response = await provider.generate(request)

            assert response.content == "This is a test response from Cursor Agent."
            assert mock_exec.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_all_retries_fail(self):
        """Test generation when all retries fail."""
        provider = CursorAgentProvider(retry_limit=2)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.side_effect = Exception("Persistent error")

            with pytest.raises(Exception) as exc_info:
                await provider.generate(request)

            assert "All 2 attempts failed" in str(exc_info.value)
            assert mock_exec.call_count == 2

    def test_supports_vision(self):
        """Test vision capability detection."""
        provider = CursorAgentProvider()

        # Cursor Agent models don't support vision
        assert provider.supports_vision("composer-1") is False
        assert provider.supports_vision("gpt-5-codex") is False

    def test_supports_code_generation(self):
        """Test code generation capability detection."""
        provider = CursorAgentProvider()

        # Models in FUNCTION_CALLING_MODELS support code generation
        assert provider.supports_code_generation("composer-1") is True
        assert provider.supports_code_generation("gpt-5-codex") is True
        # Models not in the set don't support it
        assert provider.supports_code_generation("unknown-model") is False

    def test_read_only_mode_by_default(self, sample_generation_request):
        """Test that read-only mode is enabled by default (no --force flag)."""
        provider = CursorAgentProvider()
        command = provider.build_command(sample_generation_request)

        # Verify --force flag is NOT present (read-only/propose-only mode)
        assert "--force" not in command
