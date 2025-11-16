"""
Tests for ClaudeProvider CLI integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from model_chorus.providers.claude_provider import ClaudeProvider
from model_chorus.providers.base_provider import GenerationRequest


class TestClaudeProvider:
    """Test suite for ClaudeProvider."""

    def test_initialization(self):
        """Test ClaudeProvider initialization."""
        provider = ClaudeProvider()
        assert provider.provider_name == "claude"
        assert provider.cli_command == "claude"
        assert provider.timeout == 120
        assert provider.retry_limit == 3

    def test_build_command_basic(self, sample_generation_request):
        """Test building basic Claude CLI command."""
        provider = ClaudeProvider()
        command = provider.build_command(sample_generation_request)

        assert command[0] == "claude"
        assert "--print" in command
        assert "--output-format" in command
        assert "json" in command
        assert "--system-prompt" in command
        # Prompt should be passed via stdin, not as command argument
        assert provider.input_data == sample_generation_request.prompt
        assert sample_generation_request.prompt not in command

    def test_build_command_with_model(self):
        """Test building command with specific model."""
        provider = ClaudeProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            metadata={"model": "opus"},
        )
        command = provider.build_command(request)

        assert "--model" in command
        assert "opus" in command

    def test_build_command_without_system_prompt(self):
        """Test building command without system prompt."""
        provider = ClaudeProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            system_prompt=None,
        )
        command = provider.build_command(request)

        assert "--system-prompt" not in command

    def test_parse_response_success(self, mock_claude_response):
        """Test parsing successful Claude CLI response."""
        provider = ClaudeProvider()
        stdout = json.dumps(mock_claude_response)
        stderr = ""
        returncode = 0

        response = provider.parse_response(stdout, stderr, returncode)

        assert response.content == "This is a test response from Claude."
        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 50
        assert response.stop_reason == "success"
        assert response.metadata["total_cost_usd"] == 0.001

    def test_parse_response_failure(self):
        """Test parsing failed CLI command."""
        provider = ClaudeProvider()
        stdout = ""
        stderr = "Error: API key not found"
        returncode = 1

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "failed with return code 1" in str(exc_info.value)
        assert "API key not found" in str(exc_info.value)

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        provider = ClaudeProvider()
        stdout = "Not valid JSON"
        stderr = ""
        returncode = 0

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "Failed to parse" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_claude_response):
        """Test successful generation with mocked subprocess."""
        provider = ClaudeProvider()
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.return_value = (
                json.dumps(mock_claude_response),
                "",
                0,
            )

            response = await provider.generate(request)

            assert response.content == "This is a test response from Claude."
            assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry(self, mock_claude_response):
        """Test generation with retry on failure."""
        provider = ClaudeProvider(retry_limit=3)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            # Fail twice, succeed on third attempt
            mock_exec.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                (json.dumps(mock_claude_response), "", 0),
            ]

            response = await provider.generate(request)

            assert response.content == "This is a test response from Claude."
            assert mock_exec.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_all_retries_fail(self):
        """Test generation when all retries fail."""
        provider = ClaudeProvider(retry_limit=2)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.side_effect = Exception("Persistent error")

            with pytest.raises(Exception) as exc_info:
                await provider.generate(request)

            assert "All 2 attempts failed" in str(exc_info.value)
            assert mock_exec.call_count == 2

    def test_supports_vision(self):
        """Test vision capability detection."""
        provider = ClaudeProvider()

        # Only sonnet and haiku support vision
        assert provider.supports_vision("sonnet") is True
        assert provider.supports_vision("haiku") is True
        assert provider.supports_vision("unknown-model") is False

    def test_supports_thinking(self):
        """Test thinking mode capability detection."""
        provider = ClaudeProvider()

        # Only sonnet and haiku support thinking mode
        assert provider.supports_thinking("sonnet") is True
        assert provider.supports_thinking("haiku") is True
        assert provider.supports_thinking("unknown-model") is False

    def test_read_only_mode_allowed_tools(self, sample_generation_request):
        """Test that read-only mode restricts tools to allowed list."""
        provider = ClaudeProvider()
        command = provider.build_command(sample_generation_request)

        # Verify allowed-tools flag is present
        assert "--allowed-tools" in command

        # Verify read-only tools are in the command
        # (they appear as separate elements after --allowed-tools)
        assert "Read" in command
        assert "Grep" in command
        assert "Glob" in command
        assert "WebSearch" in command
        assert "WebFetch" in command
        assert "Task" in command

    def test_read_only_mode_disallowed_tools(self, sample_generation_request):
        """Test that read-only mode blocks write operations."""
        provider = ClaudeProvider()
        command = provider.build_command(sample_generation_request)

        # Verify disallowed-tools flag is present
        assert "--disallowed-tools" in command

        # Verify write tools are in the command
        # (they appear as separate elements after --disallowed-tools)
        assert "Write" in command
        assert "Edit" in command
        assert "Bash" in command
