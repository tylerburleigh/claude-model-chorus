"""
Tests for CodexProvider CLI integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from modelchorus.providers.codex_provider import CodexProvider
from modelchorus.providers.base_provider import GenerationRequest


class TestCodexProvider:
    """Test suite for CodexProvider."""

    def test_initialization(self):
        """Test CodexProvider initialization."""
        provider = CodexProvider()
        assert provider.provider_name == "codex"
        assert provider.cli_command == "codex"
        assert provider.timeout == 120
        assert provider.retry_limit == 3

    def test_build_command_basic(self, sample_generation_request):
        """Test building basic Codex CLI command."""
        provider = CodexProvider()
        command = provider.build_command(sample_generation_request)

        assert command[0] == "codex"
        assert command[1] == "exec"
        assert "--json" in command
        assert sample_generation_request.prompt in command

    def test_build_command_with_model(self):
        """Test building command with specific model."""
        provider = CodexProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            metadata={"model": "gpt4"},
        )
        command = provider.build_command(request)

        assert "--model" in command
        assert "gpt4" in command

    def test_build_command_with_images(self):
        """Test building command with image attachments."""
        provider = CodexProvider()
        request = GenerationRequest(
            prompt="Test prompt",
            images=["/path/to/image.png"],
        )
        command = provider.build_command(request)

        assert "--image" in command
        assert "/path/to/image.png" in command

    def test_parse_response_success(self, mock_codex_response):
        """Test parsing successful Codex CLI JSONL response."""
        provider = CodexProvider()
        stdout = mock_codex_response
        stderr = ""
        returncode = 0

        response = provider.parse_response(stdout, stderr, returncode)

        assert response.content == "This is a test response from Codex."
        assert response.model == "gpt-5-codex"
        assert response.usage["input_tokens"] == 15
        assert response.usage["output_tokens"] == 45
        assert response.usage["cached_input_tokens"] == 0
        assert response.stop_reason == "completed"
        assert response.metadata["thread_id"] == "test-thread-123"

    def test_parse_response_failure(self):
        """Test parsing failed CLI command."""
        provider = CodexProvider()
        stdout = ""
        stderr = "Error: Model not found"
        returncode = 2

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "failed with return code 2" in str(exc_info.value)
        assert "Model not found" in str(exc_info.value)

    def test_parse_response_invalid_jsonl(self):
        """Test parsing invalid JSONL response."""
        provider = CodexProvider()
        stdout = "Not valid JSONL\nAlso not valid"
        stderr = ""
        returncode = 0

        with pytest.raises(ValueError) as exc_info:
            provider.parse_response(stdout, stderr, returncode)

        assert "Failed to parse" in str(exc_info.value)

    def test_parse_response_missing_agent_message(self):
        """Test parsing JSONL without agent_message event."""
        provider = CodexProvider()
        stdout = """{"type":"thread.started","thread_id":"test-123"}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":20}}"""
        stderr = ""
        returncode = 0

        response = provider.parse_response(stdout, stderr, returncode)

        # Should handle gracefully with empty content
        assert response.content == ""
        assert response.usage["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_codex_response):
        """Test successful generation with mocked subprocess."""
        provider = CodexProvider()
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.return_value = (
                mock_codex_response,
                "",
                0,
            )

            response = await provider.generate(request)

            assert response.content == "This is a test response from Codex."
            assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry(self, mock_codex_response):
        """Test generation with retry on failure."""
        provider = CodexProvider(retry_limit=3)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            # Fail twice, succeed on third attempt
            mock_exec.side_effect = [
                Exception("Connection reset"),
                Exception("Request timeout"),
                (mock_codex_response, "", 0),
            ]

            response = await provider.generate(request)

            assert response.content == "This is a test response from Codex."
            assert mock_exec.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_all_retries_fail(self):
        """Test generation when all retries fail."""
        provider = CodexProvider(retry_limit=2)
        request = GenerationRequest(prompt="Test prompt")

        with patch.object(provider, "execute_command") as mock_exec:
            mock_exec.side_effect = Exception("Fatal error")

            with pytest.raises(Exception) as exc_info:
                await provider.generate(request)

            assert "All 2 attempts failed" in str(exc_info.value)
            assert mock_exec.call_count == 2

    def test_supports_vision(self):
        """Test vision capability detection."""
        provider = CodexProvider()

        assert provider.supports_vision("gpt4") is True
        assert provider.supports_vision("gpt4-turbo") is True
        assert provider.supports_vision("gpt35-turbo") is False

    def test_supports_function_calling(self):
        """Test function calling capability detection."""
        provider = CodexProvider()

        assert provider.supports_function_calling("gpt4") is True
        assert provider.supports_function_calling("gpt4-turbo") is True
        assert provider.supports_function_calling("gpt35-turbo") is True
