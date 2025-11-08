"""
Integration tests for GeminiProvider.

These tests verify that the Gemini CLI integration works correctly with real CLI calls.
"""

import pytest
import subprocess
from model_chorus.providers.gemini_provider import GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest


class TestGeminiIntegration:
    """Integration tests for Gemini provider."""

    @pytest.fixture
    def provider(self):
        """Create a Gemini provider instance."""
        return GeminiProvider(timeout=30, retry_limit=1)

    @pytest.fixture
    def simple_request(self):
        """Create a simple generation request."""
        return GenerationRequest(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.7,
            max_tokens=10,
        )

    def test_gemini_cli_available(self):
        """Test that the Gemini CLI is available."""
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert result.returncode == 0, "Gemini CLI should be available"
        except FileNotFoundError:
            pytest.skip("Gemini CLI not installed")

    def test_build_command_basic(self, provider, simple_request):
        """Test that build_command creates correct CLI arguments."""
        command = provider.build_command(simple_request)

        # Should not include "chat" subcommand
        assert "chat" not in command, "Gemini CLI doesn't use 'chat' subcommand"

        # Should not include --temperature (not supported)
        assert "--temperature" not in command, "Gemini CLI doesn't support --temperature"

        # Should not include --json (uses --output-format instead)
        assert "--json" not in command, "Gemini CLI uses -o json, not --json"

        # Should include -o json
        assert "-o" in command and "json" in command, "Should use -o json for JSON output"

        # Prompt should be positional (not --prompt)
        assert simple_request.prompt in command, "Prompt should be in command"
        assert "--prompt" not in command, "Should use positional prompt, not --prompt"

    def test_build_command_with_model(self, provider):
        """Test command building with model metadata."""
        request = GenerationRequest(
            prompt="Test prompt",
            metadata={"model": "pro"},
        )
        command = provider.build_command(request)

        assert "-m" in command or "--model" in command, "Should include model flag"
        assert "pro" in command, "Should include model name"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_simple_query(self, provider, simple_request):
        """Test actual generation with Gemini CLI."""
        try:
            # Check if Gemini CLI is available
            subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pytest.skip("Gemini CLI not available or not working")

        # Make actual request
        response = await provider.generate(simple_request)

        # Verify response structure
        assert response is not None, "Should get a response"
        assert response.content, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        assert response.model, "Response should include model name"

        # Response should be "4" or similar
        assert "4" in response.content, "Should correctly answer 2+2=4"

    @pytest.mark.integration
    def test_parse_response_format(self, provider):
        """Test that parse_response handles Gemini CLI JSON format correctly."""
        # Simulate Gemini CLI JSON output format
        stdout = """{
            "response": "Test response content",
            "stats": {
                "models": {
                    "gemini-2.5-pro": {
                        "tokens": {
                            "prompt": 10,
                            "candidates": 20,
                            "total": 30
                        }
                    }
                }
            }
        }"""

        response = provider.parse_response(stdout, "", 0)

        assert response.content == "Test response content"
        assert response.model == "gemini-2.5-pro"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 20
        assert response.usage["total_tokens"] == 30

    def test_parse_response_error_handling(self, provider):
        """Test error handling in parse_response."""
        # Test with non-zero return code
        with pytest.raises(ValueError, match="Gemini CLI failed"):
            provider.parse_response("", "Error message", 1)

        # Test with invalid JSON
        with pytest.raises(ValueError, match="Failed to parse"):
            provider.parse_response("not valid json", "", 0)

    def test_supports_vision(self, provider):
        """Test vision capability detection."""
        assert provider.supports_vision("pro") is True
        assert provider.supports_vision("flash") is True
        assert provider.supports_vision("ultra") is True

    def test_supports_thinking(self, provider):
        """Test thinking mode capability detection."""
        assert provider.supports_thinking("pro") is True
        assert provider.supports_thinking("ultra") is True
        assert provider.supports_thinking("flash") is False
