"""
Integration tests for GeminiProvider.

These tests verify that the Gemini CLI integration works correctly with real CLI calls.
"""

import subprocess

import pytest

# Import provider availability from shared test helpers
from test_helpers import GEMINI_AVAILABLE

from model_chorus.providers.base_provider import GenerationRequest
from model_chorus.providers.gemini_provider import GeminiProvider


@pytest.mark.requires_gemini
@pytest.mark.skipif(
    not GEMINI_AVAILABLE,
    reason="Gemini not available (config disabled or CLI not found)",
)
class TestGeminiIntegration:
    """Integration tests for Gemini provider."""

    @pytest.fixture
    def provider(self):
        """Create a Gemini provider instance.

        Automatically configured to use the fastest model (flash) for tests.
        """
        provider_instance = GeminiProvider(timeout=30, retry_limit=1)

        # Wrap the generate method to inject the fast model
        original_generate = provider_instance.generate

        async def generate_with_fast_model(request):
            # Use gemini-2.5-flash (fastest) if no model specified
            if "model" not in request.metadata:
                request.metadata["model"] = "gemini-2.5-flash"
            return await original_generate(request)

        provider_instance.generate = generate_with_fast_model
        return provider_instance

    @pytest.fixture
    def simple_request(self):
        """Create a simple generation request."""
        return GenerationRequest(
            prompt="What is 2+2? Answer with just the number.",
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
        assert (
            "--temperature" not in command
        ), "Gemini CLI doesn't support --temperature"

        # Should not include --json (uses --output-format instead)
        assert (
            "--json" not in command
        ), "Gemini CLI uses --output-format json, not --json"

        # Should include --output-format json
        assert (
            "--output-format" in command and "json" in command
        ), "Should use --output-format json for JSON output"

        # Prompt is passed as positional argument (after all flags)
        # The -p flag only works with shell=True, not with subprocess.exec
        # Gemini prepends "Human: " to prompts, so check for that
        assert (
            command[-1].endswith(simple_request.prompt)
            or simple_request.prompt in command[-1]
        ), f"Prompt should be in command as positional argument. Got: {command[-1]}"
        assert (
            "-p" not in command
        ), "Should not use -p flag (use positional arg instead)"

        # Verify input_data is NOT set (we use positional args, not stdin)
        command_result = provider.build_command(simple_request)  # Reset state
        assert (
            not hasattr(provider, "input_data") or provider.input_data is None
        ), "Provider should not use input_data (positional args used instead)"

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
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
        ):
            pytest.skip("Gemini CLI not available or not working")

        # Make actual request
        try:
            response = await provider.generate(simple_request)
        except Exception as e:
            # Skip if Gemini API is unavailable (e.g., error code 144, API key issues)
            if (
                "144" in str(e)
                or "API" in str(e).upper()
                or "authentication" in str(e).lower()
            ):
                pytest.skip(f"Gemini API unavailable or authentication failed: {e}")
            raise  # Re-raise if it's a different error

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
        # Gemini models support vision
        assert provider.supports_vision("gemini-2.5-pro") is True
        assert provider.supports_vision("gemini-2.5-flash") is True
        # Models not in VISION_MODELS don't support vision
        assert provider.supports_vision("other-model") is False

    def test_supports_thinking(self, provider):
        """Test thinking mode capability detection."""
        # Pro and Flash support thinking mode
        assert provider.supports_thinking("gemini-2.5-pro") is True
        assert provider.supports_thinking("gemini-2.5-flash") is True
        # Models not in THINKING_MODELS don't support thinking
        assert provider.supports_thinking("other-model") is False
