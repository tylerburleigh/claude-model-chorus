"""
Test CLI providers implement the ModelProvider interface correctly.

This test suite verifies that all CLI-based providers (CLIProvider, ClaudeProvider,
CodexProvider, GeminiProvider, CursorAgentProvider) properly implement the
ModelProvider abstract interface.
"""

import sys
from pathlib import Path

# Add model_chorus to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model-chorus" / "src"))

import pytest
from model_chorus.providers import (
    ModelProvider,
    CLIProvider,
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    CursorAgentProvider,
    GenerationRequest,
)


class TestCLIProvidersImplementInterface:
    """Test that all CLI providers implement the ModelProvider interface."""

    @pytest.fixture
    def providers(self):
        """Create instances of all CLI providers for testing."""
        return {
            "claude": ClaudeProvider(),
            "codex": CodexProvider(),
            "gemini": GeminiProvider(),
            "cursor-agent": CursorAgentProvider(),
        }

    def test_all_providers_inherit_from_base(self, providers):
        """Verify all providers inherit from ModelProvider."""
        for name, provider in providers.items():
            assert isinstance(provider, ModelProvider), (
                f"{name} provider must be an instance of ModelProvider"
            )

    def test_all_providers_inherit_from_cli_provider(self, providers):
        """Verify all concrete providers inherit from CLIProvider."""
        for name, provider in providers.items():
            assert isinstance(provider, CLIProvider), (
                f"{name} provider must be an instance of CLIProvider"
            )

    def test_all_providers_have_generate_method(self, providers):
        """Verify all providers have the generate() method."""
        for name, provider in providers.items():
            assert hasattr(provider, "generate"), (
                f"{name} provider must have generate() method"
            )
            assert callable(provider.generate), (
                f"{name} provider generate() must be callable"
            )

    def test_all_providers_have_build_command_method(self, providers):
        """Verify all providers have the build_command() method."""
        for name, provider in providers.items():
            assert hasattr(provider, "build_command"), (
                f"{name} provider must have build_command() method"
            )
            assert callable(provider.build_command), (
                f"{name} provider build_command() must be callable"
            )

    def test_all_providers_have_parse_response_method(self, providers):
        """Verify all providers have the parse_response() method."""
        for name, provider in providers.items():
            assert hasattr(provider, "parse_response"), (
                f"{name} provider must have parse_response() method"
            )
            assert callable(provider.parse_response), (
                f"{name} provider parse_response() must be callable"
            )

    def test_all_providers_have_supports_vision_method(self, providers):
        """Verify all providers have the supports_vision() method."""
        for name, provider in providers.items():
            assert hasattr(provider, "supports_vision"), (
                f"{name} provider must have supports_vision() method"
            )
            assert callable(provider.supports_vision), (
                f"{name} provider supports_vision() must be callable"
            )

    def test_all_providers_have_get_available_models_method(self, providers):
        """Verify all providers have the get_available_models() method."""
        for name, provider in providers.items():
            assert hasattr(provider, "get_available_models"), (
                f"{name} provider must have get_available_models() method"
            )
            assert callable(provider.get_available_models), (
                f"{name} provider get_available_models() must be callable"
            )

    def test_all_providers_return_models(self, providers):
        """Verify all providers return a non-empty list of available models."""
        for name, provider in providers.items():
            models = provider.get_available_models()
            assert isinstance(models, list), (
                f"{name} provider get_available_models() must return a list"
            )
            assert len(models) > 0, (
                f"{name} provider must have at least one available model"
            )

    def test_all_providers_have_provider_name(self, providers):
        """Verify all providers have a provider_name attribute."""
        for name, provider in providers.items():
            assert hasattr(provider, "provider_name"), (
                f"{name} provider must have provider_name attribute"
            )
            assert isinstance(provider.provider_name, str), (
                f"{name} provider provider_name must be a string"
            )
            assert len(provider.provider_name) > 0, (
                f"{name} provider provider_name must not be empty"
            )

    def test_all_providers_have_cli_command(self, providers):
        """Verify all providers have a cli_command attribute."""
        for name, provider in providers.items():
            assert hasattr(provider, "cli_command"), (
                f"{name} provider must have cli_command attribute"
            )
            assert isinstance(provider.cli_command, str), (
                f"{name} provider cli_command must be a string"
            )
            assert len(provider.cli_command) > 0, (
                f"{name} provider cli_command must not be empty"
            )

    def test_build_command_returns_list(self, providers):
        """Verify build_command() returns a list of command parts."""
        request = GenerationRequest(prompt="test prompt")

        for name, provider in providers.items():
            command = provider.build_command(request)
            assert isinstance(command, list), (
                f"{name} provider build_command() must return a list"
            )
            assert len(command) > 0, (
                f"{name} provider build_command() must return non-empty list"
            )
            assert all(isinstance(part, str) for part in command), (
                f"{name} provider build_command() must return list of strings"
            )

    def test_build_command_includes_cli_command(self, providers):
        """Verify build_command() starts with the CLI command."""
        request = GenerationRequest(prompt="test prompt")

        for name, provider in providers.items():
            command = provider.build_command(request)
            assert command[0] == provider.cli_command, (
                f"{name} provider build_command() must start with cli_command"
            )

    def test_supports_vision_returns_bool(self, providers):
        """Verify supports_vision() returns a boolean."""
        for name, provider in providers.items():
            models = provider.get_available_models()
            for model in models:
                result = provider.supports_vision(model.model_id)
                assert isinstance(result, bool), (
                    f"{name} provider supports_vision() must return a boolean"
                )
