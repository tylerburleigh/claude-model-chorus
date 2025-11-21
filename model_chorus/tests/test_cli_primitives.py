"""
Unit tests for CLI primitives module.

Tests verify the three primitive classes that extract common patterns from
CLI command implementations:
- ProviderResolver: Provider initialization with error handling
- WorkflowContext: Config resolution and prompt validation
- OutputFormatter: Standardized console output formatting
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import typer

from model_chorus.cli.primitives import (
    OutputFormatter,
    ProviderResolver,
    WorkflowContext,
)

# Import actual exceptions from their proper locations
from model_chorus.providers.cli_provider import (
    ProviderDisabledError,
    ProviderUnavailableError,
)


class TestProviderResolver:
    """Test suite for ProviderResolver class."""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_init_stores_dependencies(self):
        """Test ProviderResolver stores all dependencies correctly."""
        config = Mock()
        get_provider_fn = Mock()
        get_install_cmd_fn = Mock()

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=get_provider_fn,
            get_install_command_fn=get_install_cmd_fn,
            verbose=True,
        )

        assert resolver.config is config
        assert resolver.get_provider_fn is get_provider_fn
        assert resolver.get_install_command_fn is get_install_cmd_fn
        assert resolver.verbose is True

    def test_init_defaults_verbose_false(self):
        """Test verbose defaults to False."""
        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=Mock(),
            get_install_command_fn=Mock(),
        )

        assert resolver.verbose is False

    # ========================================================================
    # resolve_provider() Tests - Success Cases
    # ========================================================================

    def test_resolve_provider_success(self):
        """Test successful provider resolution."""
        mock_provider = Mock()
        get_provider_fn = Mock(return_value=mock_provider)

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=False,
        )

        result = resolver.resolve_provider("claude", timeout=30.0)

        assert result is mock_provider
        get_provider_fn.assert_called_once_with("claude", timeout=30)

    def test_resolve_provider_casts_timeout_to_int(self):
        """Test timeout is cast to int when calling provider function."""
        get_provider_fn = Mock(return_value=Mock())

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
        )

        resolver.resolve_provider("gemini", timeout=45.7)

        get_provider_fn.assert_called_once_with("gemini", timeout=45)

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_provider_verbose_output(self, mock_console):
        """Test verbose mode prints initialization messages."""
        mock_provider = Mock()
        get_provider_fn = Mock(return_value=mock_provider)

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=True,
        )

        resolver.resolve_provider("claude", timeout=30.0)

        assert mock_console.print.call_count == 2
        mock_console.print.assert_any_call(
            "[cyan]Initializing provider: claude[/cyan]"
        )
        mock_console.print.assert_any_call(
            "[green]✓ claude initialized[/green]"
        )

    # ========================================================================
    # resolve_provider() Tests - Error Cases
    # ========================================================================

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_provider_disabled_error(self, mock_console):
        """Test ProviderDisabledError shows helpful message and exits."""
        get_provider_fn = Mock(
            side_effect=ProviderDisabledError(
                "Provider 'gemini' is disabled in configuration"
            )
        )

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
        )

        with pytest.raises(typer.Exit) as exc_info:
            resolver.resolve_provider("gemini", timeout=30.0)

        assert exc_info.value.exit_code == 1
        assert mock_console.print.call_count == 2
        mock_console.print.assert_any_call(
            "[red]Error: Provider 'gemini' is disabled in configuration[/red]"
        )

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_provider_unavailable_error(self, mock_console):
        """Test ProviderUnavailableError shows installation instructions."""
        get_provider_fn = Mock(
            side_effect=ProviderUnavailableError(
                provider_name="gemini",
                reason="gemini CLI not found",
                suggestions=["Install gemini CLI", "Add to PATH"],
            )
        )
        get_install_cmd_fn = Mock(return_value="pip install gemini-cli")

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=get_install_cmd_fn,
        )

        with pytest.raises(typer.Exit) as exc_info:
            resolver.resolve_provider("gemini", timeout=30.0)

        assert exc_info.value.exit_code == 1

        # Verify error messages were printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("gemini CLI not found" in call for call in calls)
        assert any("Installation" in call for call in calls)
        assert any("pip install gemini-cli" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_provider_generic_error(self, mock_console):
        """Test generic exceptions show error and exit."""
        get_provider_fn = Mock(side_effect=RuntimeError("Unexpected error"))

        resolver = ProviderResolver(
            config=Mock(),
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
        )

        with pytest.raises(typer.Exit) as exc_info:
            resolver.resolve_provider("claude", timeout=30.0)

        assert exc_info.value.exit_code == 1
        mock_console.print.assert_called_once()
        assert "Failed to initialize claude" in str(mock_console.print.call_args)

    # ========================================================================
    # resolve_fallback_providers() Tests
    # ========================================================================

    def test_resolve_fallback_providers_success(self):
        """Test successful fallback provider initialization."""
        config = Mock()
        config.get_workflow_fallback_providers.return_value = [
            "gemini",
            "codex",
        ]

        provider1 = Mock()
        provider2 = Mock()
        get_provider_fn = Mock(side_effect=[provider1, provider2])

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=False,
        )

        result = resolver.resolve_fallback_providers(
            workflow_name="consensus",
            exclude_provider="claude",
            timeout=30.0,
        )

        assert result == [provider1, provider2]
        config.get_workflow_fallback_providers.assert_called_once_with(
            "consensus", exclude_provider="claude"
        )

    def test_resolve_fallback_providers_empty_list(self):
        """Test handling of no fallback providers in config."""
        config = Mock()
        config.get_workflow_fallback_providers.return_value = []

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=Mock(),
            get_install_command_fn=Mock(),
        )

        result = resolver.resolve_fallback_providers(
            workflow_name="chat",
            exclude_provider="claude",
            timeout=30.0,
        )

        assert result == []

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_fallback_providers_skips_disabled(self, mock_console):
        """Test disabled fallback providers are skipped silently."""
        from model_chorus.providers.cli_provider import ProviderDisabledError

        config = Mock()
        config.get_workflow_fallback_providers.return_value = [
            "gemini",
            "codex",
        ]

        provider2 = Mock()
        get_provider_fn = Mock(
            side_effect=[
                ProviderDisabledError("gemini disabled"),
                provider2,
            ]
        )

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=True,
        )

        result = resolver.resolve_fallback_providers(
            workflow_name="consensus",
            exclude_provider="claude",
            timeout=30.0,
        )

        assert result == [provider2]
        assert any(
            "Skipping disabled fallback provider gemini" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_fallback_providers_handles_errors(self, mock_console):
        """Test fallback provider errors are caught and logged."""
        config = Mock()
        config.get_workflow_fallback_providers.return_value = [
            "gemini",
            "codex",
        ]

        provider2 = Mock()
        get_provider_fn = Mock(
            side_effect=[
                RuntimeError("Connection failed"),
                provider2,
            ]
        )

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=True,
        )

        result = resolver.resolve_fallback_providers(
            workflow_name="consensus",
            exclude_provider="claude",
            timeout=30.0,
        )

        assert result == [provider2]
        assert any(
            "Could not initialize fallback gemini" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_fallback_providers_verbose_output(self, mock_console):
        """Test verbose mode shows fallback initialization progress."""
        config = Mock()
        config.get_workflow_fallback_providers.return_value = ["gemini"]

        provider = Mock()
        get_provider_fn = Mock(return_value=provider)

        resolver = ProviderResolver(
            config=config,
            get_provider_fn=get_provider_fn,
            get_install_command_fn=Mock(),
            verbose=True,
        )

        resolver.resolve_fallback_providers(
            workflow_name="consensus",
            exclude_provider="claude",
            timeout=30.0,
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Initializing fallback providers" in call for call in calls)
        assert any("✓ gemini initialized (fallback)" in call for call in calls)


class TestWorkflowContext:
    """Test suite for WorkflowContext class."""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_init_stores_dependencies(self):
        """Test WorkflowContext stores all dependencies correctly."""
        config = Mock()
        construct_fn = Mock()

        context = WorkflowContext(
            workflow_name="chat",
            config=config,
            construct_prompt_with_files_fn=construct_fn,
        )

        assert context.workflow_name == "chat"
        assert context.config is config
        assert context.construct_prompt_with_files_fn is construct_fn

    # ========================================================================
    # validate_and_get_prompt() Tests
    # ========================================================================

    def test_validate_prompt_from_positional_arg(self):
        """Test prompt validation with positional argument."""
        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.validate_and_get_prompt(
            prompt_arg="Hello world",
            prompt_flag=None,
        )

        assert result == "Hello world"

    def test_validate_prompt_from_flag(self):
        """Test prompt validation with --prompt flag."""
        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.validate_and_get_prompt(
            prompt_arg=None,
            prompt_flag="What is 2+2?",
        )

        assert result == "What is 2+2?"

    @patch("model_chorus.cli.primitives.console")
    def test_validate_prompt_missing_both(self, mock_console):
        """Test error when neither prompt arg nor flag provided."""
        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=Mock(),
        )

        with pytest.raises(typer.Exit) as exc_info:
            context.validate_and_get_prompt(
                prompt_arg=None,
                prompt_flag=None,
            )

        assert exc_info.value.exit_code == 1
        mock_console.print.assert_called_once()
        assert "Prompt is required" in str(mock_console.print.call_args)

    @patch("model_chorus.cli.primitives.console")
    def test_validate_prompt_both_provided(self, mock_console):
        """Test error when both prompt arg and flag provided."""
        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=Mock(),
        )

        with pytest.raises(typer.Exit) as exc_info:
            context.validate_and_get_prompt(
                prompt_arg="Hello",
                prompt_flag="World",
            )

        assert exc_info.value.exit_code == 1
        mock_console.print.assert_called_once()
        assert "Cannot specify prompt both" in str(mock_console.print.call_args)

    # ========================================================================
    # resolve_config_defaults() Tests
    # ========================================================================

    def test_resolve_config_defaults_all_provided(self):
        """Test config resolution when all values provided via CLI."""
        config = Mock()

        context = WorkflowContext(
            workflow_name="chat",
            config=config,
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.resolve_config_defaults(
            provider="gemini",
            system="Custom system prompt",
            timeout=60.0,
        )

        assert result == {
            "provider": "gemini",
            "system": "Custom system prompt",
            "timeout": 60.0,
        }
        # Config should not be consulted when all values provided
        assert not config.get_workflow_default_provider.called
        assert not config.get_workflow_default.called

    def test_resolve_config_defaults_from_config(self):
        """Test config resolution pulls defaults from config."""
        config = Mock()
        config.get_workflow_default_provider.return_value = "claude"
        config.get_workflow_default.side_effect = lambda wf, key, default: {
            "system_prompt": "Be helpful",
            "timeout": 120.0,
        }.get(key, default)

        context = WorkflowContext(
            workflow_name="chat",
            config=config,
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.resolve_config_defaults()

        assert result == {
            "provider": "claude",
            "system": "Be helpful",
            "timeout": 120.0,
        }
        config.get_workflow_default_provider.assert_called_once_with(
            "chat", "claude"
        )

    def test_resolve_config_defaults_custom_default_provider(self):
        """Test custom default provider fallback."""
        config = Mock()
        config.get_workflow_default_provider.return_value = "gemini"
        config.get_workflow_default.side_effect = lambda wf, key, default: default

        context = WorkflowContext(
            workflow_name="consensus",
            config=config,
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.resolve_config_defaults(default_provider="gemini")

        assert result["provider"] == "gemini"
        config.get_workflow_default_provider.assert_called_once_with(
            "consensus", "gemini"
        )

    @patch("model_chorus.cli.primitives.console")
    def test_resolve_config_defaults_disabled_provider(self, mock_console):
        """Test error when default provider is disabled."""
        config = Mock()
        config.get_workflow_default_provider.return_value = None

        context = WorkflowContext(
            workflow_name="chat",
            config=config,
            construct_prompt_with_files_fn=Mock(),
        )

        with pytest.raises(typer.Exit) as exc_info:
            context.resolve_config_defaults()

        assert exc_info.value.exit_code == 1
        assert mock_console.print.call_count == 2
        assert any(
            "Default provider for 'chat' workflow is disabled" in str(call)
            for call in mock_console.print.call_args_list
        )

    def test_resolve_config_defaults_partial_override(self):
        """Test config resolution with partial CLI overrides."""
        config = Mock()
        config.get_workflow_default_provider.return_value = "claude"
        config.get_workflow_default.side_effect = lambda wf, key, default: {
            "system_prompt": "Be concise",
            "timeout": 90.0,
        }.get(key, default)

        context = WorkflowContext(
            workflow_name="chat",
            config=config,
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.resolve_config_defaults(
            provider="gemini",  # Override
            system=None,  # Use config
            timeout=None,  # Use config
        )

        assert result == {
            "provider": "gemini",
            "system": "Be concise",
            "timeout": 90.0,
        }

    # ========================================================================
    # create_memory() Tests
    # ========================================================================

    def test_create_memory(self):
        """Test memory creation returns ConversationMemory instance."""
        from model_chorus.core.conversation import ConversationMemory

        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=Mock(),
        )

        result = context.create_memory()

        assert isinstance(result, ConversationMemory)

    # ========================================================================
    # prepare_prompt_with_files() Tests
    # ========================================================================

    def test_prepare_prompt_with_files_no_files(self):
        """Test prompt preparation without files."""
        construct_fn = Mock(return_value="Hello world")

        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=construct_fn,
        )

        result = context.prepare_prompt_with_files(
            prompt="Hello world",
            files=None,
        )

        assert result == "Hello world"
        construct_fn.assert_called_once_with("Hello world", None)

    def test_prepare_prompt_with_files_with_files(self):
        """Test prompt preparation with file context."""
        construct_fn = Mock(return_value="[file contents]\n\nHello world")

        context = WorkflowContext(
            workflow_name="chat",
            config=Mock(),
            construct_prompt_with_files_fn=construct_fn,
        )

        result = context.prepare_prompt_with_files(
            prompt="Hello world",
            files=["file1.txt", "file2.txt"],
        )

        assert result == "[file contents]\n\nHello world"
        construct_fn.assert_called_once_with(
            "Hello world", ["file1.txt", "file2.txt"]
        )


class TestOutputFormatter:
    """Test suite for OutputFormatter class."""

    # ========================================================================
    # display_workflow_start() Tests - Basic Cases
    # ========================================================================

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_minimal(self, mock_console):
        """Test minimal workflow start display (workflow + prompt only)."""
        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt="What is 2+2?",
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Starting new chat workflow" in call for call in calls)
        assert any("What is 2+2?" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_with_provider(self, mock_console):
        """Test workflow start display includes provider."""
        OutputFormatter.display_workflow_start(
            workflow_name="consensus",
            prompt="Analyze this",
            provider="claude",
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Provider: claude" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_continuation(self, mock_console):
        """Test continuation mode shows different header."""
        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt="Follow up question",
            continuation_id="thread-123",
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Continuing chat workflow" in call for call in calls)
        assert any("Thread ID: thread-123" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_with_files(self, mock_console):
        """Test workflow start display includes file list."""
        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt="Review these files",
            files=["file1.py", "file2.py"],
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Files: file1.py, file2.py" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_truncates_long_prompt(self, mock_console):
        """Test long prompts are truncated in display."""
        long_prompt = "x" * 150

        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt=long_prompt,
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        # Should contain truncated version with ellipsis
        assert any("x" * 100 + "..." in call for call in calls)
        # Should not contain full prompt
        assert not any(long_prompt in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_does_not_truncate_short_prompt(
        self, mock_console
    ):
        """Test short prompts are not truncated."""
        short_prompt = "Short prompt"

        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt=short_prompt,
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Short prompt" in call for call in calls)
        # Don't check for absence of "..." as it may appear in other output

    # ========================================================================
    # display_workflow_start() Tests - Additional Parameters
    # ========================================================================

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_with_kwargs(self, mock_console):
        """Test additional parameters are displayed with title case."""
        OutputFormatter.display_workflow_start(
            workflow_name="consensus",
            prompt="Test",
            num_to_consult=3,
            strategy="best-of-n",
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Num To Consult: 3" in call for call in calls)
        assert any("Strategy: best-of-n" in call for call in calls)

    @patch("model_chorus.cli.primitives.console")
    def test_display_workflow_start_skips_none_kwargs(self, mock_console):
        """Test None-valued additional parameters are not displayed."""
        OutputFormatter.display_workflow_start(
            workflow_name="chat",
            prompt="Test",
            optional_param=None,
            visible_param="value",
        )

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Visible Param: value" in call for call in calls)
        assert not any("Optional Param" in call for call in calls)

    # ========================================================================
    # write_json_output() Tests
    # ========================================================================

    def test_write_json_output_creates_file(self, tmp_path):
        """Test JSON output file is created with correct content."""
        output_path = tmp_path / "output.json"
        test_data = {"result": "success", "value": 42}

        OutputFormatter.write_json_output(output_path, test_data)

        assert output_path.exists()

        import json
        with output_path.open() as f:
            written_data = json.load(f)

        assert written_data == test_data

    def test_write_json_output_creates_parent_dirs(self, tmp_path):
        """Test parent directories are created if missing."""
        output_path = tmp_path / "nested" / "dirs" / "output.json"
        test_data = {"result": "success"}

        OutputFormatter.write_json_output(output_path, test_data)

        assert output_path.exists()
        assert output_path.parent.exists()

    @patch("model_chorus.cli.primitives.console")
    def test_write_json_output_shows_confirmation(
        self, mock_console, tmp_path
    ):
        """Test confirmation message is shown after writing."""
        output_path = tmp_path / "output.json"

        OutputFormatter.write_json_output(output_path, {})

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Results saved to" in call for call in calls)
        assert any(str(output_path) in call for call in calls)

    def test_write_json_output_handles_string_path(self, tmp_path):
        """Test string paths are accepted (not just Path objects)."""
        output_path = str(tmp_path / "output.json")
        test_data = {"result": "success"}

        OutputFormatter.write_json_output(output_path, test_data)

        from pathlib import Path
        assert Path(output_path).exists()

    def test_write_json_output_formatted_with_indent(self, tmp_path):
        """Test JSON output is formatted with indentation."""
        output_path = tmp_path / "output.json"
        test_data = {"key": "value", "nested": {"data": 123}}

        OutputFormatter.write_json_output(output_path, test_data)

        content = output_path.read_text()
        # Indented JSON should have newlines
        assert "\n" in content
        # Should have indentation spaces
        assert "  " in content
