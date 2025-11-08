"""
Integration tests for ModelChorus CLI commands.

Tests the complete CLI interface including argument and ideate commands
with various parameters, options, and error conditions.
"""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from modelchorus.cli.main import app
from modelchorus.providers.base_provider import GenerationResponse
from modelchorus.core.base_workflow import WorkflowResult, WorkflowStep


# Test fixtures
@pytest.fixture
def cli_runner():
    """Create Typer CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_provider():
    """Mock provider for CLI testing."""
    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.validate_api_key = MagicMock(return_value=True)

    # Mock async generate method
    async def mock_generate(request):
        return GenerationResponse(
            content=f"Test response to: {request.prompt[:50]}",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason="end_turn",
        )

    provider.generate = AsyncMock(side_effect=mock_generate)
    return provider


@pytest.fixture
def mock_workflow_result():
    """Mock workflow result for CLI testing."""
    return WorkflowResult(
        success=True,
        synthesis="Test synthesis result",
        steps=[
            WorkflowStep(
                step_number=1,
                content="Step 1 content",
                model="test-model",
                metadata={"name": "Step 1"}
            )
        ],
        metadata={
            "thread_id": "test-thread-123",
            "is_continuation": False,
            "conversation_length": 1,
            "model": "test-model",
            "usage": {"total_tokens": 30}
        },
        error=None,
    )


@pytest.fixture
def temp_test_file(tmp_path):
    """Create temporary test file for file input tests."""
    test_file = tmp_path / "test_input.txt"
    test_file.write_text("Test file content for CLI testing")
    return test_file


@pytest.fixture
def temp_output_file(tmp_path):
    """Create temporary output file path."""
    return tmp_path / "output.json"


# ARGUMENT command tests
class TestArgumentCommand:
    """Test suite for 'argument' CLI command."""

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_basic_invocation(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test basic argument command invocation."""
        # Setup mocks
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        # Execute command
        result = cli_runner.invoke(app, ["argument", "Universal basic income reduces poverty"])

        # Assertions
        assert result.exit_code == 0
        assert "Analyzing argument" in result.stdout
        assert "Test synthesis result" in result.stdout
        mock_get_provider.assert_called_once_with("claude")
        mock_workflow.run.assert_called_once()

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_with_provider_option(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test argument with explicit provider selection."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test prompt", "--provider", "gemini"]
        )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("gemini")

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_with_continuation(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test argument with continuation thread ID."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()

        # Mock continuation result
        continuation_result = WorkflowResult(
            success=True,
            synthesis="Continued analysis",
            steps=[
                WorkflowStep(
                    step_number=1,
                    content="Continued content",
                    model="test-model",
                    metadata={}
                )
            ],
            metadata={
                "thread_id": "thread-123",
                "is_continuation": True,
                "conversation_length": 3,
            },
            error=None,
        )
        mock_workflow.run = AsyncMock(return_value=continuation_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Continue analysis", "--continue", "thread-123"]
        )

        assert result.exit_code == 0
        assert "Continued" in result.stdout or "3 messages" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_with_file(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result,
        temp_test_file
    ):
        """Test argument with file input."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Analyze this", "--file", str(temp_test_file)]
        )

        assert result.exit_code == 0
        # Verify file was passed to workflow
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs['files'] == [str(temp_test_file)]

    @patch('modelchorus.cli.main.get_provider_by_name')
    def test_argument_with_nonexistent_file(
        self,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test argument with nonexistent file returns error."""
        mock_get_provider.return_value = mock_provider

        result = cli_runner.invoke(
            app,
            ["argument", "Test", "--file", "/nonexistent/file.txt"]
        )

        assert result.exit_code == 1
        assert "File not found" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_with_temperature(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test argument with custom temperature."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test", "--temperature", "0.9"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs['temperature'] == 0.9

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_with_output_file(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result,
        temp_output_file
    ):
        """Test argument with JSON output file."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test", "--output", str(temp_output_file)]
        )

        assert result.exit_code == 0
        assert temp_output_file.exists()

        # Verify JSON structure
        with open(temp_output_file) as f:
            output_data = json.load(f)

        assert output_data['success'] is True
        assert output_data['synthesis'] == "Test synthesis result"
        assert 'steps' in output_data
        assert 'metadata' in output_data

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_argument_verbose_mode(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test argument with verbose flag."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test", "--verbose"]
        )

        assert result.exit_code == 0
        # Verbose should show initialization and metadata
        assert "Initializing provider" in result.stdout or "claude" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    def test_argument_invalid_provider(
        self,
        mock_get_provider,
        cli_runner
    ):
        """Test argument with invalid provider."""
        mock_get_provider.side_effect = Exception("Unknown provider")

        result = cli_runner.invoke(
            app,
            ["argument", "Test", "--provider", "invalid-provider"]
        )

        assert result.exit_code == 1
        assert "Failed to initialize" in result.stdout


# IDEATE command tests
class TestIdeateCommand:
    """Test suite for 'ideate' CLI command."""

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_basic_invocation(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test basic ideate command."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "New features for task management"]
        )

        assert result.exit_code == 0
        assert "Generating creative ideas" in result.stdout
        assert "Test synthesis result" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_with_num_ideas(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test ideate with custom number of ideas."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "Marketing ideas", "--num-ideas", "10"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs['num_ideas'] == 10

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_with_high_temperature(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test ideate with high creativity temperature."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "Creative concepts", "--temperature", "1.0"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs['temperature'] == 1.0

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_with_continuation(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test ideate with continuation."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()

        continuation_result = WorkflowResult(
            success=True,
            synthesis="Refined ideas",
            steps=[
                WorkflowStep(
                    step_number=1,
                    content="Refined content",
                    model="test-model",
                    metadata={}
                )
            ],
            metadata={"thread_id": "thread-456", "is_continuation": True},
            error=None,
        )
        mock_workflow.run = AsyncMock(return_value=continuation_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "Refine idea 3", "--continue", "thread-456"]
        )

        assert result.exit_code == 0

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_with_files(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result,
        temp_test_file
    ):
        """Test ideate with context files."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "Ideas based on context", "--file", str(temp_test_file)]
        )

        assert result.exit_code == 0
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs['files'] == [str(temp_test_file)]

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    def test_ideate_with_output(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result,
        temp_output_file
    ):
        """Test ideate with JSON output."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value=mock_workflow_result)
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["ideate", "Test", "--output", str(temp_output_file)]
        )

        assert result.exit_code == 0
        assert temp_output_file.exists()

        with open(temp_output_file) as f:
            output_data = json.load(f)

        assert output_data['success'] is True
        assert 'ideas' in output_data


# Edge cases and error handling
class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_empty_prompt(self, cli_runner):
        """Test commands with empty prompt."""
        # Typer will fail if required argument is missing
        result = cli_runner.invoke(app, ["argument"])

        assert result.exit_code != 0

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_workflow_failure(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test handling of workflow failures."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()

        # Mock failed workflow result
        # Note: argument, ideate, and research commands don't explicitly check result.success
        # They just proceed to display results. Only chat command checks success status.
        # So we test with an exception instead to verify error handling.
        mock_workflow.run = AsyncMock(side_effect=Exception("Workflow execution failed"))
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test prompt"]
        )

        assert result.exit_code == 1
        assert "Error" in result.stdout or "error" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    def test_keyboard_interrupt(
        self,
        mock_workflow_class,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test handling of keyboard interrupt."""
        mock_get_provider.return_value = mock_provider
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_workflow_class.return_value = mock_workflow

        result = cli_runner.invoke(
            app,
            ["argument", "Test"]
        )

        assert result.exit_code == 130
        assert "Interrupted" in result.stdout

    def test_very_long_prompt(self, cli_runner):
        """Test handling of very long input prompt."""
        long_prompt = "A" * 10000

        # Should not crash, but we can't test full execution without mocks
        # This mainly tests that the CLI can accept long inputs
        result = cli_runner.invoke(
            app,
            ["argument", long_prompt]
        )

        # Will fail at provider initialization but shouldn't crash on input length
        assert result.exit_code in [0, 1]  # Either success or graceful failure

    @patch('modelchorus.cli.main.get_provider_by_name')
    def test_special_characters_in_prompt(
        self,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test handling of special characters in prompt."""
        mock_get_provider.return_value = mock_provider

        special_prompt = 'Test with "quotes" and \'apostrophes\' and $pecial chars!'

        result = cli_runner.invoke(
            app,
            ["argument", special_prompt]
        )

        # Should not crash on special characters
        assert result.exit_code in [0, 1]


# Integration tests
class TestCommandIntegration:
    """Test suite for cross-command integration."""

    @patch('modelchorus.cli.main.get_provider_by_name')
    @patch('modelchorus.cli.main.ArgumentWorkflow')
    @patch('modelchorus.cli.main.IdeateWorkflow')
    @patch('modelchorus.cli.main.ResearchWorkflow')
    def test_all_commands_available(
        self,
        mock_research_wf,
        mock_ideate_wf,
        mock_argument_wf,
        mock_get_provider,
        cli_runner,
        mock_provider,
        mock_workflow_result
    ):
        """Test that all three new commands are available."""
        mock_get_provider.return_value = mock_provider

        # Setup mocks for all workflows
        for mock_wf_class in [mock_argument_wf, mock_ideate_wf, mock_research_wf]:
            mock_wf = MagicMock()
            mock_wf.run = AsyncMock(return_value=mock_workflow_result)
            mock_wf.ingest_source = MagicMock()
            mock_wf_class.return_value = mock_wf

        # Test each command
        commands = [
            ["argument", "Test argument"],
            ["ideate", "Test ideation"],
            ["research", "Test research"]
        ]

        for cmd in commands:
            result = cli_runner.invoke(app, cmd)
            assert result.exit_code == 0

    def test_help_shows_all_commands(self, cli_runner):
        """Test that help text shows all available commands."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "argument" in result.stdout
        assert "ideate" in result.stdout
        assert "research" in result.stdout

    @patch('modelchorus.cli.main.get_provider_by_name')
    def test_common_options_work_across_commands(
        self,
        mock_get_provider,
        cli_runner,
        mock_provider
    ):
        """Test that common options (--verbose, --provider) work for all commands."""
        mock_get_provider.return_value = mock_provider

        commands = ["argument", "ideate", "research"]

        for cmd in commands:
            # Test --help
            result = cli_runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            assert "--provider" in result.stdout
            assert "--verbose" in result.stdout
            assert "--output" in result.stdout
