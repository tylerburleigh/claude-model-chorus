"""
CLI primitive classes for ModelChorus command-line interface.

This module provides reusable components that extract common patterns from
CLI command implementations, reducing duplication and improving maintainability.
"""

from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..core.config import ModelChorusConfig
    from ..providers.base_provider import ModelProvider

console = Console()


class ProviderResolver:
    """
    Handles provider initialization and fallback provider setup.

    Extracts the common pattern of:
    1. Initializing primary provider with error handling
    2. Loading and initializing fallback providers from config
    3. Providing helpful error messages for disabled/unavailable providers
    """

    def __init__(
        self,
        config: "ModelChorusConfig",
        get_provider_fn,
        get_install_command_fn,
        verbose: bool = False,
    ):
        """
        Initialize ProviderResolver.

        Args:
            config: ModelChorus configuration object
            get_provider_fn: Function to get provider instance by name
                (signature: get_provider_by_name(name: str, timeout: int))
            get_install_command_fn: Function to get provider installation command
                (signature: get_install_command(provider: str))
            verbose: Whether to show detailed initialization messages
        """
        self.config = config
        self.get_provider_fn = get_provider_fn
        self.get_install_command_fn = get_install_command_fn
        self.verbose = verbose

    def resolve_provider(
        self, provider_name: str, timeout: float
    ) -> "ModelProvider":
        """
        Initialize and return provider instance with comprehensive error handling.

        Args:
            provider_name: Name of provider to initialize
            timeout: Timeout in seconds for provider operations

        Returns:
            Initialized ModelProvider instance

        Raises:
            SystemExit: If provider cannot be initialized (via typer.Exit)
        """
        import typer

        from ..providers.cli_provider import (
            ProviderDisabledError,
            ProviderUnavailableError,
        )

        if self.verbose:
            console.print(f"[cyan]Initializing provider: {provider_name}[/cyan]")

        try:
            provider_instance = self.get_provider_fn(
                provider_name, timeout=int(timeout)
            )
            if self.verbose:
                console.print(f"[green]✓ {provider_name} initialized[/green]")
            return provider_instance

        except ProviderDisabledError as e:
            # Provider disabled in config
            console.print(f"[red]Error: {e}[/red]")
            console.print(
                f"\n[yellow]To fix this, edit .claude/model_chorus_config.yaml and set '{provider_name}: enabled: true'[/yellow]"
            )
            raise typer.Exit(1)

        except ProviderUnavailableError as e:
            # Provider CLI not available - show helpful error message
            console.print(f"[red]Error: {e.reason}[/red]\n")
            if e.suggestions:
                console.print("[yellow]To fix this:[/yellow]")
                for suggestion in e.suggestions:
                    console.print(f"  • {suggestion}")
            console.print(
                f"\n[yellow]Installation:[/yellow] {self.get_install_command_fn(provider_name)}"
            )
            console.print(
                "\n[dim]Run 'model-chorus list-providers --check' to see which providers are available[/dim]"
            )
            raise typer.Exit(1)

        except Exception as e:
            console.print(
                f"[red]Failed to initialize {provider_name}: {e}[/red]"
            )
            raise typer.Exit(1)

    def resolve_fallback_providers(
        self,
        workflow_name: str,
        exclude_provider: str,
        timeout: float,
    ) -> list["ModelProvider"]:
        """
        Initialize fallback providers from config with error handling.

        Args:
            workflow_name: Name of workflow to get fallback providers for
            exclude_provider: Primary provider name to exclude from fallbacks
            timeout: Timeout in seconds for provider operations

        Returns:
            List of successfully initialized fallback provider instances
        """
        from ..providers.cli_provider import ProviderDisabledError

        # Load fallback providers from config and filter by enabled status
        fallback_provider_names = self.config.get_workflow_fallback_providers(
            workflow_name, exclude_provider=exclude_provider
        )

        fallback_providers: list["ModelProvider"] = []

        if fallback_provider_names and self.verbose:
            console.print(
                f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]"
            )

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = self.get_provider_fn(
                    fallback_name, timeout=int(timeout)
                )
                fallback_providers.append(fallback_instance)
                if self.verbose:
                    console.print(
                        f"[green]✓ {fallback_name} initialized (fallback)[/green]"
                    )
            except ProviderDisabledError:
                if self.verbose:
                    console.print(
                        f"[yellow]⚠ Skipping disabled fallback provider {fallback_name}[/yellow]"
                    )
            except Exception as e:
                if self.verbose:
                    console.print(
                        f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]"
                    )

        return fallback_providers


class WorkflowContext:
    """
    Manages workflow execution context including config, prompt validation, and memory setup.

    Extracts common patterns of:
    1. Prompt validation (arg vs flag)
    2. Config resolution with workflow defaults
    3. Conversation memory initialization
    4. File context ingestion
    """

    def __init__(
        self,
        workflow_name: str,
        config: "ModelChorusConfig",
        construct_prompt_with_files_fn,
    ):
        """
        Initialize WorkflowContext.

        Args:
            workflow_name: Name of workflow (e.g., "chat", "consensus")
            config: ModelChorus configuration object
            construct_prompt_with_files_fn: Function to construct prompts with files
                (signature: construct_prompt_with_files(prompt: str, files: list[str] | None))
        """
        self.workflow_name = workflow_name
        self.config = config
        self.construct_prompt_with_files_fn = construct_prompt_with_files_fn

    def validate_and_get_prompt(
        self, prompt_arg: str | None, prompt_flag: str | None
    ) -> str:
        """
        Validate prompt arguments and return the final prompt string.

        Args:
            prompt_arg: Positional prompt argument
            prompt_flag: --prompt flag value

        Returns:
            The validated prompt string

        Raises:
            SystemExit: If prompt validation fails (via typer.Exit)
        """
        import typer

        if prompt_arg is None and prompt_flag is None:
            console.print(
                "[red]Error: Prompt is required (provide as positional argument or use --prompt)[/red]"
            )
            raise typer.Exit(1)

        if prompt_arg is not None and prompt_flag is not None:
            console.print(
                "[red]Error: Cannot specify prompt both as positional argument and --prompt flag[/red]"
            )
            raise typer.Exit(1)

        prompt = prompt_arg or prompt_flag
        assert prompt is not None  # Validated above
        return prompt

    def resolve_config_defaults(
        self,
        provider: str | None = None,
        system: str | None = None,
        timeout: float | None = None,
        default_provider: str = "claude",
    ) -> dict[str, str | float | None]:
        """
        Resolve configuration defaults for provider, system prompt, and timeout.

        Args:
            provider: CLI-provided provider name (overrides config)
            system: CLI-provided system prompt (overrides config)
            timeout: CLI-provided timeout (overrides config)
            default_provider: Fallback provider if not in config

        Returns:
            Dictionary with resolved 'provider', 'system', and 'timeout' values

        Raises:
            SystemExit: If default provider is disabled (via typer.Exit)
        """
        import typer

        resolved_provider = provider
        if resolved_provider is None:
            resolved_provider = self.config.get_workflow_default_provider(
                self.workflow_name, default_provider
            )
            if resolved_provider is None:
                console.print(
                    f"[red]Error: Default provider for '{self.workflow_name}' workflow is disabled.[/red]"
                )
                console.print(
                    "[yellow]Enable it in .claude/model_chorus_config.yaml or specify --provider[/yellow]"
                )
                raise typer.Exit(1)

        resolved_system = system
        if resolved_system is None:
            resolved_system = self.config.get_workflow_default(
                self.workflow_name, "system_prompt", None
            )

        resolved_timeout = timeout
        if resolved_timeout is None:
            resolved_timeout = self.config.get_workflow_default(
                self.workflow_name, "timeout", 120.0
            )

        return {
            "provider": resolved_provider,
            "system": resolved_system,
            "timeout": resolved_timeout,
        }

    def create_memory(self):
        """
        Create and return a new ConversationMemory instance.

        Returns:
            ConversationMemory instance for workflow use
        """
        from ..core.conversation import ConversationMemory

        return ConversationMemory()

    def prepare_prompt_with_files(
        self, prompt: str, files: list[str] | None
    ) -> str:
        """
        Prepare final prompt by ingesting file contents if provided.

        Args:
            prompt: Base prompt string
            files: Optional list of file paths to include in context

        Returns:
            Final prompt with file contents prepended (if files provided)
        """
        return self.construct_prompt_with_files_fn(prompt, files)


class OutputFormatter:
    """
    Standardizes console output formatting for workflow commands.

    Extracts common patterns for:
    1. Displaying workflow start/continuation messages
    2. Showing execution parameters (prompt, provider, files, etc.)
    3. Consistent formatting and truncation
    """

    @staticmethod
    def display_workflow_start(
        workflow_name: str,
        prompt: str,
        provider: str | None = None,
        continuation_id: str | None = None,
        files: list[str] | None = None,
        **kwargs,
    ):
        """
        Display standardized workflow start information.

        Args:
            workflow_name: Name of workflow being executed
            prompt: The prompt being sent
            provider: Provider name (optional)
            continuation_id: Thread/session ID for continuation (optional)
            files: List of file paths included (optional)
            **kwargs: Additional parameters to display (e.g., num_to_consult, strategy)
        """
        # Display header
        if continuation_id:
            console.print(
                f"\n[bold cyan]Continuing {workflow_name} workflow...[/bold cyan]"
            )
        else:
            console.print(
                f"\n[bold cyan]Starting new {workflow_name} workflow...[/bold cyan]"
            )

        # Display prompt (truncated if too long)
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        console.print(f"Prompt: {truncated_prompt}")

        # Display provider if provided
        if provider:
            console.print(f"Provider: {provider}")

        # Display continuation ID if provided
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")

        # Display files if provided
        if files:
            console.print(f"Files: {', '.join(files)}")

        # Display any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                # Convert key from snake_case to Title Case for display
                display_key = key.replace("_", " ").title()
                console.print(f"{display_key}: {value}")

        console.print()

    @staticmethod
    def write_json_output(output_path, result_data: dict):
        """
        Write workflow result to JSON file.

        Args:
            output_path: Path object or string path to output file
            result_data: Dictionary of result data to write
        """
        import json
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w") as f:
            json.dump(result_data, f, indent=2, default=str)

        console.print(f"\n[green]✓ Results saved to {output_file}[/green]")
