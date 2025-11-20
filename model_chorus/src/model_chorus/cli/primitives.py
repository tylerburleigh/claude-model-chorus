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
