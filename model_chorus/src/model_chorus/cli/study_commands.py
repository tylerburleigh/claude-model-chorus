"""
CLI commands for STUDY workflow.

This module provides Typer commands for persona-based collaborative research
using the StudyWorkflow. Supports new investigations and continuation of
existing research sessions.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich import print as rprint

from ..providers import (
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    CursorAgentProvider,
)
from ..providers.cli_provider import ProviderUnavailableError
from ..workflows.study import StudyWorkflow
from ..core.conversation import ConversationMemory
from ..core.config import get_config_loader

console = Console()

# Initialize config loader
_config_loader = None


def get_config():
    """Get the config loader instance, initializing if needed."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_config_loader()
        try:
            _config_loader.load_config()
        except Exception:
            # If config fails to load, continue with defaults
            pass
    return _config_loader


def get_install_command(provider: str) -> str:
    """Get installation command for a provider CLI.

    Args:
        provider: Provider name (claude, gemini, codex, cursor-agent)

    Returns:
        Installation command string
    """
    commands = {
        "claude": "curl -fsSL https://claude.ai/install.sh | bash",
        "gemini": "npm install -g @google/gemini-cli",
        "codex": "npm install -g @openai/codex",
        "cursor-agent": "curl https://cursor.com/install -fsSL | bash",
    }
    return commands.get(provider.lower(), "See provider documentation")


def get_provider_by_name(name: str):
    """Get provider instance by name."""
    providers = {
        "claude": ClaudeProvider,
        "codex": CodexProvider,
        "gemini": GeminiProvider,
        "cursor-agent": CursorAgentProvider,
    }

    provider_class = providers.get(name.lower())
    if not provider_class:
        console.print(f"[red]Error: Unknown provider '{name}'[/red]")
        console.print(f"Available providers: {', '.join(providers.keys())}")
        raise typer.Exit(1)

    return provider_class()


def study(
    scenario: str = typer.Option(..., "--scenario", help="Investigation description or research question"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
        help="Thread ID to continue an existing investigation",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="File paths to include in research context (can specify multiple times)",
    ),
    personas: Optional[List[str]] = typer.Option(
        None,
        "--persona",
        help="Specific personas to use (can specify multiple times)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Temperature for generation (0.0-1.0). Defaults to config or 0.7",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for result (JSON format)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed execution information",
    ),
    skip_provider_check: bool = typer.Option(
        False,
        "--skip-provider-check",
        help="Skip provider availability check (faster startup)",
    ),
):
    """
    Conduct persona-based collaborative research.

    The STUDY workflow provides multi-persona investigation with role-based
    orchestration, enabling collaborative exploration of complex topics through
    specialized personas with distinct expertise.

    Example:
        # Start new investigation
        model-chorus study --scenario "Explore authentication system patterns"

        # Continue investigation
        model-chorus study --scenario "Deep dive into OAuth 2.0" --continue thread-id-123

        # Include files
        model-chorus study --scenario "Analyze this codebase" -f src/auth.py -f tests/test_auth.py

        # Specify personas
        model-chorus study --scenario "Security analysis" --persona SecurityExpert --persona Architect
    """
    try:
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_default_provider('study', 'claude')
        if temperature is None:
            temperature = config.get_workflow_default('study', 'temperature', 0.7)
        if max_tokens is None:
            max_tokens = config.get_workflow_default('study', 'max_tokens', None)
        if system is None:
            system = config.get_workflow_default('study', 'system_prompt', None)

        # Create provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider)
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except ProviderUnavailableError as e:
            # Provider CLI not available - show helpful error message
            console.print(f"[red]Error: {e.reason}[/red]\n")
            if e.suggestions:
                console.print("[yellow]To fix this:[/yellow]")
                for suggestion in e.suggestions:
                    console.print(f"  • {suggestion}")
            console.print(f"\n[yellow]Installation:[/yellow] {get_install_command(provider)}")
            console.print(f"\n[dim]Run 'model-chorus list-providers --check' to see which providers are available[/dim]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider}: {e}[/red]")
            raise typer.Exit(1)

        # Load fallback providers from config
        fallback_provider_names = config.get_workflow_default('study', 'fallback_providers', [])
        fallback_providers = []

        if fallback_provider_names and verbose:
            console.print(f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]")

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = get_provider_by_name(fallback_name)
                fallback_providers.append(fallback_instance)
                if verbose:
                    console.print(f"[green]✓ {fallback_name} initialized (fallback)[/green]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]")

        # Create conversation memory (in-memory for now)
        memory = ConversationMemory()

        # Create workflow
        workflow_config = {}
        if personas:
            # Convert persona names to persona configurations
            workflow_config['personas'] = [
                {"name": p, "role": "investigator"}
                for p in personas
            ]

        workflow = StudyWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            conversation_memory=memory,
            config=workflow_config,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Validate files exist
        if files:
            for file_path in files:
                if not Path(file_path).exists():
                    console.print(f"[red]Error: File not found: {file_path}[/red]")
                    raise typer.Exit(1)

        # Display investigation info
        console.print(f"\n[bold cyan]{'Continuing' if continuation_id else 'Starting'} STUDY Investigation[/bold cyan]")
        console.print(f"Scenario: {scenario[:100]}{'...' if len(scenario) > 100 else ''}")
        console.print(f"Provider: {provider}")
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")
        if files:
            console.print(f"Files: {', '.join(files)}")
        if personas:
            console.print(f"Personas: {', '.join(personas)}")
        console.print()

        # Build kwargs for workflow
        workflow_kwargs = {}
        if personas:
            workflow_kwargs['personas'] = workflow_config.get('personas', [])
        if system:
            workflow_kwargs['system_prompt'] = system
        if temperature is not None:
            workflow_kwargs['temperature'] = temperature
        if max_tokens is not None:
            workflow_kwargs['max_tokens'] = max_tokens

        # Run async workflow
        result = asyncio.run(
            workflow.run(
                prompt=scenario,
                continuation_id=continuation_id,
                files=files,
                skip_provider_check=skip_provider_check,
                **workflow_kwargs
            )
        )

        # Display results
        if result.success:
            console.print(f"[bold green]✓ STUDY investigation completed[/bold green]\n")

            # Show thread info
            thread_id = result.metadata.get('thread_id')
            is_continuation = result.metadata.get('is_continuation', False)
            personas_used = result.metadata.get('personas_used', [])

            console.print(f"[cyan]Thread ID:[/cyan] {thread_id}")
            if not is_continuation:
                console.print("[cyan]Status:[/cyan] New investigation started")
            else:
                console.print(f"[cyan]Status:[/cyan] Investigation continued")
            if personas_used:
                console.print(f"[cyan]Personas:[/cyan] {', '.join(personas_used)}")
            console.print()

            # Show investigation steps
            if result.steps:
                console.print("[bold]Investigation Steps:[/bold]\n")
                for i, step in enumerate(result.steps, 1):
                    persona_name = step.metadata.get('persona', f'Step {i}') if hasattr(step, 'metadata') and step.metadata else f'Step {i}'
                    console.print(f"[bold cyan]{persona_name}:[/bold cyan]")
                    console.print(step.content)
                    console.print()

            # Show synthesis
            console.print("[bold]Research Synthesis:[/bold]\n")
            console.print(result.synthesis)

            # Show usage info if available
            usage = result.metadata.get('usage', {})
            if usage and verbose:
                console.print(f"\n[dim]Tokens: {usage.get('total_tokens', 'N/A')}[/dim]")

            # Save to file if requested
            if output:
                output_data = {
                    "scenario": scenario,
                    "provider": provider,
                    "thread_id": thread_id,
                    "is_continuation": is_continuation,
                    "personas_used": personas_used,
                    "steps": [
                        {
                            "persona": step.metadata.get('persona', f'Step {i}') if hasattr(step, 'metadata') and step.metadata else f'Step {i}',
                            "content": step.content,
                            "metadata": step.metadata if hasattr(step, 'metadata') else {}
                        }
                        for i, step in enumerate(result.steps, 1)
                    ],
                    "synthesis": result.synthesis,
                    "model": result.metadata.get('model'),
                    "usage": usage,
                }
                if files:
                    output_data["files"] = files

                output.write_text(json.dumps(output_data, indent=2))
                console.print(f"\n[green]✓ Result saved to {output}[/green]")

            console.print(f"\n[dim]To continue this investigation, use: --continue {thread_id}[/dim]")

        else:
            console.print(f"[red]✗ STUDY investigation failed: {result.error}[/red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)
