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

# Create study command group
study_app = typer.Typer(
    name="study",
    help="Persona-based collaborative research workflows",
    add_completion=False,
)

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


@study_app.command()
def start(
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
        model-chorus study start --scenario "Explore authentication system patterns"

        # Continue investigation
        model-chorus study start --scenario "Deep dive into OAuth 2.0" --continue thread-id-123

        # Include files
        model-chorus study start --scenario "Analyze this codebase" -f src/auth.py -f tests/test_auth.py

        # Specify personas
        model-chorus study start --scenario "Security analysis" --persona SecurityExpert --persona Architect
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


@study_app.command(name="next")
def study_next(
    investigation: str = typer.Option(..., "--investigation", help="Investigation ID (thread ID) to continue"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="Additional file paths to include in context (can specify multiple times)",
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
    Continue an existing STUDY investigation.

    This command automatically continues an investigation using the existing
    thread context and personas. It's a convenience wrapper around the study
    command with automatic continuation.

    The command retrieves the investigation's conversation history and prompts
    the next investigation step based on the current state.

    Example:
        # Continue investigation with automatic next step
        model-chorus study-next --investigation thread-id-123

        # Continue with additional files
        model-chorus study-next --investigation thread-id-123 -f new_data.py

        # Continue with specific provider
        model-chorus study-next --investigation thread-id-123 -p gemini
    """
    try:
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_default_provider('study', 'claude')

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

        # Create conversation memory
        memory = ConversationMemory()

        # Try to retrieve existing thread
        thread = memory.get_thread(investigation)
        if not thread:
            console.print(f"[red]Error: Investigation not found: {investigation}[/red]")
            console.print("[yellow]Make sure you're using the correct thread ID from a previous investigation.[/yellow]")
            console.print("\nTo start a new investigation, use: model-chorus study --scenario \"...\"")
            raise typer.Exit(1)

        # Create workflow
        workflow = StudyWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            conversation_memory=memory,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Validate files exist
        if files:
            for file_path in files:
                if not Path(file_path).exists():
                    console.print(f"[red]Error: File not found: {file_path}[/red]")
                    raise typer.Exit(1)

        # Display continuation info
        console.print(f"\n[bold cyan]Continuing STUDY Investigation[/bold cyan]")
        console.print(f"Investigation ID: {investigation}")
        console.print(f"Provider: {provider}")
        console.print(f"Previous messages: {len(thread.messages)}")
        if files:
            console.print(f"Additional files: {', '.join(files)}")
        console.print()

        # Generate automatic next step prompt
        next_step_prompt = "Continue the investigation by analyzing the next aspect or exploring deeper into the current findings."

        # Build kwargs for workflow
        workflow_kwargs = {}
        if max_tokens is not None:
            workflow_kwargs['max_tokens'] = max_tokens

        # Run async workflow with continuation
        result = asyncio.run(
            workflow.run(
                prompt=next_step_prompt,
                continuation_id=investigation,
                files=files,
                skip_provider_check=skip_provider_check,
                **workflow_kwargs
            )
        )

        # Display results
        if result.success:
            console.print(f"[bold green]✓ Investigation step completed[/bold green]\n")

            # Show thread info
            thread_id = result.metadata.get('thread_id')
            personas_used = result.metadata.get('personas_used', [])

            console.print(f"[cyan]Investigation ID:[/cyan] {thread_id}")
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
                    "investigation_id": investigation,
                    "provider": provider,
                    "thread_id": thread_id,
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

            console.print(f"\n[dim]To continue this investigation, use: model-chorus study-next --investigation {thread_id}[/dim]")

        else:
            console.print(f"[red]✗ Investigation continuation failed: {result.error}[/red]")
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


@study_app.command(name="view")
def study_view(
    investigation: str = typer.Option(..., "--investigation", help="Investigation ID (thread ID) to view"),
    persona: Optional[str] = typer.Option(
        None,
        "--persona",
        help="Filter by specific persona (optional)",
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        help="Show complete conversation history",
    ),
    format_json: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
):
    """
    View memory and conversation history for a STUDY investigation.

    This command displays the conversation memory for an investigation,
    including all messages, persona contributions, and investigation metadata.
    Useful for reviewing investigation history and debugging.

    Example:
        # View investigation summary
        model-chorus study-view --investigation thread-id-123

        # View all messages
        model-chorus study-view --investigation thread-id-123 --show-all

        # Filter by persona
        model-chorus study-view --investigation thread-id-123 --persona Researcher

        # Output as JSON
        model-chorus study-view --investigation thread-id-123 --json
    """
    try:
        # Create conversation memory
        memory = ConversationMemory()

        # Try to retrieve existing thread
        thread = memory.get_thread(investigation)
        if not thread:
            console.print(f"[red]Error: Investigation not found: {investigation}[/red]")
            console.print("[yellow]Make sure you're using the correct thread ID from a previous investigation.[/yellow]")
            console.print("\nTo start a new investigation, use: model-chorus study --scenario \"...\"")
            raise typer.Exit(1)

        # Extract messages and metadata
        messages = thread.messages
        metadata = thread.metadata if hasattr(thread, 'metadata') else {}

        # Filter by persona if specified
        if persona:
            messages = [
                msg for msg in messages
                if msg.metadata and msg.metadata.get('persona', '').lower() == persona.lower()
            ]
            if not messages:
                console.print(f"[yellow]No messages found for persona '{persona}' in investigation {investigation}[/yellow]")
                console.print(f"Total messages in thread: {len(thread.messages)}")
                raise typer.Exit(0)

        # JSON output format
        if format_json:
            output_data = {
                "investigation_id": investigation,
                "total_messages": len(thread.messages),
                "filtered_messages": len(messages) if persona else len(thread.messages),
                "metadata": metadata,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content[:200] + "..." if len(msg.content) > 200 and not show_all else msg.content,
                        "metadata": msg.metadata if hasattr(msg, 'metadata') else {},
                        "timestamp": (
                            msg.timestamp if isinstance(msg.timestamp, str)
                            else msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat')
                            else None
                        ) if hasattr(msg, 'timestamp') and msg.timestamp else None
                    }
                    for msg in messages
                ]
            }
            if persona:
                output_data["filter_persona"] = persona

            console.print(json.dumps(output_data, indent=2))
            return

        # Display header
        console.print(f"\n[bold cyan]STUDY Investigation Memory[/bold cyan]")
        console.print(f"[cyan]Investigation ID:[/cyan] {investigation}")
        console.print(f"[cyan]Total Messages:[/cyan] {len(thread.messages)}")
        if persona:
            console.print(f"[cyan]Filtered Persona:[/cyan] {persona} ({len(messages)} messages)")
        console.print()

        # Display metadata if available and verbose
        if metadata and verbose:
            console.print("[bold]Investigation Metadata:[/bold]")
            for key, value in metadata.items():
                console.print(f"  {key}: {value}")
            console.print()

        # Display messages
        if not messages:
            console.print("[yellow]No messages found in this investigation.[/yellow]")
        else:
            console.print("[bold]Conversation History:[/bold]\n")
            for i, msg in enumerate(messages, 1):
                # Extract persona from metadata if available
                msg_persona = msg.metadata.get('persona', 'Unknown') if hasattr(msg, 'metadata') and msg.metadata else 'Unknown'

                # Handle timestamp (could be datetime object or string)
                if hasattr(msg, 'timestamp') and msg.timestamp:
                    if isinstance(msg.timestamp, str):
                        timestamp = msg.timestamp
                    else:
                        timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = 'N/A'

                # Display message header
                console.print(f"[bold cyan]Message {i}[/bold cyan]")
                console.print(f"  [dim]Role:[/dim] {msg.role}")
                console.print(f"  [dim]Persona:[/dim] {msg_persona}")
                if verbose:
                    console.print(f"  [dim]Timestamp:[/dim] {timestamp}")

                # Display content (truncated unless show_all)
                content = msg.content
                if not show_all and len(content) > 200:
                    content = content[:200] + "..."
                console.print(f"\n{content}\n")

                # Show separator
                if i < len(messages):
                    console.print("[dim]" + "─" * 60 + "[/dim]\n")

        # Show usage tips
        if not show_all and any(len(msg.content) > 200 for msg in messages):
            console.print(f"\n[dim]Use --show-all to see complete messages[/dim]")
        if not persona and len(thread.messages) > 1:
            console.print(f"[dim]Use --persona <name> to filter by persona[/dim]")
        if not format_json:
            console.print(f"[dim]Use --json for machine-readable output[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)
