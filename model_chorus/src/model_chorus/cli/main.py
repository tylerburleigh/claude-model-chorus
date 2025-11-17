"""
CLI interface for ModelChorus.

Typer-based command-line application for running multi-model workflows.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..providers import (
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    CursorAgentProvider,
    GenerationRequest,
)
from ..providers.cli_provider import ProviderUnavailableError
from ..workflows import ArgumentWorkflow, ChatWorkflow, ConsensusWorkflow, ConsensusStrategy, IdeateWorkflow, ThinkDeepWorkflow
from ..core.conversation import ConversationMemory
from ..core.config import get_claude_config_loader
from ..core.progress import set_progress_enabled
from model_chorus import __version__
from .study_commands import study_app


class ProviderDisabledError(Exception):
    """Raised when attempting to use a disabled provider."""
    pass

app = typer.Typer(
    name="model-chorus",
    help="Multi-model AI workflow orchestration",
    add_completion=False,
)
console = Console()


def _find_project_root(start_path: Path) -> Path:
    """Traverse upward from the starting path to locate the project root."""
    for parent in start_path.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Fallback: use the top-most parent (filesystem root)
    return start_path.parents[-1]


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

# Ordered list so that more specific mappings run before broader fallbacks.
LEGACY_PATH_MAPPINGS: List[Tuple[Path, Path]] = [
    (Path("src/claude_skills/sdd_toolkit"), Path("model_chorus/src/model_chorus")),
    (Path("src/claude_skills"), Path("model_chorus/src/model_chorus")),
]


def _format_path_for_display(path: Path) -> str:
    """Return a repository-relative path when possible for cleaner display."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_context_files(
    files: Optional[Union[str, Sequence[str]]]
) -> Tuple[List[str], List[str], List[str]]:
    """Normalize and resolve context file paths with legacy mapping support.

    Args:
        files: Raw string (comma-separated) or iterable of file paths.

    Returns:
        Tuple containing:
            resolved_paths: Paths that exist after normalization/mapping.
            remapped_notices: Descriptions of legacy paths that were remapped.
            missing_files: Original entries that could not be resolved.
    """
    if not files:
        return [], [], []

    if isinstance(files, str):
        entries = [part.strip() for part in files.split(",") if part.strip()]
    else:
        entries = [str(item).strip() for item in files if str(item).strip()]

    resolved_paths: List[str] = []
    remapped_notices: List[str] = []
    missing_files: List[str] = []
    cwd = Path.cwd()

    for original in entries:
        normalized_str = original.strip()
        normalized_path = Path(normalized_str.replace("\\", "/"))
        raw_path = Path(normalized_str).expanduser()

        candidate_paths = []
        remapped_target: Optional[Path] = None

        if raw_path.is_absolute():
            candidate_paths.append(raw_path.resolve(strict=False))
        else:
            candidate_paths.append((PROJECT_ROOT / normalized_path).resolve(strict=False))
            candidate_paths.append((cwd / normalized_path).resolve(strict=False))

        for legacy_prefix, new_prefix in LEGACY_PATH_MAPPINGS:
            try:
                suffix = normalized_path.relative_to(legacy_prefix)
            except ValueError:
                continue

            remapped_target = (PROJECT_ROOT / new_prefix / suffix).resolve(strict=False)
            candidate_paths.insert(0, remapped_target)
            break

        unique_candidates: List[Path] = []
        seen = set()
        for candidate in candidate_paths:
            key = candidate.as_posix()
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        existing_path = next((path for path in unique_candidates if path.exists()), None)

        if existing_path:
            display_path = _format_path_for_display(existing_path)
            resolved_paths.append(display_path)
            if remapped_target and existing_path == remapped_target and original != display_path:
                remapped_notices.append(f"{original} -> {display_path}")
        else:
            missing_files.append(original)

    return resolved_paths, remapped_notices, missing_files

# Register study command group
app.add_typer(study_app, name='study')

# Initialize config loader
_config_loader = None


def get_config():
    """Get the config loader instance, initializing if needed."""
    global _config_loader
    if _config_loader is None:
        _config_loader = get_claude_config_loader()
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


def get_provider_by_name(name: str, timeout: int = 120):
    """Get provider instance by name.

    Args:
        name: Provider name (claude, gemini, codex, cursor-agent)
        timeout: Timeout in seconds for provider operations (default: 120)

    Returns:
        Provider instance

    Raises:
        ProviderDisabledError: If provider is disabled in config
        typer.Exit: If provider is unknown
    """
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

    # Check if provider is enabled in config
    config = get_config()
    if not config.is_provider_enabled(name.lower()):
        raise ProviderDisabledError(
            f"Provider '{name}' is disabled in .claude/model_chorus_config.yaml. "
            f"Enable it in the config file or run setup to reconfigure."
        )

    return provider_class(timeout=timeout)


@app.command()
def chat(
    prompt_arg: Optional[str] = typer.Argument(None, help="Message to send to the AI model"),
    prompt_flag: Optional[str] = typer.Option(None, "--prompt", help="Message to send to the AI model (alternative to positional)"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
        "--session-id",
        help="Thread ID to continue an existing conversation",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="File paths to include in conversation context (can specify multiple times)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout per provider in seconds. Defaults to config or 120.0",
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
    Chat with a single AI model with conversation continuity.

    Example:
        # Start new conversation (positional prompt)
        model-chorus chat "What is quantum computing?" --provider claude

        # Start new conversation (flag prompt)
        model-chorus chat --prompt "What is quantum computing?" --provider claude

        # Continue conversation
        model-chorus chat "Give me an example" --continue thread-id-123

        # Include files
        model-chorus chat "Review this code" -f src/main.py -f tests/test_main.py
    """
    try:
        # Validate prompt input
        if prompt_arg is None and prompt_flag is None:
            console.print("[red]Error: Prompt is required (provide as positional argument or use --prompt)[/red]")
            raise typer.Exit(1)
        if prompt_arg is not None and prompt_flag is not None:
            console.print("[red]Error: Cannot specify prompt both as positional argument and --prompt flag[/red]")
            raise typer.Exit(1)

        prompt = prompt_arg or prompt_flag
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_workflow_default_provider('chat', 'claude')
            if provider is None:
                console.print("[red]Error: Default provider for 'chat' workflow is disabled.[/red]")
                console.print("[yellow]Enable it in .claude/model_chorus_config.yaml or specify --provider[/yellow]")
                raise typer.Exit(1)
        if system is None:
            system = config.get_workflow_default('chat', 'system_prompt', None)
        if timeout is None:
            timeout = config.get_workflow_default('chat', 'timeout', 120.0)

        # Create provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider, timeout=int(timeout))
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except ProviderDisabledError as e:
            # Provider disabled in config
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"\n[yellow]To fix this, edit .claude/model_chorus_config.yaml and set '{provider}: enabled: true'[/yellow]")
            raise typer.Exit(1)
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

        # Load fallback providers from config and filter by enabled status
        fallback_provider_names = config.get_workflow_fallback_providers('chat', exclude_provider=provider)

        fallback_providers = []
        if fallback_provider_names and verbose:
            console.print(f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]")

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = get_provider_by_name(fallback_name, timeout=int(timeout))
                fallback_providers.append(fallback_instance)
                if verbose:
                    console.print(f"[green]✓ {fallback_name} initialized (fallback)[/green]")
            except ProviderDisabledError:
                if verbose:
                    console.print(f"[yellow]⚠ Skipping disabled fallback provider {fallback_name}[/yellow]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]")

        # Create conversation memory (in-memory for now)
        memory = ConversationMemory()

        # Create workflow
        workflow = ChatWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            conversation_memory=memory,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        final_prompt = construct_prompt_with_files(prompt, files)

        # Display conversation info
        console.print(f"\n[bold cyan]{'Continuing' if continuation_id else 'Starting new'} chat conversation...[/bold cyan]")
        console.print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        console.print(f"Provider: {provider}")
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")
        if files:
            console.print(f"Files: {', '.join(files)}")
        console.print()

        # Run async workflow
        result = asyncio.run(
            workflow.run(
                prompt=final_prompt,
                continuation_id=continuation_id,
                files=files,
                skip_provider_check=skip_provider_check,
                system_prompt=system,
            )
        )

        # Display results
        if result.success:
            console.print(f"[bold green]✓ Chat completed[/bold green]\n")

            # Show thread info
            thread_id = result.metadata.get('thread_id')
            is_continuation = result.metadata.get('is_continuation', False)
            conv_length = result.metadata.get('conversation_length', 0)

            console.print(f"[cyan]Thread ID:[/cyan] {thread_id}")
            if not is_continuation:
                console.print("[cyan]Status:[/cyan] New conversation started")
            else:
                console.print(f"[cyan]Status:[/cyan] Continued ({conv_length} messages in thread)")
            console.print()

            # Show response
            console.print("[bold]Response:[/bold]\n")
            console.print(result.synthesis)

            # Show usage info if available
            usage = result.metadata.get('usage', {})
            if usage and verbose:
                console.print(f"\n[dim]Tokens: {usage.get('total_tokens', 'N/A')}[/dim]")

            # Save to file if requested
            if output:
                output_data = {
                    "prompt": prompt,
                    "provider": provider,
                    "thread_id": thread_id,
                    "is_continuation": is_continuation,
                    "response": result.synthesis,
                    "model": result.metadata.get('model'),
                    "usage": usage,
                    "conversation_length": conv_length,
                }
                if files:
                    output_data["files"] = files

                output.write_text(json.dumps(output_data, indent=2))
                console.print(f"\n[green]✓ Result saved to {output}[/green]")

            console.print(f"\n[dim]To continue this conversation, use: --continue {thread_id}[/dim]")

        else:
            console.print(f"[red]✗ Chat failed: {result.error}[/red]")
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


@app.command()
def argument(
    prompt_arg: Optional[str] = typer.Argument(None, help="Argument, claim, or question to analyze"),
    prompt_flag: Optional[str] = typer.Option(None, "--prompt", help="Argument, claim, or question to analyze (alternative to positional)"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
        "--session-id",
        help="Thread ID to continue an existing conversation",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="File paths to include in conversation context (can specify multiple times)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout per provider in seconds. Defaults to config or 120.0",
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
    Analyze arguments through structured dialectical reasoning.

    The argument workflow uses role-based orchestration to examine claims from
    multiple perspectives: Creator (thesis), Skeptic (critique), and Moderator (synthesis).

    Example:
        # Analyze an argument (positional prompt)
        model-chorus argument "Universal basic income would reduce poverty"

        # Analyze an argument (flag prompt)
        model-chorus argument --prompt "Universal basic income would reduce poverty"

        # Continue analysis
        model-chorus argument "What about inflation?" --continue thread-id-123

        # Include supporting files
        model-chorus argument "Review this proposal" -f proposal.md -f data.csv
    """
    try:
        # Validate prompt input
        if prompt_arg is None and prompt_flag is None:
            console.print("[red]Error: Prompt is required (provide as positional argument or use --prompt)[/red]")
            raise typer.Exit(1)
        if prompt_arg is not None and prompt_flag is not None:
            console.print("[red]Error: Cannot specify prompt both as positional argument and --prompt flag[/red]")
            raise typer.Exit(1)

        prompt = prompt_arg or prompt_flag
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_workflow_default_provider('argument', 'claude')
            if provider is None:
                console.print("[red]Error: Default provider for 'argument' workflow is disabled.[/red]")
                console.print("[yellow]Enable it in .claude/model_chorus_config.yaml or specify --provider[/yellow]")
                raise typer.Exit(1)
        if timeout is None:
            timeout = config.get_workflow_default('argument', 'timeout', 120.0)

        # Create provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider, timeout=int(timeout))
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except ProviderDisabledError as e:
            # Provider disabled in config
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"\n[yellow]To fix this, edit .claude/model_chorus_config.yaml and set '{provider}: enabled: true'[/yellow]")
            raise typer.Exit(1)
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

        # Load fallback providers from config and filter by enabled status
        fallback_provider_names = config.get_workflow_fallback_providers('argument', exclude_provider=provider)

        fallback_providers = []
        if fallback_provider_names and verbose:
            console.print(f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]")

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = get_provider_by_name(fallback_name, timeout=int(timeout))
                fallback_providers.append(fallback_instance)
                if verbose:
                    console.print(f"[green]✓ {fallback_name} initialized (fallback)[/green]")
            except ProviderDisabledError:
                if verbose:
                    console.print(f"[yellow]⚠ Skipping disabled fallback provider {fallback_name}[/yellow]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]")

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow
        workflow = ArgumentWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            conversation_memory=memory,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        final_prompt = construct_prompt_with_files(prompt, files)

        # Display analysis info
        console.print(f"\n[bold cyan]Analyzing argument through dialectical reasoning...[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]\n")

        # Build config
        config = {}
        if system:
            config['system_prompt'] = system

        # Execute workflow
        result = asyncio.run(
            workflow.run(
                prompt=final_prompt,
                continuation_id=continuation_id,
                files=files,
                skip_provider_check=skip_provider_check,
                **config
            )
        )

        # Display results
        console.print("\n[bold green]Analysis Complete[/bold green]")
        console.print(f"\n[bold]Dialectical Analysis:[/bold]")

        # Show each role's perspective
        if result.steps:
            for i, step in enumerate(result.steps, 1):
                role_name = step.name if hasattr(step, 'name') else f"Step {i}"
                console.print(f"\n[bold cyan]{role_name}:[/bold cyan]")
                console.print(step.content)

        # Show synthesis
        if result.synthesis:
            console.print(f"\n[bold magenta]Final Synthesis:[/bold magenta]")
            console.print(result.synthesis)

        # Show metadata
        if verbose and result.metadata:
            console.print(f"\n[dim]Thread ID: {result.metadata.get('thread_id', 'N/A')}[/dim]")
            console.print(f"[dim]Model: {result.metadata.get('model', 'N/A')}[/dim]")

        # Save output if requested
        if output:
            output_data = {
                'success': result.success,
                'synthesis': result.synthesis,
                'steps': [
                    {
                        'name': step.name if hasattr(step, 'name') else f'Step {i}',
                        'content': step.content,
                        'metadata': step.metadata if hasattr(step, 'metadata') else {}
                    }
                    for i, step in enumerate(result.steps, 1)
                ],
                'metadata': result.metadata,
            }

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]Results saved to: {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)


@app.command()
def ideate(
    prompt_arg: Optional[str] = typer.Argument(None, help="Topic or problem to brainstorm ideas for"),
    prompt_flag: Optional[str] = typer.Option(None, "--prompt", help="Topic or problem to brainstorm ideas for (alternative to positional)"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
        "--session-id",
        help="Thread ID to continue an existing ideation session",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="File paths to include in context (can specify multiple times)",
    ),
    num_ideas: int = typer.Option(
        5,
        "--num-ideas",
        "-n",
        help="Number of ideas to generate",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout per provider in seconds. Defaults to config or 120.0",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format)",
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
    Generate creative ideas through structured brainstorming.

    The ideate workflow uses enhanced creative prompting to generate diverse
    and innovative ideas for any topic or problem.

    Example:
        # Generate ideas (positional prompt)
        model-chorus ideate "New features for a task management app"

        # Generate ideas (flag prompt)
        model-chorus ideate --prompt "New features for a task management app"

        # Control creativity and quantity
        model-chorus ideate "Marketing campaign ideas" -n 10

        # Continue brainstorming
        model-chorus ideate "Refine the third idea" --continue thread-id-123
    """
    try:
        # Validate prompt input
        if prompt_arg is None and prompt_flag is None:
            console.print("[red]Error: Prompt is required (provide as positional argument or use --prompt)[/red]")
            raise typer.Exit(1)
        if prompt_arg is not None and prompt_flag is not None:
            console.print("[red]Error: Cannot specify prompt both as positional argument and --prompt flag[/red]")
            raise typer.Exit(1)

        prompt = prompt_arg or prompt_flag
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_workflow_default_provider('ideate', 'claude')
            if provider is None:
                console.print("[red]Error: Default provider for 'ideate' workflow is disabled.[/red]")
                console.print("[yellow]Enable it in .claude/model_chorus_config.yaml or specify --provider[/yellow]")
                raise typer.Exit(1)
        if timeout is None:
            timeout = config.get_workflow_default('ideate', 'timeout', 120.0)

        # Create provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider, timeout=int(timeout))
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except ProviderDisabledError as e:
            # Provider disabled in config
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"\n[yellow]To fix this, edit .claude/model_chorus_config.yaml and set '{provider}: enabled: true'[/yellow]")
            raise typer.Exit(1)
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

        # Load fallback providers from config and filter by enabled status
        fallback_provider_names = config.get_workflow_fallback_providers('ideate', exclude_provider=provider)

        fallback_providers = []
        if fallback_provider_names and verbose:
            console.print(f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]")

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = get_provider_by_name(fallback_name, timeout=int(timeout))
                fallback_providers.append(fallback_instance)
                if verbose:
                    console.print(f"[green]✓ {fallback_name} initialized (fallback)[/green]")
            except ProviderDisabledError:
                if verbose:
                    console.print(f"[yellow]⚠ Skipping disabled fallback provider {fallback_name}[/yellow]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]")

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow
        workflow = IdeateWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            conversation_memory=memory,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        final_prompt = construct_prompt_with_files(prompt, files)

        # Display ideation info
        console.print(f"\n[bold cyan]Generating creative ideas...[/bold cyan]")
        console.print(f"[dim]Topic: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
        console.print(f"[dim]Ideas requested: {num_ideas}[/dim]\n")

        # Build config
        config = {
            'num_ideas': num_ideas,
        }
        if system:
            config['system_prompt'] = system

        # Execute workflow
        result = asyncio.run(
            workflow.run(
                prompt=final_prompt,
                continuation_id=continuation_id,
                files=files,
                skip_provider_check=skip_provider_check,
                **config
            )
        )

        # Display results
        console.print("\n[bold green]Ideation Complete[/bold green]")

        # Show ideas from steps
        if result.steps:
            for i, step in enumerate(result.steps, 1):
                console.print(f"\n[bold cyan]{step.name if hasattr(step, 'name') else f'Idea {i}'}:[/bold cyan]")
                console.print(step.content)

        # Show synthesis
        if result.synthesis:
            console.print(f"\n[bold magenta]Summary & Recommendations:[/bold magenta]")
            console.print(result.synthesis)

        # Show metadata
        if verbose and result.metadata:
            console.print(f"\n[dim]Thread ID: {result.metadata.get('thread_id', 'N/A')}[/dim]")
            console.print(f"[dim]Model: {result.metadata.get('model', 'N/A')}[/dim]")

        # Save output if requested
        if output:
            output_data = {
                'success': result.success,
                'synthesis': result.synthesis,
                'ideas': [
                    {
                        'name': step.name if hasattr(step, 'name') else f'Idea {i}',
                        'content': step.content,
                        'metadata': step.metadata if hasattr(step, 'metadata') else {}
                    }
                    for i, step in enumerate(result.steps, 1)
                ],
                'metadata': result.metadata,
            }

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]Results saved to: {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)

def construct_prompt_with_files(prompt: str, files: Optional[List[str]]) -> str:
    """Construct a prompt by prepending the content of files."""
    if not files:
        return prompt

    file_content = "The user has provided the following file(s) for context:\n\n"
    for file_path in files:
        if not Path(file_path).exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        with open(file_path, "r") as f:
            file_content += f"--- {file_path} ---\n{f.read()}\n\n"
    
    return f"{prompt}\n\n{file_content}"


@app.command()
def consensus(
    prompt_arg: Optional[str] = typer.Argument(None, help="Prompt to send to all models"),
    prompt_flag: Optional[str] = typer.Option(None, "--prompt", help="Prompt to send to all models (alternative to positional)"),
    num_to_consult: Optional[int] = typer.Option(
        None,
        "--num-to-consult",
        "-m",
        help="Number of successful responses required (models to consult). Defaults to config or 2",
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Consensus strategy: all_responses, first_valid, majority, weighted, synthesize. Defaults to config or 'all_responses'",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="File paths to include in conversation context (can specify multiple times)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout per provider in seconds. Defaults to config or 120.0",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format)",
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
    Run consensus workflow with priority-based provider selection.

    Tries providers in priority order until num_to_consult successful responses
    are obtained. If a provider fails, automatically falls back to the next
    provider in the priority list.

    Example:
        # Positional prompt
        model-chorus consensus "Explain quantum computing" --num-to-consult 2

        # Flag prompt
        model-chorus consensus --prompt "Explain quantum computing" --num-to-consult 2
    """
    try:
        # Validate prompt input
        if prompt_arg is None and prompt_flag is None:
            console.print("[red]Error: Prompt is required (provide as positional argument or use --prompt)[/red]")
            raise typer.Exit(1)
        if prompt_arg is not None and prompt_flag is not None:
            console.print("[red]Error: Cannot specify prompt both as positional argument and --prompt flag[/red]")
            raise typer.Exit(1)

        prompt = prompt_arg or prompt_flag
        # Apply config defaults if values not provided
        config = get_config()

        # Get priority list from config
        provider_priority = config.get_workflow_provider_priority('consensus')

        # Get num_to_consult from CLI or config
        if num_to_consult is None:
            num_to_consult = config.get_workflow_num_to_consult('consensus', fallback=2)

        if strategy is None:
            strategy = config.get_workflow_default('consensus', 'strategy', 'all_responses')
        if timeout is None:
            timeout = config.get_workflow_default('consensus', 'timeout', 120.0)
        if system is None:
            system = config.get_workflow_default('consensus', 'system_prompt', None)

        # Validate strategy
        try:
            strategy_enum = ConsensusStrategy[strategy.upper()]
        except KeyError:
            console.print(f"[red]Error: Invalid strategy '{strategy}'[/red]")
            console.print(
                f"Valid strategies: {', '.join([s.name.lower() for s in ConsensusStrategy])}"
            )
            raise typer.Exit(1)

        final_prompt = construct_prompt_with_files(prompt, files)

        # Validate we have enough providers
        if not provider_priority:
            console.print("[red]Error: No providers configured for consensus workflow[/red]")
            console.print("[yellow]Please configure provider_priority in .claude/model_chorus_config.yaml[/yellow]")
            raise typer.Exit(1)

        if len(provider_priority) < num_to_consult:
            console.print(
                f"[red]Error: Not enough providers available. Need {num_to_consult} but only "
                f"{len(provider_priority)} enabled: {', '.join(provider_priority)}[/red]"
            )
            console.print("[yellow]Please enable more providers or reduce num_to_consult[/yellow]")
            raise typer.Exit(1)

        # Create provider instances for ALL providers in priority list (for fallback)
        if verbose:
            console.print(f"[cyan]Provider priority order: {', '.join(provider_priority)}[/cyan]")
            console.print(f"[cyan]Will consult {num_to_consult} providers (with automatic fallback)[/cyan]\n")

        provider_instances = []
        for provider_name in provider_priority:
            try:
                provider = get_provider_by_name(provider_name)
                provider_instances.append(provider)
                if verbose:
                    console.print(f"[green]✓ {provider_name} initialized[/green]")
            except ProviderDisabledError as e:
                # This shouldn't happen since we filtered above, but handle it anyway
                console.print(f"[yellow]Skipping disabled provider {provider_name}[/yellow]")
            except Exception as e:
                console.print(f"[red]Failed to initialize {provider_name}: {e}[/red]")
                raise typer.Exit(1)

        # Create workflow with dynamic fallback support
        workflow = ConsensusWorkflow(
            providers=provider_instances,
            strategy=strategy_enum,
            default_timeout=timeout,
            num_to_consult=num_to_consult,
        )

        # Apply provider-level metadata defaults (e.g., model overrides) from config
        for provider_config in workflow.provider_configs:
            provider_key = provider_config.provider.provider_name.lower()
            override_model = config.get_provider_model(provider_key)
            if override_model:
                metadata = dict(provider_config.metadata) if provider_config.metadata else {}
                if "model" not in metadata:
                    metadata["model"] = override_model
                    provider_config.metadata = metadata
                    if verbose:
                        console.print(
                            f"[cyan]Applied model override for {provider_key}: {override_model}[/cyan]"
                        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Create request
        request = GenerationRequest(
            prompt=final_prompt,
            system_prompt=system,
        )

        # Execute workflow
        console.print(f"\n[bold cyan]Executing consensus workflow...[/bold cyan]")
        console.print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        console.print(f"Providers: {len(provider_priority)} available (need {num_to_consult} successful)")
        console.print(f"Strategy: {strategy_enum.value}\n")

        # Run async workflow
        result = asyncio.run(workflow.execute(request, strategy=strategy_enum))

        # Display results
        console.print(f"[bold green]✓ Workflow completed[/bold green]\n")

        # Show summary table
        table = Table(title="Consensus Results")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Response Length", justify="right")

        # Show all providers that were tried (successful and failed)
        tried_providers = list(result.provider_results.keys()) + result.failed_providers
        for provider_name in tried_providers:
            if provider_name in result.provider_results:
                response = result.provider_results[provider_name]
                table.add_row(
                    provider_name,
                    "✓ Success",
                    f"{len(response.content)} chars",
                )
            else:
                table.add_row(
                    provider_name,
                    "✗ Failed",
                    "-",
                )

        console.print(table)
        console.print()

        # Show consensus response
        if result.consensus_response:
            if result.strategy_used == ConsensusStrategy.ALL_RESPONSES:
                console.print("[bold]Individual Provider Responses:[/bold]\n")
            else:
                console.print("[bold]Consensus Response:[/bold]\n")
            console.print(result.consensus_response)
        else:
            console.print("[yellow]No consensus response generated[/yellow]")

        # Save to file if requested
        if output:
            output_data = {
                "prompt": prompt,
                "strategy": strategy_enum.value,
                "providers": providers,
                "consensus_response": result.consensus_response,
                "responses": {
                    name: {
                        "content": resp.content,
                        "model": resp.model,
                        "usage": resp.usage,
                        "stop_reason": resp.stop_reason,
                    }
                    for name, resp in result.provider_results.items()
                },
                "failed_providers": result.failed_providers,
                "metadata": result.metadata,
            }

            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n[green]✓ Results saved to {output}[/green]")

        # Exit with appropriate code
        if result.failed_providers:
            console.print(
                f"\n[yellow]Warning: {len(result.failed_providers)} provider(s) failed[/yellow]"
            )
            if len(result.all_responses) == 0:
                console.print("[red]Error: All providers failed[/red]")
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


@app.command()
def thinkdeep(
    step: str = typer.Option(..., "--step", help="Investigation step description"),
    step_number: int = typer.Option(..., "--step-number", help="Current step index (starts at 1)"),
    total_steps: int = typer.Option(..., "--total-steps", help="Estimated total investigation steps"),
    next_step_required: bool = typer.Option(
        False,
        "--next-step-required",
        help="Continue investigation with another step (omit for final step)",
    ),
    findings: str = typer.Option(..., "--findings", help="What was discovered in this step"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Provider to use (claude, gemini, codex, cursor-agent). Defaults to config or 'claude'",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continuation-id", "--continue", "-c", "--session-id",
        help="Resume previous investigation thread",
    ),
    hypothesis: Optional[str] = typer.Option(
        None,
        "--hypothesis",
        help="Current working theory about the problem",
    ),
    confidence: str = typer.Option(
        "exploring",
        "--confidence",
        help="Confidence level (exploring, low, medium, high, very_high, almost_certain, certain)",
    ),
    files_checked: Optional[str] = typer.Option(
        None,
        "--files-checked",
        help="Comma-separated list of files examined",
    ),
    relevant_files: Optional[str] = typer.Option(
        None,
        "--relevant-files",
        help="Comma-separated list of files relevant to findings",
    ),
    thinking_mode: Optional[str] = typer.Option(
        None,
        "--thinking-mode",
        help="Reasoning depth (minimal, low, medium, high, max). Defaults to config or 'medium'",
    ),
    use_assistant_model: bool = typer.Option(
        True,
        "--use-assistant-model/--no-use-assistant-model",
        help="Enable expert validation (default: enabled)",
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
    Start a ThinkDeep investigation for systematic problem analysis.

    ThinkDeep provides multi-step investigation with explicit hypothesis tracking,
    confidence progression, and state management across investigation steps.

    Example:
        # Start new investigation
        model-chorus thinkdeep --step "Investigate why API latency increased" --step-number 1 --total-steps 3 --next-step-required --findings "Examining deployment logs" --confidence exploring

        # Continue investigation
        model-chorus thinkdeep --continuation-id "thread-123" --step "Check database query performance" --step-number 2 --total-steps 3 --next-step-required --findings "Found N+1 query pattern" --confidence medium --hypothesis "N+1 queries causing slowdown"

        # Final step (omit --next-step-required)
        model-chorus thinkdeep --continuation-id "thread-123" --step "Verify fix resolves issue" --step-number 3 --total-steps 3 --findings "Latency reduced to baseline" --confidence high --hypothesis "Confirmed: N+1 queries were root cause"
    """
    try:
        # Apply config defaults if values not provided
        config = get_config()
        if provider is None:
            provider = config.get_workflow_default_provider('thinkdeep', 'claude')
            if provider is None:
                console.print("[red]Error: Default provider for 'thinkdeep' workflow is disabled.[/red]")
                console.print("[yellow]Enable it in .claude/model_chorus_config.yaml or specify --provider[/yellow]")
                raise typer.Exit(1)
        if thinking_mode is None:
            thinking_mode = config.get_workflow_default('thinkdeep', 'thinking_mode', 'medium')

        # Read timeout from config
        timeout = config.get_workflow_default('thinkdeep', 'timeout', 120.0)

        # Validate confidence level
        valid_confidence_levels = ['exploring', 'low', 'medium', 'high', 'very_high', 'almost_certain', 'certain']
        if confidence not in valid_confidence_levels:
            console.print(f"[red]Error: Invalid confidence level '{confidence}'. Must be one of: {', '.join(valid_confidence_levels)}[/red]")
            raise typer.Exit(1)

        # Validate thinking mode
        valid_thinking_modes = ['minimal', 'low', 'medium', 'high', 'max']
        if thinking_mode not in valid_thinking_modes:
            console.print(f"[red]Error: Invalid thinking mode '{thinking_mode}'. Must be one of: {', '.join(valid_thinking_modes)}[/red]")
            raise typer.Exit(1)

        # Create primary provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider, timeout=int(timeout))
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except ProviderDisabledError as e:
            # Provider disabled in config
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"\n[yellow]To fix this, edit .claude/model_chorus_config.yaml and set '{provider}: enabled: true'[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider}: {e}[/red]")
            raise typer.Exit(1)

        # Validate provider supports requested parameters
        if thinking_mode and thinking_mode != 'medium':  # medium is default
            # Check if provider is gemini (which doesn't support thinking_mode)
            if provider == 'gemini':
                console.print(f"[yellow]Warning: Gemini provider does not support --thinking-mode parameter[/yellow]")
                console.print(f"[yellow]The parameter will be ignored. Use --provider claude or --provider codex for thinking mode support.[/yellow]")
                if not skip_provider_check:
                    console.print(f"[yellow]Proceeding anyway... (use Ctrl+C to cancel)[/yellow]")
                    import time
                    time.sleep(2)  # Give user time to cancel

        # Load fallback providers from config and filter by enabled status
        fallback_provider_names = config.get_workflow_fallback_providers('thinkdeep', exclude_provider=provider)

        fallback_providers = []
        if fallback_provider_names and verbose:
            console.print(f"[cyan]Initializing fallback providers: {', '.join(fallback_provider_names)}[/cyan]")

        for fallback_name in fallback_provider_names:
            try:
                fallback_instance = get_provider_by_name(fallback_name, timeout=int(timeout))
                fallback_providers.append(fallback_instance)
                if verbose:
                    console.print(f"[green]✓ {fallback_name} initialized (fallback)[/green]")
            except ProviderDisabledError:
                if verbose:
                    console.print(f"[yellow]⚠ Skipping disabled fallback provider {fallback_name}[/yellow]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠ Could not initialize fallback {fallback_name}: {e}[/yellow]")

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow config
        config = {
            'enable_expert_validation': use_assistant_model
        }

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=provider_instance,
            fallback_providers=fallback_providers,
            expert_provider=None,  # Expert will be handled by workflow if enabled
            conversation_memory=memory,
            config=config,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Parse files_checked if provided
        files_list = None
        if files_checked:
            resolved_files, remapped_files, missing_files = resolve_context_files(files_checked)
            if remapped_files:
                for notice in remapped_files:
                    console.print(f"[cyan]Remapped legacy file path: {notice}[/cyan]")
            if missing_files:
                console.print(f"[yellow]Warning: Skipping missing context file(s): {', '.join(missing_files)}[/yellow]")
            files_list = resolved_files or None

        # Parse relevant files if provided
        relevant_files_list = None
        if relevant_files:
            resolved_relevant, remapped_relevant, missing_relevant = resolve_context_files(relevant_files)
            if remapped_relevant:
                for notice in remapped_relevant:
                    console.print(f"[cyan]Remapped relevant file path: {notice}[/cyan]")
            if missing_relevant:
                console.print(f"[red]Error: Relevant file(s) not found: {', '.join(missing_relevant)}[/red]")
                raise typer.Exit(1)
            relevant_files_list = resolved_relevant or None

        # Display investigation info
        console.print(f"\n[bold cyan]{'Continuing' if continuation_id else 'Starting'} ThinkDeep Investigation[/bold cyan]")
        console.print(f"Step {step_number}/{total_steps}: {step}")
        console.print(f"Provider: {provider}")
        console.print(f"Confidence: {confidence}")
        if hypothesis:
            console.print(f"Hypothesis: {hypothesis}")
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")
        if files_list:
            console.print(f"Files: {', '.join(files_list)}")
        if relevant_files_list:
            console.print(f"Relevant files: {', '.join(relevant_files_list)}")
        console.print()

        # Run async workflow with multi-step parameters
        result = asyncio.run(
            workflow.run(
                step=step,
                step_number=step_number,
                total_steps=total_steps,
                next_step_required=next_step_required,
                findings=findings,
                hypothesis=hypothesis,
                confidence=confidence,
                continuation_id=continuation_id,
                files=files_list,
                relevant_files=relevant_files_list,
                skip_provider_check=skip_provider_check,
                thinking_mode=thinking_mode,
            )
        )

        # Display results
        if result.success:
            console.print(f"[bold green]✓ Investigation step {step_number}/{total_steps} completed[/bold green]\n")

            # Show step info
            thread_id = result.metadata.get('thread_id')
            returned_continuation_id = result.metadata.get('continuation_id', thread_id)

            console.print(f"[cyan]Continuation ID:[/cyan] {returned_continuation_id}")
            console.print(f"[cyan]Step:[/cyan] {step_number}/{total_steps}")
            console.print(f"[cyan]Confidence Level:[/cyan] {confidence}")
            if hypothesis:
                console.print(f"[cyan]Hypothesis:[/cyan] {hypothesis}")
            console.print(f"[cyan]Findings:[/cyan] {findings}")
            if files_list:
                console.print(f"[cyan]Files Examined:[/cyan] {len(files_list)}")
            relevant_files_this_step = result.metadata.get('relevant_files_this_step') or []
            if relevant_files_this_step:
                console.print(f"[cyan]Relevant Files (this step):[/cyan] {', '.join(relevant_files_this_step)}")
            cumulative_relevant_files = result.metadata.get('relevant_files') or []
            if cumulative_relevant_files:
                console.print(f"[cyan]Relevant Files (cumulative):[/cyan] {', '.join(cumulative_relevant_files)}")
            console.print()

            # Show response
            console.print("[bold]Investigation Analysis:[/bold]\n")
            console.print(result.synthesis)

            # Show expert validation if present
            if use_assistant_model and len(result.steps) > 1:
                console.print("\n[bold]Expert Validation:[/bold]\n")
                console.print(result.steps[-1].content)

            # Show usage info if available
            usage = result.metadata.get('usage', {})
            if usage and verbose:
                console.print(f"\n[dim]Tokens: {usage.get('total_tokens', 'N/A')}[/dim]")

            # Save to file if requested
            if output:
                output_data = {
                    "step": step,
                    "step_number": step_number,
                    "total_steps": total_steps,
                    "next_step_required": next_step_required,
                    "findings": findings,
                    "hypothesis": hypothesis,
                    "confidence": confidence,
                    "provider": provider,
                    "continuation_id": returned_continuation_id,
                    "response": result.synthesis,
                    "usage": usage,
                }
                if files_list:
                    output_data["files"] = files_list
                if relevant_files_this_step:
                    output_data["relevant_files_this_step"] = relevant_files_this_step
                cumulative_relevant_files = result.metadata.get('relevant_files') or []
                if cumulative_relevant_files:
                    output_data["relevant_files"] = cumulative_relevant_files
                if use_assistant_model and len(result.steps) > 1:
                    output_data["expert_validation"] = result.steps[-1].content

                output.write_text(json.dumps(output_data, indent=2))
                console.print(f"\n[green]✓ Result saved to {output}[/green]")

            # Show next step guidance
            if next_step_required:
                console.print(f"\n[dim]To continue this investigation:[/dim]")
                console.print(f"[dim]  model-chorus thinkdeep --continuation-id {returned_continuation_id} --step \"Next investigation step\" --step-number {step_number + 1} --total-steps {total_steps} ...[/dim]")
            else:
                console.print(f"\n[green]✓ Investigation complete ({total_steps} steps)[/green]")

        else:
            console.print(f"[red]✗ Investigation failed: {result.error}[/red]")
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


@app.command(name="thinkdeep-status")
def thinkdeep_status(
    thread_id: str = typer.Argument(..., help="Thread ID of the investigation to inspect"),
    show_steps: bool = typer.Option(
        False,
        "--steps",
        help="Show all investigation steps",
    ),
    show_files: bool = typer.Option(
        False,
        "--files",
        help="Show all examined files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including evidence",
    ),
):
    """
    Inspect the state of an ongoing ThinkDeep investigation.

    Shows current hypotheses, confidence level, investigation progress,
    and optionally all steps and examined files.

    Example:
        # View basic status
        model-chorus thinkdeep-status thread-id-123

        # View with all steps
        model-chorus thinkdeep-status thread-id-123 --steps

        # View with files
        model-chorus thinkdeep-status thread-id-123 --files --verbose
    """
    try:
        # Create conversation memory to access thread
        memory = ConversationMemory()

        # Create a dummy workflow to access state
        # (We need a provider but won't use it for inspection)
        dummy_provider = ClaudeProvider()
        workflow = ThinkDeepWorkflow(
            provider=dummy_provider,
            conversation_memory=memory,
        )

        # Get investigation state
        state = workflow.get_investigation_state(thread_id)

        if not state:
            console.print(f"[red]Error: Investigation thread not found: {thread_id}[/red]")
            console.print("[yellow]Make sure you're using the correct thread ID from a previous investigation.[/yellow]")
            raise typer.Exit(1)

        # Get investigation summary
        summary = workflow.get_investigation_summary(thread_id)

        # Display header
        console.print(f"\n[bold cyan]Investigation Status[/bold cyan]")
        console.print(f"[cyan]Thread ID:[/cyan] {thread_id}\n")

        # Display summary metrics
        console.print(f"[cyan]Confidence Level:[/cyan] {state.current_confidence}")
        console.print(f"[cyan]Total Steps:[/cyan] {len(state.steps)}")
        console.print(f"[cyan]Total Hypotheses:[/cyan] {len(state.hypotheses)}")
        console.print(f"[cyan]Files Examined:[/cyan] {len(state.relevant_files)}")

        if summary:
            console.print(f"[cyan]Active Hypotheses:[/cyan] {summary['active_hypotheses']}")
            console.print(f"[cyan]Validated Hypotheses:[/cyan] {summary['validated_hypotheses']}")
            console.print(f"[cyan]Disproven Hypotheses:[/cyan] {summary['disproven_hypotheses']}")
            console.print(f"[cyan]Complete:[/cyan] {'Yes' if summary['is_complete'] else 'No'}")

        # Display hypotheses
        if state.hypotheses:
            console.print("\n[bold]Hypotheses:[/bold]")
            for i, hyp in enumerate(state.hypotheses, 1):
                status_color = {
                    'active': 'yellow',
                    'validated': 'green',
                    'disproven': 'red'
                }.get(hyp.status, 'white')
                console.print(f"  {i}. [{status_color}]{hyp.status.upper()}[/{status_color}] {hyp.hypothesis}")
                if hyp.evidence and verbose:
                    console.print(f"     [dim]Evidence ({len(hyp.evidence)}):[/dim]")
                    for evidence in hyp.evidence:
                        console.print(f"       • {evidence}")
        else:
            console.print("\n[yellow]No hypotheses yet.[/yellow]")

        # Display investigation steps if requested
        if show_steps and state.steps:
            console.print("\n[bold]Investigation Steps:[/bold]")
            for i, step in enumerate(state.steps, 1):
                console.print(f"\n  [cyan]Step {step.step_number}:[/cyan]")
                console.print(f"  Confidence: {step.confidence}")
                console.print(f"  Files checked: {len(step.files_checked)}")
                if verbose:
                    console.print(f"  Findings: {step.findings}")
                else:
                    # Truncate findings if not verbose
                    findings_preview = step.findings[:150] + "..." if len(step.findings) > 150 else step.findings
                    console.print(f"  Findings: {findings_preview}")

        # Display examined files if requested
        if show_files and state.relevant_files:
            console.print("\n[bold]Examined Files:[/bold]")
            for file in state.relevant_files:
                console.print(f"  • {file}")

        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)


@app.command(name="config")
def config_cmd(
    subcommand: str = typer.Argument(
        ...,
        help="Config subcommand: show, validate, or init"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
):
    """
    Manage ModelChorus configuration.

    Subcommands:
        show     - Display current effective configuration
        validate - Validate .model-chorusrc file
        init     - Generate sample .model-chorusrc file

    Examples:
        model-chorus config show
        model-chorus config validate
        model-chorus config init
    """
    try:
        if subcommand == "show":
            _config_show(verbose)
        elif subcommand == "validate":
            _config_validate(verbose)
        elif subcommand == "init":
            _config_init(verbose)
        else:
            console.print(f"[red]Error: Unknown config subcommand '{subcommand}'[/red]")
            console.print("Valid subcommands: show, validate, init")
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


def _config_show(verbose: bool):
    """Show current effective configuration."""
    loader = get_config()
    config_obj = loader.get_config()

    console.print("\n[bold cyan]ModelChorus Configuration[/bold cyan]\n")

    # Show config file location
    if loader.config_path:
        console.print(f"[green]✓ Config file found:[/green] {loader.config_path}\n")
    else:
        console.print("[yellow]⚠ No config file found (using defaults)[/yellow]\n")

    # Show global defaults
    console.print("[bold]Global Defaults:[/bold]")
    if config_obj.default_provider:
        console.print(f"  default_provider: {config_obj.default_provider}")
    else:
        console.print("  default_provider: [dim](not set)[/dim]")

    if config_obj.generation:
        console.print("  generation:")
        if config_obj.generation.temperature is not None:
            console.print(f"    temperature: {config_obj.generation.temperature}")
        if config_obj.generation.max_tokens is not None:
            console.print(f"    max_tokens: {config_obj.generation.max_tokens}")
        if config_obj.generation.timeout is not None:
            console.print(f"    timeout: {config_obj.generation.timeout}")
        if config_obj.generation.system_prompt is not None:
            console.print(f"    system_prompt: {config_obj.generation.system_prompt[:50]}...")

    # Show workflow-specific config
    if config_obj.workflows:
        console.print("\n[bold]Workflow-Specific Configuration:[/bold]")
        for workflow_name, workflow_config in config_obj.workflows.items():
            console.print(f"\n  [cyan]{workflow_name}:[/cyan]")
            config_dict = workflow_config.model_dump(exclude_none=True)
            for key, value in config_dict.items():
                if isinstance(value, list):
                    console.print(f"    {key}: {', '.join(value)}")
                else:
                    console.print(f"    {key}: {value}")

    # Show effective defaults for each workflow if verbose
    if verbose:
        console.print("\n[bold]Effective Defaults by Workflow:[/bold]")
        workflows = ['chat', 'consensus', 'thinkdeep', 'argument', 'ideate', 'research']
        for workflow in workflows:
            console.print(f"\n  [cyan]{workflow}:[/cyan]")
            if workflow == 'consensus':
                providers = loader.get_default_providers(workflow, ['claude', 'gemini'])
                console.print(f"    providers: {', '.join(providers)}")
            else:
                provider = loader.get_workflow_default_provider(workflow, 'claude')
                console.print(f"    provider: {provider if provider else '[disabled]'}")
            temp = loader.get_workflow_default(workflow, 'temperature', 0.7)
            console.print(f"    temperature: {temp}")

    console.print()


def _config_validate(verbose: bool):
    """Validate .model-chorusrc file."""
    loader = get_config()

    # Find config file
    config_path = loader.find_config_file()

    if not config_path:
        console.print("[yellow]⚠ No .model-chorusrc file found in current directory or parent directories[/yellow]")
        console.print("\nSearched for: .model-chorusrc, .model-chorusrc.yaml, .model-chorusrc.yml, .model-chorusrc.json")
        console.print("\nTo create a sample config file, run: model-chorus config init")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Validating config file:[/cyan] {config_path}\n")

    # Try to load and validate
    try:
        loader.load_config(config_path)
        console.print("[green]✓ Configuration is valid![/green]\n")

        if verbose:
            console.print("Configuration details:")
            _config_show(verbose=False)
    except Exception as e:
        console.print(f"[red]✗ Configuration is invalid:[/red]\n")
        console.print(f"  {str(e)}\n")
        raise typer.Exit(1)


def _config_init(verbose: bool):
    """Initialize a sample .model-chorusrc file."""
    config_filename = ".model-chorusrc"
    config_path = Path.cwd() / config_filename

    if config_path.exists():
        console.print(f"[yellow]⚠ Config file already exists:[/yellow] {config_path}")
        console.print("\nTo overwrite, delete the existing file first.")
        raise typer.Exit(1)

    # Read the example config from the package
    example_config = """# ModelChorus Configuration
# Supported formats: YAML or JSON

# Default provider for all workflows
default_provider: claude

# Global generation parameters
generation:
  temperature: 0.7
  max_tokens: 2000
  timeout: 120.0

# Workflow-specific overrides
workflows:
  chat:
    provider: claude
    temperature: 0.7

  consensus:
    providers:
      - claude
      - gemini
    strategy: synthesize
    temperature: 0.7

  thinkdeep:
    provider: claude
    thinking_mode: medium
    temperature: 0.6

  research:
    providers:
      - claude
      - gemini
    citation_style: academic
    depth: thorough
    temperature: 0.5
"""

    try:
        with open(config_path, 'w') as f:
            f.write(example_config)

        console.print(f"[green]✓ Created config file:[/green] {config_path}\n")
        console.print("You can now edit this file to customize your ModelChorus configuration.")
        console.print("\nTo validate your config, run: model-chorus config validate")

        if verbose:
            console.print(f"\n[dim]Config file contents:[/dim]")
            console.print(example_config)
    except Exception as e:
        console.print(f"[red]✗ Failed to create config file:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list_providers(
    check: bool = typer.Option(
        False,
        "--check",
        help="Check if provider CLIs are actually installed and working"
    )
):
    """List all available providers and their models.

    Use --check to verify which providers are actually installed and working.
    """
    providers = {
        "claude": ClaudeProvider(),
        "codex": CodexProvider(),
        "gemini": GeminiProvider(),
        "cursor-agent": CursorAgentProvider(),
    }

    console.print("\n[bold]Available Providers:[/bold]\n")

    async def check_provider_async(name: str, provider):
        """Check a single provider's availability."""
        is_available, error = await provider.check_availability()
        return name, provider, is_available, error

    # If --check flag is set, test provider availability
    if check:
        console.print("[dim]Checking provider availability...[/dim]\n")

        # Run all availability checks concurrently
        async def check_all():
            tasks = [check_provider_async(name, prov) for name, prov in providers.items()]
            return await asyncio.gather(*tasks)

        results = asyncio.run(check_all())

        for name, provider, is_available, error in results:
            # Status indicator
            if is_available:
                status = "[green]✓ Installed and working[/green]"
            else:
                status = f"[red]✗ Not available[/red]"

            console.print(f"[cyan]● {name}[/cyan]")
            console.print(f"  Status: {status}")
            console.print(f"  Provider: {provider.provider_name}")
            console.print(f"  CLI Command: {provider.cli_command}")

            if not is_available:
                console.print(f"  [yellow]Issue:[/yellow] {error}")
                console.print(f"  [yellow]Install:[/yellow] {get_install_command(name)}")
            else:
                models = provider.get_available_models()
                console.print(f"  Models ({len(models)}):")
                for model in models:
                    capabilities = [cap.value for cap in model.capabilities]
                    console.print(f"    - {model.model_id}: {', '.join(capabilities)}")

            console.print()
    else:
        # Original behavior without availability checking
        for name, provider in providers.items():
            console.print(f"[cyan]● {name}[/cyan]")
            console.print(f"  Provider: {provider.provider_name}")
            console.print(f"  CLI Command: {provider.cli_command}")

            models = provider.get_available_models()
            console.print(f"  Models ({len(models)}):")

            for model in models:
                capabilities = [cap.value for cap in model.capabilities]
                console.print(f"    - {model.model_id}: {', '.join(capabilities)}")

            console.print()

        console.print("[dim]Use --check to verify which providers are actually installed[/dim]\n")




@app.command()
def version():
    """Show version information."""
    console.print("[bold]ModelChorus[/bold] - Multi-Model AI Workflow Orchestration")
    console.print(f"Version: {__version__}")
    console.print("\nProviders:")
    console.print("  - Claude (Anthropic)")
    console.print("  - Codex (OpenAI)")
    console.print("  - Gemini (Google)")
    console.print("  - Cursor Agent")


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
