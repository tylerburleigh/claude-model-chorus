"""
CLI interface for ModelChorus.

Typer-based command-line application for running multi-model workflows.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

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
from ..workflows import ChatWorkflow, ConsensusWorkflow, ConsensusStrategy, ThinkDeepWorkflow
from ..core.conversation import ConversationMemory

app = typer.Typer(
    name="modelchorus",
    help="Multi-model AI workflow orchestration",
    add_completion=False,
)
console = Console()


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


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Message to send to the AI model"),
    provider: str = typer.Option(
        "claude",
        "--provider",
        "-p",
        help="Provider to use (claude, gemini, codex, cursor-agent)",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
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
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature for generation (0.0-1.0)",
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
):
    """
    Chat with a single AI model with conversation continuity.

    Example:
        # Start new conversation
        modelchorus chat "What is quantum computing?" -p claude

        # Continue conversation
        modelchorus chat "Give me an example" --continue thread-id-123

        # Include files
        modelchorus chat "Review this code" -f src/main.py -f tests/test_main.py
    """
    try:
        # Create provider instance
        if verbose:
            console.print(f"[cyan]Initializing provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider)
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider}: {e}[/red]")
            raise typer.Exit(1)

        # Create conversation memory (in-memory for now)
        memory = ConversationMemory()

        # Create workflow
        workflow = ChatWorkflow(
            provider=provider_instance,
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
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens,
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
def consensus(
    prompt: str = typer.Argument(..., help="Prompt to send to all models"),
    providers: List[str] = typer.Option(
        ["claude", "gemini"],
        "--provider",
        "-p",
        help="Providers to use (can specify multiple times)",
    ),
    strategy: str = typer.Option(
        "all_responses",
        "--strategy",
        "-s",
        help="Consensus strategy: all_responses, first_valid, majority, weighted, synthesize",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature for generation (0.0-1.0)",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    timeout: float = typer.Option(
        120.0,
        "--timeout",
        help="Timeout per provider in seconds",
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
):
    """
    Run consensus workflow across multiple AI models.

    Example:
        modelchorus consensus "Explain quantum computing" -p claude -p gemini -s synthesize
    """
    try:
        # Validate strategy
        try:
            strategy_enum = ConsensusStrategy[strategy.upper()]
        except KeyError:
            console.print(f"[red]Error: Invalid strategy '{strategy}'[/red]")
            console.print(
                f"Valid strategies: {', '.join([s.name.lower() for s in ConsensusStrategy])}"
            )
            raise typer.Exit(1)

        # Create provider instances
        if verbose:
            console.print(f"[cyan]Creating providers: {', '.join(providers)}[/cyan]")

        provider_instances = []
        for provider_name in providers:
            try:
                provider = get_provider_by_name(provider_name)
                provider_instances.append(provider)
                if verbose:
                    console.print(f"[green]✓ {provider_name} initialized[/green]")
            except Exception as e:
                console.print(f"[red]Failed to initialize {provider_name}: {e}[/red]")
                raise typer.Exit(1)

        # Create workflow
        workflow = ConsensusWorkflow(
            providers=provider_instances,
            strategy=strategy_enum,
            default_timeout=timeout,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Create request
        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Execute workflow
        console.print(f"\n[bold cyan]Executing consensus workflow...[/bold cyan]")
        console.print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        console.print(f"Providers: {len(provider_instances)}")
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

        for provider_name in providers:
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
    prompt: str = typer.Argument(..., help="Investigation problem statement or query"),
    provider: str = typer.Option(
        "claude",
        "--provider",
        "-p",
        help="Provider to use for investigation (claude, gemini, codex, cursor-agent)",
    ),
    expert_provider: Optional[str] = typer.Option(
        None,
        "--expert",
        "-e",
        help="Expert provider for validation (optional, uses different model for validation)",
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
        help="File paths to examine during investigation (can specify multiple times)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature for generation (0.0-1.0)",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    disable_expert: bool = typer.Option(
        False,
        "--disable-expert",
        help="Disable expert validation even if expert provider is specified",
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
):
    """
    Start a ThinkDeep investigation for systematic problem analysis.

    ThinkDeep provides extended reasoning with hypothesis tracking, confidence
    progression, and optional expert validation.

    Example:
        # Start new investigation
        modelchorus thinkdeep "Why is authentication failing?" -p claude

        # Continue investigation
        modelchorus thinkdeep "Check async patterns" --continue thread-id-123

        # Include files and expert validation
        modelchorus thinkdeep "Analyze bug" -f src/auth.py -e gemini
    """
    try:
        # Create primary provider instance
        if verbose:
            console.print(f"[cyan]Initializing primary provider: {provider}[/cyan]")

        try:
            provider_instance = get_provider_by_name(provider)
            if verbose:
                console.print(f"[green]✓ {provider} initialized[/green]")
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider}: {e}[/red]")
            raise typer.Exit(1)

        # Create expert provider if specified
        expert_instance = None
        if expert_provider:
            if verbose:
                console.print(f"[cyan]Initializing expert provider: {expert_provider}[/cyan]")
            try:
                expert_instance = get_provider_by_name(expert_provider)
                if verbose:
                    console.print(f"[green]✓ {expert_provider} initialized as expert[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to initialize expert provider: {e}[/yellow]")
                console.print("[yellow]Continuing without expert validation[/yellow]")

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow config
        config = {}
        if disable_expert:
            config['enable_expert_validation'] = False

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=provider_instance,
            expert_provider=expert_instance,
            conversation_memory=memory,
            config=config if config else None,
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
        console.print(f"\n[bold cyan]{'Continuing' if continuation_id else 'Starting new'} ThinkDeep investigation...[/bold cyan]")
        console.print(f"Problem: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        console.print(f"Primary Provider: {provider}")
        if expert_instance:
            expert_status = "disabled" if disable_expert else "enabled"
            console.print(f"Expert Provider: {expert_provider} ({expert_status})")
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")
        if files:
            console.print(f"Files: {', '.join(files)}")
        console.print()

        # Run async workflow
        result = asyncio.run(
            workflow.run(
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        # Display results
        if result.success:
            console.print(f"[bold green]✓ Investigation step completed[/bold green]\n")

            # Show thread info
            thread_id = result.metadata.get('thread_id')
            is_continuation = result.metadata.get('is_continuation', False)
            step_number = result.metadata.get('investigation_step', 1)
            confidence = result.metadata.get('confidence', 'exploring')
            hypotheses_count = result.metadata.get('hypotheses_count', 0)
            expert_performed = result.metadata.get('expert_validation_performed', False)
            files_examined = result.metadata.get('files_examined', 0)

            console.print(f"[cyan]Thread ID:[/cyan] {thread_id}")
            console.print(f"[cyan]Investigation Step:[/cyan] {step_number}")
            console.print(f"[cyan]Confidence Level:[/cyan] {confidence}")
            console.print(f"[cyan]Hypotheses:[/cyan] {hypotheses_count}")
            console.print(f"[cyan]Files Examined:[/cyan] {files_examined}")
            if expert_performed:
                console.print(f"[cyan]Expert Validation:[/cyan] ✓ Performed")

            # Show hypothesis details if continuing investigation
            if is_continuation and hypotheses_count > 0:
                console.print("\n[bold]Current Hypotheses:[/bold]")
                # Get investigation state to show hypotheses
                state = workflow.get_investigation_state(thread_id)
                if state and state.hypotheses:
                    for i, hyp in enumerate(state.hypotheses, 1):
                        status_color = {
                            'active': 'yellow',
                            'validated': 'green',
                            'disproven': 'red'
                        }.get(hyp.status, 'white')
                        console.print(f"  {i}. [{status_color}]{hyp.status.upper()}[/{status_color}] {hyp.hypothesis}")
                        if hyp.evidence and verbose:
                            console.print(f"     Evidence: {', '.join(hyp.evidence[:3])}")
            console.print()

            # Show response
            console.print("[bold]Investigation Findings:[/bold]\n")
            console.print(result.synthesis)

            # Show expert validation if present
            if expert_performed and len(result.steps) > 1:
                console.print("\n[bold]Expert Validation:[/bold]\n")
                console.print(result.steps[-1].content)

            # Show usage info if available
            usage = result.metadata.get('usage', {})
            if usage and verbose:
                console.print(f"\n[dim]Tokens: {usage.get('total_tokens', 'N/A')}[/dim]")

            # Save to file if requested
            if output:
                output_data = {
                    "prompt": prompt,
                    "provider": provider,
                    "expert_provider": expert_provider,
                    "thread_id": thread_id,
                    "is_continuation": is_continuation,
                    "investigation_step": step_number,
                    "confidence": confidence,
                    "hypotheses_count": hypotheses_count,
                    "expert_validation_performed": expert_performed,
                    "response": result.synthesis,
                    "model": result.metadata.get('model'),
                    "usage": usage,
                }
                if files:
                    output_data["files"] = files
                if expert_performed and len(result.steps) > 1:
                    output_data["expert_validation"] = result.steps[-1].content

                output.write_text(json.dumps(output_data, indent=2))
                console.print(f"\n[green]✓ Result saved to {output}[/green]")

            console.print(f"\n[dim]To continue this investigation, use: --continue {thread_id}[/dim]")

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


@app.command()
def list_providers():
    """List all available providers and their models."""
    providers = {
        "claude": ClaudeProvider(),
        "codex": CodexProvider(),
        "gemini": GeminiProvider(),
        "cursor-agent": CursorAgentProvider(),
    }

    console.print("\n[bold]Available Providers:[/bold]\n")

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


@app.command()
def version():
    """Show version information."""
    console.print("[bold]ModelChorus[/bold] - Multi-Model AI Workflow Orchestration")
    console.print("Version: 0.1.0")
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
