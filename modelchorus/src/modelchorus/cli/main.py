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
from ..workflows import ArgumentWorkflow, ChatWorkflow, ConsensusWorkflow, ConsensusStrategy, IdeateWorkflow, ResearchWorkflow, ThinkDeepWorkflow
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
def argument(
    prompt: str = typer.Argument(..., help="Argument, claim, or question to analyze"),
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
    Analyze arguments through structured dialectical reasoning.

    The argument workflow uses role-based orchestration to examine claims from
    multiple perspectives: Creator (thesis), Skeptic (critique), and Moderator (synthesis).

    Example:
        # Analyze an argument
        modelchorus argument "Universal basic income would reduce poverty"

        # Continue analysis
        modelchorus argument "What about inflation?" --continue thread-id-123

        # Include supporting files
        modelchorus argument "Review this proposal" -f proposal.md -f data.csv
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

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow
        workflow = ArgumentWorkflow(
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

        # Display analysis info
        console.print(f"\n[bold cyan]Analyzing argument through dialectical reasoning...[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]\n")

        # Build config
        config = {}
        if system:
            config['system_prompt'] = system
        if temperature is not None:
            config['temperature'] = temperature
        if max_tokens is not None:
            config['max_tokens'] = max_tokens

        # Execute workflow
        result = asyncio.run(
            workflow.run(
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
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
    prompt: str = typer.Argument(..., help="Topic or problem to brainstorm ideas for"),
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
    temperature: float = typer.Option(
        0.9,
        "--temperature",
        "-t",
        help="Temperature for generation (0.0-1.0, higher = more creative)",
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
    Generate creative ideas through structured brainstorming.

    The ideate workflow uses enhanced creative prompting to generate diverse
    and innovative ideas for any topic or problem.

    Example:
        # Generate ideas
        modelchorus ideate "New features for a task management app"

        # Control creativity and quantity
        modelchorus ideate "Marketing campaign ideas" -n 10 -t 1.0

        # Continue brainstorming
        modelchorus ideate "Refine the third idea" --continue thread-id-123
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

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow
        workflow = IdeateWorkflow(
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

        # Display ideation info
        console.print(f"\n[bold cyan]Generating creative ideas...[/bold cyan]")
        console.print(f"[dim]Topic: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
        console.print(f"[dim]Ideas requested: {num_ideas}[/dim]\n")

        # Build config
        config = {
            'num_ideas': num_ideas,
            'temperature': temperature,
        }
        if system:
            config['system_prompt'] = system
        if max_tokens is not None:
            config['max_tokens'] = max_tokens

        # Execute workflow
        result = asyncio.run(
            workflow.run(
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
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


@app.command()
def research(
    prompt: str = typer.Argument(..., help="Research question or topic to investigate"),
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
        help="Thread ID to continue an existing research session",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--file",
        "-f",
        help="Source files to include in research (can specify multiple times)",
    ),
    citation_style: str = typer.Option(
        "informal",
        "--citation-style",
        help="Citation format: informal, academic, or technical",
    ),
    research_depth: str = typer.Option(
        "thorough",
        "--depth",
        "-d",
        help="Research depth: shallow, moderate, thorough, or comprehensive",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        help="System prompt for context",
    ),
    temperature: float = typer.Option(
        0.5,
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
        help="Output file for research dossier (JSON format)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed execution information",
    ),
):
    """
    Conduct systematic research with evidence extraction and citations.

    The research workflow gathers information from multiple sources, extracts evidence,
    validates claims, and generates a comprehensive research dossier with proper citations.

    Example:
        # Basic research
        modelchorus research "What are the latest trends in AI orchestration?"

        # With source files and citations
        modelchorus research "Analyze user feedback" -f feedback.txt --citation-style academic

        # Deep research with comprehensive analysis
        modelchorus research "Security vulnerabilities in microservices" --depth comprehensive
    """
    try:
        # Validate citation style
        valid_styles = ['informal', 'academic', 'technical']
        if citation_style not in valid_styles:
            console.print(f"[red]Error: Invalid citation style '{citation_style}'[/red]")
            console.print(f"Valid styles: {', '.join(valid_styles)}")
            raise typer.Exit(1)

        # Validate research depth
        valid_depths = ['shallow', 'moderate', 'thorough', 'comprehensive']
        if research_depth not in valid_depths:
            console.print(f"[red]Error: Invalid research depth '{research_depth}'[/red]")
            console.print(f"Valid depths: {', '.join(valid_depths)}")
            raise typer.Exit(1)

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

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow
        workflow = ResearchWorkflow(
            provider=provider_instance,
            conversation_memory=memory,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Validate and ingest source files
        if files:
            for file_path in files:
                path = Path(file_path)
                if not path.exists():
                    console.print(f"[red]Error: File not found: {file_path}[/red]")
                    raise typer.Exit(1)

                # Read file content and ingest as source
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    workflow.ingest_source(
                        title=path.name,
                        url=str(path.absolute()),
                        source_type='document',
                        credibility='high'
                    )

                    if verbose:
                        console.print(f"[green]✓ Ingested source: {path.name}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")

        # Display research info
        console.print(f"\n[bold cyan]Conducting systematic research...[/bold cyan]")
        console.print(f"[dim]Question: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
        console.print(f"[dim]Depth: {research_depth} | Citations: {citation_style}[/dim]")
        if files:
            console.print(f"[dim]Sources: {len(files)} file(s)[/dim]")
        console.print()

        # Build config
        config = {
            'citation_style': citation_style,
            'research_depth': research_depth,
            'temperature': temperature,
        }
        if system:
            config['system_prompt'] = system
        if max_tokens is not None:
            config['max_tokens'] = max_tokens

        # Execute workflow
        result = asyncio.run(
            workflow.run(
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
                **config
            )
        )

        # Display results
        console.print("\n[bold green]Research Complete[/bold green]")

        # Show research findings
        if result.steps:
            for i, step in enumerate(result.steps, 1):
                console.print(f"\n[bold cyan]{step.name if hasattr(step, 'name') else f'Finding {i}'}:[/bold cyan]")
                console.print(step.content)

        # Show synthesis/dossier
        if result.synthesis:
            console.print(f"\n[bold magenta]Research Dossier:[/bold magenta]")
            console.print(result.synthesis)

        # Show metadata
        if verbose and result.metadata:
            console.print(f"\n[dim]Thread ID: {result.metadata.get('thread_id', 'N/A')}[/dim]")
            console.print(f"[dim]Model: {result.metadata.get('model', 'N/A')}[/dim]")
            if 'sources_analyzed' in result.metadata:
                console.print(f"[dim]Sources analyzed: {result.metadata['sources_analyzed']}[/dim]")

        # Save output if requested
        if output:
            output_data = {
                'success': result.success,
                'research_question': prompt,
                'dossier': result.synthesis,
                'findings': [
                    {
                        'name': step.name if hasattr(step, 'name') else f'Finding {i}',
                        'content': step.content,
                        'metadata': step.metadata if hasattr(step, 'metadata') else {}
                    }
                    for i, step in enumerate(result.steps, 1)
                ],
                'metadata': result.metadata,
                'citation_style': citation_style,
                'research_depth': research_depth,
            }

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]Research dossier saved to: {output}[/green]")

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
    step: str = typer.Option(..., "--step", help="Investigation step description"),
    step_number: int = typer.Option(..., "--step-number", help="Current step index (starts at 1)"),
    total_steps: int = typer.Option(..., "--total-steps", help="Estimated total investigation steps"),
    next_step_required: bool = typer.Option(..., "--next-step-required", help="Whether more steps are needed (true/false)"),
    findings: str = typer.Option(..., "--findings", help="What was discovered in this step"),
    model: str = typer.Option(
        "claude",
        "--model",
        help="AI model to use (claude, gemini, codex, cursor-agent)",
    ),
    continuation_id: Optional[str] = typer.Option(
        None,
        "--continuation-id",
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
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Creativity level (0.0-1.0)",
    ),
    thinking_mode: str = typer.Option(
        "medium",
        "--thinking-mode",
        help="Reasoning depth (minimal, low, medium, high, max)",
    ),
    use_assistant_model: bool = typer.Option(
        True,
        "--use-assistant-model",
        help="Enable expert validation (default: true)",
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

    ThinkDeep provides multi-step investigation with explicit hypothesis tracking,
    confidence progression, and state management across investigation steps.

    Example:
        # Start new investigation
        modelchorus thinkdeep --step "Investigate why API latency increased" --step-number 1 --total-steps 3 --next-step-required true --findings "Examining deployment logs" --confidence exploring

        # Continue investigation
        modelchorus thinkdeep --continuation-id "thread-123" --step "Check database query performance" --step-number 2 --total-steps 3 --next-step-required true --findings "Found N+1 query pattern" --confidence medium --hypothesis "N+1 queries causing slowdown"

        # Final step
        modelchorus thinkdeep --continuation-id "thread-123" --step "Verify fix resolves issue" --step-number 3 --total-steps 3 --next-step-required false --findings "Latency reduced to baseline" --confidence high --hypothesis "Confirmed: N+1 queries were root cause"
    """
    try:
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
            console.print(f"[cyan]Initializing model: {model}[/cyan]")

        try:
            provider_instance = get_provider_by_name(model)
            if verbose:
                console.print(f"[green]✓ {model} initialized[/green]")
        except Exception as e:
            console.print(f"[red]Failed to initialize {model}: {e}[/red]")
            raise typer.Exit(1)

        # Create conversation memory
        memory = ConversationMemory()

        # Create workflow config
        config = {
            'enable_expert_validation': use_assistant_model
        }

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=provider_instance,
            expert_provider=None,  # Expert will be handled by workflow if enabled
            conversation_memory=memory,
            config=config,
        )

        if verbose:
            console.print(f"[cyan]Workflow: {workflow}[/cyan]")

        # Parse files_checked if provided
        files_list = None
        if files_checked:
            files_list = [f.strip() for f in files_checked.split(',') if f.strip()]
            # Validate files exist
            for file_path in files_list:
                if not Path(file_path).exists():
                    console.print(f"[red]Error: File not found: {file_path}[/red]")
                    raise typer.Exit(1)

        # Display investigation info
        console.print(f"\n[bold cyan]{'Continuing' if continuation_id else 'Starting'} ThinkDeep Investigation[/bold cyan]")
        console.print(f"Step {step_number}/{total_steps}: {step}")
        console.print(f"Model: {model}")
        console.print(f"Confidence: {confidence}")
        if hypothesis:
            console.print(f"Hypothesis: {hypothesis}")
        if continuation_id:
            console.print(f"Thread ID: {continuation_id}")
        if files_list:
            console.print(f"Files: {', '.join(files_list)}")
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
                temperature=temperature,
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
                    "model": model,
                    "continuation_id": returned_continuation_id,
                    "response": result.synthesis,
                    "usage": usage,
                }
                if files_list:
                    output_data["files"] = files_list
                if use_assistant_model and len(result.steps) > 1:
                    output_data["expert_validation"] = result.steps[-1].content

                output.write_text(json.dumps(output_data, indent=2))
                console.print(f"\n[green]✓ Result saved to {output}[/green]")

            # Show next step guidance
            if next_step_required:
                console.print(f"\n[dim]To continue this investigation:[/dim]")
                console.print(f"[dim]  modelchorus thinkdeep --continuation-id {returned_continuation_id} --step \"Next investigation step\" --step-number {step_number + 1} --total-steps {total_steps} ...[/dim]")
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
        modelchorus thinkdeep-status thread-id-123

        # View with all steps
        modelchorus thinkdeep-status thread-id-123 --steps

        # View with files
        modelchorus thinkdeep-status thread-id-123 --files --verbose
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
