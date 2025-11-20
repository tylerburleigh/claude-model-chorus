"""
Progress reporting utilities for ModelChorus workflows.

Provides simple stderr-based progress reporting to enable real-time
feedback during multi-step workflow execution.
"""

import sys

from rich.console import Console

# Create a console that outputs to stderr for progress updates
_progress_console = Console(file=sys.stderr, highlight=False)

# Global flag to control progress output
_progress_enabled = True


def set_progress_enabled(enabled: bool) -> None:
    """
    Enable or disable progress output.

    Args:
        enabled: True to enable progress output, False to disable
    """
    global _progress_enabled
    _progress_enabled = enabled


def is_progress_enabled() -> bool:
    """Check if progress output is enabled."""
    return _progress_enabled


def emit_progress(message: str, prefix: str | None = None, style: str = "cyan") -> None:
    """
    Emit a progress update to stderr.

    Progress messages are sent to stderr so they don't interfere with
    stdout output (which may be parsed as JSON or redirected).

    Args:
        message: Progress message to display
        prefix: Optional prefix (e.g., role name, stage name)
        style: Rich style for the message (default: cyan)

    Example:
        >>> emit_progress("Analyzing argument...", prefix="Creator")
        [Creator] Analyzing argument...

        >>> emit_progress("Generating response...")
        Generating response...
    """
    if not _progress_enabled:
        return

    if prefix:
        formatted = f"[bold {style}][{prefix}][/bold {style}] {message}"
    else:
        formatted = f"[{style}]{message}[/{style}]"

    _progress_console.print(formatted)


def emit_stage(stage: str) -> None:
    """
    Emit a workflow stage indicator.

    Args:
        stage: Stage name (e.g., "Creator", "Skeptic", "Moderator")

    Example:
        >>> emit_stage("Creator")
        [Creator] Generating initial perspective...
    """
    emit_progress("Starting...", prefix=stage, style="bold cyan")


def emit_provider_start(provider: str) -> None:
    """
    Emit progress for provider execution start.

    Args:
        provider: Provider name

    Example:
        >>> emit_provider_start("claude")
        [claude] Executing...
    """
    emit_progress("Executing...", prefix=provider, style="yellow")


def emit_provider_complete(provider: str, duration: float | None = None) -> None:
    """
    Emit progress for provider execution completion.

    Args:
        provider: Provider name
        duration: Optional execution duration in seconds

    Example:
        >>> emit_provider_complete("claude", 2.3)
        [claude] Complete (2.3s)
    """
    if duration is not None:
        message = f"Complete ({duration:.1f}s)"
    else:
        message = "Complete"

    emit_progress(message, prefix=provider, style="green")


def emit_workflow_start(workflow: str, estimated_duration: str | None = None) -> None:
    """
    Emit workflow start with optional time estimate.

    Args:
        workflow: Workflow name
        estimated_duration: Optional human-readable duration estimate

    Example:
        >>> emit_workflow_start("argument", "15-30s")
        Starting argument workflow (estimated: 15-30s)...
    """
    if estimated_duration:
        message = f"Starting {workflow} workflow (estimated: {estimated_duration})..."
    else:
        message = f"Starting {workflow} workflow..."

    emit_progress(message, style="bold magenta")


def emit_workflow_complete(workflow: str) -> None:
    """
    Emit workflow completion.

    Args:
        workflow: Workflow name

    Example:
        >>> emit_workflow_complete("argument")
        ✓ argument workflow complete
    """
    emit_progress(f"✓ {workflow} workflow complete", style="bold green")
