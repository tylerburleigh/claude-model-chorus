"""Common prompt utilities for ModelChorus workflows.

This module provides reusable prompt generation and enhancement functions
for all ModelChorus workflows, particularly those that invoke external
CLI tools in read-only mode.
"""

from typing import Optional


def get_read_only_system_prompt() -> str:
    """
    Generate system prompt informing AI of read-only tool constraints.

    This prompt should be prepended to workflow-specific system prompts
    to set expectations about available tool capabilities when operating
    via external CLI tools.

    The constraints are enforced at the CLI provider level (no --yolo flag
    for Gemini, etc.), but this prompt ensures the AI understands the
    limitations upfront, preventing wasted API tokens on failed attempts.

    Returns:
        System prompt text describing read-only constraints
    """
    return """ENVIRONMENT CONSTRAINTS:
You are operating via an external CLI tool in READ-ONLY mode.

Available tools:
- read_file: Read file contents
- web_fetch: Fetch and process web pages
- glob: Search for files by pattern
- grep: Search file contents with regex

UNAVAILABLE - These will not work (and will be blocked):
- write_file / edit_file / modify_file: Cannot write or modify files
- run_command / shell / execute: Cannot run system commands
- Any operations that modify, delete, or write files/data

YOUR ROLE: Analysis, recommendations, insights, and planning only.
Do NOT attempt any write or modification operations - they will fail.
Focus on understanding, explaining, and suggesting improvements."""


def prepend_system_constraints(custom_prompt: Optional[str] = None) -> str:
    """
    Prepend read-only environment constraints to a custom system prompt.

    Combines the read-only constraints with any workflow-specific
    system prompt instructions. Constraints are prepended so they
    establish context before task-specific guidance.

    Args:
        custom_prompt: Optional workflow-specific system prompt.
                      If None, returns just the constraints.

    Returns:
        Combined system prompt with constraints prepended

    Example:
        >>> ideation_prompt = "You are a creative brainstorming expert..."
        >>> final_prompt = prepend_system_constraints(ideation_prompt)
        >>> # Returns constraints + ideation prompt
    """
    constraints = get_read_only_system_prompt()

    if custom_prompt:
        return f"{constraints}\n\n{custom_prompt}"

    return constraints
