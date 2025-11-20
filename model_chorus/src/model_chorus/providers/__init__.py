"""
Provider integrations for ModelChorus.

This module contains abstractions and implementations for different
AI model providers via CLI tools.
"""

from .base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    ModelConfig,
    ModelProvider,
)
from .claude_provider import ClaudeProvider
from .cli_provider import CLIProvider
from .codex_provider import CodexProvider
from .cursor_agent_provider import CursorAgentProvider
from .gemini_provider import GeminiProvider

__all__ = [
    "ModelProvider",
    "ModelConfig",
    "ModelCapability",
    "GenerationRequest",
    "GenerationResponse",
    "CLIProvider",
    "ClaudeProvider",
    "CodexProvider",
    "GeminiProvider",
    "CursorAgentProvider",
]
