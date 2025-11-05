"""
Provider integrations for ModelChorus.

This module contains abstractions and implementations for different
AI model providers via CLI tools.
"""

from .base_provider import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    GenerationRequest,
    GenerationResponse,
)
from .cli_provider import CLIProvider

__all__ = [
    "ModelProvider",
    "ModelConfig",
    "ModelCapability",
    "GenerationRequest",
    "GenerationResponse",
    "CLIProvider",
]
