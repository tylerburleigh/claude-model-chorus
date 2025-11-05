"""
Provider integrations for ModelChorus.

This module contains abstractions and implementations for different
AI model providers (Anthropic, OpenAI, etc.).
"""

from .anthropic_provider import AnthropicProvider
from .base_provider import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    GenerationRequest,
    GenerationResponse,
)

__all__ = [
    "AnthropicProvider",
    "ModelProvider",
    "ModelConfig",
    "ModelCapability",
    "GenerationRequest",
    "GenerationResponse",
]
