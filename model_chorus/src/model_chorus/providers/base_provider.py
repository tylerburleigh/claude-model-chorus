"""
Base provider abstract class for ModelChorus.

This module defines the abstract base class that all model provider implementations
must inherit from, providing a consistent interface for different AI providers
(Anthropic, OpenAI, Google, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ModelCapability(Enum):
    """Enumeration of model capabilities."""

    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    THINKING = "thinking"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    capabilities: List[ModelCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    images: Optional[List[str]] = None
    continuation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage information with explicit fields for type safety.

    This dataclass provides a standardized way to track token consumption
    across different AI providers, with support for caching and provider-
    specific metadata.

    Attributes:
        input_tokens: Number of tokens in the input prompt/context.
        output_tokens: Number of tokens in the generated response.
        cached_input_tokens: Number of input tokens retrieved from cache
            (provider-dependent, e.g., OpenAI prompt caching).
        total_tokens: Total token count, typically input + output tokens.
            May or may not include cached tokens depending on provider.
        metadata: Provider-specific additional usage information (e.g.,
            cost, rate limits, model-specific metrics).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """
    Abstract base class for all model providers.

    All provider implementations (Anthropic, OpenAI, Google, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        provider_name: Name of the provider (e.g., "anthropic", "openai")
        api_key: API key for authentication
        config: Provider-specific configuration
    """

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model provider.

        Args:
            provider_name: Name of the provider
            api_key: API key for authentication (can be None if using env vars)
            config: Optional provider-specific configuration
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.config = config or {}
        self._available_models: List[ModelConfig] = []

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text based on the request.

        This method must be implemented by all subclasses to handle text generation
        using the provider's API.

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with the generated content

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @abstractmethod
    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific model supports vision capabilities.

        Args:
            model_id: The model identifier to check

        Returns:
            True if the model supports vision, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement supports_vision()")

    def get_available_models(self) -> List[ModelConfig]:
        """
        Get list of available models for this provider.

        Returns:
            List of ModelConfig objects describing available models
        """
        return self._available_models

    def supports_capability(self, model_id: str, capability: ModelCapability) -> bool:
        """
        Check if a model supports a specific capability.

        Args:
            model_id: The model identifier to check
            capability: The capability to check for

        Returns:
            True if the model supports the capability, False otherwise
        """
        for model in self._available_models:
            if model.model_id == model_id:
                return capability in model.capabilities
        return False

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is set and potentially valid.

        Subclasses can override this for more sophisticated validation.

        Returns:
            True if API key appears valid, False otherwise
        """
        return self.api_key is not None and len(self.api_key) > 0

    def set_model_list(self, models: List[ModelConfig]) -> None:
        """
        Set the list of available models for this provider.

        Args:
            models: List of ModelConfig objects
        """
        self._available_models = models

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"
