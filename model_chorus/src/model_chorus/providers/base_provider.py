"""
Base provider abstract class for ModelChorus.

This module defines the abstract base class that all model provider implementations
must inherit from, providing a consistent interface for different AI providers
(Anthropic, OpenAI, Google, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    max_tokens: int | None = None
    capabilities: list[ModelCapability] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    images: list[str] | None = None
    continuation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage information with explicit fields for type safety.

    This dataclass provides a standardized way to track token consumption
    across different AI providers, with support for caching and provider-
    specific metadata.

    Supports both attribute access (usage.input_tokens) and dict-like access
    (usage['input_tokens']) for backward compatibility.

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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Support dict-like read access: usage['input_tokens'].

        Args:
            key: Field name to access.

        Returns:
            Value of the requested field.

        Raises:
            KeyError: If key is not a valid TokenUsage field.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' is not a valid TokenUsage field")

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dict-like write access: usage['input_tokens'] = 100.

        Args:
            key: Field name to set.
            value: Value to assign to the field.

        Raises:
            KeyError: If key is not a valid TokenUsage field.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid TokenUsage field")

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-like get with default: usage.get('input_tokens', 0).

        Args:
            key: Field name to retrieve.
            default: Default value if field doesn't exist.

        Returns:
            Field value if it exists, otherwise default.
        """
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        """Return field names like dict.keys().

        Returns:
            List of field names in the TokenUsage dataclass.
        """
        return [
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "total_tokens",
            "metadata",
        ]

    def values(self) -> list[Any]:
        """Return field values like dict.values().

        Returns:
            List of current field values.
        """
        return [
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.total_tokens,
            self.metadata,
        ]

    def items(self) -> zip:
        """Return (key, value) pairs like dict.items().

        Returns:
            Zip iterator of (field_name, field_value) tuples.
        """
        return zip(self.keys(), self.values())


@dataclass
class GenerationResponse:
    """Response from text generation with standardized structure across providers.

    This dataclass provides a unified response format for all AI providers (Claude,
    Gemini, OpenAI Codex, etc.), supporting conversation continuation via thread_id,
    token usage tracking, and debugging capabilities.

    Attributes:
        content: The generated text content from the model.
        model: Model identifier that generated this response (e.g., "claude-3-opus",
            "gemini-pro", "gpt-4").
        usage: Token usage information as a TokenUsage dataclass. Supports both
            attribute access (usage.input_tokens) and dict-like access
            (usage['input_tokens']) for backward compatibility.
        stop_reason: Reason generation stopped (e.g., "end_turn", "max_tokens",
            "stop_sequence"). Provider-specific values, may be None.
        metadata: Provider-specific additional metadata (e.g., safety ratings,
            citations, model version details).
        thread_id: Conversation continuation identifier for multi-turn interactions.
            Provider-specific mapping:
            - Claude: Maps from CLI response 'session_id' field
            - Cursor: Maps from CLI response 'session_id' field
            - Codex (OpenAI): Maps from CLI response 'thread_id' field
            - Gemini: Always None (does not support conversation continuation)
            Used to maintain context across multiple generation requests.
        provider: Name of the provider that generated this response. Valid values:
            "claude", "gemini", "codex", "cursor". Useful for multi-provider
            workflows and debugging.
        stderr: Standard error output captured from CLI-based providers. Contains
            warning messages, debug output, or error details. Empty string if no
            errors, None if not captured. Only populated for CLI providers (Claude,
            Gemini, Codex).
        duration_ms: Request duration in milliseconds, measured from request start
            to response completion. Useful for performance monitoring, latency
            analysis, and cost optimization. None if not measured.
        raw_response: Complete raw response from the provider as returned by the
            CLI or API. Useful for debugging, testing provider-specific features,
            and understanding response structure. May contain sensitive data.
            None if not captured.

    Example:
        Basic usage::

            response = GenerationResponse(
                content="Hello, world!",
                model="claude-3-opus-20240229",
                provider="claude"
            )
            response.usage['input_tokens'] = 10
            response.usage['output_tokens'] = 5

        With conversation continuation::

            # First turn
            resp1 = GenerationResponse(
                content="Initial response",
                model="gpt-4",
                thread_id="thread_abc123",
                provider="codex"
            )

            # Follow-up turn using same thread_id
            resp2 = GenerationResponse(
                content="Follow-up response",
                model="gpt-4",
                thread_id="thread_abc123",  # Same ID for continuation
                provider="codex"
            )
    """

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    provider: str | None = None
    stderr: str | None = None
    duration_ms: int | None = None
    raw_response: dict[str, Any] | None = None


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
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
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
        self._available_models: list[ModelConfig] = []

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

    def get_available_models(self) -> list[ModelConfig]:
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

    def set_model_list(self, models: list[ModelConfig]) -> None:
        """
        Set the list of available models for this provider.

        Args:
            models: List of ModelConfig objects
        """
        self._available_models = models

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"
