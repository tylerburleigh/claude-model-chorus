"""
Provider middleware for ModelChorus.

This module provides a middleware pattern for wrapping model providers with
additional functionality like retries, rate limiting, and observability.

Middleware can be chained together to compose multiple behaviors:
    provider = RateLimitMiddleware(
        RetryMiddleware(
            ClaudeProvider()
        )
    )
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .base_provider import GenerationRequest, GenerationResponse, ModelProvider

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """
    Abstract base class for provider middleware.

    Middleware wraps a ModelProvider to add cross-cutting concerns like
    retries, rate limiting, logging, or circuit breakers. Middleware can
    be chained together to compose multiple behaviors.

    Example:
        provider = RetryMiddleware(
            LoggingMiddleware(
                ClaudeProvider()
            )
        )
    """

    def __init__(self, provider: ModelProvider):
        """
        Initialize middleware with a provider.

        Args:
            provider: The model provider to wrap (can be another middleware)
        """
        self.provider = provider

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using the wrapped provider.

        Subclasses must implement this to add their middleware behavior
        before/after calling the wrapped provider.

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with generated content

        Raises:
            Exception: Provider-specific or middleware-specific errors
        """
        raise NotImplementedError("Subclasses must implement generate()")


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
            - delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
        jitter: Whether to add random jitter to delays (default: True)
            - Helps prevent thundering herd when multiple requests retry simultaneously
        retryable_exceptions: Tuple of exception types to retry (default: (Exception,))
        permanent_error_patterns: List of error message patterns that indicate
            permanent failures that should not be retried (default: common auth/validation errors)

    Examples:
        # Default config: 3 retries with exponential backoff
        config = RetryConfig()

        # Aggressive retries for flaky network
        config = RetryConfig(max_retries=5, base_delay=0.5)

        # Conservative with longer delays
        config = RetryConfig(max_retries=2, base_delay=2.0, max_delay=120.0)
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    permanent_error_patterns: list[str] = field(
        default_factory=lambda: [
            "command not found",
            "permission denied",
            "not found in path",
            "invalid api key",
            "unauthorized",
            "authentication failed",
            "400",  # Bad request
            "401",  # Unauthorized
            "403",  # Forbidden
            "404",  # Not found
            "invalid request",
            "validation error",
        ]
    )


class RetryMiddleware(Middleware):
    """
    Middleware that adds retry logic with exponential backoff.

    Automatically retries failed generation requests with configurable
    exponential backoff. Distinguishes between transient errors (retryable)
    and permanent errors (fail immediately).

    Example:
        # Use default retry config (3 retries, exponential backoff)
        provider = RetryMiddleware(ClaudeProvider())

        # Custom retry config
        config = RetryConfig(max_retries=5, base_delay=2.0)
        provider = RetryMiddleware(ClaudeProvider(), config)

        # Use like a regular provider
        response = await provider.generate(request)
    """

    def __init__(
        self, provider: ModelProvider, config: RetryConfig | None = None
    ):
        """
        Initialize retry middleware.

        Args:
            provider: The model provider to wrap
            config: Optional RetryConfig for customizing retry behavior
        """
        super().__init__(provider)
        self.config = config or RetryConfig()

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable or permanent.

        Args:
            error: The exception to check

        Returns:
            True if the error is transient and should be retried, False for permanent errors
        """
        # Check if error type is in the retryable list
        if not isinstance(error, self.config.retryable_exceptions):
            return False

        # Check for permanent error patterns in error message
        error_str = str(error).lower()
        for pattern in self.config.permanent_error_patterns:
            if pattern.lower() in error_str:
                logger.debug(
                    f"Error matches permanent pattern '{pattern}': {error}"
                )
                return False

        # If we get here, the error is retryable
        return True

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff: base_delay * (multiplier ^ attempt)
        delay = self.config.base_delay * (
            self.config.backoff_multiplier**attempt
        )

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled (±25% randomness)
        if self.config.jitter:
            import random

            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure positive delay

        return delay

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text with automatic retry on transient failures.

        Retries failed requests using exponential backoff. Permanent errors
        (auth failures, validation errors) fail immediately without retry.

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with generated content

        Raises:
            Exception: If all retry attempts fail or if a permanent error occurs
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} for provider "
                    f"{self.provider.__class__.__name__}"
                )

                # Try the actual generation
                response = await self.provider.generate(request)

                if attempt > 0:
                    logger.info(
                        f"Generation successful on retry attempt {attempt + 1}"
                    )

                return response

            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(
                        f"Permanent error encountered, not retrying: {e}"
                    )
                    raise

                # Log the failure
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {e}"
                )

                # Don't wait after the last attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        error_msg = f"All {self.config.max_retries + 1} attempts failed. Last error: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg) from last_exception
