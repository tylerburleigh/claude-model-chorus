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


# Structured Error Types


class ProviderError(Exception):
    """Base exception for provider-related errors.

    All provider middleware errors should inherit from this base class
    to enable consistent error handling across the middleware stack.

    Attributes:
        provider_name: Name of the provider where the error occurred
        original_error: The underlying exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize provider error.

        Args:
            message: Human-readable error description
            provider_name: Name of the provider (if available)
            original_error: The underlying exception (if any)
        """
        super().__init__(message)
        self.provider_name = provider_name
        self.original_error = original_error


class ConfigError(ProviderError):
    """Configuration-related errors.

    Raised when middleware or provider configuration is invalid,
    missing required fields, or contains conflicting settings.

    Examples:
        - Invalid retry configuration (negative values, etc.)
        - Missing required circuit breaker thresholds
        - Conflicting rate limit settings
    """

    def __init__(
        self,
        message: str,
        config_field: str | None = None,
        invalid_value: Any | None = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Human-readable error description
            config_field: Name of the problematic config field
            invalid_value: The invalid value that caused the error
        """
        super().__init__(message)
        self.config_field = config_field
        self.invalid_value = invalid_value


class RetryExhaustedError(ProviderError):
    """Raised when all retry attempts have been exhausted.

    Contains details about the retry attempts and the final error
    that prevented success.

    Attributes:
        attempts: Number of attempts made (including initial try)
        last_error: The exception from the final attempt
        provider_name: Name of the provider that failed
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception,
        provider_name: str | None = None,
    ):
        """
        Initialize retry exhausted error.

        Args:
            message: Human-readable error description
            attempts: Number of attempts made
            last_error: The exception from the final attempt
            provider_name: Name of the provider
        """
        super().__init__(message, provider_name=provider_name, original_error=last_error)
        self.attempts = attempts
        self.last_error = last_error


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
        error_msg = f"All {self.config.max_retries + 1} attempts failed"
        logger.error(f"{error_msg}. Last error: {last_exception}")
        raise RetryExhaustedError(
            message=error_msg,
            attempts=self.config.max_retries + 1,
            last_error=last_exception,
            provider_name=self.provider.__class__.__name__,
        ) from last_exception


# Circuit Breaker Implementation

from enum import Enum
from time import time
from typing import Callable


class CircuitState(Enum):
    """States of a circuit breaker.

    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ProviderError):
    """Raised when circuit breaker is open and request is rejected.

    Attributes:
        provider_name: Name of the provider with open circuit
        recovery_time: Seconds until circuit will try half-open state
    """

    def __init__(self, provider_name: str, recovery_time: float):
        """
        Initialize circuit open error.

        Args:
            provider_name: Name of the provider with open circuit
            recovery_time: Seconds until circuit will try half-open state
        """
        message = (
            f"Circuit breaker is OPEN for provider '{provider_name}'. "
            f"Retry in {recovery_time:.1f} seconds."
        )
        super().__init__(message, provider_name=provider_name)
        self.recovery_time = recovery_time


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit (default: 5)
        success_threshold: Number of consecutive successes in half-open to close circuit (default: 2)
        recovery_timeout: Seconds to wait before trying half-open state (default: 60.0)
        monitored_exceptions: Tuple of exception types that count as failures (default: (Exception,))
        excluded_exceptions: Tuple of exception types that don't count as failures
            (e.g., validation errors that shouldn't trigger circuit)

    Examples:
        # Default config: 5 failures, 60s timeout
        config = CircuitBreakerConfig()

        # Aggressive protection
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)

        # Conservative (allow more failures)
        config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=120.0)
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout: float = 60.0
    monitored_exceptions: tuple[type[Exception], ...] = (Exception,)
    excluded_exceptions: tuple[type[Exception], ...] = ()


class CircuitBreakerMiddleware(Middleware):
    """
    Middleware that implements circuit breaker pattern for fault tolerance.

    Protects against cascading failures by tracking provider health and
    "opening the circuit" when failures exceed threshold. Gives failing
    providers time to recover before allowing requests through again.

    State Machine:
        CLOSED (normal) --[failures >= threshold]--> OPEN (rejecting)
        OPEN --[timeout elapsed]--> HALF_OPEN (testing)
        HALF_OPEN --[success >= threshold]--> CLOSED
        HALF_OPEN --[any failure]--> OPEN

    Example:
        # Use default config (5 failures, 60s timeout)
        provider = CircuitBreakerMiddleware(ClaudeProvider())

        # Custom config
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        provider = CircuitBreakerMiddleware(ClaudeProvider(), config)

        # Compose with retry middleware (circuit breaker wraps retry)
        provider = CircuitBreakerMiddleware(
            RetryMiddleware(ClaudeProvider())
        )

        try:
            response = await provider.generate(request)
        except CircuitOpenError as e:
            print(f"Circuit open, retry in {e.recovery_time}s")
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        """
        Initialize circuit breaker middleware.

        Args:
            provider: The model provider to wrap
            config: Optional CircuitBreakerConfig for customizing behavior
            on_state_change: Optional callback for state changes (old_state, new_state)
        """
        super().__init__(provider)
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()  # Thread-safe state changes

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count."""
        return self._success_count

    async def _transition_to(self, new_state: CircuitState) -> None:
        """
        Transition to a new circuit state.

        Args:
            new_state: The state to transition to
        """
        old_state = self._state
        if old_state != new_state:
            self._state = new_state

            # Reset counters on state change
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
                self._last_failure_time = None
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
                # Keep failure count for logging

            logger.info(
                f"Circuit breaker state: {old_state.value} -> {new_state.value} "
                f"(provider: {self.provider.__class__.__name__})"
            )

            # Call state change callback if provided
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def _should_attempt_request(self) -> bool:
        """
        Check if request should be attempted based on circuit state.

        Returns:
            True if request should proceed, False if circuit is open
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.HALF_OPEN:
            return True

        # State is OPEN - check if recovery timeout has elapsed
        if self._last_failure_time is None:
            return True

        time_since_failure = time() - self._last_failure_time
        if time_since_failure >= self.config.recovery_timeout:
            # Timeout elapsed, allow one request through (will transition to HALF_OPEN)
            return True

        return False

    def _time_until_recovery(self) -> float:
        """
        Calculate seconds until circuit can try half-open state.

        Returns:
            Seconds until recovery (0 if ready now)
        """
        if self._last_failure_time is None:
            return 0.0

        time_since_failure = time() - self._last_failure_time
        remaining = self.config.recovery_timeout - time_since_failure
        return max(0.0, remaining)

    def _is_monitored_exception(self, error: Exception) -> bool:
        """
        Check if exception should be counted as failure.

        Args:
            error: The exception to check

        Returns:
            True if exception counts as failure, False otherwise
        """
        # Check if excluded
        if isinstance(error, self.config.excluded_exceptions):
            return False

        # Check if monitored
        return isinstance(error, self.config.monitored_exceptions)

    async def _record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker success in HALF_OPEN: "
                    f"{self._success_count}/{self.config.success_threshold}"
                )

                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                if self._failure_count > 0:
                    logger.debug("Circuit breaker: Resetting failure count after success")
                    self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """
        Record a failed request.

        Args:
            error: The exception that caused the failure
        """
        if not self._is_monitored_exception(error):
            logger.debug(f"Exception not monitored by circuit breaker: {type(error).__name__}")
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time()

            if self._state == CircuitState.CLOSED:
                logger.debug(
                    f"Circuit breaker failure: "
                    f"{self._failure_count}/{self.config.failure_threshold}"
                )

                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN immediately opens circuit
                logger.warning("Circuit breaker: Failure during HALF_OPEN, reopening circuit")
                await self._transition_to(CircuitState.OPEN)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text with circuit breaker protection.

        Fails fast with CircuitOpenError when circuit is open. Records
        successes and failures to manage circuit state transitions.

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with generated content

        Raises:
            CircuitOpenError: If circuit is open and not ready to retry
            Exception: If provider request fails
        """
        # Check if request should be attempted
        if not self._should_attempt_request():
            recovery_time = self._time_until_recovery()
            raise CircuitOpenError(
                self.provider.__class__.__name__,
                recovery_time
            )

        # If OPEN and timeout elapsed, transition to HALF_OPEN
        async with self._lock:
            if self._state == CircuitState.OPEN:
                time_since_failure = (
                    time() - self._last_failure_time
                    if self._last_failure_time
                    else float("inf")
                )
                if time_since_failure >= self.config.recovery_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)

        # Attempt the request
        try:
            response = await self.provider.generate(request)
            await self._record_success()
            return response

        except Exception as e:
            await self._record_failure(e)
            raise
