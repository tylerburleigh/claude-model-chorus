"""
Tests for provider middleware (retry, circuit breaker).

This module tests the middleware components that wrap model providers
to add resilience features like automatic retries and circuit breakers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_chorus.providers.base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
)
from model_chorus.providers.middleware import (
    CircuitBreakerConfig,
    CircuitBreakerMiddleware,
    CircuitOpenError,
    CircuitState,
    ConfigError,
    Middleware,
    ProviderError,
    RetryConfig,
    RetryExhaustedError,
    RetryMiddleware,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock ModelProvider for testing middleware."""
    provider = MagicMock(spec=ModelProvider)
    provider.provider_name = "test-provider"
    provider.__class__.__name__ = "MockProvider"
    return provider


@pytest.fixture
def sample_request():
    """Create a sample GenerationRequest for testing."""
    return GenerationRequest(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def sample_response():
    """Create a sample GenerationResponse for testing."""
    return GenerationResponse(
        content="Test response",
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 20},
        stop_reason="end_turn",
    )


# ============================================================================
# Error Classes Tests
# ============================================================================


class TestProviderError:
    """Test ProviderError base exception."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = ProviderError("Test error")
        assert str(error) == "Test error"
        assert error.provider_name is None
        assert error.original_error is None

    def test_full_initialization(self):
        """Test error with all attributes."""
        original = ValueError("Original error")
        error = ProviderError("Test error", provider_name="claude", original_error=original)
        assert str(error) == "Test error"
        assert error.provider_name == "claude"
        assert error.original_error is original


class TestConfigError:
    """Test ConfigError exception."""

    def test_basic_initialization(self):
        """Test config error initialization."""
        error = ConfigError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert error.config_field is None
        assert error.invalid_value is None

    def test_full_initialization(self):
        """Test config error with field and value."""
        error = ConfigError(
            "Invalid retry count",
            config_field="max_retries",
            invalid_value=-1,
        )
        assert str(error) == "Invalid retry count"
        assert error.config_field == "max_retries"
        assert error.invalid_value == -1


class TestRetryExhaustedError:
    """Test RetryExhaustedError exception."""

    def test_initialization(self):
        """Test retry exhausted error initialization."""
        last_error = RuntimeError("Final attempt failed")
        error = RetryExhaustedError(
            "All retries failed",
            attempts=4,
            last_error=last_error,
            provider_name="claude",
        )
        assert str(error) == "All retries failed"
        assert error.attempts == 4
        assert error.last_error is last_error
        assert error.provider_name == "claude"
        assert error.original_error is last_error


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)
        assert len(config.permanent_error_patterns) > 0

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_multiplier=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False

    def test_permanent_error_patterns(self):
        """Test default permanent error patterns."""
        config = RetryConfig()
        patterns = config.permanent_error_patterns

        # Check for common permanent error patterns
        assert "invalid api key" in patterns
        assert "unauthorized" in patterns
        assert "authentication failed" in patterns
        assert "400" in patterns  # Bad request
        assert "401" in patterns  # Unauthorized
        assert "403" in patterns  # Forbidden
        assert "404" in patterns  # Not found


# ============================================================================
# RetryMiddleware Tests
# ============================================================================


class TestRetryMiddleware:
    """Test RetryMiddleware implementation."""

    def test_initialization_default_config(self, mock_provider):
        """Test retry middleware initialization with default config."""
        middleware = RetryMiddleware(mock_provider)
        assert middleware.provider is mock_provider
        assert middleware.config.max_retries == 3
        assert middleware.config.base_delay == 1.0

    def test_initialization_custom_config(self, mock_provider):
        """Test retry middleware initialization with custom config."""
        config = RetryConfig(max_retries=5, base_delay=2.0)
        middleware = RetryMiddleware(mock_provider, config)
        assert middleware.provider is mock_provider
        assert middleware.config.max_retries == 5
        assert middleware.config.base_delay == 2.0

    def test_is_retryable_error_transient(self, mock_provider):
        """Test detection of retryable (transient) errors."""
        middleware = RetryMiddleware(mock_provider)

        # Generic exceptions should be retryable
        assert middleware._is_retryable_error(RuntimeError("Network timeout")) is True
        assert middleware._is_retryable_error(ConnectionError("Connection lost")) is True

    def test_is_retryable_error_permanent(self, mock_provider):
        """Test detection of permanent (non-retryable) errors."""
        middleware = RetryMiddleware(mock_provider)

        # Errors matching permanent patterns should not be retryable
        assert middleware._is_retryable_error(ValueError("Invalid API key")) is False
        assert middleware._is_retryable_error(RuntimeError("401 Unauthorized")) is False
        assert middleware._is_retryable_error(Exception("Authentication failed")) is False
        assert middleware._is_retryable_error(Exception("400 Bad Request")) is False

    def test_is_retryable_error_non_exception_type(self, mock_provider):
        """Test non-retryable exception types."""
        config = RetryConfig(retryable_exceptions=(RuntimeError,))
        middleware = RetryMiddleware(mock_provider, config)

        # RuntimeError is retryable
        assert middleware._is_retryable_error(RuntimeError("Network error")) is True

        # ValueError is not in retryable list
        assert middleware._is_retryable_error(ValueError("Parse error")) is False

    def test_calculate_delay_exponential_backoff(self, mock_provider):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=60.0,
            jitter=False,  # Disable jitter for predictable testing
        )
        middleware = RetryMiddleware(mock_provider, config)

        # Delays should increase exponentially: 1, 2, 4, 8, ...
        assert middleware._calculate_delay(0) == 1.0
        assert middleware._calculate_delay(1) == 2.0
        assert middleware._calculate_delay(2) == 4.0
        assert middleware._calculate_delay(3) == 8.0

    def test_calculate_delay_max_cap(self, mock_provider):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=5.0,
            jitter=False,
        )
        middleware = RetryMiddleware(mock_provider, config)

        # Delay should be capped at max_delay
        assert middleware._calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self, mock_provider):
        """Test delay calculation with jitter enabled."""
        config = RetryConfig(base_delay=10.0, jitter=True)
        middleware = RetryMiddleware(mock_provider, config)

        delay = middleware._calculate_delay(0)
        # With 25% jitter, delay should be in range [7.5, 12.5]
        assert 7.5 <= delay <= 12.5
        # Should always be positive
        assert delay > 0

    @pytest.mark.asyncio
    async def test_generate_success_first_attempt(
        self, mock_provider, sample_request, sample_response
    ):
        """Test successful generation on first attempt."""
        mock_provider.generate = AsyncMock(return_value=sample_response)
        middleware = RetryMiddleware(mock_provider)

        response = await middleware.generate(sample_request)

        assert response is sample_response
        assert mock_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_success_after_retries(
        self, mock_provider, sample_request, sample_response
    ):
        """Test successful generation after transient failures."""
        # Fail twice, succeed on third attempt
        mock_provider.generate = AsyncMock(
            side_effect=[
                RuntimeError("Network timeout"),
                RuntimeError("Connection reset"),
                sample_response,
            ]
        )

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        middleware = RetryMiddleware(mock_provider, config)

        response = await middleware.generate(sample_request)

        assert response is sample_response
        assert mock_provider.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_permanent_error_no_retry(
        self, mock_provider, sample_request
    ):
        """Test that permanent errors fail immediately without retry."""
        mock_provider.generate = AsyncMock(
            side_effect=ValueError("Invalid API key")
        )

        middleware = RetryMiddleware(mock_provider)

        with pytest.raises(ValueError, match="Invalid API key"):
            await middleware.generate(sample_request)

        # Should fail immediately (1 attempt, no retries)
        assert mock_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_all_retries_exhausted(
        self, mock_provider, sample_request
    ):
        """Test RetryExhaustedError when all retries fail."""
        error = RuntimeError("Persistent network error")
        mock_provider.generate = AsyncMock(side_effect=error)

        config = RetryConfig(max_retries=2, base_delay=0.01)
        middleware = RetryMiddleware(mock_provider, config)

        with pytest.raises(RetryExhaustedError) as exc_info:
            await middleware.generate(sample_request)

        assert exc_info.value.attempts == 3  # initial + 2 retries
        assert exc_info.value.last_error is error
        assert exc_info.value.provider_name == "MockProvider"
        assert mock_provider.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_respects_backoff_delay(
        self, mock_provider, sample_request, sample_response
    ):
        """Test that retry middleware waits between attempts."""
        call_times = []

        async def track_time(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise RuntimeError("Transient error")
            return sample_response

        mock_provider.generate = AsyncMock(side_effect=track_time)

        config = RetryConfig(max_retries=3, base_delay=0.05, jitter=False)
        middleware = RetryMiddleware(mock_provider, config)

        await middleware.generate(sample_request)

        # Should have 3 calls
        assert len(call_times) == 3

        # Check delays between calls (should be ~0.05s, ~0.1s)
        # Allow some tolerance for timing
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.04 <= delay1 <= 0.08  # ~0.05s delay
        assert 0.08 <= delay2 <= 0.15  # ~0.1s delay (exponential)


# ============================================================================
# CircuitBreakerConfig Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.recovery_timeout == 60.0
        assert config.monitored_exceptions == (Exception,)
        assert config.excluded_exceptions == ()

    def test_custom_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            recovery_timeout=30.0,
            monitored_exceptions=(RuntimeError,),
            excluded_exceptions=(ValueError,),
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.recovery_timeout == 30.0
        assert config.monitored_exceptions == (RuntimeError,)
        assert config.excluded_exceptions == (ValueError,)


# ============================================================================
# CircuitBreakerMiddleware Tests
# ============================================================================


class TestCircuitBreakerMiddleware:
    """Test CircuitBreakerMiddleware implementation."""

    def test_initialization_default_config(self, mock_provider):
        """Test circuit breaker initialization with default config."""
        middleware = CircuitBreakerMiddleware(mock_provider)
        assert middleware.provider is mock_provider
        assert middleware.state == CircuitState.CLOSED
        assert middleware.failure_count == 0
        assert middleware.success_count == 0
        assert middleware.config.failure_threshold == 5

    def test_initialization_custom_config(self, mock_provider):
        """Test circuit breaker initialization with custom config."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        middleware = CircuitBreakerMiddleware(mock_provider, config)
        assert middleware.config.failure_threshold == 3
        assert middleware.config.recovery_timeout == 30.0

    def test_initialization_with_callback(self, mock_provider):
        """Test circuit breaker initialization with state change callback."""
        callback = MagicMock()
        middleware = CircuitBreakerMiddleware(mock_provider, on_state_change=callback)
        assert middleware.on_state_change is callback

    @pytest.mark.asyncio
    async def test_generate_success_in_closed_state(
        self, mock_provider, sample_request, sample_response
    ):
        """Test successful generation in CLOSED state."""
        mock_provider.generate = AsyncMock(return_value=sample_response)
        middleware = CircuitBreakerMiddleware(mock_provider)

        response = await middleware.generate(sample_request)

        assert response is sample_response
        assert middleware.state == CircuitState.CLOSED
        assert middleware.failure_count == 0

    @pytest.mark.asyncio
    async def test_failure_count_increases_on_error(
        self, mock_provider, sample_request
    ):
        """Test that failure count increases on monitored errors."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(failure_threshold=3)
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # First failure
        with pytest.raises(RuntimeError):
            await middleware.generate(sample_request)
        assert middleware.failure_count == 1
        assert middleware.state == CircuitState.CLOSED

        # Second failure
        with pytest.raises(RuntimeError):
            await middleware.generate(sample_request)
        assert middleware.failure_count == 2
        assert middleware.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self, mock_provider, sample_request
    ):
        """Test circuit opens after failure threshold reached."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(failure_threshold=3)
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Generate failures to reach threshold
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        # Circuit should now be OPEN
        assert middleware.state == CircuitState.OPEN
        assert middleware.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_open_rejects_requests(
        self, mock_provider, sample_request
    ):
        """Test that OPEN circuit rejects requests with CircuitOpenError."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60.0)
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        assert middleware.state == CircuitState.OPEN

        # Next request should fail fast with CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            await middleware.generate(sample_request)

        assert exc_info.value.provider_name == "MockProvider"
        assert exc_info.value.recovery_time > 0

        # Provider should not be called (fail fast)
        assert mock_provider.generate.call_count == 2  # Only the initial failures

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(
        self, mock_provider, sample_request, sample_response
    ):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.05,  # 50ms timeout for faster test
        )
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        assert middleware.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.06)

        # Next request should transition to HALF_OPEN
        mock_provider.generate = AsyncMock(return_value=sample_response)
        response = await middleware.generate(sample_request)

        assert response is sample_response
        assert middleware.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_after_success_threshold(
        self, mock_provider, sample_request, sample_response
    ):
        """Test circuit closes after success threshold in HALF_OPEN."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            recovery_timeout=0.05,
        )
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        # Wait for recovery timeout
        await asyncio.sleep(0.06)

        # Make successful requests to close circuit
        mock_provider.generate = AsyncMock(return_value=sample_response)

        # First success in HALF_OPEN
        await middleware.generate(sample_request)
        assert middleware.state == CircuitState.HALF_OPEN
        assert middleware.success_count == 1

        # Second success should close circuit
        await middleware.generate(sample_request)
        assert middleware.state == CircuitState.CLOSED
        assert middleware.success_count == 0  # Reset on transition
        assert middleware.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(
        self, mock_provider, sample_request, sample_response
    ):
        """Test circuit reopens immediately on failure in HALF_OPEN."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.05,
        )
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        # Wait for recovery timeout and transition to HALF_OPEN
        await asyncio.sleep(0.06)
        mock_provider.generate = AsyncMock(return_value=sample_response)
        await middleware.generate(sample_request)
        assert middleware.state == CircuitState.HALF_OPEN

        # Failure in HALF_OPEN should immediately reopen circuit
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error again"))
        with pytest.raises(RuntimeError):
            await middleware.generate(sample_request)

        assert middleware.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_counted(
        self, mock_provider, sample_request
    ):
        """Test that excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # ValueError should not increase failure count
        mock_provider.generate = AsyncMock(side_effect=ValueError("Validation error"))

        for _ in range(3):
            with pytest.raises(ValueError):
                await middleware.generate(sample_request)

        # Circuit should remain CLOSED
        assert middleware.state == CircuitState.CLOSED
        assert middleware.failure_count == 0

    @pytest.mark.asyncio
    async def test_state_change_callback_invoked(
        self, mock_provider, sample_request
    ):
        """Test that state change callback is invoked on transitions."""
        callback = MagicMock()
        config = CircuitBreakerConfig(failure_threshold=2)
        middleware = CircuitBreakerMiddleware(
            mock_provider,
            config,
            on_state_change=callback,
        )

        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        # Callback should be invoked once (CLOSED -> OPEN)
        callback.assert_called_once_with(CircuitState.CLOSED, CircuitState.OPEN)

    @pytest.mark.asyncio
    async def test_success_resets_failure_count_in_closed(
        self, mock_provider, sample_request, sample_response
    ):
        """Test that success resets failure count in CLOSED state."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))
        config = CircuitBreakerConfig(failure_threshold=5)
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Generate some failures
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await middleware.generate(sample_request)

        assert middleware.failure_count == 3
        assert middleware.state == CircuitState.CLOSED

        # Now succeed
        mock_provider.generate = AsyncMock(return_value=sample_response)
        await middleware.generate(sample_request)

        # Failure count should be reset
        assert middleware.failure_count == 0
        assert middleware.state == CircuitState.CLOSED

    def test_is_monitored_exception(self, mock_provider):
        """Test exception monitoring logic."""
        config = CircuitBreakerConfig(
            monitored_exceptions=(RuntimeError, ConnectionError),
            excluded_exceptions=(ValueError,),
        )
        middleware = CircuitBreakerMiddleware(mock_provider, config)

        # Monitored exceptions
        assert middleware._is_monitored_exception(RuntimeError("Error")) is True
        assert middleware._is_monitored_exception(ConnectionError("Error")) is True

        # Excluded exception (even if in monitored)
        assert middleware._is_monitored_exception(ValueError("Error")) is False

        # Not monitored
        assert middleware._is_monitored_exception(TypeError("Error")) is False


# ============================================================================
# Middleware Chaining Tests
# ============================================================================


class TestMiddlewareChaining:
    """Test chaining multiple middleware together."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(
        self, mock_provider, sample_request, sample_response
    ):
        """Test composing retry middleware with circuit breaker."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        # Chain: CircuitBreaker wraps Retry wraps Provider
        retry_middleware = RetryMiddleware(
            mock_provider,
            RetryConfig(max_retries=3, base_delay=0.01),
        )
        circuit_middleware = CircuitBreakerMiddleware(
            retry_middleware,
            CircuitBreakerConfig(failure_threshold=5),
        )

        response = await circuit_middleware.generate(sample_request)

        assert response is sample_response
        assert circuit_middleware.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_retry_exhausted_counts_as_circuit_failure(
        self, mock_provider, sample_request
    ):
        """Test that retry exhaustion counts as failure for circuit breaker."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Error"))

        retry_middleware = RetryMiddleware(
            mock_provider,
            RetryConfig(max_retries=2, base_delay=0.01),
        )
        circuit_middleware = CircuitBreakerMiddleware(
            retry_middleware,
            CircuitBreakerConfig(failure_threshold=2),
        )

        # First request: retry exhausted, circuit counts as 1 failure
        with pytest.raises(RetryExhaustedError):
            await circuit_middleware.generate(sample_request)

        assert circuit_middleware.failure_count == 1

        # Second request: retry exhausted again, circuit opens
        with pytest.raises(RetryExhaustedError):
            await circuit_middleware.generate(sample_request)

        assert circuit_middleware.state == CircuitState.OPEN
