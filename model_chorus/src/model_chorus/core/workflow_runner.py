"""
Workflow execution runner for ModelChorus.

This module provides the WorkflowRunner class that orchestrates workflow execution
with provider fallback, error handling, and telemetry hooks. It extracts and
enhances the fallback logic previously embedded in BaseWorkflow.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers import GenerationRequest, GenerationResponse, ModelProvider

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of workflow execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some providers failed but one succeeded


@dataclass
class ExecutionMetrics:
    """Metrics collected during workflow execution.

    Attributes:
        started_at: Timestamp when execution started
        completed_at: Timestamp when execution completed (None if still running)
        duration_ms: Execution duration in milliseconds (None if still running)
        provider_attempts: List of (provider_name, success, error) tuples for each attempt
        total_attempts: Total number of provider attempts made
        successful_provider: Name of the provider that succeeded (None if all failed)
        status: Overall execution status (SUCCESS, FAILURE, or PARTIAL)
        input_tokens: Total input tokens used (if available from response)
        output_tokens: Total output tokens generated (if available from response)
        custom_metrics: Dictionary for custom metrics added by telemetry handlers
    """

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: int | None = None
    provider_attempts: list[tuple[str, bool, str | None]] = field(default_factory=list)
    total_attempts: int = 0
    successful_provider: str | None = None
    status: ExecutionStatus | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def mark_complete(self) -> None:
        """Mark execution as complete and calculate duration."""
        self.completed_at = datetime.now()
        self.duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )

    def add_attempt(
        self, provider_name: str, success: bool, error: str | None = None
    ) -> None:
        """
        Record a provider attempt.

        Args:
            provider_name: Name of the provider attempted
            success: Whether the attempt succeeded
            error: Error message if attempt failed (None if successful)
        """
        self.provider_attempts.append((provider_name, success, error))
        self.total_attempts += 1
        if success:
            self.successful_provider = provider_name

    def set_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Set token usage metrics.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def add_custom_metric(self, key: str, value: Any) -> None:
        """
        Add a custom metric.

        Args:
            key: Metric name
            value: Metric value
        """
        self.custom_metrics[key] = value


class WorkflowRunner:
    """
    Orchestrates workflow execution with provider fallback and telemetry.

    The WorkflowRunner extracts provider fallback logic from BaseWorkflow
    and adds structured error handling, execution metrics, and extension hooks
    for telemetry/observability.

    Key Features:
    - Automatic provider fallback with configurable retry logic
    - Execution metrics collection (timing, attempts, success/failure)
    - Extension hooks for telemetry integration
    - Structured error handling with detailed context

    Example:
        >>> runner = WorkflowRunner()
        >>> response, metrics = await runner.execute(
        ...     request,
        ...     primary_provider,
        ...     fallback_providers=[backup1, backup2]
        ... )
        >>> print(f"Used {metrics.successful_provider} after {metrics.total_attempts} attempts")
        >>> print(f"Duration: {metrics.duration_ms}ms")
    """

    def __init__(self, telemetry_enabled: bool = False):
        """
        Initialize workflow runner.

        Args:
            telemetry_enabled: Whether to enable telemetry hooks (default: False)
        """
        self.telemetry_enabled = telemetry_enabled
        self._telemetry_callbacks: list[
            Callable[[ExecutionMetrics, Any], None]
        ] = []

    def register_telemetry_callback(
        self, callback: Callable[[ExecutionMetrics, Any], None]
    ) -> None:
        """
        Register a telemetry callback to be invoked on execution events.

        Callbacks receive (metrics, context) where context is either:
        - GenerationResponse for successful executions
        - Exception for failed executions

        Args:
            callback: Callable that takes (ExecutionMetrics, context)

        Example:
            >>> def my_telemetry(metrics, context):
            ...     print(f"Duration: {metrics.duration_ms}ms")
            ...     if isinstance(context, GenerationResponse):
            ...         print(f"Success with {context.provider}")
            >>> runner = WorkflowRunner(telemetry_enabled=True)
            >>> runner.register_telemetry_callback(my_telemetry)
        """
        self._telemetry_callbacks.append(callback)

    def unregister_telemetry_callback(
        self, callback: Callable[[ExecutionMetrics, Any], None]
    ) -> bool:
        """
        Unregister a previously registered telemetry callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        try:
            self._telemetry_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    async def execute(
        self,
        request: "GenerationRequest",
        primary_provider: "ModelProvider",
        fallback_providers: list["ModelProvider"] | None = None,
    ) -> tuple["GenerationResponse", ExecutionMetrics]:
        """
        Execute generation request with automatic fallback.

        Attempts to generate using the primary provider. If that fails, automatically
        tries each fallback provider in order until one succeeds. Collects detailed
        execution metrics throughout the process.

        Args:
            request: Generation request to execute
            primary_provider: Primary provider to try first
            fallback_providers: Optional list of fallback providers to try if primary fails

        Returns:
            Tuple of (response, execution_metrics)

        Raises:
            Exception: If all providers fail, raises exception with details about all attempts

        Example:
            >>> runner = WorkflowRunner()
            >>> response, metrics = await runner.execute(
            ...     request, primary_provider, [fallback1, fallback2]
            ... )
            >>> if metrics.total_attempts > 1:
            ...     logger.warning(
            ...         f"Primary failed, succeeded with {metrics.successful_provider}"
            ...     )
        """
        from ..providers.cli_provider import ProviderUnavailableError

        metrics = ExecutionMetrics()
        fallback_providers = fallback_providers or []
        all_providers = [primary_provider] + fallback_providers
        last_exception: Exception | None = None

        for i, provider in enumerate(all_providers):
            try:
                logger.info(
                    f"Attempting provider {provider.provider_name} "
                    f"({i+1}/{len(all_providers)})"
                )

                # Execute generation
                response = await provider.generate(request)

                # Record successful attempt
                metrics.add_attempt(provider.provider_name, success=True)

                # Set execution status based on whether fallback was used
                if i > 0:
                    metrics.status = ExecutionStatus.PARTIAL
                else:
                    metrics.status = ExecutionStatus.SUCCESS

                # Extract token usage from response if available
                if hasattr(response, "usage") and response.usage:
                    metrics.set_token_usage(
                        input_tokens=getattr(response.usage, "input_tokens", 0),
                        output_tokens=getattr(response.usage, "output_tokens", 0),
                    )

                metrics.mark_complete()

                # Log fallback usage
                if i > 0:
                    logger.warning(
                        f"Primary provider failed, succeeded with fallback: "
                        f"{provider.provider_name}"
                    )

                # Call telemetry hooks if enabled
                if self.telemetry_enabled:
                    self._on_execution_complete(metrics, response)

                return response, metrics

            except ProviderUnavailableError as e:
                # Permanent error - provider CLI not available
                error_msg = f"Provider unavailable: {e.reason}"
                metrics.add_attempt(provider.provider_name, success=False, error=error_msg)
                logger.error(f"{provider.provider_name} unavailable: {e.reason}")
                last_exception = e

            except Exception as e:
                # Other error - could be transient or permanent
                error_msg = str(e)[:200]  # Truncate long errors
                metrics.add_attempt(provider.provider_name, success=False, error=error_msg)
                logger.warning(f"{provider.provider_name} failed: {str(e)[:100]}")
                last_exception = e

        # All providers failed
        metrics.status = ExecutionStatus.FAILURE
        metrics.mark_complete()

        # Call telemetry hooks for failure
        if self.telemetry_enabled:
            self._on_execution_failed(metrics, last_exception)

        # Construct detailed error message
        error_details = "\n".join(
            f"  - {name}: {'Success' if success else error or 'Unknown error'}"
            for name, success, error in metrics.provider_attempts
        )
        error_msg = (
            f"All {len(all_providers)} providers failed after {metrics.duration_ms}ms.\n"
            f"Attempts:\n{error_details}\n"
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise Exception(error_msg) from last_exception

    def _on_execution_complete(
        self, metrics: ExecutionMetrics, response: "GenerationResponse"
    ) -> None:
        """
        Hook called when execution completes successfully.

        Invokes all registered telemetry callbacks with the metrics and response.
        Subclasses can override this to add custom telemetry behavior.

        Args:
            metrics: Execution metrics collected
            response: The successful generation response
        """
        logger.debug(
            f"Execution complete: {metrics.successful_provider} "
            f"in {metrics.duration_ms}ms after {metrics.total_attempts} attempts "
            f"(tokens: {metrics.input_tokens} in / {metrics.output_tokens} out)"
        )

        # Invoke all registered telemetry callbacks
        for callback in self._telemetry_callbacks:
            try:
                callback(metrics, response)
            except Exception as e:
                logger.error(
                    f"Telemetry callback {callback.__name__} failed: {e}",
                    exc_info=True,
                )

    def _on_execution_failed(
        self, metrics: ExecutionMetrics, exception: Exception | None
    ) -> None:
        """
        Hook called when all providers fail.

        Invokes all registered telemetry callbacks with the metrics and exception.
        Subclasses can override this to add custom error handling or alerting.

        Args:
            metrics: Execution metrics collected
            exception: The final exception that caused failure
        """
        logger.debug(
            f"Execution failed: all {metrics.total_attempts} attempts failed "
            f"in {metrics.duration_ms}ms"
        )

        # Invoke all registered telemetry callbacks
        for callback in self._telemetry_callbacks:
            try:
                callback(metrics, exception)
            except Exception as e:
                logger.error(
                    f"Telemetry callback {callback.__name__} failed: {e}",
                    exc_info=True,
                )
