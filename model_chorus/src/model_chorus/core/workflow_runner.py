"""
Workflow execution runner for ModelChorus.

This module provides the WorkflowRunner class that orchestrates workflow execution
with provider fallback, error handling, and telemetry hooks. It extracts and
enhances the fallback logic previously embedded in BaseWorkflow.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers import GenerationRequest, GenerationResponse, ModelProvider

logger = logging.getLogger(__name__)


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
    """

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: int | None = None
    provider_attempts: list[tuple[str, bool, str | None]] = field(default_factory=list)
    total_attempts: int = 0
    successful_provider: str | None = None

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

        Subclasses or telemetry integrations can override this to collect
        additional metrics, send to monitoring systems, etc.

        Args:
            metrics: Execution metrics collected
            response: The successful generation response
        """
        logger.debug(
            f"Execution complete: {metrics.successful_provider} "
            f"in {metrics.duration_ms}ms after {metrics.total_attempts} attempts"
        )

    def _on_execution_failed(
        self, metrics: ExecutionMetrics, exception: Exception | None
    ) -> None:
        """
        Hook called when all providers fail.

        Subclasses or telemetry integrations can override this to log
        failures, send alerts, update dashboards, etc.

        Args:
            metrics: Execution metrics collected
            exception: The final exception that caused failure
        """
        logger.debug(
            f"Execution failed: all {metrics.total_attempts} attempts failed "
            f"in {metrics.duration_ms}ms"
        )
