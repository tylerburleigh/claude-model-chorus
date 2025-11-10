"""
Base workflow abstract class for ModelChorus.

This module defines the abstract base class that all workflow implementations
must inherit from, providing a consistent interface for multi-model orchestration.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from .conversation import ConversationMemory
from .models import ConversationThread, ConversationMessage

if TYPE_CHECKING:
    from ..providers import ModelProvider, GenerationRequest, GenerationResponse
    from ..providers.cli_provider import ProviderUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow execution."""

    step_number: int
    content: str
    model: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    success: bool
    steps: List[WorkflowStep] = field(default_factory=list)
    synthesis: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step_number: int, content: str, model: Optional[str] = None,
                 **metadata) -> None:
        """Add a step to the workflow result."""
        step = WorkflowStep(
            step_number=step_number,
            content=content,
            model=model,
            metadata=metadata
        )
        self.steps.append(step)


class BaseWorkflow(ABC):
    """
    Abstract base class for all ModelChorus workflows.

    All workflow implementations (thinkdeep, debug, consensus, etc.) must inherit
    from this class and implement the run() method.

    Attributes:
        name: Human-readable name of the workflow
        description: Brief description of what this workflow does
        config: Configuration dictionary for the workflow
        conversation_memory: Optional ConversationMemory instance for multi-turn conversations
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize the base workflow.

        Args:
            name: Human-readable workflow name
            description: Brief description of the workflow
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations
        """
        self.name = name
        self.description = description
        self.config = config or {}
        self.conversation_memory = conversation_memory
        self._result: Optional[WorkflowResult] = None

    @abstractmethod
    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        """
        Execute the workflow with the given prompt.

        This method must be implemented by all subclasses to define the specific
        workflow behavior.

        Args:
            prompt: The input prompt/task for the workflow
            **kwargs: Additional workflow-specific parameters

        Returns:
            WorkflowResult containing the execution results

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement run()")

    def synthesize(self, steps: List[WorkflowStep], model: Optional[str] = None) -> str:
        """
        Synthesize final results from workflow steps.

        This is a helper method that can be used by workflow implementations
        to generate a final synthesis from multiple steps. The default implementation
        simply concatenates step content, but subclasses can override for more
        sophisticated synthesis.

        Args:
            steps: List of workflow steps to synthesize
            model: Optional model to use for synthesis (if AI-powered)

        Returns:
            Synthesized text combining the workflow steps
        """
        if not steps:
            return "No steps to synthesize."

        # Default implementation: simple concatenation
        synthesis_parts = []
        for step in steps:
            synthesis_parts.append(f"Step {step.step_number}:\n{step.content}\n")

        return "\n".join(synthesis_parts)

    def get_result(self) -> Optional[WorkflowResult]:
        """
        Get the most recent workflow execution result.

        Returns:
            The last WorkflowResult, or None if no execution has occurred
        """
        return self._result

    def validate_config(self) -> bool:
        """
        Validate the workflow configuration.

        Subclasses can override this to add specific configuration validation.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

    # ========================================================================
    # Provider Fallback and Availability Methods
    # ========================================================================

    async def _execute_with_fallback(
        self,
        request: "GenerationRequest",
        primary_provider: "ModelProvider",
        fallback_providers: Optional[List["ModelProvider"]] = None
    ) -> tuple["GenerationResponse", str, List[str]]:
        """
        Execute generation with automatic fallback to alternative providers.

        Attempts to generate using the primary provider. If that fails, automatically
        tries each fallback provider in order until one succeeds.

        Args:
            request: Generation request to execute
            primary_provider: Primary provider to try first
            fallback_providers: Optional list of fallback providers to try if primary fails

        Returns:
            Tuple of (response, successful_provider_name, failed_provider_names)

        Raises:
            Exception: If all providers fail

        Example:
            >>> response, used, failed = await self._execute_with_fallback(
            ...     request, primary_provider, [fallback1, fallback2]
            ... )
            >>> if failed:
            ...     logger.warning(f"Providers failed: {failed}, used: {used}")
        """
        from ..providers.cli_provider import ProviderUnavailableError

        fallback_providers = fallback_providers or []
        all_providers = [primary_provider] + fallback_providers
        failed_providers = []
        last_exception = None

        for i, provider in enumerate(all_providers):
            try:
                logger.info(
                    f"Attempting provider {provider.provider_name} "
                    f"({i+1}/{len(all_providers)})"
                )
                response = await provider.generate(request)

                if i > 0:
                    logger.warning(
                        f"Primary provider failed, succeeded with fallback: "
                        f"{provider.provider_name}"
                    )

                return response, provider.provider_name, failed_providers

            except ProviderUnavailableError as e:
                # Permanent error - provider CLI not available
                failed_providers.append(provider.provider_name)
                logger.error(
                    f"{provider.provider_name} unavailable: {e.reason}"
                )
                last_exception = e

            except Exception as e:
                # Other error - could be transient or permanent
                failed_providers.append(provider.provider_name)
                logger.warning(
                    f"{provider.provider_name} failed: {str(e)[:100]}"
                )
                last_exception = e

        # All providers failed
        error_msg = (
            f"All {len(all_providers)} providers failed. "
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise Exception(error_msg)

    async def check_provider_availability(
        self,
        primary_provider: "ModelProvider",
        fallback_providers: Optional[List["ModelProvider"]] = None
    ) -> tuple[bool, List[str], List[tuple[str, str]]]:
        """
        Check availability of all providers before starting workflow.

        Tests each provider's CLI availability concurrently. This allows the workflow
        to fail fast if no providers are available, or to warn the user if some
        providers are unavailable.

        Args:
            primary_provider: Primary provider to check
            fallback_providers: Optional list of fallback providers to check

        Returns:
            Tuple of (at_least_one_available, available_providers, unavailable_providers)
            - at_least_one_available: True if at least one provider is available
            - available_providers: List of available provider names
            - unavailable_providers: List of (provider_name, error_message) tuples

        Example:
            >>> has_provider, available, unavailable = await self.check_provider_availability(
            ...     primary, [fallback1, fallback2]
            ... )
            >>> if not has_provider:
            ...     raise Exception("No providers available")
            >>> if unavailable:
            ...     logger.warning(f"Some providers unavailable: {unavailable}")
        """
        fallback_providers = fallback_providers or []

        # Deduplicate providers by name to avoid checking the same provider twice
        # This can happen when --model flag specifies a provider that's also in fallback list
        seen_names = set()
        unique_providers = []
        for provider in [primary_provider] + fallback_providers:
            if provider.provider_name not in seen_names:
                seen_names.add(provider.provider_name)
                unique_providers.append(provider)

        all_providers = unique_providers

        available = []
        unavailable = []

        # Check all providers concurrently
        async def check_one(provider: "ModelProvider"):
            """Check a single provider's availability."""
            is_available, error = await provider.check_availability()
            return provider.provider_name, is_available, error

        tasks = [check_one(p) for p in all_providers]
        results = await asyncio.gather(*tasks)

        for name, is_available, error in results:
            if is_available:
                available.append(name)
            else:
                unavailable.append((name, error))

        return len(available) > 0, available, unavailable

    # ========================================================================
    # Conversation Support Methods (task-1-5-2)
    # ========================================================================

    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """
        Retrieve conversation thread by ID.

        Convenience wrapper for conversation_memory.get_thread().
        Returns None if conversation_memory not available or thread not found.

        Args:
            thread_id: Thread ID to retrieve

        Returns:
            ConversationThread if found, None otherwise

        Example:
            >>> thread = workflow.get_thread(continuation_id)
            >>> if thread:
            ...     print(f"Found thread with {len(thread.messages)} messages")
        """
        if not self.conversation_memory:
            return None
        return self.conversation_memory.get_thread(thread_id)

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        **kwargs
    ) -> bool:
        """
        Add message to conversation thread.

        Convenience wrapper for conversation_memory.add_message().
        Returns False if conversation_memory not available or operation fails.

        Args:
            thread_id: Thread to add message to
            role: Message role ('user' or 'assistant')
            content: Message content
            **kwargs: Additional arguments passed to ConversationMemory.add_message()
                     (files, workflow_name, model_provider, model_name, metadata)

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = workflow.add_message(
            ...     thread_id,
            ...     "assistant",
            ...     "Analysis complete",
            ...     model_name="gpt-5"
            ... )
        """
        if not self.conversation_memory:
            return False

        if content is None:
            logger.warning(
                "Skipping conversation message for workflow '%s' - content is None (role=%s, thread_id=%s)",
                self.name,
                role,
                thread_id,
            )
            return False

        if not isinstance(content, str):
            logger.warning(
                "Skipping conversation message for workflow '%s' - content has non-string type %s (role=%s, thread_id=%s)",
                self.name,
                type(content).__name__,
                role,
                thread_id,
            )
            return False

        if not content.strip():
            logger.warning(
                "Skipping conversation message for workflow '%s' - blank content after stripping (role=%s, thread_id=%s)",
                self.name,
                role,
                thread_id,
            )
            return False

        return self.conversation_memory.add_message(thread_id, role, content, **kwargs)

    def resume_conversation(self, thread_id: str) -> Optional[List[ConversationMessage]]:
        """
        Resume conversation from existing thread.

        Retrieves thread and returns messages for context. This is a convenience
        method combining get_thread() and message extraction.

        Args:
            thread_id: Thread ID to resume from

        Returns:
            List of messages if thread found, None otherwise

        Example:
            >>> messages = workflow.resume_conversation(continuation_id)
            >>> if messages:
            ...     print(f"Resuming with {len(messages)} messages of context")
            ...     for msg in messages[-5:]:  # Last 5 messages
            ...         print(f"{msg.role}: {msg.content[:50]}...")
        """
        if not self.conversation_memory:
            return None

        thread = self.conversation_memory.get_thread(thread_id)
        if not thread:
            return None

        return thread.messages

    def __repr__(self) -> str:
        """String representation of the workflow."""
        return f"{self.__class__.__name__}(name='{self.name}')"
