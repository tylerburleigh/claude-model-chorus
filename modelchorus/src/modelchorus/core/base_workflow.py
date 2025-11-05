"""
Base workflow abstract class for ModelChorus.

This module defines the abstract base class that all workflow implementations
must inherit from, providing a consistent interface for multi-model orchestration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .conversation import ConversationMemory
from .models import ConversationThread, ConversationMessage


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
