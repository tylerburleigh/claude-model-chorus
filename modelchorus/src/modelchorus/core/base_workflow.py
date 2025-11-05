"""
Base workflow abstract class for ModelChorus.

This module defines the abstract base class that all workflow implementations
must inherit from, providing a consistent interface for multi-model orchestration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


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
    """

    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base workflow.

        Args:
            name: Human-readable workflow name
            description: Brief description of the workflow
            config: Optional configuration dictionary
        """
        self.name = name
        self.description = description
        self.config = config or {}
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

    def __repr__(self) -> str:
        """String representation of the workflow."""
        return f"{self.__class__.__name__}(name='{self.name}')"
