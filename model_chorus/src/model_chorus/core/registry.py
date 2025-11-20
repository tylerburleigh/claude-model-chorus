"""
Workflow registry for ModelChorus plugin system.

This module provides a registry for dynamically registering and loading
workflow implementations, enabling a flexible plugin architecture.
"""

from typing import Dict, Type, Optional, Callable
import inspect
from .base_workflow import BaseWorkflow


class WorkflowRegistry:
    """
    Registry for workflow implementations.

    Provides a plugin system for registering and retrieving workflow classes
    dynamically. Workflows can be registered using the @register decorator or
    programmatically via register_workflow().

    Example:
        ```python
        @WorkflowRegistry.register("thinkdeep")
        class ThinkDeepWorkflow(BaseWorkflow):
            async def run(self, prompt: str, **kwargs):
                # Implementation
                pass

        # Later, retrieve the workflow
        workflow_class = WorkflowRegistry.get("thinkdeep")
        workflow = workflow_class("My Workflow", "Description")
        ```
    """

    _workflows: Dict[str, Type[BaseWorkflow]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator for registering a workflow class.

        Args:
            name: Unique name for the workflow (e.g., "thinkdeep", "debug")

        Returns:
            Decorator function that registers the workflow class

        Raises:
            ValueError: If a workflow with this name is already registered
            TypeError: If the registered class doesn't inherit from BaseWorkflow

        Example:
            ```python
            @WorkflowRegistry.register("myworkflow")
            class MyWorkflow(BaseWorkflow):
                pass
            ```
        """

        def decorator(workflow_class: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
            # Validate that the class inherits from BaseWorkflow
            if not inspect.isclass(workflow_class):
                raise TypeError(f"Expected a class, got {type(workflow_class)}")

            if not issubclass(workflow_class, BaseWorkflow):
                raise TypeError(
                    f"Workflow class {workflow_class.__name__} must inherit from BaseWorkflow"
                )

            # Check for duplicate registration
            if name in cls._workflows:
                raise ValueError(
                    f"Workflow '{name}' is already registered with class "
                    f"{cls._workflows[name].__name__}"
                )

            # Register the workflow
            cls._workflows[name] = workflow_class
            return workflow_class

        return decorator

    @classmethod
    def register_workflow(cls, name: str, workflow_class: Type[BaseWorkflow]) -> None:
        """
        Programmatically register a workflow class.

        Alternative to using the @register decorator. Useful when you need to
        register workflows dynamically or from configuration.

        Args:
            name: Unique name for the workflow
            workflow_class: The workflow class to register

        Raises:
            ValueError: If a workflow with this name is already registered
            TypeError: If the class doesn't inherit from BaseWorkflow

        Example:
            ```python
            WorkflowRegistry.register_workflow("custom", MyCustomWorkflow)
            ```
        """
        # Validate the class
        if not inspect.isclass(workflow_class):
            raise TypeError(f"Expected a class, got {type(workflow_class)}")

        if not issubclass(workflow_class, BaseWorkflow):
            raise TypeError(
                f"Workflow class {workflow_class.__name__} must inherit from BaseWorkflow"
            )

        # Check for duplicate registration
        if name in cls._workflows:
            raise ValueError(
                f"Workflow '{name}' is already registered with class "
                f"{cls._workflows[name].__name__}"
            )

        cls._workflows[name] = workflow_class

    @classmethod
    def get(cls, name: str) -> Type[BaseWorkflow]:
        """
        Retrieve a registered workflow class by name.

        Args:
            name: Name of the workflow to retrieve

        Returns:
            The workflow class

        Raises:
            KeyError: If no workflow is registered with this name

        Example:
            ```python
            WorkflowClass = WorkflowRegistry.get("thinkdeep")
            workflow = WorkflowClass("name", "description")
            ```
        """
        if name not in cls._workflows:
            available = ", ".join(cls._workflows.keys()) if cls._workflows else "none"
            raise KeyError(
                f"No workflow registered with name '{name}'. " f"Available workflows: {available}"
            )

        return cls._workflows[name]

    @classmethod
    def list_workflows(cls) -> Dict[str, Type[BaseWorkflow]]:
        """
        Get all registered workflows.

        Returns:
            Dictionary mapping workflow names to their classes

        Example:
            ```python
            workflows = WorkflowRegistry.list_workflows()
            for name, workflow_class in workflows.items():
                print(f"{name}: {workflow_class.__name__}")
            ```
        """
        return cls._workflows.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a workflow is registered.

        Args:
            name: Name of the workflow to check

        Returns:
            True if the workflow is registered, False otherwise

        Example:
            ```python
            if WorkflowRegistry.is_registered("thinkdeep"):
                workflow = WorkflowRegistry.get("thinkdeep")
            ```
        """
        return name in cls._workflows

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a workflow from the registry.

        Args:
            name: Name of the workflow to unregister

        Raises:
            KeyError: If no workflow is registered with this name

        Example:
            ```python
            WorkflowRegistry.unregister("myworkflow")
            ```
        """
        if name not in cls._workflows:
            raise KeyError(f"No workflow registered with name '{name}'")

        del cls._workflows[name]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered workflows.

        Useful for testing or resetting the registry state.

        Example:
            ```python
            WorkflowRegistry.clear()
            ```
        """
        cls._workflows.clear()
