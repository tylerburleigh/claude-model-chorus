"""
Workflow registry for ModelChorus plugin system.

This module provides a registry for dynamically registering and loading
workflow implementations, enabling a flexible plugin architecture with
workflow discovery and metadata management.
"""

import inspect
from collections.abc import Callable

from .base_workflow import BaseWorkflow
from .models import WorkflowMetadata


class WorkflowRegistry:
    """
    Registry for workflow implementations with metadata support.

    Provides a plugin system for registering and retrieving workflow classes
    dynamically with rich metadata for discovery and documentation. Workflows
    can be registered using the @register decorator or programmatically.

    Example:
        ```python
        @WorkflowRegistry.register(
            "thinkdeep",
            description="Extended reasoning with systematic investigation",
            version="2.0.0",
            author="ModelChorus Team"
        )
        class ThinkDeepWorkflow(BaseWorkflow):
            async def run(self, prompt: str, **kwargs):
                # Implementation
                pass

        # List all workflows
        workflows = WorkflowRegistry.list_workflows()

        # Get workflow info
        info = WorkflowRegistry.get_workflow_info("thinkdeep")
        ```
    """

    _workflows: dict[str, type[BaseWorkflow]] = {}
    _metadata: dict[str, WorkflowMetadata] = {}

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

        def decorator(workflow_class: type[BaseWorkflow]) -> type[BaseWorkflow]:
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
    def register_workflow(cls, name: str, workflow_class: type[BaseWorkflow]) -> None:
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
    def get(cls, name: str) -> type[BaseWorkflow]:
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
                f"No workflow registered with name '{name}'. "
                f"Available workflows: {available}"
            )

        return cls._workflows[name]

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
        if name in cls._metadata:
            del cls._metadata[name]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered workflows and metadata.

        Useful for testing or resetting the registry state.

        Example:
            ```python
            WorkflowRegistry.clear()
            ```
        """
        cls._workflows.clear()
        cls._metadata.clear()

    @classmethod
    def list_workflows(cls) -> list[str]:
        """
        List all registered workflow names.

        Returns:
            List of workflow names in alphabetical order

        Example:
            ```python
            workflows = WorkflowRegistry.list_workflows()
            print(f"Available workflows: {', '.join(workflows)}")
            ```
        """
        return sorted(cls._workflows.keys())

    @classmethod
    def get_workflow_info(cls, name: str) -> WorkflowMetadata | None:
        """
        Get metadata for a registered workflow.

        Args:
            name: Name of the workflow

        Returns:
            WorkflowMetadata if found, None otherwise

        Example:
            ```python
            info = WorkflowRegistry.get_workflow_info("consensus")
            if info:
                print(f"{info.name}: {info.description}")
                print(f"Version: {info.version}")
                print(f"Author: {info.author}")
            ```
        """
        return cls._metadata.get(name)

    @classmethod
    def register_metadata(
        cls,
        name: str,
        description: str,
        version: str = "1.0.0",
        author: str = "Unknown",
        category: str | None = None,
        parameters: list[str] | None = None,
        examples: list[str] | None = None,
    ) -> None:
        """
        Register or update metadata for a workflow.

        Can be called independently or as part of workflow registration.
        Useful for adding metadata to workflows registered without it.

        Args:
            name: Workflow name
            description: Workflow description
            version: Semantic version (default: "1.0.0")
            author: Workflow author (default: "Unknown")
            category: Optional category
            parameters: Optional list of parameter names
            examples: Optional usage examples

        Example:
            ```python
            WorkflowRegistry.register_metadata(
                "consensus",
                description="Multi-model consultation",
                version="2.0.0",
                author="ModelChorus Team",
                category="consultation",
                parameters=["strategy", "providers"],
                examples=["model-chorus consensus 'prompt' --strategy vote"]
            )
            ```
        """
        metadata = WorkflowMetadata(
            name=name,
            description=description,
            version=version,
            author=author,
            category=category,
            parameters=parameters or [],
            examples=examples or [],
        )
        cls._metadata[name] = metadata
