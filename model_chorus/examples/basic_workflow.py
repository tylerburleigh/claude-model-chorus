"""
Basic workflow example for ModelChorus.

This example demonstrates how to create and execute a simple workflow
using the ModelChorus framework.
"""

import asyncio
from model_chorus.core import BaseWorkflow, WorkflowResult, WorkflowRegistry, WorkflowRequest


# Define a custom workflow
@WorkflowRegistry.register("example")
class ExampleWorkflow(BaseWorkflow):
    """
    Example workflow that demonstrates basic workflow structure.

    This is a placeholder implementation showing the minimal structure
    needed to create a working workflow.
    """

    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        """
        Execute the example workflow.

        Args:
            prompt: The input prompt to process
            **kwargs: Additional workflow parameters

        Returns:
            WorkflowResult with the execution results
        """
        # Create a result object
        result = WorkflowResult(success=True)

        # Add a workflow step
        result.add_step(
            step_number=1,
            content=f"Processing prompt: {prompt}",
            model="example-model"
        )

        # Add another step
        result.add_step(
            step_number=2,
            content="Analysis complete",
            model="example-model"
        )

        # Generate synthesis
        result.synthesis = self.synthesize(result.steps)

        return result


async def main():
    """Main entry point for the example."""

    # Create a workflow request
    request = WorkflowRequest(
        prompt="What are the benefits of multi-model AI workflows?",
        models=["example-model"],
        config={"example_param": "value"}
    )

    print("ModelChorus - Basic Workflow Example")
    print("=" * 50)
    print(f"\nPrompt: {request.prompt}")
    print(f"Models: {request.models}")
    print(f"Config: {request.config}")

    # Retrieve the workflow from the registry
    workflow_class = WorkflowRegistry.get("example")

    # Create a workflow instance
    workflow = workflow_class(
        name="Example Workflow",
        description="A simple demonstration workflow"
    )

    print(f"\nExecuting workflow: {workflow.name}")
    print(f"Description: {workflow.description}")

    # Execute the workflow
    result = await workflow.run(request.prompt, **request.config)

    # Display results
    print("\n" + "=" * 50)
    print("Workflow Results:")
    print("=" * 50)
    print(f"\nSuccess: {result.success}")
    print(f"\nSteps executed: {len(result.steps)}")

    for step in result.steps:
        print(f"\nStep {step.step_number}:")
        print(f"  Model: {step.model}")
        print(f"  Content: {step.content}")

    print(f"\nSynthesis:")
    print(result.synthesis)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
