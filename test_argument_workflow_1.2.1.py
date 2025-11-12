
import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    agent_input = {
        "prompt": "Should we migrate to Kubernetes?",
        "temperature": 0.7
    }

    # Execute
    result = await workflow.run(**agent_input)

    # Verify
    if result.success:
        print("Agent invocation successful.")
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Provider: {result.metadata.get('provider')}")
        print(f"Roles: {result.metadata.get('roles_executed')}")
        print(f"\nResult:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

