import asyncio
from model_chorus.workflows.ideate import IdeateWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Simulated agent call
    agent_input = {
        "prompt": "New product ideas for developers",
        "num_ideas": 5
    }

    # Agent invokes skill
    result = await workflow.run(**agent_input)

    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Ideas requested: {result.metadata.get('num_ideas_requested')}")
        print(f"Ideas generated: {result.metadata.get('num_ideas_generated')}")
        print(f"\nIdeas:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

