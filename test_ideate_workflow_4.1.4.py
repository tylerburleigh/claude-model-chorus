import asyncio
from model_chorus.workflows.ideate import IdeateWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Execute
    result = await workflow.run(
        prompt="Ways to improve developer productivity",
        num_ideas=12,
        temperature=0.7
    )

    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Ideas requested: {result.metadata.get('num_ideas_requested')}")
        print(f"Ideas generated: {result.metadata.get('num_ideas_generated')}")
        print(f"\nIdeas:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

