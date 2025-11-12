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
    try:
        result = await workflow.run(
            prompt="Test",
            num_ideas=0  # Invalid
        )
        if not result.success:
            print(f"✅ Expected error: {result.error}")
        else:
            print("Unexpected success.")
    except ValueError as e:
        print(f"✅ Expected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
