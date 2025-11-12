import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory
from pydantic import ValidationError

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # Execute
    result = await workflow.run(
        prompt="Test prompt",
        temperature=2.5  # Invalid
    )

    if not result.success:
        print(f"✅ Expected error: {result.error}")
    else:
        print("Unexpected success.")

if __name__ == "__main__":
    asyncio.run(main())