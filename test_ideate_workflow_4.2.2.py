import asyncio
from model_chorus.workflows.ideate import IdeateWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Initial
    result1 = await workflow.run(
        prompt="Features for a code editor",
        num_ideas=5
    )

    if not result1.success:
        print(f"Round 1 failed: {result1.error}")
        return

    print("--- Round 1 Complete ---")
    print(f"Session ID: {result1.metadata.get('thread_id')}")
    print(f"Ideas:\n{result1.synthesis}")
    print("------------------------\n")

    # Continue
    agent_input = {
        "prompt": "Focus on AI-assisted features",
        "continuation_id": result1.metadata.get('thread_id'),
        "num_ideas": 3
    }

    result2 = await workflow.run(**agent_input)

    if not result2.success:
        print(f"Round 2 failed: {result2.error}")
        return

    print("--- Round 2 Complete ---")
    print(f"Session ID: {result2.metadata.get('thread_id')}")
    print(f"Ideas:\n{result2.synthesis}")
    print("------------------------")


if __name__ == "__main__":
    asyncio.run(main())

