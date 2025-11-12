
import asyncio
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Execute
    result = await workflow.run(
        prompt="Explain the difference between async and sync Python code.",
        temperature=0.5
    )

    # Verify
    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Response:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

