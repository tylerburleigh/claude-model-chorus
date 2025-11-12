
import asyncio
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers import GeminiProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = GeminiProvider()
    memory = ConversationMemory()
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Execute
    result = await workflow.run(
        prompt="What are the benefits of functional programming?"
    )

    # Verify
    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Provider: {result.metadata.get('provider')}")
        print(f"Response:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

