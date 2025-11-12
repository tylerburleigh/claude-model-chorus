
import asyncio
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Turn 1
    result1 = await workflow.run(
        prompt="What is a Python decorator?"
    )
    session_id = result1.metadata.get('thread_id')
    print(f"Turn 1 Session ID: {session_id}")
    print(f"Turn 1 Response:\n{result1.synthesis}")

    # Turn 2
    result2 = await workflow.run(
        prompt="Can you show me a practical example?",
        continuation_id=session_id
    )
    print(f"Turn 2 Session ID: {result2.metadata.get('thread_id')}")
    print(f"Turn 2 Response:\n{result2.synthesis}")

    # Turn 3
    result3 = await workflow.run(
        prompt="How does it compare to class-based approaches?",
        continuation_id=session_id
    )
    print(f"Turn 3 Session ID: {result3.metadata.get('thread_id')}")
    print(f"Turn 3 Response:\n{result3.synthesis}")

if __name__ == "__main__":
    asyncio.run(main())
