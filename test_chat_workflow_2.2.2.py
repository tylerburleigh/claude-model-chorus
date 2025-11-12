
import asyncio
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Initial
    result1 = await workflow.run(prompt="What is REST?")
    session_id = result1.metadata.get('thread_id')
    print(f"Initial Session ID: {session_id}")

    # Continue
    agent_input = {
        "prompt": "How does it differ from GraphQL?",
        "continuation_id": session_id
    }

    result2 = await workflow.run(**agent_input)

    # Verify
    if result2.success:
        print("Agent threading successful.")
        print(f"Continuation Session ID: {result2.metadata.get('thread_id')}")
        print(f"Same session: {result2.metadata.get('thread_id') == session_id}")
        print(f"\nRound 2 result:\n{result2.synthesis}")
    else:
        print(f"Workflow failed: {result2.error}")

if __name__ == "__main__":
    asyncio.run(main())

