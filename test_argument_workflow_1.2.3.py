
import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # First call
    result1 = await workflow.run(prompt="Should we use GraphQL?")
    session_id = result1.metadata.get('thread_id')
    print(f"Initial Session ID: {session_id}")

    # Continuation via agent
    agent_input = {
        "prompt": "What about REST vs GraphQL?",
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
