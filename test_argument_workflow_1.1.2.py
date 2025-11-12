
import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # Execute first run to get a session_id
    result1 = await workflow.run(
        prompt="Should remote work be mandatory for software companies?",
        temperature=0.7
    )
    
    session_id = result1.metadata.get('thread_id')
    print(f"Initial Session ID: {session_id}")

    # Execute second run
    result2 = await workflow.run(
        prompt="What about hybrid work models as a compromise?",
        continuation_id=session_id,
        temperature=0.7
    )

    # Verify
    if result2.success:
        print(f"Continuation Session ID: {result2.metadata.get('thread_id')}")
        print(f"Same session: {result2.metadata.get('thread_id') == session_id}")
        print(f"\nRound 2 result:\n{result2.synthesis}")
    else:
        print(f"Workflow failed: {result2.error}")

if __name__ == "__main__":
    asyncio.run(main())
