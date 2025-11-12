
import asyncio
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    agent_input = {
        "prompt": "How do I handle errors in async Python?",
    }

    # Execute
    result = await workflow.run(**agent_input)

    # Verify
    if result.success:
        print("Agent invocation successful.")
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Provider: {result.metadata.get('provider')}")
        print(f"Response:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())

