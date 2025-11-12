import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # Execute
    result = await workflow.run(
        prompt="Should remote work be mandatory for software companies?",
        temperature=0.7
    )

    # Verify
    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Provider: {result.metadata.get('provider')}")
        print(f"Roles: {result.metadata.get('roles_executed')}")
        
        # The result format has changed. Now we print the synthesis.
        # The synthesis is a concatenation of steps.
        # For a more detailed output, we can iterate through the steps.
        
        creator_step = next((step for step in result.steps if step.metadata.get('role') == 'creator'), None)
        skeptic_step = next((step for step in result.steps if step.metadata.get('role') == 'skeptic'), None)
        moderator_step = next((step for step in result.steps if step.metadata.get('role') == 'moderator'), None)

        print("\nResult:")
        if creator_step:
            print("CREATOR (Pro):")
            print(creator_step.content)
            print("-" * 20)
        if skeptic_step:
            print("SKEPTIC (Con):")
            print(skeptic_step.content)
            print("-" * 20)
        if moderator_step:
            print("MODERATOR (Synthesis):")
            print(moderator_step.content)
            print("-" * 20)

    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())