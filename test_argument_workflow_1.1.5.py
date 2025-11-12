
import asyncio
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def main():
    # Setup
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    long_prompt = """
Evaluate this comprehensive proposal for organizational change:
1. Restructure teams by product lines instead of functional areas
2. Implement quarterly OKR cycles with strict accountability
3. Mandate 20% time for innovation projects
4. Establish cross-functional squads with embedded QA
5. Move to continuous deployment with feature flags
6. Create centralized platform team for shared infrastructure
Consider technical debt, team morale, hiring constraints, and market competition.
"""

    # Execute
    result = await workflow.run(prompt=long_prompt, temperature=0.7)

    # Verify
    if result.success:
        print(f"Session ID: {result.metadata.get('thread_id')}")
        print(f"Provider: {result.metadata.get('provider')}")
        print(f"Roles: {result.metadata.get('roles_executed')}")
        print(f"\nResult:\n{result.synthesis}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
