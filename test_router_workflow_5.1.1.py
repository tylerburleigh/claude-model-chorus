import asyncio
from model_chorus.workflows.router import RouterWorkflow
from model_chorus.providers import ClaudeProvider

async def main():
    # Note: Router may be implemented differently; adjust based on actual API
    provider = ClaudeProvider()
    router = RouterWorkflow(provider=provider)

    result = await router.run(
        user_request="What is the difference between REST and GraphQL?"
    )

    if result.success:
        print(f"Recommended workflow: {result.synthesis.get('workflow')}")
        print(f"Confidence: {result.synthesis.get('confidence')}")
        print(f"Rationale: {result.synthesis.get('rationale')}")
        print(f"Parameters: {result.synthesis.get('parameters')}")
    else:
        print(f"Workflow failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
