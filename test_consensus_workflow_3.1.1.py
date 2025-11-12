import asyncio
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

async def main():
    # Setup
    claude_provider = ClaudeProvider()
    gemini_provider = GeminiProvider()
    workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

    # Execute
    request = GenerationRequest(prompt="What are the key principles of good API design?")
    result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

    # Verify
    if result.consensus_response:
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Providers succeeded: {[p.provider_name for p in workflow.get_providers()]}")
        print(f"\nResult:\n{result.consensus_response}")
    else:
        print("Workflow failed.")

if __name__ == "__main__":
    asyncio.run(main())