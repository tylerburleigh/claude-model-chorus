
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
    request = GenerationRequest(prompt="What is the time complexity of binary search?")
    result = await workflow.execute(request, strategy=ConsensusStrategy.FIRST_VALID)

    # Verify
    if result.consensus_response:
        print(f"First valid response:\n{result.consensus_response}")
        print(f"First provider: {list(result.provider_results.keys())[0]}")
    else:
        print("Workflow failed.")

if __name__ == "__main__":
    asyncio.run(main())

