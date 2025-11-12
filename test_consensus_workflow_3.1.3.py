
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
    request = GenerationRequest(prompt="Is Python pass-by-value or pass-by-reference?")
    result = await workflow.execute(request, strategy=ConsensusStrategy.MAJORITY)

    # Verify
    if result.consensus_response:
        print(f"Majority answer:\n{result.consensus_response}")
    else:
        print("Workflow failed.")

if __name__ == "__main__":
    asyncio.run(main())
