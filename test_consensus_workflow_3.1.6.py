
import asyncio
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider
from model_chorus.providers.base_provider import GenerationRequest

async def main():
    # Setup
    claude_provider = ClaudeProvider()
    workflow = ConsensusWorkflow(providers=[claude_provider])

    # Execute
    request = GenerationRequest(prompt="Test prompt")
    result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

    # Verify
    if result.consensus_response:
        print(f"Result:\n{result.consensus_response}")
    else:
        print("Workflow failed.")

if __name__ == "__main__":
    asyncio.run(main())

