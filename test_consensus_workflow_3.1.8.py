
import asyncio
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider
from model_chorus.providers.base_provider import GenerationRequest, ModelProvider, GenerationResponse

class SlowProvider(ModelProvider):
    def __init__(self):
        super().__init__("slow-provider", "slow-model")

    async def generate(self, request: GenerationRequest, **kwargs) -> GenerationResponse:
        await asyncio.sleep(10)
        return GenerationResponse(content="This should not be returned", model="slow-model")

    def supports_vision(self) -> bool:
        return False

async def main():
    # Setup
    claude_provider = ClaudeProvider()
    slow_provider = SlowProvider()
    workflow = ConsensusWorkflow(providers=[claude_provider, slow_provider], default_timeout=5.0)

    # Execute
    request = GenerationRequest(prompt="Complex question requiring deep analysis...")
    result = await workflow.execute(request)

    # Verify
    print(f"Succeeded: {[p for p in result.provider_results.keys()]}")
    print(f"Failed: {result.failed_providers}")
    if result.consensus_response:
        print(f"Result:\n{result.consensus_response}")
    else:
        print("Workflow failed to produce a result.")

if __name__ == "__main__":
    asyncio.run(main())
