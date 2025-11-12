import asyncio
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest, ModelProvider

class InvalidProvider(ModelProvider):
    def __init__(self):
        super().__init__("invalid-provider", "invalid-model")

    async def generate(self, request: GenerationRequest, **kwargs) -> str:
        raise ValueError("This provider is invalid")

    def supports_vision(self) -> bool:
        return False

async def main():
    # Setup
    claude_provider = ClaudeProvider()
    invalid_provider = InvalidProvider()
    workflow = ConsensusWorkflow(providers=[claude_provider, invalid_provider])

    # Execute
    request = GenerationRequest(prompt="Test prompt")
    result = await workflow.execute(request, strategy=ConsensusStrategy.SYNTHESIZE)

    # Verify
    print(f"Succeeded: {[p.provider_name for p in workflow.get_providers() if p.provider_name not in result.failed_providers]}")
    print(f"Failed: {result.failed_providers}")
    if result.consensus_response:
        print(f"Result:\n{result.consensus_response}")
    else:
        print("Workflow failed to produce a result.")

if __name__ == "__main__":
    asyncio.run(main())