import asyncio
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

async def main():
    # Setup
    claude_provider = ClaudeProvider()
    gemini_provider = GeminiProvider()
    workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

    # Read file content
    with open("test_data/files/sample.py", "r") as f:
        file_content = f.read()

    # Execute
    request = GenerationRequest(
        prompt=f"What improvements would you suggest for this code?\n\n```python\n{file_content}\n```")
    result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

    # Verify
    if result.consensus_response:
        print(f"Result:\n{result.consensus_response}")
    else:
        print("Workflow failed.")

if __name__ == "__main__":
    asyncio.run(main())