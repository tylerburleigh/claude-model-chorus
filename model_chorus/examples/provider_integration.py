"""
Provider integration example for ModelChorus.

This example demonstrates how to integrate and use different AI providers
with ModelChorus.
"""

import asyncio

from model_chorus.providers import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    ModelConfig,
    ModelProvider,
)


class ExampleProvider(ModelProvider):
    """
    Example provider implementation.

    This is a placeholder showing how to implement a custom provider
    for ModelChorus. In a real implementation, this would integrate
    with an actual AI provider's API.
    """

    def __init__(self, api_key: str = None):
        """Initialize the example provider."""
        super().__init__(provider_name="example", api_key=api_key or "example-key")

        # Define available models
        models = [
            ModelConfig(
                model_id="example-basic",
                temperature=0.7,
                max_tokens=1000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.STREAMING,
                ],
            ),
            ModelConfig(
                model_id="example-vision",
                temperature=0.7,
                max_tokens=2000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                ],
            ),
        ]

        self.set_model_list(models)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text based on the request.

        This is a placeholder implementation. In a real provider,
        this would call the actual API.

        Args:
            request: GenerationRequest with prompt and parameters

        Returns:
            GenerationResponse with generated content
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        # Generate a response (placeholder)
        content = f"Response to: {request.prompt}"

        if request.system_prompt:
            content = f"[System: {request.system_prompt}] {content}"

        return GenerationResponse(
            content=content,
            model="example-basic",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="complete",
        )

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if the model supports vision.

        Args:
            model_id: Model identifier

        Returns:
            True if model supports vision, False otherwise
        """
        return "vision" in model_id.lower()


async def main():
    """Main entry point for the provider example."""

    print("ModelChorus - Provider Integration Example")
    print("=" * 50)

    # Create a provider instance
    provider = ExampleProvider(api_key="your-api-key-here")

    print(f"\nProvider: {provider.provider_name}")
    print(f"API Key Valid: {provider.validate_api_key()}")

    # List available models
    print("\nAvailable Models:")
    for model in provider.get_available_models():
        print(f"\n  {model.model_id}")
        print(f"    Max Tokens: {model.max_tokens}")
        print(f"    Capabilities: {[cap.value for cap in model.capabilities]}")

    # Check capabilities
    print("\nCapability Checks:")
    print(
        f"  example-basic supports vision: {provider.supports_vision('example-basic')}"
    )
    print(
        f"  example-vision supports vision: {provider.supports_vision('example-vision')}"
    )
    print(
        f"  example-basic supports text: {provider.supports_capability('example-basic', ModelCapability.TEXT_GENERATION)}"
    )

    # Create a generation request
    request = GenerationRequest(
        prompt="Explain the benefits of multi-model workflows",
        system_prompt="You are a helpful AI assistant",
        temperature=0.7,
        max_tokens=500,
    )

    print("\n" + "=" * 50)
    print("Generation Request:")
    print("=" * 50)
    print(f"\nPrompt: {request.prompt}")
    print(f"System: {request.system_prompt}")
    print(f"Temperature: {request.temperature}")
    print(f"Max Tokens: {request.max_tokens}")

    # Generate response
    print("\nGenerating response...")
    response = await provider.generate(request)

    print("\n" + "=" * 50)
    print("Generation Response:")
    print("=" * 50)
    print(f"\nModel: {response.model}")
    print(f"Content: {response.content}")
    print(f"Usage: {response.usage}")
    print(f"Stop Reason: {response.stop_reason}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
