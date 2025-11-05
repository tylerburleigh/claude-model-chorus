"""
Anthropic provider implementation for ModelChorus.

This module provides integration with Anthropic's Claude models through their official
Python SDK. It supports text generation, vision capabilities, and streaming for
Claude 3 model family.
"""

import os
from typing import Optional

from anthropic import Anthropic, AsyncAnthropic

from .base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    ModelConfig,
    ModelProvider,
)


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude provider implementation.

    This provider integrates with Anthropic's Claude models, supporting text generation,
    vision capabilities, and extended thinking modes. It uses the official Anthropic
    Python SDK for API communication.

    Attributes:
        client: AsyncAnthropic client instance for API calls
        provider_name: Always "anthropic"
        api_key: Anthropic API key (from parameter or ANTHROPIC_API_KEY env var)
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            **kwargs: Additional configuration passed to parent class

        Raises:
            ValueError: If API key is not provided and ANTHROPIC_API_KEY is not set
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Initialize parent class
        super().__init__(
            provider_name="anthropic",
            api_key=resolved_api_key,
            config=kwargs
        )

        # Initialize Anthropic async client
        if resolved_api_key:
            self.client = AsyncAnthropic(api_key=resolved_api_key)
        else:
            # Allow initialization without key, but generate() will fail
            self.client = None

        # Define available Claude models with their capabilities
        models = [
            ModelConfig(
                model_id="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                    ModelCapability.THINKING,
                ],
                metadata={
                    "context_window": 200000,
                    "description": "Most capable Claude 3.5 model with extended thinking",
                }
            ),
            ModelConfig(
                model_id="claude-3-5-haiku-20241022",
                temperature=0.7,
                max_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 200000,
                    "description": "Fast and efficient Claude 3.5 model",
                }
            ),
            ModelConfig(
                model_id="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 200000,
                    "description": "Most powerful Claude 3 model for complex tasks",
                }
            ),
        ]

        self.set_model_list(models)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using Anthropic's Claude models.

        This method handles the conversion from ModelChorus's GenerationRequest format
        to Anthropic's message format, makes the API call, and converts the response
        back to GenerationResponse format.

        Args:
            request: GenerationRequest containing prompt and generation parameters

        Returns:
            GenerationResponse with generated content and metadata

        Raises:
            ValueError: If API key is not configured
            Exception: For API errors from Anthropic SDK
        """
        if not self.client:
            raise ValueError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY "
                "environment variable or pass api_key to constructor."
            )

        # Build messages array - convert prompt to user message
        messages = [
            {
                "role": "user",
                "content": request.prompt
            }
        ]

        # Handle images for vision-enabled models
        if request.images:
            # Convert to Anthropic's vision format
            content_blocks = []
            for image in request.images:
                if image.startswith("data:"):
                    # Base64 encoded image
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",  # Default, could be enhanced
                            "data": image.split(",")[1] if "," in image else image
                        }
                    })
                else:
                    # URL
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image
                        }
                    })

            # Add text prompt after images
            content_blocks.append({
                "type": "text",
                "text": request.prompt
            })

            messages[0]["content"] = content_blocks

        # Prepare API parameters
        api_params = {
            "model": request.metadata.get("model_id", "claude-3-5-sonnet-20241022"),
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }

        # Add optional parameters
        if request.system_prompt:
            api_params["system"] = request.system_prompt

        if request.temperature is not None:
            api_params["temperature"] = request.temperature

        # Make API call
        try:
            response = await self.client.messages.create(**api_params)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Anthropic API error: {str(e)}") from e

        # Extract content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        # Build usage information
        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            }

        # Return formatted response
        return GenerationResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason if hasattr(response, "stop_reason") else None,
            metadata={
                "id": response.id if hasattr(response, "id") else None,
                "type": response.type if hasattr(response, "type") else None,
            }
        )

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a Claude model supports vision capabilities.

        Vision is supported by Claude 3 Opus and Claude 3.5 Sonnet, but not by
        Claude 3.5 Haiku.

        Args:
            model_id: The model identifier to check

        Returns:
            True if the model supports vision, False otherwise
        """
        # Claude 3.5 Sonnet and Claude 3 Opus support vision
        vision_models = ["sonnet", "opus"]
        return any(vm in model_id.lower() for vm in vision_models)
