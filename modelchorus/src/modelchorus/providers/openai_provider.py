"""
OpenAI provider implementation for ModelChorus.

This module provides integration with OpenAI's GPT models through their official
Python SDK. It supports text generation, vision capabilities, and function calling
for GPT-4 and GPT-3.5 model families.
"""

import asyncio
import os
from typing import Optional
import logging

from openai import AsyncOpenAI, OpenAIError, APIError, RateLimitError, APITimeoutError

from .base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    ModelConfig,
    ModelProvider,
)
from .anthropic_provider import (
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
)

# Configure logger
logger = logging.getLogger(__name__)


class OpenAIProvider(ModelProvider):
    """
    OpenAI GPT provider implementation.

    This provider integrates with OpenAI's GPT models, supporting text generation,
    vision capabilities, and function calling. It uses the official OpenAI
    Python SDK for API communication.

    Attributes:
        client: AsyncOpenAI client instance for API calls
        provider_name: Always "openai"
        api_key: OpenAI API key (from parameter or OPENAI_API_KEY env var)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            timeout: Request timeout in seconds (default: 60.0)
            **kwargs: Additional configuration passed to parent class

        Raises:
            ValueError: If API key is not provided and OPENAI_API_KEY is not set
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Store retry configuration
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize parent class
        super().__init__(
            provider_name="openai",
            api_key=resolved_api_key,
            config=kwargs
        )

        # Initialize OpenAI async client
        if resolved_api_key:
            self.client = AsyncOpenAI(
                api_key=resolved_api_key,
                timeout=timeout,
                max_retries=0,  # We handle retries ourselves
            )
        else:
            # Allow initialization without key, but generate() will fail
            self.client = None

        # Define available GPT models with their capabilities
        models = [
            ModelConfig(
                model_id="gpt-4-turbo",
                temperature=0.7,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 128000,
                    "description": "Most capable GPT-4 model with vision and function calling",
                }
            ),
            ModelConfig(
                model_id="gpt-4o",
                temperature=0.7,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 128000,
                    "description": "Latest GPT-4 optimized model with multimodal capabilities",
                }
            ),
            ModelConfig(
                model_id="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 16385,
                    "description": "Fast and efficient GPT-3.5 model",
                }
            ),
        ]

        self.set_model_list(models)

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            ProviderError: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limit exceeded, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderRateLimitError(
                        f"Rate limit exceeded after {self.max_retries + 1} attempts"
                    ) from e

            except APITimeoutError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Request timeout, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderTimeoutError(
                        f"Request timed out after {self.max_retries + 1} attempts"
                    ) from e

            except OpenAIError as e:
                # Check if it's an authentication error
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    raise ProviderAuthenticationError(
                        "API authentication failed. Check your API key."
                    ) from e

                # Generic OpenAI error - retry if not the last attempt
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"API error, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderError(
                        f"API request failed after {self.max_retries + 1} attempts: {str(e)}"
                    ) from e

            except Exception as e:
                # Catch-all for unexpected errors
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Unexpected error, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderError(
                        f"Request failed after {self.max_retries + 1} attempts: {str(e)}"
                    ) from e

        # Should never reach here, but just in case
        if last_exception:
            raise ProviderError(
                f"Request failed: {str(last_exception)}"
            ) from last_exception

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using OpenAI's GPT models.

        This method handles the conversion from ModelChorus's GenerationRequest format
        to OpenAI's chat completion format, makes the API call, and converts the response
        back to GenerationResponse format.

        Args:
            request: GenerationRequest containing prompt and generation parameters

        Returns:
            GenerationResponse with generated content and metadata

        Raises:
            ValueError: If API key is not configured
            ProviderError: For API errors from OpenAI SDK
        """
        if not self.client:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY "
                "environment variable or pass api_key to constructor."
            )

        # Build messages array - convert prompt to user message
        messages = []

        # Add system message if provided
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })

        # Handle images for vision-enabled models
        if request.images:
            # OpenAI vision format with content array
            content_parts = []

            # Add images first
            for image in request.images:
                if image.startswith("data:"):
                    # Base64 encoded image
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    })
                elif image.startswith("http"):
                    # URL
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    })

            # Add text prompt
            content_parts.append({
                "type": "text",
                "text": request.prompt
            })

            messages.append({
                "role": "user",
                "content": content_parts
            })
        else:
            # Simple text message
            messages.append({
                "role": "user",
                "content": request.prompt
            })

        # Prepare API parameters
        api_params = {
            "model": request.metadata.get("model_id", "gpt-4-turbo"),
            "messages": messages,
        }

        # Add optional parameters
        if request.max_tokens:
            api_params["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            api_params["temperature"] = request.temperature

        # Make API call with retry logic
        async def _make_request():
            """Internal function for making the API request."""
            return await self.client.chat.completions.create(**api_params)

        response = await self._retry_with_backoff(_make_request)

        # Extract content from response
        content = response.choices[0].message.content if response.choices else ""

        # Build usage information
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # Return formatted response
        return GenerationResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.choices[0].finish_reason if response.choices else None,
            metadata={
                "id": response.id if hasattr(response, "id") else None,
                "created": response.created if hasattr(response, "created") else None,
            }
        )

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a GPT model supports vision capabilities.

        Vision is supported by GPT-4 Turbo and GPT-4o, but not by GPT-3.5.

        Args:
            model_id: The model identifier to check

        Returns:
            True if the model supports vision, False otherwise
        """
        # GPT-4 models with vision support
        vision_models = ["gpt-4-turbo", "gpt-4o", "gpt-4-vision"]
        return any(vm in model_id.lower() for vm in vision_models)
