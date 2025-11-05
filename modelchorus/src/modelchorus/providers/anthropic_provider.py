"""
Anthropic provider implementation for ModelChorus.

This module provides integration with Anthropic's Claude models through their official
Python SDK. It supports text generation, vision capabilities, and streaming for
Claude 3 model family.
"""

import asyncio
import os
from typing import Optional
import logging

from anthropic import (
    AsyncAnthropic,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)

from .base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelCapability,
    ModelConfig,
    ModelProvider,
)

# Configure logger
logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when API authentication fails."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when API request times out."""
    pass


class ProviderConnectionError(ProviderError):
    """Raised when connection to API fails."""
    pass


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

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            timeout: Request timeout in seconds (default: 60.0)
            **kwargs: Additional configuration passed to parent class

        Raises:
            ValueError: If API key is not provided and ANTHROPIC_API_KEY is not set
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Store retry configuration
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize parent class
        super().__init__(
            provider_name="anthropic",
            api_key=resolved_api_key,
            config=kwargs
        )

        # Initialize Anthropic async client with timeout
        if resolved_api_key:
            self.client = AsyncAnthropic(
                api_key=resolved_api_key,
                timeout=timeout,
                max_retries=0,  # We handle retries ourselves for better control
            )
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

    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ):
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

            except APIConnectionError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Connection error, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderConnectionError(
                        f"Connection failed after {self.max_retries + 1} attempts"
                    ) from e

            except AuthenticationError as e:
                # Don't retry authentication errors
                raise ProviderAuthenticationError(
                    "API authentication failed. Check your API key."
                ) from e

            except BadRequestError as e:
                # Don't retry bad requests - the input is invalid
                raise ProviderError(
                    f"Invalid request: {str(e)}"
                ) from e

            except APIError as e:
                # Generic API error - retry if not the last attempt
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

        # Should never reach here, but just in case
        if last_exception:
            raise ProviderError(
                f"Request failed: {str(last_exception)}"
            ) from last_exception

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

        # Make API call with retry logic
        async def _make_request():
            """Internal function for making the API request."""
            return await self.client.messages.create(**api_params)

        response = await self._retry_with_backoff(_make_request)

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
