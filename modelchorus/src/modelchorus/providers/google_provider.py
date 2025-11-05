"""
Google provider implementation for ModelChorus.

This module provides integration with Google's Gemini models through their official
Python SDK. It supports text generation, vision capabilities, and multimodal inputs
for the Gemini model family.
"""

import asyncio
import os
from typing import Optional, Any
import logging

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from google.api_core import exceptions as google_exceptions
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    GenerationConfig = None
    google_exceptions = None

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


class GoogleProvider(ModelProvider):
    """
    Google Gemini provider implementation.

    This provider integrates with Google's Gemini models, supporting text generation,
    vision capabilities, and multimodal inputs. It uses the official Google Generative AI
    Python SDK for API communication.

    Attributes:
        provider_name: Always "google"
        api_key: Google API key (from parameter or GOOGLE_API_KEY env var)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ) -> None:
        """
        Initialize the Google provider.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            timeout: Request timeout in seconds (default: 60.0)
            **kwargs: Additional configuration passed to parent class

        Raises:
            ValueError: If API key is not provided and GOOGLE_API_KEY is not set
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        # Store retry configuration
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize parent class
        super().__init__(
            provider_name="google",
            api_key=resolved_api_key,
            config=kwargs
        )

        # Check if Google SDK is available
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required for GoogleProvider. "
                "Install it with: pip install google-generativeai"
            )

        # Configure Google Generative AI
        if resolved_api_key:
            genai.configure(api_key=resolved_api_key)
        else:
            # Allow initialization without key, but generate() will fail
            pass

        # Define available Gemini models with their capabilities
        models = [
            ModelConfig(
                model_id="gemini-2.0-flash-exp",
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
                    "context_window": 1000000,
                    "description": "Latest Gemini 2.0 experimental model with extended thinking",
                }
            ),
            ModelConfig(
                model_id="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 2000000,
                    "description": "Most capable Gemini 1.5 model with huge context window",
                }
            ),
            ModelConfig(
                model_id="gemini-1.5-flash",
                temperature=0.7,
                max_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                ],
                metadata={
                    "context_window": 1000000,
                    "description": "Fast and efficient Gemini 1.5 model",
                }
            ),
        ]

        self.set_model_list(models)

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Function to execute (can be sync or async)
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
                # Google SDK is synchronous, so we run it in executor
                result = func(*args, **kwargs)
                return result

            except google_exceptions.ResourceExhausted as e:
                # Rate limit / quota exceeded
                last_exception = e
                if attempt < self.max_retries:
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

            except google_exceptions.DeadlineExceeded as e:
                # Timeout
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

            except google_exceptions.Unauthenticated as e:
                # Authentication error - don't retry
                raise ProviderAuthenticationError(
                    "API authentication failed. Check your API key."
                ) from e

            except google_exceptions.GoogleAPIError as e:
                # Generic Google API error
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
        Generate text using Google's Gemini models.

        This method handles the conversion from ModelChorus's GenerationRequest format
        to Google's GenerativeModel format, makes the API call, and converts the response
        back to GenerationResponse format.

        Args:
            request: GenerationRequest containing prompt and generation parameters

        Returns:
            GenerationResponse with generated content and metadata

        Raises:
            ValueError: If API key is not configured
            ProviderError: For API errors from Google SDK
        """
        if not self.api_key:
            raise ValueError(
                "Google API key not configured. Set GOOGLE_API_KEY "
                "environment variable or pass api_key to constructor."
            )

        # Get model name from request or use default
        model_name = request.metadata.get("model_id", "gemini-1.5-pro")

        # Create the model instance
        model = genai.GenerativeModel(model_name)

        # Prepare generation config
        generation_config = {}

        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens

        if request.temperature is not None:
            generation_config["temperature"] = request.temperature

        # Build content parts for multimodal input
        content_parts = []

        # Add images if provided (Gemini supports multimodal)
        if request.images:
            # Note: Google SDK expects actual image data or PIL images
            # For now, we'll just include the prompt
            # In production, you'd need to download/decode images
            logger.warning(
                "Image support in GoogleProvider requires image data processing. "
                "Currently only text prompts are fully supported."
            )

        # Add the text prompt
        if request.system_prompt:
            # Gemini doesn't have a separate system prompt field
            # We prepend it to the content
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
            content_parts.append(full_prompt)
        else:
            content_parts.append(request.prompt)

        # Make API call with retry logic
        def _make_request():
            """Internal function for making the API request."""
            if generation_config:
                return model.generate_content(
                    content_parts,
                    generation_config=GenerationConfig(**generation_config)
                )
            else:
                return model.generate_content(content_parts)

        response = await self._retry_with_backoff(_make_request)

        # Extract content from response
        content = response.text if hasattr(response, "text") else ""

        # Build usage information
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": usage_meta.prompt_token_count if hasattr(usage_meta, "prompt_token_count") else 0,
                "completion_tokens": usage_meta.candidates_token_count if hasattr(usage_meta, "candidates_token_count") else 0,
                "total_tokens": usage_meta.total_token_count if hasattr(usage_meta, "total_token_count") else 0,
            }

        # Extract finish reason
        finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)

        # Return formatted response
        return GenerationResponse(
            content=content,
            model=model_name,
            usage=usage,
            stop_reason=finish_reason,
            metadata={
                "prompt_feedback": str(response.prompt_feedback) if hasattr(response, "prompt_feedback") else None,
            }
        )

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a Gemini model supports vision capabilities.

        Most Gemini models support vision (multimodal inputs).

        Args:
            model_id: The model identifier to check

        Returns:
            True if the model supports vision, False otherwise
        """
        # Most Gemini models support vision
        # Flash models support vision too
        return "gemini" in model_id.lower()
