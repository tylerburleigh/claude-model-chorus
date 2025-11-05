"""
Gemini CLI provider for ModelChorus.

This module provides integration with Google's Gemini models via the `gemini` CLI tool.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .cli_provider import CLIProvider
from .base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelConfig,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class GeminiProvider(CLIProvider):
    """
    Provider for Google's Gemini models via the `gemini` CLI tool.

    This provider wraps the `gemini` command-line interface to enable
    text generation with Gemini models (Pro, Flash, Ultra).

    Supported features:
    - Text generation with customizable prompts
    - System instructions
    - Temperature and token control
    - Vision capabilities (multimodal support)
    - Function calling
    - Thinking/reasoning mode

    Example:
        >>> provider = GeminiProvider()
        >>> request = GenerationRequest(
        ...     prompt="Explain machine learning",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> response = await provider.generate(request)
        >>> print(response.content)
    """

    # Model capability mappings
    VISION_MODELS = {"pro", "ultra", "flash"}  # All Gemini models support vision
    THINKING_MODELS = {"pro", "ultra"}  # Models that support extended thinking

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        retry_limit: int = 3,
    ):
        """
        Initialize the Gemini provider.

        Args:
            api_key: API key for Google AI (optional if set via env vars)
            config: Provider-specific configuration
            timeout: Command execution timeout in seconds (default: 120)
            retry_limit: Maximum retry attempts for failed commands (default: 3)
        """
        super().__init__(
            provider_name="gemini",
            cli_command="gemini",
            api_key=api_key,
            config=config,
            timeout=timeout,
            retry_limit=retry_limit,
        )

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize the list of available Gemini models with their capabilities."""
        models = [
            ModelConfig(
                model_id="pro",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.THINKING,
                ],
                metadata={"family": "gemini-2.0", "size": "large"},
            ),
            ModelConfig(
                model_id="flash",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "gemini-2.0", "size": "medium"},
            ),
            ModelConfig(
                model_id="ultra",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.THINKING,
                ],
                metadata={"family": "gemini-ultra", "size": "xlarge"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Gemini generation request.

        Constructs a command like:
            gemini "prompt text" --model pro -o json

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution
        """
        command = [self.cli_command]

        # Add model from metadata if specified (must come before prompt)
        if "model" in request.metadata:
            command.extend(["-m", request.metadata["model"]])

        # Note: Gemini CLI doesn't support --temperature or --max-tokens via CLI flags
        # These parameters are controlled through the Gemini CLI settings/config

        # Add prompt as positional argument (must be after flags, before options)
        command.append(request.prompt)

        # Add JSON output format for easier parsing
        command.extend(["-o", "json"])

        logger.debug(f"Built Gemini command: {' '.join(command)}")
        logger.warning(
            "Note: Gemini CLI does not support temperature, max_tokens, system_prompt, "
            "thinking mode, or images via command-line arguments. "
            "These features require Gemini CLI configuration or are not supported."
        )
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `gemini` CLI returns JSON output with -o json flag:
        {
            "response": "...",
            "stats": {
                "models": {
                    "gemini-2.5-pro": {
                        "tokens": {"prompt": 10, "candidates": 50, "total": 60}
                    }
                }
            }
        }

        Args:
            stdout: Standard output from CLI command
            stderr: Standard error from CLI command
            returncode: Process exit code

        Returns:
            GenerationResponse with parsed content

        Raises:
            ValueError: If output cannot be parsed or command failed
        """
        # Check for command failure
        if returncode != 0:
            error_msg = f"Gemini CLI failed with return code {returncode}"
            if stderr:
                error_msg += f"\nError: {stderr}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Parse JSON output
        try:
            data = json.loads(stdout)

            # Extract model name from stats if available
            model_name = "unknown"
            usage = {}
            if "stats" in data and "models" in data["stats"]:
                # Get the first (and usually only) model from stats
                models = data["stats"]["models"]
                if models:
                    model_name = list(models.keys())[0]
                    model_stats = models[model_name]
                    # Convert token stats to standard usage format
                    if "tokens" in model_stats:
                        tokens = model_stats["tokens"]
                        usage = {
                            "input_tokens": tokens.get("prompt", 0),
                            "output_tokens": tokens.get("candidates", 0),
                            "total_tokens": tokens.get("total", 0),
                        }

            response = GenerationResponse(
                content=data.get("response", ""),
                model=model_name,
                usage=usage,
                stop_reason=None,  # Gemini CLI doesn't provide finish_reason
                metadata={
                    "stats": data.get("stats"),
                    "raw_response": data,
                },
            )

            logger.info(
                f"Successfully parsed Gemini response: {len(response.content)} chars, "
                f"model={response.model}"
            )
            return response

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Gemini CLI JSON output: {e}\nOutput: {stdout[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific Gemini model supports vision capabilities.

        Args:
            model_id: The model identifier (e.g., "pro", "flash", "ultra")

        Returns:
            True if the model supports vision, False otherwise
        """
        return model_id in self.VISION_MODELS

    def supports_thinking(self, model_id: str) -> bool:
        """
        Check if a specific Gemini model supports thinking mode.

        Args:
            model_id: The model identifier

        Returns:
            True if the model supports thinking mode, False otherwise
        """
        return model_id in self.THINKING_MODELS
