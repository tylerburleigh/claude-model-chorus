"""
Codex CLI provider for ModelChorus.

This module provides integration with OpenAI's Codex models via the `codex` CLI tool.
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


class CodexProvider(CLIProvider):
    """
    Provider for OpenAI's Codex models via the `codex` CLI tool.

    This provider wraps the `codex` command-line interface to enable
    code generation and completion with Codex models (GPT-4, GPT-3.5, etc.).

    Supported features:
    - Text and code generation
    - System prompts
    - Temperature and token control
    - Function calling capabilities
    - Vision support (model-dependent)

    Example:
        >>> provider = CodexProvider()
        >>> request = GenerationRequest(
        ...     prompt="Write a Python function to sort a list",
        ...     temperature=0.5,
        ...     max_tokens=500
        ... )
        >>> response = await provider.generate(request)
        >>> print(response.content)
    """

    # Model capability mappings
    VISION_MODELS = {"gpt4", "gpt4-turbo"}  # Models that support vision
    FUNCTION_CALLING_MODELS = {"gpt4", "gpt4-turbo", "gpt35-turbo"}  # Models with function calling

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        retry_limit: int = 3,
    ):
        """
        Initialize the Codex provider.

        Args:
            api_key: API key for OpenAI (optional if set via env vars)
            config: Provider-specific configuration
            timeout: Command execution timeout in seconds (default: 120)
            retry_limit: Maximum retry attempts for failed commands (default: 3)
        """
        super().__init__(
            provider_name="codex",
            cli_command="codex",
            api_key=api_key,
            config=config,
            timeout=timeout,
            retry_limit=retry_limit,
        )

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize the list of available Codex/GPT models with their capabilities."""
        models = [
            ModelConfig(
                model_id="gpt4",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "gpt-4", "size": "large"},
            ),
            ModelConfig(
                model_id="gpt4-turbo",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "gpt-4-turbo", "size": "large"},
            ),
            ModelConfig(
                model_id="gpt35-turbo",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "gpt-3.5-turbo", "size": "medium"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Codex generation request.

        Constructs a command like:
            codex chat --prompt "..." --model gpt4 --temperature 0.7

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution
        """
        command = [self.cli_command, "chat"]

        # Add prompt
        command.extend(["--prompt", request.prompt])

        # Add system prompt if provided
        if request.system_prompt:
            command.extend(["--system", request.system_prompt])

        # Add model from metadata if specified
        if "model" in request.metadata:
            command.extend(["--model", request.metadata["model"]])

        # Add temperature
        temperature = request.temperature if request.temperature is not None else 0.7
        command.extend(["--temperature", str(temperature)])

        # Add max tokens if specified
        if request.max_tokens:
            command.extend(["--max-tokens", str(request.max_tokens)])

        # Add images if provided (vision capability)
        if request.images:
            for image_path in request.images:
                command.extend(["--image", image_path])

        # Add JSON output format for easier parsing
        command.append("--json")

        logger.debug(f"Built Codex command: {' '.join(command)}")
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `codex` CLI returns JSON output with --json flag:
        {
            "content": "...",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
            "finish_reason": "stop"
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
            error_msg = f"Codex CLI failed with return code {returncode}"
            if stderr:
                error_msg += f"\nError: {stderr}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Parse JSON output
        try:
            data = json.loads(stdout)

            # Convert OpenAI usage format to our standard format
            usage = {}
            if "usage" in data:
                usage_data = data["usage"]
                usage = {
                    "input_tokens": usage_data.get("prompt_tokens", 0),
                    "output_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }

            response = GenerationResponse(
                content=data.get("content", ""),
                model=data.get("model", "unknown"),
                usage=usage,
                stop_reason=data.get("finish_reason"),
                metadata={
                    "raw_response": data,
                },
            )

            logger.info(
                f"Successfully parsed Codex response: {len(response.content)} chars, "
                f"model={response.model}, stop_reason={response.stop_reason}"
            )
            return response

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Codex CLI JSON output: {e}\nOutput: {stdout[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific Codex/GPT model supports vision capabilities.

        Args:
            model_id: The model identifier (e.g., "gpt4", "gpt4-turbo", "gpt35-turbo")

        Returns:
            True if the model supports vision, False otherwise
        """
        return model_id in self.VISION_MODELS

    def supports_function_calling(self, model_id: str) -> bool:
        """
        Check if a specific Codex/GPT model supports function calling.

        Args:
            model_id: The model identifier

        Returns:
            True if the model supports function calling, False otherwise
        """
        return model_id in self.FUNCTION_CALLING_MODELS
