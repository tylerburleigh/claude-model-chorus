"""
Cursor Agent CLI provider for ModelChorus.

This module provides integration with Cursor's AI agent via the `cursor-agent` CLI tool.
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


class CursorAgentProvider(CLIProvider):
    """
    Provider for Cursor's AI agent via the `cursor-agent` CLI tool.

    This provider wraps the `cursor-agent` command-line interface to enable
    code-focused AI interactions with Cursor's agent models.

    Supported features:
    - Code generation and completion
    - Natural language to code conversion
    - Code explanation and documentation
    - System prompts for context
    - Temperature control
    - Working directory support for context-aware code generation

    Example:
        >>> provider = CursorAgentProvider()
        >>> request = GenerationRequest(
        ...     prompt="Write a function to validate email addresses",
        ...     temperature=0.3,
        ...     max_tokens=500,
        ...     metadata={"working_directory": "/path/to/project"}
        ... )
        >>> response = await provider.generate(request)
        >>> print(response.content)
    """

    # Model capability mappings
    # Cursor agent is primarily focused on code generation
    FUNCTION_CALLING_MODELS = {"default", "fast", "premium"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        retry_limit: int = 3,
    ):
        """
        Initialize the Cursor Agent provider.

        Args:
            api_key: API key for Cursor (optional if set via env vars)
            config: Provider-specific configuration
            timeout: Command execution timeout in seconds (default: 120)
            retry_limit: Maximum retry attempts for failed commands (default: 3)
        """
        super().__init__(
            provider_name="cursor-agent",
            cli_command="cursor-agent",
            api_key=api_key,
            config=config,
            timeout=timeout,
            retry_limit=retry_limit,
        )

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize the list of available Cursor agent models with their capabilities."""
        models = [
            ModelConfig(
                model_id="default",
                temperature=0.3,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "cursor", "focus": "code", "speed": "medium"},
            ),
            ModelConfig(
                model_id="fast",
                temperature=0.3,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "cursor", "focus": "code", "speed": "fast"},
            ),
            ModelConfig(
                model_id="premium",
                temperature=0.3,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "cursor", "focus": "code", "speed": "slow", "quality": "high"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Cursor Agent generation request.

        Constructs a command like:
            cursor-agent chat --prompt "..." --model default --temperature 0.3

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

        # Add temperature (default to 0.3 for code generation)
        temperature = request.temperature if request.temperature is not None else 0.3
        command.extend(["--temperature", str(temperature)])

        # Add max tokens if specified
        if request.max_tokens:
            command.extend(["--max-tokens", str(request.max_tokens)])

        # Add working directory if specified (for context-aware code generation)
        if "working_directory" in request.metadata:
            command.extend(["--working-directory", request.metadata["working_directory"]])

        # Add JSON output format for easier parsing
        command.append("--json")

        # SECURITY: We intentionally do NOT add --force flag here
        # Without --force, Cursor Agent operates in propose-only mode (read-only)
        # Changes are suggested but not applied, providing safe operation

        logger.debug(f"Built Cursor Agent command: {' '.join(command)}")
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `cursor-agent` CLI returns JSON output with --json flag:
        {
            "content": "...",
            "model": "cursor-default",
            "usage": {"input_tokens": 10, "output_tokens": 50},
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
            error_msg = f"Cursor Agent CLI failed with return code {returncode}"
            if stderr:
                error_msg += f"\nError: {stderr}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Parse JSON output
        try:
            data = json.loads(stdout)

            response = GenerationResponse(
                content=data.get("content", ""),
                model=data.get("model", "unknown"),
                usage=data.get("usage", {}),
                stop_reason=data.get("finish_reason"),
                metadata={
                    "code_blocks": data.get("code_blocks"),
                    "language": data.get("language"),
                    "raw_response": data,
                },
            )

            logger.info(
                f"Successfully parsed Cursor Agent response: {len(response.content)} chars, "
                f"model={response.model}, stop_reason={response.stop_reason}"
            )
            return response

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Cursor Agent CLI JSON output: {e}\nOutput: {stdout[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific Cursor Agent model supports vision capabilities.

        Currently, Cursor Agent models do not support vision.

        Args:
            model_id: The model identifier

        Returns:
            False (Cursor Agent does not support vision)
        """
        return False

    def supports_code_generation(self, model_id: str) -> bool:
        """
        Check if a specific Cursor Agent model supports code generation.

        Args:
            model_id: The model identifier

        Returns:
            True (all Cursor Agent models are optimized for code)
        """
        return model_id in self.FUNCTION_CALLING_MODELS
