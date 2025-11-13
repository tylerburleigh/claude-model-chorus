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
    TokenUsage,
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
    FUNCTION_CALLING_MODELS = {"composer-1", "gpt-5-codex"}

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
                model_id="composer-1",
                temperature=0.3,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={"tier": "default", "speed": "250_tokens_per_sec"},
            ),
            ModelConfig(
                model_id="gpt-5-codex",
                temperature=0.3,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={"tier": "codex"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Cursor Agent generation request.

        Constructs a command like:
            cursor-agent -p --output-format json --model default "prompt text"

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution
        """
        command = [self.cli_command]

        # Add print flag for non-interactive mode
        command.append("-p")

        # Add JSON output format for easier parsing
        command.extend(["--output-format", "json"])

        # Add model from metadata if specified
        if "model" in request.metadata:
            command.extend(["--model", request.metadata["model"]])

        # SECURITY: We intentionally do NOT add --force flag here
        # Without --force, Cursor Agent operates in propose-only mode (read-only)
        # Changes are suggested but not applied, providing safe operation

        # Add prompt as positional argument at the end
        # Combine system prompt and user prompt if both provided
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            full_prompt = request.prompt

        command.append(full_prompt)

        logger.debug(f"Built Cursor Agent command: {' '.join(command)}")
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `cursor-agent` CLI returns JSON output with --output-format json:
        {
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": 1696,
            "duration_api_ms": 1696,
            "result": "...",
            "session_id": "...",
            "request_id": "..."
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

            # Extract content from the "result" field
            content = data.get("result", "")

            # Check if this is an error response
            if data.get("is_error", False):
                error_msg = f"Cursor Agent returned error: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            response = GenerationResponse(
                content=content,
                model=data.get("model", "cursor-agent"),
                usage=data.get("usage", {}),
                stop_reason=data.get("subtype"),
                metadata={
                    "session_id": data.get("session_id"),
                    "request_id": data.get("request_id"),
                    "duration_ms": data.get("duration_ms"),
                    "duration_api_ms": data.get("duration_api_ms"),
                    "raw_response": data,
                },
            )

            logger.info(
                f"Successfully parsed Cursor Agent response: {len(response.content)} chars, "
                f"session_id={data.get('session_id')}"
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
