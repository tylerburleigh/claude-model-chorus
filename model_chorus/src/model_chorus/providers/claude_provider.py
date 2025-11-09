"""
Claude CLI provider for ModelChorus.

This module provides integration with Anthropic's Claude models via the `claude` CLI tool.
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


class ClaudeProvider(CLIProvider):
    """
    Provider for Anthropic's Claude models via the `claude` CLI tool.

    This provider wraps the `claude` command-line interface to enable
    text generation with Claude models (Opus, Sonnet, Haiku).

    Supported features:
    - Text generation with customizable prompts
    - System prompts
    - Temperature and token control
    - Vision capabilities (model-dependent)
    - Thinking mode (extended reasoning)

    Example:
        >>> provider = ClaudeProvider()
        >>> request = GenerationRequest(
        ...     prompt="Explain quantum computing",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> response = await provider.generate(request)
        >>> print(response.content)
    """

    # Model capability mappings
    VISION_MODELS = {"opus", "sonnet"}  # Models that support vision
    THINKING_MODELS = {"opus", "sonnet"}  # Models that support thinking mode

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        retry_limit: int = 3,
    ):
        """
        Initialize the Claude provider.

        Args:
            api_key: API key for Anthropic (optional if set via env vars)
            config: Provider-specific configuration
            timeout: Command execution timeout in seconds (default: 120)
            retry_limit: Maximum retry attempts for failed commands (default: 3)
        """
        super().__init__(
            provider_name="claude",
            cli_command="claude",
            api_key=api_key,
            config=config,
            timeout=timeout,
            retry_limit=retry_limit,
        )

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize the list of available Claude models with their capabilities."""
        models = [
            ModelConfig(
                model_id="opus",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.THINKING,
                ],
                metadata={"family": "claude-3", "size": "large"},
            ),
            ModelConfig(
                model_id="sonnet",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.THINKING,
                ],
                metadata={"family": "claude-3.5", "size": "medium"},
            ),
            ModelConfig(
                model_id="haiku",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                metadata={"family": "claude-3", "size": "small"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Claude generation request.

        Constructs a command like:
            claude --print [prompt] --output-format json --allowed-tools Read Grep ... --system-prompt "..."

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution
        """
        command = [self.cli_command]

        # Use print mode for non-interactive output
        command.append("--print")

        # Add prompt as positional argument (must come early, before tool restriction flags)
        command.append(request.prompt)

        # Use JSON output format for easier parsing
        command.extend(["--output-format", "json"])

        # Add read-only tool restrictions for security
        # Allow only read-only and information-gathering tools
        readonly_tools = ["Read", "Grep", "Glob", "WebSearch", "WebFetch", "Task", "Explore"]
        command.extend(["--allowed-tools"] + readonly_tools)

        # Explicitly block write operations
        write_tools = ["Write", "Edit", "Bash"]
        command.extend(["--disallowed-tools"] + write_tools)

        # Add system prompt if provided
        if request.system_prompt:
            command.extend(["--system-prompt", request.system_prompt])

        # Add model from metadata if specified
        if "model" in request.metadata:
            command.extend(["--model", request.metadata["model"]])

        # Note: Claude CLI doesn't support --temperature, --max-tokens flags
        # These would need to be set via config or are model defaults

        logger.debug(f"Built Claude command: {' '.join(command)}")
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `claude --print --output-format json` returns:
        {
            "type": "result",
            "subtype": "success",
            "result": "...",  # The actual response content
            "usage": {"input_tokens": 10, "output_tokens": 50, ...},
            "modelUsage": {"claude-sonnet-4-5-20250929": {...}},
            ...
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
            error_msg = f"Claude CLI failed with return code {returncode}"
            if stderr:
                error_msg += f"\nError: {stderr}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Parse JSON output
        try:
            data = json.loads(stdout)

            # Extract the result content
            content = data.get("result", "")

            # Extract model info from modelUsage if available
            model_usage = data.get("modelUsage", {})
            model = list(model_usage.keys())[0] if model_usage else "unknown"

            # Extract usage info
            usage = data.get("usage", {})

            response = GenerationResponse(
                content=content,
                model=model,
                usage=usage,
                stop_reason=data.get("subtype"),  # "success" or error type
                metadata={
                    "raw_response": data,
                    "duration_ms": data.get("duration_ms"),
                    "total_cost_usd": data.get("total_cost_usd"),
                },
            )

            logger.info(
                f"Successfully parsed Claude response: {len(response.content)} chars, "
                f"model={response.model}"
            )
            return response

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Claude CLI JSON output: {e}\nOutput: {stdout[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific Claude model supports vision capabilities.

        Args:
            model_id: The model identifier (e.g., "opus", "sonnet", "haiku")

        Returns:
            True if the model supports vision, False otherwise
        """
        return model_id in self.VISION_MODELS

    def supports_thinking(self, model_id: str) -> bool:
        """
        Check if a specific Claude model supports thinking mode.

        Args:
            model_id: The model identifier (e.g., "opus", "sonnet", "haiku")

        Returns:
            True if the model supports thinking mode, False otherwise
        """
        return model_id in self.THINKING_MODELS
