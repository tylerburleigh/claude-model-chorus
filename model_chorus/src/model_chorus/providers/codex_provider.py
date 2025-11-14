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
    FUNCTION_CALLING_MODELS = {"gpt-5-codex", "gpt-5-codex-mini", "gpt-5"}  # Models with function calling

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
                model_id="gpt-5-codex",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={"tier": "primary", "optimized_for": "codex"},
            ),
            ModelConfig(
                model_id="gpt-5-codex-mini",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={"tier": "fast", "optimized_for": "codex"},
            ),
            ModelConfig(
                model_id="gpt-5",
                temperature=0.7,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                ],
                metadata={"tier": "general"},
            ),
        ]
        self.set_model_list(models)

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Codex generation request.

        Constructs a command like:
            codex exec --json --model gpt4 --sandbox read-only "..."

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution
        """
        command = [self.cli_command, "exec"]

        # Use JSON output for easier parsing
        command.append("--json")

        # Add read-only sandbox mode for security
        # NOTE: --sandbox read-only in exec mode runs non-interactively without approval prompts
        command.extend(["--sandbox", "read-only"])

        # Add model from metadata if specified
        if "model" in request.metadata:
            command.extend(["--model", request.metadata["model"]])

        # Note: Codex CLI uses config for most parameters
        # Temperature, max tokens, etc. are set via -c config overrides
        # For simplicity, we'll use defaults

        # Add images if provided (vision capability)
        if request.images:
            for image_path in request.images:
                command.extend(["--image", image_path])

        # Set input_data for stdin piping (exec mode expects input via stdin)
        self.input_data = request.prompt

        logger.debug(f"Built Codex command: {' '.join(command)}")
        return command

    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        The `codex exec --json` returns JSONL (JSON Lines) format with events:
        {"type":"thread.started","thread_id":"..."}
        {"type":"turn.started"}
        {"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"..."}}
        {"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":50}}

        Args:
            stdout: Standard output from CLI command (JSONL format)
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

        # Parse JSONL output
        try:
            content = ""
            usage = {}
            thread_id = None

            # Process each line as a JSON event
            for line in stdout.strip().split('\n'):
                if not line:
                    continue

                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")
                elif event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        content = item.get("text", "")
                elif event_type == "turn.completed":
                    usage_data = event.get("usage", {})
                    usage = {
                        "input_tokens": usage_data.get("input_tokens", 0),
                        "output_tokens": usage_data.get("output_tokens", 0),
                        "cached_input_tokens": usage_data.get("cached_input_tokens", 0),
                    }

            response = GenerationResponse(
                content=content,
                model="gpt-5-codex",  # Default model from help output
                usage=usage,
                stop_reason="completed",
                metadata={
                    "thread_id": thread_id,
                },
            )

            logger.info(
                f"Successfully parsed Codex response: {len(response.content)} chars"
            )
            return response

        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Failed to parse Codex CLI JSONL output: {e}\nOutput: {stdout[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific Codex/GPT model supports vision capabilities.

        Args:
            model_id: The model identifier

        Returns:
            False - Codex models don't support vision
        """
        return False

    def supports_function_calling(self, model_id: str) -> bool:
        """
        Check if a specific Codex/GPT model supports function calling.

        Args:
            model_id: The model identifier

        Returns:
            True if the model supports function calling, False otherwise
        """
        return model_id in self.FUNCTION_CALLING_MODELS
