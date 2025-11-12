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
    # Note: THINKING_MODELS removed - Gemini CLI doesn't support thinking_mode parameter

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
                    # Note: THINKING removed - Gemini CLI doesn't support thinking_mode via CLI args
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
                    # Note: THINKING removed - Gemini CLI doesn't support thinking_mode via CLI args
                ],
                metadata={"family": "gemini-ultra", "size": "xlarge"},
            ),
        ]
        self.set_model_list(models)

    async def check_availability(self) -> tuple[bool, Optional[str]]:
        """
        Check if Gemini CLI is available and working.

        Override parent implementation to use --help instead of --version,
        which is faster and more reliable for Gemini CLI.

        Returns:
            Tuple of (is_available, error_message)
            - is_available: True if CLI is available and working
            - error_message: None if available, otherwise description of the issue
        """
        import asyncio

        try:
            # Try running --help with shorter timeout (gemini --version can be slow)
            process = await asyncio.create_subprocess_exec(
                self.cli_command,
                "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Use shorter timeout for Gemini (2 seconds instead of 5)
                await asyncio.wait_for(process.communicate(), timeout=2.0)
                return (True, None)
            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                return (False, f"CLI command '{self.cli_command}' timed out during availability check")

        except FileNotFoundError:
            return (False, f"CLI command '{self.cli_command}' not found in PATH")
        except PermissionError:
            return (False, f"Permission denied to execute '{self.cli_command}'")
        except Exception as e:
            return (False, f"Failed to check '{self.cli_command}' availability: {str(e)}")

    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a Gemini generation request.

        Constructs a command like:
            gemini "prompt text" --model pro --output-format json

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts for subprocess execution

        Raises:
            ValueError: If unsupported parameters are provided
        """
        # Validate parameters - Gemini CLI has limited parameter support
        unsupported_params = []

        if request.temperature is not None:
            unsupported_params.append(f"temperature={request.temperature}")

        if request.max_tokens is not None:
            unsupported_params.append(f"max_tokens={request.max_tokens}")

        if request.system_prompt is not None:
            # System prompt gets merged into prompt but not as a separate parameter
            # We'll allow this with a warning since we handle it by concatenation
            logger.info("System prompt will be merged with user prompt (Gemini CLI limitation)")

        thinking_mode = request.metadata.get('thinking_mode') if request.metadata else None
        if thinking_mode and thinking_mode != 'medium':  # medium is default, so it's ok
            unsupported_params.append(f"thinking_mode={thinking_mode}")

        if unsupported_params:
            raise ValueError(
                f"Gemini CLI provider does not support the following parameters: {', '.join(unsupported_params)}. "
                f"Please remove these parameters or use a different provider (claude, codex) that supports them."
            )

        command = [self.cli_command]

        # Build full prompt (system + user prompt)
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # Add prompt using the -p flag for non-interactive mode
        command.extend(["-p", full_prompt])

        # Add model from metadata if specified
        if "model" in request.metadata:
            command.extend(["-m", request.metadata["model"]])

        # Add JSON output format for easier parsing
        command.extend(["--output-format", "json"])

        # SECURITY: We intentionally do NOT add --yolo flag here
        # In non-interactive mode without --yolo, Gemini defaults to read-only tools only
        # This provides safe operation when used by ModelChorus workflows

        logger.debug(f"Built Gemini command: {' '.join(command)}")
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

            # Extract content from Gemini CLI response format
            content = data.get("response", data.get("content", ""))

            # Extract model info from stats if available
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
                content=content,
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
