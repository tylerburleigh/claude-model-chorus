"""
CLI-based provider for ModelChorus.

This module provides the base class for providers that interact with AI models
via command-line interface tools (e.g., claude CLI, gemini CLI, codex CLI).
"""

import asyncio
import json
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base_provider import (
    ModelProvider,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class CLIProvider(ModelProvider):
    """
    Base class for CLI-based model providers.

    This class extends ModelProvider to handle interactions with AI models
    via CLI tools, providing common functionality for:
    - Building CLI commands
    - Executing commands asynchronously via subprocess
    - Parsing CLI output
    - Error handling and retries

    Subclasses must implement:
    - build_command(): Construct the CLI command for a request
    - parse_response(): Parse CLI output into GenerationResponse

    Attributes:
        cli_command: Base CLI command (e.g., "claude", "gemini")
        timeout: Command timeout in seconds
        retry_limit: Maximum number of retry attempts
    """

    def __init__(
        self,
        provider_name: str,
        cli_command: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        retry_limit: int = 3,
    ):
        """
        Initialize the CLI provider.

        Args:
            provider_name: Name of the provider (e.g., "claude", "gemini")
            cli_command: Base CLI command to execute
            api_key: API key for authentication (optional, may use env vars)
            config: Provider-specific configuration
            timeout: Command execution timeout in seconds (default: 120)
            retry_limit: Maximum retry attempts for failed commands (default: 3)
        """
        super().__init__(provider_name, api_key, config)
        self.cli_command = cli_command
        self.timeout = timeout
        self.retry_limit = retry_limit

    @abstractmethod
    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a generation request.

        Subclasses must implement this to construct the appropriate command
        for their specific CLI tool.

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            List of command parts (e.g., ["claude", "chat", "--prompt", "..."])

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement build_command()")

    @abstractmethod
    def parse_response(self, stdout: str, stderr: str, returncode: int) -> GenerationResponse:
        """
        Parse CLI output into a GenerationResponse.

        Subclasses must implement this to handle their CLI tool's output format.

        Args:
            stdout: Standard output from CLI command
            stderr: Standard error from CLI command
            returncode: Process exit code

        Returns:
            GenerationResponse with parsed content

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If output cannot be parsed
        """
        raise NotImplementedError("Subclasses must implement parse_response()")

    async def execute_command(self, command: List[str]) -> tuple[str, str, int]:
        """
        Execute a CLI command asynchronously with timeout and error handling.

        Args:
            command: List of command parts to execute

        Returns:
            Tuple of (stdout, stderr, returncode)

        Raises:
            asyncio.TimeoutError: If command exceeds timeout
            Exception: For other execution errors
        """
        logger.debug(f"Executing command: {' '.join(command)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            returncode = process.returncode or 0

            logger.debug(
                f"Command completed with return code {returncode}, "
                f"stdout length: {len(stdout_str)}, stderr length: {len(stderr_str)}"
            )

            return stdout_str, stderr_str, returncode

        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using the CLI tool.

        This method orchestrates the generation process:
        1. Build the CLI command
        2. Execute the command with retries
        3. Parse the response
        4. Return the result

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with generated content

        Raises:
            Exception: If all retry attempts fail
        """
        command = self.build_command(request)
        last_exception = None

        for attempt in range(self.retry_limit):
            try:
                stdout, stderr, returncode = await self.execute_command(command)
                response = self.parse_response(stdout, stderr, returncode)

                logger.info(
                    f"Generation successful on attempt {attempt + 1}/{self.retry_limit}"
                )
                return response

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_limit} failed: {e}"
                )

                if attempt < self.retry_limit - 1:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        error_msg = f"All {self.retry_limit} attempts failed. Last error: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def supports_vision(self, model_id: str) -> bool:
        """
        Check if a specific model supports vision capabilities.

        Default implementation checks the model's capabilities list.
        Subclasses can override for more sophisticated checks.

        Args:
            model_id: The model identifier to check

        Returns:
            True if the model supports vision, False otherwise
        """
        from .base_provider import ModelCapability

        return self.supports_capability(model_id, ModelCapability.VISION)

    def __repr__(self) -> str:
        """String representation of the CLI provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider='{self.provider_name}', "
            f"cli_command='{self.cli_command}'"
            f")"
        )
