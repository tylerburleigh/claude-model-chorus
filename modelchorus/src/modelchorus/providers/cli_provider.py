"""
CLI-based provider for ModelChorus.

This module provides the base class for providers that interact with AI models
via command-line interface tools (e.g., claude CLI, gemini CLI, codex CLI).
"""

import asyncio
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_provider import (
    ModelProvider,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """Provider CLI is not available or cannot be used."""

    def __init__(self, provider_name: str, reason: str, suggestions: Optional[List[str]] = None):
        """
        Initialize provider unavailability error.

        Args:
            provider_name: Name of the provider (e.g., "claude")
            reason: Reason for unavailability
            suggestions: List of suggestions for fixing the issue
        """
        self.provider_name = provider_name
        self.reason = reason
        self.suggestions = suggestions or []
        super().__init__(f"{provider_name}: {reason}")


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

    async def check_availability(self) -> tuple[bool, Optional[str]]:
        """
        Check if this provider's CLI is available and working.

        Returns:
            Tuple of (is_available, error_message)
            - is_available: True if CLI is available and working
            - error_message: None if available, otherwise description of the issue
        """
        try:
            # Try running --version or --help with short timeout
            process = await asyncio.create_subprocess_exec(
                self.cli_command,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(process.communicate(), timeout=5.0)
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
            return (False, f"Error checking availability: {str(e)}")

    def _load_conversation_context(self, continuation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation context from continuation_id.

        Reads the conversation thread file and serializes it for CLI invocation.
        Returns None if the thread file doesn't exist or can't be loaded.

        Args:
            continuation_id: Thread ID to load conversation context for

        Returns:
            Dictionary with conversation context or None if not found
        """
        try:
            # Default conversations directory
            conversations_dir = Path.home() / ".modelchorus" / "conversations"
            thread_file = conversations_dir / f"{continuation_id}.json"

            if not thread_file.exists():
                logger.warning(f"Conversation thread not found: {continuation_id}")
                return None

            with open(thread_file, "r") as f:
                thread_data = json.load(f)

            logger.debug(f"Loaded conversation context for {continuation_id}")
            return thread_data

        except Exception as e:
            logger.error(f"Error loading conversation context: {e}")
            return None

    @abstractmethod
    def build_command(self, request: GenerationRequest) -> List[str]:
        """
        Build the CLI command for a generation request.

        Subclasses must implement this to construct the appropriate command
        for their specific CLI tool.

        If request.continuation_id is present, subclasses can use
        self._load_conversation_context(continuation_id) to retrieve
        conversation history for multi-turn conversations.

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
            ProviderUnavailableError: If CLI is not available or cannot be executed
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

        except FileNotFoundError:
            # CLI command not found in PATH
            error_msg = f"CLI command '{self.cli_command}' not found in PATH"
            logger.error(error_msg)
            raise ProviderUnavailableError(
                self.provider_name,
                f"CLI command '{self.cli_command}' not installed or not in PATH",
                suggestions=[
                    f"Install the {self.provider_name} CLI tool",
                    f"Ensure '{self.cli_command}' is in your system PATH",
                    "Or use a different provider with --provider flag",
                ]
            )
        except PermissionError:
            # No permission to execute CLI command
            error_msg = f"Permission denied to execute '{self.cli_command}'"
            logger.error(error_msg)
            raise ProviderUnavailableError(
                self.provider_name,
                f"Permission denied to execute '{self.cli_command}'",
                suggestions=[
                    f"Check file permissions for '{self.cli_command}'",
                    "Or use a different provider with --provider flag",
                ]
            )
        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            raise

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable or permanent.

        Args:
            error: The exception to check

        Returns:
            True if the error is transient and should be retried, False otherwise
        """
        # Permanent errors that should not be retried
        permanent_error_types = (
            ProviderUnavailableError,  # CLI not found, permission denied
            FileNotFoundError,  # Command not found
            PermissionError,  # Permission issues
        )

        if isinstance(error, permanent_error_types):
            return False

        # Check for specific error messages indicating permanent failures
        error_str = str(error).lower()
        permanent_error_patterns = [
            "command not found",
            "permission denied",
            "not found in path",
            "invalid api key",
            "unauthorized",
            "authentication failed",
            "400",  # Bad request
            "401",  # Unauthorized
            "403",  # Forbidden
        ]

        for pattern in permanent_error_patterns:
            if pattern in error_str:
                return False

        # Timeout and network errors are retryable
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using the CLI tool.

        This method orchestrates the generation process:
        1. Build the CLI command
        2. Execute the command with retries (only for transient errors)
        3. Parse the response
        4. Return the result

        Args:
            request: GenerationRequest containing prompt and parameters

        Returns:
            GenerationResponse with generated content

        Raises:
            ProviderUnavailableError: If provider CLI is not available
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

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"Permanent error encountered, not retrying: {e}")
                    raise

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
