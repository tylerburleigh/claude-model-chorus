"""
Context ingestion service for ModelChorus.

Provides controlled file reading with size limits and chunking support
for safe context injection into LLM prompts.

Key Features:
- File size limits to prevent context overflow
- Safe file reading with encoding detection
- Binary file detection and handling
- Path validation and normalization
- Configurable size thresholds
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import chardet

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_MAX_FILE_SIZE_KB = 100
DEFAULT_WARN_FILE_SIZE_KB = 50


class FileTooLargeError(Exception):
    """Raised when a file exceeds the maximum size limit."""
    pass


class BinaryFileError(Exception):
    """Raised when attempting to read a binary file as text."""
    pass


class ContextIngestionService:
    """
    Manages safe file reading for context injection.

    Provides controlled access to file contents with size limits
    and proper error handling to prevent context window overflow.

    Attributes:
        max_file_size_kb: Maximum file size in kilobytes
        warn_file_size_kb: File size threshold for warnings
    """

    def __init__(
        self,
        max_file_size_kb: int = DEFAULT_MAX_FILE_SIZE_KB,
        warn_file_size_kb: int = DEFAULT_WARN_FILE_SIZE_KB,
    ):
        """
        Initialize the context ingestion service.

        Args:
            max_file_size_kb: Maximum file size in KB (default: 100)
            warn_file_size_kb: Size threshold for warnings in KB (default: 50)

        Raises:
            ValueError: If max_file_size_kb <= 0 or warn_file_size_kb > max_file_size_kb
        """
        if max_file_size_kb <= 0:
            raise ValueError("max_file_size_kb must be positive")
        if warn_file_size_kb > max_file_size_kb:
            raise ValueError("warn_file_size_kb cannot exceed max_file_size_kb")

        self.max_file_size_kb = max_file_size_kb
        self.warn_file_size_kb = warn_file_size_kb

    def read_file(self, file_path: str | Path) -> str:
        """
        Read file contents with size validation.

        Args:
            file_path: Path to the file to read

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            FileTooLargeError: If file exceeds size limit
            BinaryFileError: If file appears to be binary
            PermissionError: If file cannot be accessed
            ValueError: If path is invalid
        """
        # Normalize and validate path
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Check file size
        file_size_kb = path.stat().st_size / 1024

        if file_size_kb > self.max_file_size_kb:
            raise FileTooLargeError(
                f"File {path} ({file_size_kb:.1f} KB) exceeds maximum size "
                f"of {self.max_file_size_kb} KB"
            )

        if file_size_kb > self.warn_file_size_kb:
            logger.warning(
                f"File {path} ({file_size_kb:.1f} KB) exceeds warning threshold "
                f"of {self.warn_file_size_kb} KB"
            )

        # Detect encoding and check if binary
        try:
            with open(path, "rb") as f:
                raw_data = f.read(8192)  # Read first 8KB for detection

            detection = chardet.detect(raw_data)
            encoding = detection.get("encoding")
            confidence = detection.get("confidence", 0)

            if encoding is None or confidence < 0.7:
                raise BinaryFileError(f"File appears to be binary: {path}")

        except (IOError, OSError) as e:
            raise PermissionError(f"Cannot access file {path}: {e}")

        # Read file with detected encoding
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return content

        except UnicodeDecodeError as e:
            raise BinaryFileError(f"Failed to decode file {path}: {e}")

    def get_file_info(self, file_path: str | Path) -> dict:
        """
        Get file metadata without reading contents.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata:
                - size_kb: File size in kilobytes
                - exceeds_max: Whether file exceeds max size
                - exceeds_warn: Whether file exceeds warning threshold
                - path: Resolved absolute path

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        file_size_kb = path.stat().st_size / 1024

        return {
            "size_kb": file_size_kb,
            "exceeds_max": file_size_kb > self.max_file_size_kb,
            "exceeds_warn": file_size_kb > self.warn_file_size_kb,
            "path": str(path),
        }

    def can_read_file(self, file_path: str | Path) -> Tuple[bool, Optional[str]]:
        """
        Check if a file can be read without actually reading it.

        Args:
            file_path: Path to check

        Returns:
            Tuple of (can_read: bool, reason: Optional[str])
            If can_read is False, reason contains the error message

        Examples:
            >>> service = ContextIngestionService()
            >>> can_read, reason = service.can_read_file("large_file.txt")
            >>> if not can_read:
            ...     print(f"Cannot read file: {reason}")
        """
        try:
            info = self.get_file_info(file_path)
            if info["exceeds_max"]:
                return False, f"File size ({info['size_kb']:.1f} KB) exceeds maximum"
            return True, None
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return False, str(e)
