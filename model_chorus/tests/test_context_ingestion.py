"""
Tests for the ContextIngestionService.

Covers file reading, size limits, encoding detection, chunking,
and error handling for the context ingestion service.
"""

import pytest
import tempfile
from pathlib import Path
from model_chorus.core.context_ingestion import (
    ContextIngestionService,
    FileTooLargeError,
    BinaryFileError,
    DEFAULT_MAX_FILE_SIZE_KB,
    DEFAULT_WARN_FILE_SIZE_KB,
)


class TestContextIngestionServiceInitialization:
    """Test service initialization and configuration."""

    def test_default_initialization(self):
        """Test service initializes with default values."""
        service = ContextIngestionService()
        assert service.max_file_size_kb == DEFAULT_MAX_FILE_SIZE_KB
        assert service.warn_file_size_kb == DEFAULT_WARN_FILE_SIZE_KB

    def test_custom_initialization(self):
        """Test service initializes with custom values."""
        service = ContextIngestionService(max_file_size_kb=200, warn_file_size_kb=100)
        assert service.max_file_size_kb == 200
        assert service.warn_file_size_kb == 100

    def test_initialization_validation_max_size_zero(self):
        """Test initialization fails with zero max_file_size_kb."""
        with pytest.raises(ValueError, match="max_file_size_kb must be positive"):
            ContextIngestionService(max_file_size_kb=0)

    def test_initialization_validation_max_size_negative(self):
        """Test initialization fails with negative max_file_size_kb."""
        with pytest.raises(ValueError, match="max_file_size_kb must be positive"):
            ContextIngestionService(max_file_size_kb=-10)

    def test_initialization_validation_warn_exceeds_max(self):
        """Test initialization fails when warn_file_size_kb exceeds max_file_size_kb."""
        with pytest.raises(ValueError, match="warn_file_size_kb cannot exceed max_file_size_kb"):
            ContextIngestionService(max_file_size_kb=50, warn_file_size_kb=100)


class TestReadFile:
    """Test basic file reading functionality."""

    def test_read_small_text_file(self):
        """Test reading a small text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            content = service.read_file(temp_file)
            assert content == "Hello, World!"
        finally:
            Path(temp_file).unlink()

    def test_read_file_with_unicode(self):
        """Test reading file with unicode characters."""
        # Create temp file path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_file = f.name

        # Write file with standard text content (chardet works best with regular ASCII/UTF-8 text)
        # Avoid emojis as they can confuse encoding detection
        content = (
            "This is a test file with various characters.\n"
            "Hello, World! Bonjour, Monde! Hola, Mundo!\n"
            "Testing accented characters: café, naïve, résumé.\n"
            "Some more text to ensure proper encoding detection.\n"
        ) * 50
        Path(temp_file).write_text(content, encoding="utf-8")

        try:
            service = ContextIngestionService()
            file_content = service.read_file(temp_file)
            assert "café" in file_content
            assert "naïve" in file_content
            assert "résumé" in file_content
        finally:
            Path(temp_file).unlink()

    def test_read_file_with_multiple_lines(self):
        """Test reading file with multiple lines."""
        test_content = "Line 1\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            content = service.read_file(temp_file)
            assert content == test_content
        finally:
            Path(temp_file).unlink()

    def test_read_file_not_found(self):
        """Test reading non-existent file raises FileNotFoundError."""
        service = ContextIngestionService()
        with pytest.raises(FileNotFoundError, match="File not found"):
            service.read_file("/nonexistent/file.txt")

    def test_read_directory_raises_error(self):
        """Test reading a directory raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ContextIngestionService()
            with pytest.raises(ValueError, match="Path is not a file"):
                service.read_file(temp_dir)

    def test_read_file_too_large(self):
        """Test reading file exceeding max size raises FileTooLargeError."""
        # Create file larger than default max (100KB)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (110 * 1024))  # 110KB
            temp_file = f.name

        try:
            service = ContextIngestionService()
            with pytest.raises(FileTooLargeError, match="exceeds maximum size"):
                service.read_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_read_file_exceeds_warning_threshold(self, caplog):
        """Test file exceeding warning threshold logs warning."""
        # Create file larger than warning threshold (50KB) but smaller than max (100KB)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (60 * 1024))  # 60KB
            temp_file = f.name

        try:
            service = ContextIngestionService()
            content = service.read_file(temp_file)
            assert len(content) == 60 * 1024
            assert "exceeds warning threshold" in caplog.text
        finally:
            Path(temp_file).unlink()

    def test_read_binary_file_raises_error(self):
        """Test reading binary file raises BinaryFileError."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            # Write realistic binary data (like a simple image header or compressed data)
            # Create data that is clearly binary with many null bytes and non-printable characters
            binary_data = (
                bytes([0x89, 0x50, 0x4E, 0x47]) + bytes(range(256)) * 20
            )  # PNG-like header + binary
            f.write(binary_data)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            with pytest.raises(BinaryFileError, match="appears to be binary"):
                service.read_file(temp_file)
        finally:
            Path(temp_file).unlink()


class TestGetFileInfo:
    """Test file metadata retrieval."""

    def test_get_file_info_small_file(self):
        """Test getting info for a small file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Small file")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            info = service.get_file_info(temp_file)

            assert "size_kb" in info
            assert "exceeds_max" in info
            assert "exceeds_warn" in info
            assert "path" in info
            assert info["exceeds_max"] is False
            assert info["exceeds_warn"] is False
        finally:
            Path(temp_file).unlink()

    def test_get_file_info_large_file(self):
        """Test getting info for a large file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (110 * 1024))  # 110KB
            temp_file = f.name

        try:
            service = ContextIngestionService()
            info = service.get_file_info(temp_file)

            assert info["exceeds_max"] is True
            assert info["exceeds_warn"] is True
            assert info["size_kb"] > 100
        finally:
            Path(temp_file).unlink()

    def test_get_file_info_nonexistent_file(self):
        """Test getting info for nonexistent file raises FileNotFoundError."""
        service = ContextIngestionService()
        with pytest.raises(FileNotFoundError):
            service.get_file_info("/nonexistent/file.txt")


class TestCanReadFile:
    """Test file readability checks."""

    def test_can_read_small_file(self):
        """Test can_read_file returns True for small file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Small file")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            can_read, reason = service.can_read_file(temp_file)
            assert can_read is True
            assert reason is None
        finally:
            Path(temp_file).unlink()

    def test_can_read_large_file(self):
        """Test can_read_file returns False for file exceeding max size."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (110 * 1024))  # 110KB
            temp_file = f.name

        try:
            service = ContextIngestionService()
            can_read, reason = service.can_read_file(temp_file)
            assert can_read is False
            assert "exceeds maximum" in reason
        finally:
            Path(temp_file).unlink()

    def test_can_read_nonexistent_file(self):
        """Test can_read_file returns False for nonexistent file."""
        service = ContextIngestionService()
        can_read, reason = service.can_read_file("/nonexistent/file.txt")
        assert can_read is False
        assert reason is not None


class TestReadFileChunked:
    """Test chunked file reading."""

    def test_read_file_chunked_single_chunk(self):
        """Test reading small file returns single chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Small file content")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            chunks = service.read_file_chunked(temp_file, chunk_size_kb=50)
            assert len(chunks) == 1
            assert chunks[0] == "Small file content"
        finally:
            Path(temp_file).unlink()

    def test_read_file_chunked_multiple_chunks(self):
        """Test reading large file returns multiple chunks."""
        # Create a file larger than chunk size
        content = "x" * (60 * 1024)  # 60KB
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService(max_file_size_kb=200)
            chunks = service.read_file_chunked(temp_file, chunk_size_kb=50)

            # Should have at least 2 chunks (60KB / 50KB)
            assert len(chunks) >= 2

            # Verify all chunks combined equal original content
            combined = "".join(chunks)
            assert combined == content
        finally:
            Path(temp_file).unlink()

    def test_read_file_chunked_with_max_chunks(self):
        """Test reading file with max_chunks limit."""
        content = "x" * (100 * 1024)  # 100KB
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService(max_file_size_kb=200)
            chunks = service.read_file_chunked(temp_file, chunk_size_kb=30, max_chunks=2)

            # Should have exactly 2 chunks due to max_chunks limit
            assert len(chunks) == 2
        finally:
            Path(temp_file).unlink()

    def test_read_file_chunked_invalid_chunk_size(self):
        """Test reading file with invalid chunk size raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            with pytest.raises(ValueError, match="chunk_size_kb must be positive"):
                service.read_file_chunked(temp_file, chunk_size_kb=0)
        finally:
            Path(temp_file).unlink()

    def test_read_file_chunked_binary_file(self):
        """Test reading binary file in chunks raises BinaryFileError."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            # Write realistic binary data
            binary_data = bytes([0x89, 0x50, 0x4E, 0x47]) + bytes(range(256)) * 20
            f.write(binary_data)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            with pytest.raises(BinaryFileError):
                service.read_file_chunked(temp_file)
        finally:
            Path(temp_file).unlink()


class TestReadFileLines:
    """Test line-by-line file reading."""

    def test_read_file_lines_all_lines(self):
        """Test reading all lines from a file."""
        content = "Line 1\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            lines = service.read_file_lines(temp_file)
            assert len(lines) == 3
            assert lines[0] == "Line 1"
            assert lines[1] == "Line 2"
            assert lines[2] == "Line 3"
        finally:
            Path(temp_file).unlink()

    def test_read_file_lines_with_max_lines(self):
        """Test reading file with max_lines limit."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            lines = service.read_file_lines(temp_file, max_lines=3)
            assert len(lines) == 3
            assert lines[0] == "Line 1"
            assert lines[2] == "Line 3"
        finally:
            Path(temp_file).unlink()

    def test_read_file_lines_skip_empty(self):
        """Test reading file with skip_empty option."""
        content = "Line 1\n\nLine 3\n\nLine 5"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            lines = service.read_file_lines(temp_file, skip_empty=True)
            assert len(lines) == 3
            assert lines[0] == "Line 1"
            assert lines[1] == "Line 3"
            assert lines[2] == "Line 5"
        finally:
            Path(temp_file).unlink()

    def test_read_file_lines_keep_empty(self):
        """Test reading file without skipping empty lines."""
        content = "Line 1\n\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            service = ContextIngestionService()
            lines = service.read_file_lines(temp_file, skip_empty=False)
            assert len(lines) == 3
            assert lines[0] == "Line 1"
            assert lines[1] == ""
            assert lines[2] == "Line 3"
        finally:
            Path(temp_file).unlink()

    def test_read_file_lines_too_large(self):
        """Test reading large file with lines raises FileTooLargeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (110 * 1024))  # 110KB
            temp_file = f.name

        try:
            service = ContextIngestionService()
            with pytest.raises(FileTooLargeError):
                service.read_file_lines(temp_file)
        finally:
            Path(temp_file).unlink()


class TestPathValidation:
    """Test path validation and normalization."""

    def test_validate_path_with_string(self):
        """Test path validation accepts string paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test")
            temp_file = f.name

        try:
            service = ContextIngestionService()
            content = service.read_file(str(temp_file))
            assert content == "Test"
        finally:
            Path(temp_file).unlink()

    def test_validate_path_with_pathlib(self):
        """Test path validation accepts Path objects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test")
            temp_file = Path(f.name)

        try:
            service = ContextIngestionService()
            content = service.read_file(temp_file)
            assert content == "Test"
        finally:
            temp_file.unlink()


class TestCustomSizeLimits:
    """Test service with custom size limits."""

    def test_custom_max_size_allows_larger_files(self):
        """Test custom max_file_size_kb allows reading larger files."""
        # Create 150KB file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (150 * 1024))
            temp_file = f.name

        try:
            # Default service (100KB max) should fail
            default_service = ContextIngestionService()
            with pytest.raises(FileTooLargeError):
                default_service.read_file(temp_file)

            # Custom service (200KB max) should succeed
            custom_service = ContextIngestionService(max_file_size_kb=200)
            content = custom_service.read_file(temp_file)
            assert len(content) == 150 * 1024
        finally:
            Path(temp_file).unlink()

    def test_custom_warn_threshold(self, caplog):
        """Test custom warn_file_size_kb threshold."""
        # Create 30KB file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * (30 * 1024))
            temp_file = f.name

        try:
            # Service with 20KB warning threshold should warn
            service = ContextIngestionService(max_file_size_kb=100, warn_file_size_kb=20)
            content = service.read_file(temp_file)
            assert "exceeds warning threshold" in caplog.text
        finally:
            Path(temp_file).unlink()
