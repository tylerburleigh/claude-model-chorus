"""
Document models and data structures for ModelChorus document processing.

This module defines the core data structures used throughout document workflows:
- DocumentFormat: Enumeration of supported document formats
- DocumentMetadata: Metadata about documents (size, encoding, timestamps)
- Document: Primary document representation with content and metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any


class DocumentFormat(Enum):
    """
    Supported document formats for parsing and processing.

    These formats are detected automatically by the DocumentParser
    and determine which format-specific parser is used.
    """
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"
    JSON = "json"
    HTML = "html"
    XML = "xml"
    DOCX = "docx"
    RTF = "rtf"
    CSV = "csv"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, extension: str) -> "DocumentFormat":
        """
        Get DocumentFormat from file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            DocumentFormat enum value, or UNKNOWN if not recognized

        Example:
            >>> DocumentFormat.from_extension(".pdf")
            DocumentFormat.PDF
            >>> DocumentFormat.from_extension("txt")
            DocumentFormat.TXT
        """
        # Normalize extension (remove leading dot, lowercase)
        ext = extension.lstrip('.').lower()

        # Map common extensions to formats
        for format in cls:
            if format.value == ext:
                return format

        return cls.UNKNOWN


@dataclass
class DocumentMetadata:
    """
    Metadata about a document.

    This captures information about the document's source, size, encoding,
    and timestamps. Useful for tracking, debugging, and optimization.

    Attributes:
        format: Detected document format
        size_bytes: File size in bytes (if from file)
        encoding: Text encoding (utf-8, ascii, etc.)
        created_at: Document creation timestamp
        modified_at: Last modification timestamp
        source_path: Original file path (if from file)
        mime_type: MIME type (e.g., "application/pdf")
        custom: Additional custom metadata
    """
    format: DocumentFormat
    size_bytes: Optional[int] = None
    encoding: Optional[str] = "utf-8"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    source_path: Optional[Path] = None
    mime_type: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Core document representation for ModelChorus.

    This is the primary data structure used throughout document processing
    workflows (DIGEST, SYNTHESIZE, REVIEW). It contains the parsed document
    content along with metadata about the document.

    Attributes:
        content: Parsed document content (text or structured data)
        metadata: Document metadata (format, size, timestamps, etc.)
        format: Shortcut to metadata.format
        source: Source identifier (file path or description)

    Example:
        >>> doc = Document(
        ...     content="Hello world",
        ...     metadata=DocumentMetadata(format=DocumentFormat.TXT),
        ...     source="example.txt"
        ... )
        >>> doc.format
        DocumentFormat.TXT
    """
    content: str
    metadata: DocumentMetadata
    source: Optional[str] = None

    @property
    def format(self) -> DocumentFormat:
        """Convenience property to access document format."""
        return self.metadata.format

    def __len__(self) -> int:
        """Return length of document content."""
        return len(self.content)

    def __repr__(self) -> str:
        """String representation of document."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(format={self.format.value}, source={self.source}, length={len(self)}, preview='{content_preview}')"
