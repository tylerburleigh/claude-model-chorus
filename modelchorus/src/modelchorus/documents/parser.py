"""
Document parser base class and format detection for ModelChorus.

This module provides:
- DocumentParser: Abstract base class for format-specific parsers
- Format detection utilities (extension-based, MIME type-based)
- ParserRegistry: Registry system for format-specific parser implementations

All document parsers must inherit from DocumentParser and implement
the abstract methods for parsing and format support.
"""

import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Type, Dict, Callable
import logging

from .models import Document, DocumentFormat, DocumentMetadata


logger = logging.getLogger(__name__)


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.

    All format-specific parsers (PDF, DOCX, HTML, etc.) must inherit from
    this class and implement the abstract methods. This ensures a consistent
    interface for document parsing across all formats.

    The parser follows an async pattern to support large files and I/O-bound
    operations without blocking.

    Example:
        >>> class TxtParser(DocumentParser):
        ...     async def parse(self, source):
        ...         # Implementation
        ...         pass
        ...
        ...     def supports_format(self, format):
        ...         return format == DocumentFormat.TXT
    """

    @abstractmethod
    async def parse(self, source: Union[str, Path]) -> Document:
        """
        Parse a document from file path or raw content.

        This method must be implemented by all subclasses to provide
        format-specific parsing logic.

        Args:
            source: File path (str or Path) or raw document content (str)

        Returns:
            Document object with parsed content and metadata

        Raises:
            FileNotFoundError: If source is a path and file doesn't exist
            ValueError: If source format is not supported by this parser
            IOError: If file cannot be read

        Example:
            >>> parser = TxtParser()
            >>> doc = await parser.parse("example.txt")
            >>> print(doc.content)
        """
        raise NotImplementedError("Subclasses must implement parse()")

    @abstractmethod
    def supports_format(self, format: DocumentFormat) -> bool:
        """
        Check if this parser supports the given format.

        Args:
            format: DocumentFormat to check

        Returns:
            True if this parser can handle the format, False otherwise

        Example:
            >>> parser = TxtParser()
            >>> parser.supports_format(DocumentFormat.TXT)
            True
            >>> parser.supports_format(DocumentFormat.PDF)
            False
        """
        raise NotImplementedError("Subclasses must implement supports_format()")

    @staticmethod
    def detect_format(source: Union[str, Path]) -> DocumentFormat:
        """
        Auto-detect document format from file path or content.

        Uses multiple detection strategies:
        1. File extension (most reliable for file paths)
        2. MIME type detection (fallback)

        Args:
            source: File path or raw content to analyze

        Returns:
            DocumentFormat enum value (UNKNOWN if cannot detect)

        Example:
            >>> DocumentParser.detect_format("document.pdf")
            DocumentFormat.PDF
            >>> DocumentParser.detect_format(Path("notes.md"))
            DocumentFormat.MARKDOWN
            >>> DocumentParser.detect_format("unknown.xyz")
            DocumentFormat.UNKNOWN
        """
        # Convert to Path if string
        if isinstance(source, str):
            # Check if it looks like a file path
            if '/' in source or '\\' in source or '.' in source:
                source_path = Path(source)
            else:
                # Probably raw content, not a path
                logger.debug("Source appears to be raw content, cannot detect format")
                return DocumentFormat.UNKNOWN
        else:
            source_path = source

        # Strategy 1: Extension-based detection
        if source_path.suffix:
            format_from_ext = DocumentFormat.from_extension(source_path.suffix)
            if format_from_ext != DocumentFormat.UNKNOWN:
                logger.debug(f"Detected format {format_from_ext.value} from extension {source_path.suffix}")
                return format_from_ext

        # Strategy 2: MIME type detection
        mime_type, _ = mimetypes.guess_type(str(source_path))
        if mime_type:
            format_from_mime = DocumentParser._format_from_mime(mime_type)
            if format_from_mime != DocumentFormat.UNKNOWN:
                logger.debug(f"Detected format {format_from_mime.value} from MIME type {mime_type}")
                return format_from_mime

        logger.warning(f"Could not detect format for {source_path}")
        return DocumentFormat.UNKNOWN

    @staticmethod
    def _format_from_mime(mime_type: str) -> DocumentFormat:
        """
        Map MIME type to DocumentFormat.

        Args:
            mime_type: MIME type string (e.g., "application/pdf")

        Returns:
            DocumentFormat enum value, or UNKNOWN if not recognized
        """
        mime_to_format = {
            "application/pdf": DocumentFormat.PDF,
            "text/plain": DocumentFormat.TXT,
            "text/markdown": DocumentFormat.MARKDOWN,
            "application/json": DocumentFormat.JSON,
            "text/html": DocumentFormat.HTML,
            "application/xhtml+xml": DocumentFormat.HTML,
            "application/xml": DocumentFormat.XML,
            "text/xml": DocumentFormat.XML,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.DOCX,
            "application/rtf": DocumentFormat.RTF,
            "text/csv": DocumentFormat.CSV,
        }

        return mime_to_format.get(mime_type, DocumentFormat.UNKNOWN)


class ParserRegistry:
    """
    Registry system for format-specific document parsers.

    This provides a centralized registry where format-specific parsers
    can register themselves. The registry then provides parser instances
    for a given document format.

    This follows the plugin/registry pattern, allowing new parsers to be
    added without modifying existing code.

    Example:
        >>> # Register a parser
        >>> @ParserRegistry.register(DocumentFormat.TXT)
        ... class TxtParser(DocumentParser):
        ...     pass
        ...
        >>> # Get parser for format
        >>> parser = ParserRegistry.get_parser(DocumentFormat.TXT)
        >>> isinstance(parser, TxtParser)
        True
    """

    _parsers: Dict[DocumentFormat, Type[DocumentParser]] = {}
    _instances: Dict[DocumentFormat, DocumentParser] = {}

    @classmethod
    def register(cls, format: DocumentFormat) -> Callable[[Type[DocumentParser]], Type[DocumentParser]]:
        """
        Decorator for registering format-specific parsers.

        This decorator registers a parser class for a specific document format.
        The parser class must inherit from DocumentParser.

        Args:
            format: DocumentFormat this parser handles

        Returns:
            Decorator function

        Example:
            >>> @ParserRegistry.register(DocumentFormat.TXT)
            ... class TxtParser(DocumentParser):
            ...     async def parse(self, source):
            ...         # Implementation
            ...         pass
            ...
            ...     def supports_format(self, format):
            ...         return format == DocumentFormat.TXT
        """
        def decorator(parser_class: Type[DocumentParser]) -> Type[DocumentParser]:
            if not issubclass(parser_class, DocumentParser):
                raise TypeError(f"{parser_class.__name__} must inherit from DocumentParser")

            cls._parsers[format] = parser_class
            logger.info(f"Registered parser {parser_class.__name__} for format {format.value}")
            return parser_class

        return decorator

    @classmethod
    def get_parser(cls, format: DocumentFormat) -> Optional[DocumentParser]:
        """
        Get parser instance for a document format.

        Returns a singleton instance of the parser for the given format.
        If no parser is registered for the format, returns None.

        Args:
            format: DocumentFormat to get parser for

        Returns:
            DocumentParser instance, or None if no parser registered

        Example:
            >>> parser = ParserRegistry.get_parser(DocumentFormat.TXT)
            >>> if parser:
            ...     doc = await parser.parse("example.txt")
        """
        if format not in cls._parsers:
            logger.warning(f"No parser registered for format {format.value}")
            return None

        # Return singleton instance (create if doesn't exist)
        if format not in cls._instances:
            parser_class = cls._parsers[format]
            cls._instances[format] = parser_class()
            logger.debug(f"Created parser instance for format {format.value}")

        return cls._instances[format]

    @classmethod
    def get_registered_formats(cls) -> list[DocumentFormat]:
        """
        Get list of all registered document formats.

        Returns:
            List of DocumentFormat values that have parsers registered

        Example:
            >>> formats = ParserRegistry.get_registered_formats()
            >>> DocumentFormat.TXT in formats
            True
        """
        return list(cls._parsers.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered parsers.

        This is primarily useful for testing. In production, parsers should
        remain registered for the lifetime of the application.
        """
        cls._parsers.clear()
        cls._instances.clear()
        logger.debug("Cleared parser registry")


async def parse_document(source: Union[str, Path], format: Optional[DocumentFormat] = None) -> Document:
    """
    Convenience function to parse a document with automatic format detection.

    This function detects the document format (if not provided), retrieves
    the appropriate parser from the registry, and parses the document.

    Args:
        source: File path or raw content to parse
        format: Optional DocumentFormat (auto-detected if not provided)

    Returns:
        Parsed Document object

    Raises:
        ValueError: If format cannot be detected or no parser registered
        FileNotFoundError: If source is a path and file doesn't exist

    Example:
        >>> # Automatic format detection
        >>> doc = await parse_document("example.pdf")
        >>>
        >>> # Explicit format
        >>> doc = await parse_document("example.txt", format=DocumentFormat.TXT)
    """
    # Detect format if not provided
    if format is None:
        format = DocumentParser.detect_format(source)

    if format == DocumentFormat.UNKNOWN:
        raise ValueError(f"Cannot detect format for source: {source}")

    # Get parser from registry
    parser = ParserRegistry.get_parser(format)
    if parser is None:
        raise ValueError(f"No parser registered for format: {format.value}")

    # Parse document
    return await parser.parse(source)
