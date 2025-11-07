"""
Document processing infrastructure for ModelChorus.

This module provides the core document parsing infrastructure including:
- Base DocumentParser abstract class
- Document models and metadata
- Format detection utilities
- Parser registry system

Document workflows (DIGEST, SYNTHESIZE, REVIEW) build on this foundation.

Example:
    >>> from modelchorus.documents import Document, DocumentFormat, parse_document
    >>>
    >>> # Parse a document with automatic format detection
    >>> doc = await parse_document("example.pdf")
    >>> print(f"Parsed {doc.format.value} document: {len(doc)} characters")
    >>>
    >>> # Register a custom parser
    >>> from modelchorus.documents import DocumentParser, ParserRegistry
    >>>
    >>> @ParserRegistry.register(DocumentFormat.TXT)
    ... class TxtParser(DocumentParser):
    ...     async def parse(self, source):
    ...         # Custom parsing logic
    ...         pass
    ...
    ...     def supports_format(self, format):
    ...         return format == DocumentFormat.TXT
"""

from .models import Document, DocumentFormat, DocumentMetadata
from .parser import DocumentParser, ParserRegistry, parse_document

__all__ = [
    # Core models
    "Document",
    "DocumentFormat",
    "DocumentMetadata",
    # Parser infrastructure
    "DocumentParser",
    "ParserRegistry",
    # Convenience functions
    "parse_document",
]
