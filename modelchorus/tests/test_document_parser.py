"""
Unit tests for document parser infrastructure.

Tests cover:
- Document models (Document, DocumentMetadata, DocumentFormat)
- Format detection (extension-based, MIME type-based)
- Parser base class and registry
- Error handling and edge cases
"""

import pytest
from pathlib import Path
from datetime import datetime

from modelchorus.documents import (
    Document,
    DocumentFormat,
    DocumentMetadata,
    DocumentParser,
    ParserRegistry,
    parse_document,
)


class TestDocumentFormat:
    """Tests for DocumentFormat enum."""

    def test_from_extension_with_dot(self):
        """Test format detection from extension with leading dot."""
        assert DocumentFormat.from_extension(".pdf") == DocumentFormat.PDF
        assert DocumentFormat.from_extension(".txt") == DocumentFormat.TXT
        assert DocumentFormat.from_extension(".md") == DocumentFormat.MARKDOWN
        assert DocumentFormat.from_extension(".json") == DocumentFormat.JSON

    def test_from_extension_without_dot(self):
        """Test format detection from extension without leading dot."""
        assert DocumentFormat.from_extension("pdf") == DocumentFormat.PDF
        assert DocumentFormat.from_extension("txt") == DocumentFormat.TXT
        assert DocumentFormat.from_extension("md") == DocumentFormat.MARKDOWN

    def test_from_extension_case_insensitive(self):
        """Test format detection is case-insensitive."""
        assert DocumentFormat.from_extension(".PDF") == DocumentFormat.PDF
        assert DocumentFormat.from_extension("TXT") == DocumentFormat.TXT
        assert DocumentFormat.from_extension(".Md") == DocumentFormat.MARKDOWN

    def test_from_extension_unknown(self):
        """Test unknown extensions return UNKNOWN."""
        assert DocumentFormat.from_extension(".xyz") == DocumentFormat.UNKNOWN
        assert DocumentFormat.from_extension("unknown") == DocumentFormat.UNKNOWN
        assert DocumentFormat.from_extension("") == DocumentFormat.UNKNOWN


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = DocumentMetadata(format=DocumentFormat.PDF)
        assert metadata.format == DocumentFormat.PDF
        assert metadata.encoding == "utf-8"  # Default value
        assert metadata.custom == {}  # Default empty dict

    def test_metadata_with_all_fields(self):
        """Test metadata with all fields populated."""
        now = datetime.now()
        metadata = DocumentMetadata(
            format=DocumentFormat.TXT,
            size_bytes=1024,
            encoding="ascii",
            created_at=now,
            modified_at=now,
            source_path=Path("/tmp/test.txt"),
            mime_type="text/plain",
            custom={"author": "Test User"}
        )
        assert metadata.format == DocumentFormat.TXT
        assert metadata.size_bytes == 1024
        assert metadata.encoding == "ascii"
        assert metadata.created_at == now
        assert metadata.source_path == Path("/tmp/test.txt")
        assert metadata.custom["author"] == "Test User"


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test basic document creation."""
        metadata = DocumentMetadata(format=DocumentFormat.TXT)
        doc = Document(
            content="Hello world",
            metadata=metadata,
            source="test.txt"
        )
        assert doc.content == "Hello world"
        assert doc.format == DocumentFormat.TXT
        assert doc.source == "test.txt"

    def test_document_length(self):
        """Test document length property."""
        metadata = DocumentMetadata(format=DocumentFormat.TXT)
        doc = Document(content="Hello", metadata=metadata)
        assert len(doc) == 5

    def test_document_repr(self):
        """Test document string representation."""
        metadata = DocumentMetadata(format=DocumentFormat.TXT)
        doc = Document(content="Short content", metadata=metadata, source="test.txt")
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "txt" in repr_str
        assert "test.txt" in repr_str

    def test_document_repr_long_content(self):
        """Test document repr truncates long content."""
        metadata = DocumentMetadata(format=DocumentFormat.TXT)
        long_content = "x" * 100
        doc = Document(content=long_content, metadata=metadata)
        repr_str = repr(doc)
        assert "..." in repr_str  # Content should be truncated


class TestDocumentParserFormatDetection:
    """Tests for DocumentParser format detection."""

    def test_detect_format_from_extension(self):
        """Test format detection from file extension."""
        assert DocumentParser.detect_format("test.pdf") == DocumentFormat.PDF
        assert DocumentParser.detect_format("notes.txt") == DocumentFormat.TXT
        assert DocumentParser.detect_format("readme.md") == DocumentFormat.MARKDOWN
        assert DocumentParser.detect_format("data.json") == DocumentFormat.JSON
        assert DocumentParser.detect_format("page.html") == DocumentFormat.HTML

    def test_detect_format_from_path_object(self):
        """Test format detection from Path object."""
        assert DocumentParser.detect_format(Path("test.pdf")) == DocumentFormat.PDF
        assert DocumentParser.detect_format(Path("/tmp/notes.txt")) == DocumentFormat.TXT
        assert DocumentParser.detect_format(Path("./readme.md")) == DocumentFormat.MARKDOWN

    def test_detect_format_unknown(self):
        """Test unknown format detection."""
        assert DocumentParser.detect_format("unknown.xyz") == DocumentFormat.UNKNOWN
        assert DocumentParser.detect_format("noextension") == DocumentFormat.UNKNOWN

    def test_detect_format_raw_content(self):
        """Test raw content (non-path) returns UNKNOWN."""
        # Raw content without path indicators should return UNKNOWN
        assert DocumentParser.detect_format("just some text") == DocumentFormat.UNKNOWN


class TestParserRegistry:
    """Tests for ParserRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ParserRegistry.clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        ParserRegistry.clear_registry()

    def test_register_parser(self):
        """Test parser registration."""
        @ParserRegistry.register(DocumentFormat.TXT)
        class MockTxtParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock",
                    metadata=DocumentMetadata(format=DocumentFormat.TXT)
                )

            def supports_format(self, format):
                return format == DocumentFormat.TXT

        # Check parser is registered
        formats = ParserRegistry.get_registered_formats()
        assert DocumentFormat.TXT in formats

    def test_get_parser(self):
        """Test retrieving parser from registry."""
        @ParserRegistry.register(DocumentFormat.TXT)
        class MockTxtParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock",
                    metadata=DocumentMetadata(format=DocumentFormat.TXT)
                )

            def supports_format(self, format):
                return format == DocumentFormat.TXT

        # Retrieve parser
        parser = ParserRegistry.get_parser(DocumentFormat.TXT)
        assert parser is not None
        assert isinstance(parser, MockTxtParser)

    def test_get_parser_singleton(self):
        """Test parser instances are singletons."""
        @ParserRegistry.register(DocumentFormat.TXT)
        class MockTxtParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock",
                    metadata=DocumentMetadata(format=DocumentFormat.TXT)
                )

            def supports_format(self, format):
                return format == DocumentFormat.TXT

        # Get parser twice
        parser1 = ParserRegistry.get_parser(DocumentFormat.TXT)
        parser2 = ParserRegistry.get_parser(DocumentFormat.TXT)

        # Should be same instance
        assert parser1 is parser2

    def test_get_parser_unregistered(self):
        """Test getting parser for unregistered format returns None."""
        parser = ParserRegistry.get_parser(DocumentFormat.PDF)
        assert parser is None

    def test_register_non_parser_raises_error(self):
        """Test registering non-DocumentParser class raises TypeError."""
        with pytest.raises(TypeError, match="must inherit from DocumentParser"):
            @ParserRegistry.register(DocumentFormat.TXT)
            class NotAParser:
                pass

    def test_get_registered_formats(self):
        """Test getting list of registered formats."""
        @ParserRegistry.register(DocumentFormat.TXT)
        class MockTxtParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock",
                    metadata=DocumentMetadata(format=DocumentFormat.TXT)
                )

            def supports_format(self, format):
                return format == DocumentFormat.TXT

        @ParserRegistry.register(DocumentFormat.MARKDOWN)
        class MockMdParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock",
                    metadata=DocumentMetadata(format=DocumentFormat.MARKDOWN)
                )

            def supports_format(self, format):
                return format == DocumentFormat.MARKDOWN

        formats = ParserRegistry.get_registered_formats()
        assert len(formats) == 2
        assert DocumentFormat.TXT in formats
        assert DocumentFormat.MARKDOWN in formats


class TestParseDocumentFunction:
    """Tests for parse_document convenience function."""

    def setup_method(self):
        """Clear registry and register mock parser before each test."""
        ParserRegistry.clear_registry()

        @ParserRegistry.register(DocumentFormat.TXT)
        class MockTxtParser(DocumentParser):
            async def parse(self, source):
                return Document(
                    content="mock content",
                    metadata=DocumentMetadata(format=DocumentFormat.TXT),
                    source=str(source)
                )

            def supports_format(self, format):
                return format == DocumentFormat.TXT

    def teardown_method(self):
        """Clear registry after each test."""
        ParserRegistry.clear_registry()

    @pytest.mark.asyncio
    async def test_parse_document_with_auto_detection(self):
        """Test parsing document with automatic format detection."""
        doc = await parse_document("test.txt")
        assert doc.format == DocumentFormat.TXT
        assert doc.content == "mock content"
        assert doc.source == "test.txt"

    @pytest.mark.asyncio
    async def test_parse_document_with_explicit_format(self):
        """Test parsing document with explicit format."""
        doc = await parse_document("test.txt", format=DocumentFormat.TXT)
        assert doc.format == DocumentFormat.TXT
        assert doc.content == "mock content"

    @pytest.mark.asyncio
    async def test_parse_document_unknown_format_raises_error(self):
        """Test parsing document with unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect format"):
            await parse_document("unknown.xyz")

    @pytest.mark.asyncio
    async def test_parse_document_no_parser_raises_error(self):
        """Test parsing document with no registered parser raises ValueError."""
        with pytest.raises(ValueError, match="No parser registered"):
            await parse_document("test.pdf")  # PDF parser not registered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
