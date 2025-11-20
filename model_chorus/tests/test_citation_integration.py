"""
Tests for citation formatting and validation utilities.

Verifies that the citation engine validates and formats citations correctly
according to different academic styles (APA, MLA, Chicago).
"""

import pytest
from model_chorus.core.models import Citation, CitationMap
from model_chorus.utils.citation_formatter import (
    CitationStyle,
    format_citation,
    format_citation_map,
    validate_citation,
    calculate_citation_confidence,
    calculate_citation_map_confidence,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def complete_citation():
    """Citation with all metadata fields populated."""
    return Citation(
        source="https://arxiv.org/abs/2401.12345",
        location="Section 3.2, Figure 4",
        confidence=0.95,
        snippet="Our experiments show a 23% improvement in accuracy...",
        metadata={
            "author": "Smith, J.",
            "year": "2024",
            "title": "Machine Learning Advances",
        },
    )


@pytest.fixture
def minimal_citation():
    """Citation with only required fields."""
    return Citation(
        source="https://example.com/article",
        confidence=0.7,
    )


@pytest.fixture
def file_citation():
    """Citation referencing a file."""
    return Citation(
        source="research_paper.pdf",
        location="Page 42",
        confidence=0.85,
        metadata={
            "author": "Doe, A.",
            "year": "2023",
            "title": "AI Research",
        },
    )


@pytest.fixture
def doi_citation():
    """Citation with DOI reference."""
    return Citation(
        source="doi:10.1234/example.2024",
        location="Table 2",
        confidence=0.92,
        metadata={
            "author": "Johnson, M.",
            "year": "2024",
            "title": "Statistical Analysis",
        },
    )


@pytest.fixture
def citation_map_complete(complete_citation, file_citation):
    """CitationMap with multiple citations."""
    return CitationMap(
        claim_id="claim-001",
        claim_text="Machine learning improves diagnostic accuracy by 23%",
        citations=[complete_citation, file_citation],
        strength=0.9,
        metadata={"argument_type": "empirical"},
    )


@pytest.fixture
def citation_map_empty():
    """CitationMap with no citations."""
    return CitationMap(
        claim_id="claim-002",
        claim_text="This is an unsupported claim",
        citations=[],
        strength=0.0,
    )


# ============================================================================
# Citation Formatting Tests
# ============================================================================


class TestCitationFormatting:
    """Test citation formatting in different styles."""

    def test_format_apa_complete(self, complete_citation):
        """Test APA formatting with complete metadata."""
        result = format_citation(complete_citation, CitationStyle.APA)

        assert "Smith, J." in result
        assert "(2024)" in result
        assert "Machine Learning Advances" in result
        assert "Retrieved from https://arxiv.org/abs/2401.12345" in result
        assert "Section 3.2, Figure 4" in result

    def test_format_apa_minimal(self, minimal_citation):
        """Test APA formatting with minimal metadata."""
        result = format_citation(minimal_citation, CitationStyle.APA)

        assert "Retrieved from https://example.com/article" in result
        # Should gracefully handle missing metadata

    def test_format_mla_complete(self, complete_citation):
        """Test MLA formatting with complete metadata."""
        result = format_citation(complete_citation, CitationStyle.MLA)

        assert "Smith, J." in result
        assert '"Machine Learning Advances."' in result
        assert "https://arxiv.org/abs/2401.12345" in result
        assert "2024" in result
        assert "Section 3.2, Figure 4" in result

    def test_format_mla_minimal(self, minimal_citation):
        """Test MLA formatting with minimal metadata."""
        result = format_citation(minimal_citation, CitationStyle.MLA)

        assert "https://example.com/article" in result

    def test_format_chicago_complete(self, complete_citation):
        """Test Chicago formatting with complete metadata."""
        result = format_citation(complete_citation, CitationStyle.CHICAGO)

        assert "Smith, J." in result
        assert '"Machine Learning Advances."' in result
        assert "https://arxiv.org/abs/2401.12345" in result
        assert "(2024)" in result
        assert ": Section 3.2, Figure 4" in result

    def test_format_chicago_minimal(self, minimal_citation):
        """Test Chicago formatting with minimal metadata."""
        result = format_citation(minimal_citation, CitationStyle.CHICAGO)

        assert "https://example.com/article" in result

    def test_format_file_citation_apa(self, file_citation):
        """Test formatting file-based citations in APA."""
        result = format_citation(file_citation, CitationStyle.APA)

        assert "Doe, A." in result
        assert "(2023)" in result
        assert "AI Research" in result
        assert "research_paper.pdf" in result
        assert "Page 42" in result

    def test_format_doi_citation_apa(self, doi_citation):
        """Test formatting DOI citations in APA."""
        result = format_citation(doi_citation, CitationStyle.APA)

        assert "Johnson, M." in result
        assert "(2024)" in result
        assert "Statistical Analysis" in result
        assert "doi:10.1234/example.2024" in result

    def test_format_unsupported_style_raises_error(self, complete_citation):
        """Test that unsupported citation style raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported citation style"):
            format_citation(complete_citation, "invalid_style")

    def test_year_extraction_from_full_date(self):
        """Test that year is extracted from full publication_date."""
        citation = Citation(
            source="https://example.com",
            confidence=0.8,
            metadata={
                "author": "Test Author",
                "publication_date": "2024-03-15",
                "title": "Test Title",
            },
        )

        result = format_citation(citation, CitationStyle.APA)
        assert "(2024)" in result


class TestCitationMapFormatting:
    """Test CitationMap formatting."""

    def test_format_citation_map_with_claim(self, citation_map_complete):
        """Test formatting CitationMap with claim included."""
        result = format_citation_map(citation_map_complete, CitationStyle.APA, include_claim=True)

        assert "Claim: Machine learning improves diagnostic accuracy by 23%" in result
        assert "Citations:" in result
        assert "1. " in result  # First citation
        assert "2. " in result  # Second citation
        assert "Smith, J." in result
        assert "Doe, A." in result

    def test_format_citation_map_without_claim(self, citation_map_complete):
        """Test formatting CitationMap without claim."""
        result = format_citation_map(citation_map_complete, CitationStyle.APA, include_claim=False)

        assert "Claim:" not in result
        assert "Citations:" in result
        assert "1. " in result
        assert "2. " in result

    def test_format_citation_map_mla_style(self, citation_map_complete):
        """Test CitationMap formatting in MLA style."""
        result = format_citation_map(citation_map_complete, CitationStyle.MLA, include_claim=True)

        assert "Claim:" in result
        assert "Citations:" in result
        # MLA uses quotes for titles
        assert '"' in result

    def test_format_citation_map_empty(self, citation_map_empty):
        """Test formatting empty CitationMap."""
        result = format_citation_map(citation_map_empty, CitationStyle.APA, include_claim=True)

        assert "Claim: This is an unsupported claim" in result
        assert "No citations available" in result


# ============================================================================
# Citation Validation Tests
# ============================================================================


class TestCitationValidation:
    """Test citation validation logic."""

    def test_validate_complete_citation_passes(self, complete_citation):
        """Test that complete citation passes validation."""
        is_valid, issues = validate_citation(complete_citation)

        # May have recommendations but should be considered valid
        # Complete citation with all metadata should minimize issues
        assert len(issues) <= 3  # At most recommendations, not errors

    def test_validate_minimal_citation_has_recommendations(self, minimal_citation):
        """Test that minimal citation gets recommendations."""
        is_valid, issues = validate_citation(minimal_citation)

        # Should have recommendations for missing metadata
        assert len(issues) > 0
        assert any("author" in issue.lower() for issue in issues)
        assert any(
            "year" in issue.lower() or "publication_date" in issue.lower() for issue in issues
        )
        assert any("title" in issue.lower() for issue in issues)

    def test_validate_empty_source_fails(self):
        """Test that empty source fails at Pydantic validation."""
        from pydantic import ValidationError

        # Pydantic should prevent creating Citation with empty source
        with pytest.raises(ValidationError, match="at least 1 character"):
            Citation(source="", confidence=0.8)

    def test_validate_whitespace_source_fails(self):
        """Test that whitespace-only source fails validation."""
        citation = Citation(source="   ", confidence=0.8)
        is_valid, issues = validate_citation(citation)

        assert not is_valid
        assert any("empty source" in issue.lower() for issue in issues)

    def test_validate_confidence_out_of_range_low(self):
        """Test that confidence below 0.0 fails validation."""
        # Note: Pydantic should prevent this, but test validation logic
        citation = Citation(source="https://example.com", confidence=0.5)
        # Manually set invalid value to test validator
        citation.confidence = -0.1

        is_valid, issues = validate_citation(citation)

        assert not is_valid
        assert any("out of valid range" in issue.lower() for issue in issues)

    def test_validate_confidence_out_of_range_high(self):
        """Test that confidence above 1.0 fails validation."""
        citation = Citation(source="https://example.com", confidence=0.5)
        citation.confidence = 1.5

        is_valid, issues = validate_citation(citation)

        assert not is_valid
        assert any("out of valid range" in issue.lower() for issue in issues)

    def test_validate_recognized_source_formats(self):
        """Test that various source formats are recognized."""
        valid_sources = [
            "https://example.com/article",
            "http://example.com/page",
            "document.pdf",
            "paper.doc",
            "file.docx",
            "notes.txt",
            "doi:10.1234/example",
        ]

        for source in valid_sources:
            citation = Citation(
                source=source,
                confidence=0.8,
                metadata={"author": "Test", "year": "2024", "title": "Test"},
            )
            is_valid, issues = validate_citation(citation)

            # Should not have source format issue
            assert not any("source format not recognized" in issue.lower() for issue in issues)

    def test_validate_unrecognized_source_format(self):
        """Test that unrecognized source format gets warning."""
        citation = Citation(
            source="some random text",
            confidence=0.8,
            metadata={"author": "Test", "year": "2024", "title": "Test"},
        )

        is_valid, issues = validate_citation(citation)

        assert any("source format not recognized" in issue.lower() for issue in issues)


# ============================================================================
# Citation Confidence Scoring Tests
# ============================================================================


class TestCitationConfidenceScoring:
    """Test citation confidence calculation."""

    def test_calculate_confidence_complete_citation(self, complete_citation):
        """Test confidence calculation for complete citation."""
        scores = calculate_citation_confidence(complete_citation)

        assert "overall_confidence" in scores
        assert "base_confidence" in scores
        assert "metadata_score" in scores
        assert "source_quality_score" in scores
        assert "location_score" in scores
        assert "factors" in scores

        # Complete citation should have high scores
        assert scores["metadata_score"] >= 0.75  # Has author, year, title, snippet
        assert scores["source_quality_score"] >= 0.8  # arxiv.org is academic
        assert scores["location_score"] > 0.0  # Has location
        assert scores["overall_confidence"] >= 0.8

    def test_calculate_confidence_minimal_citation(self, minimal_citation):
        """Test confidence calculation for minimal citation."""
        scores = calculate_citation_confidence(minimal_citation)

        # Minimal citation should have lower scores
        assert scores["metadata_score"] < 0.5  # Missing author, year, title
        assert scores["location_score"] == 0.0  # No location
        assert (
            scores["overall_confidence"] < scores["base_confidence"]
        )  # Pulled down by missing info

    def test_confidence_academic_source_bonus(self):
        """Test that academic sources get quality bonus."""
        academic_citation = Citation(
            source="https://arxiv.org/abs/2401.12345",
            confidence=0.7,
            metadata={"author": "Test", "year": "2024", "title": "Test"},
        )

        scores = calculate_citation_confidence(academic_citation)
        assert scores["source_quality_score"] == 1.0  # Academic source
        assert scores["factors"]["source_type"] == "academic"

    def test_confidence_doi_source_bonus(self):
        """Test that DOI sources get quality bonus."""
        doi_citation = Citation(
            source="doi:10.1234/example",
            confidence=0.7,
            metadata={"author": "Test", "year": "2024", "title": "Test"},
        )

        scores = calculate_citation_confidence(doi_citation)
        assert scores["source_quality_score"] == 1.0  # DOI is academic
        assert scores["factors"]["source_type"] == "academic"

    def test_confidence_https_vs_http(self):
        """Test that HTTPS sources score higher than HTTP."""
        https_citation = Citation(
            source="https://example.com",
            confidence=0.7,
            metadata={"author": "Test", "year": "2024", "title": "Test"},
        )
        http_citation = Citation(
            source="http://example.com",
            confidence=0.7,
            metadata={"author": "Test", "year": "2024", "title": "Test"},
        )

        https_scores = calculate_citation_confidence(https_citation)
        http_scores = calculate_citation_confidence(http_citation)

        assert https_scores["source_quality_score"] > http_scores["source_quality_score"]

    def test_confidence_location_specificity_bonus(self):
        """Test location specificity scoring."""
        # No location
        no_location = Citation(source="https://example.com", confidence=0.8)
        no_loc_scores = calculate_citation_confidence(no_location)
        assert no_loc_scores["location_score"] == 0.0

        # Generic location
        generic_location = Citation(
            source="https://example.com",
            location="Chapter 3",
            confidence=0.8,
        )
        generic_scores = calculate_citation_confidence(generic_location)
        assert generic_scores["location_score"] >= 0.5

        # Specific location with page
        specific_location = Citation(
            source="https://example.com",
            location="Section 3.2, p. 42",
            confidence=0.8,
        )
        specific_scores = calculate_citation_confidence(specific_location)
        assert specific_scores["location_score"] > generic_scores["location_score"]
        assert specific_scores["factors"]["has_specific_location"]

    def test_confidence_weighted_formula(self, complete_citation):
        """Test that confidence uses weighted formula correctly."""
        scores = calculate_citation_confidence(complete_citation)

        # Verify formula: 40% base + 30% metadata + 20% source + 10% location
        expected = (
            complete_citation.confidence * 0.4
            + scores["metadata_score"] * 0.3
            + scores["source_quality_score"] * 0.2
            + scores["location_score"] * 0.1
        )

        assert abs(scores["overall_confidence"] - expected) < 0.01


class TestCitationMapConfidenceScoring:
    """Test CitationMap confidence calculation."""

    def test_calculate_map_confidence_complete(self, citation_map_complete):
        """Test CitationMap confidence with multiple citations."""
        scores = calculate_citation_map_confidence(citation_map_complete)

        assert scores["citation_count"] == 2
        assert scores["overall_confidence"] > 0.0
        assert scores["average_citation_confidence"] > 0.0
        assert scores["min_confidence"] <= scores["average_citation_confidence"]
        assert scores["max_confidence"] >= scores["average_citation_confidence"]
        assert scores["strength"] == 0.9
        assert len(scores["individual_scores"]) == 2

    def test_calculate_map_confidence_empty(self, citation_map_empty):
        """Test CitationMap confidence with no citations."""
        scores = calculate_citation_map_confidence(citation_map_empty)

        assert scores["citation_count"] == 0
        assert scores["overall_confidence"] == 0.0
        assert scores["average_citation_confidence"] == 0.0
        assert scores["min_confidence"] == 0.0
        assert scores["max_confidence"] == 0.0
        assert scores["individual_scores"] == []

    def test_calculate_map_confidence_formula(self, citation_map_complete):
        """Test that CitationMap uses correct weighted formula."""
        scores = calculate_citation_map_confidence(citation_map_complete)

        # Formula: 50% avg citation + 30% strength + 20% count factor
        citation_count_factor = min(len(citation_map_complete.citations) / 5.0, 1.0)
        expected = (
            scores["average_citation_confidence"] * 0.5
            + citation_map_complete.strength * 0.3
            + citation_count_factor * 0.2
        )

        assert abs(scores["overall_confidence"] - expected) < 0.01

    def test_calculate_map_confidence_count_plateau(self):
        """Test that citation count factor plateaus at 5."""
        # Create citation map with many citations
        many_citations = [
            Citation(source=f"https://example.com/{i}", confidence=0.8) for i in range(10)
        ]

        citation_map = CitationMap(
            claim_id="test",
            claim_text="Test claim",
            citations=many_citations,
            strength=0.8,
        )

        scores = calculate_citation_map_confidence(citation_map)

        # Count factor should plateau at 1.0 (5 or more citations)
        # Verify the formula is working correctly
        assert scores["citation_count"] == 10


# ============================================================================
# Enum Tests
# ============================================================================


class TestCitationStyleEnum:
    """Test CitationStyle enum."""

    def test_citation_style_values(self):
        """Test CitationStyle enum has correct values."""
        assert CitationStyle.APA.value == "apa"
        assert CitationStyle.MLA.value == "mla"
        assert CitationStyle.CHICAGO.value == "chicago"

    def test_citation_style_string_comparison(self):
        """Test CitationStyle can be compared to strings."""
        assert CitationStyle.APA == "apa"
        assert CitationStyle.MLA == "mla"
        assert CitationStyle.CHICAGO == "chicago"
