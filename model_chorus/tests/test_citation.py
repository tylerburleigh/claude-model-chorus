"""
Unit tests for Citation and CitationMap models.

Tests verify citation tracking functionality including:
- Citation model creation, validation, and field constraints
- CitationMap model for claim-to-citation mapping
- Model serialization and deserialization
- Edge cases and validation errors
- Integration scenarios for evidence tracking
"""

import json

import pytest
from pydantic import ValidationError

from model_chorus.core.models import Citation, CitationMap


class TestCitation:
    """Test suite for Citation model."""

    def test_citation_creation(self):
        """Test basic citation creation with all fields."""
        citation = Citation(
            source="https://arxiv.org/abs/2401.12345",
            location="Section 3.2, Figure 4",
            confidence=0.95,
            snippet="Our experiments show a 23% improvement in accuracy...",
            metadata={
                "author": "Smith et al.",
                "publication_date": "2024-01-15",
                "citation_type": "academic_paper",
            },
        )

        assert citation.source == "https://arxiv.org/abs/2401.12345"
        assert citation.location == "Section 3.2, Figure 4"
        assert citation.confidence == 0.95
        assert (
            citation.snippet == "Our experiments show a 23% improvement in accuracy..."
        )
        assert citation.metadata["author"] == "Smith et al."
        assert citation.metadata["publication_date"] == "2024-01-15"
        assert citation.metadata["citation_type"] == "academic_paper"

    def test_citation_minimal_creation(self):
        """Test citation with only required fields."""
        citation = Citation(source="https://example.com/doc", confidence=0.8)

        assert citation.source == "https://example.com/doc"
        assert citation.confidence == 0.8
        assert citation.location is None
        assert citation.snippet is None
        assert citation.metadata == {}

    def test_citation_empty_source(self):
        """Test that empty source fails validation."""
        with pytest.raises(ValidationError):
            Citation(source="", confidence=0.5)  # min_length=1 constraint

    def test_citation_confidence_bounds(self):
        """Test confidence value bounds (0.0-1.0)."""
        # Valid boundary values
        citation_zero = Citation(source="test", confidence=0.0)
        assert citation_zero.confidence == 0.0

        citation_one = Citation(source="test", confidence=1.0)
        assert citation_one.confidence == 1.0

        citation_mid = Citation(source="test", confidence=0.5)
        assert citation_mid.confidence == 0.5

        # Invalid: confidence < 0
        with pytest.raises(ValidationError):
            Citation(source="test", confidence=-0.1)

        # Invalid: confidence > 1
        with pytest.raises(ValidationError):
            Citation(source="test", confidence=1.1)

    def test_citation_various_source_types(self):
        """Test citations from different source types."""
        sources = [
            ("https://arxiv.org/abs/2401.12345", "URL"),
            ("/path/to/document.pdf", "File path"),
            ("document-id-12345", "Document ID"),
            ("ISBN:978-0-123456-78-9", "ISBN"),
            ("DOI:10.1000/xyz123", "DOI"),
        ]

        for source, source_type in sources:
            citation = Citation(source=source, confidence=0.9)
            assert citation.source == source

    def test_citation_location_formats(self):
        """Test various location format styles."""
        locations = [
            "Section 3.2",
            "Page 45",
            "Line 123",
            "Paragraph 7",
            "Figure 4",
            "Table 2",
            "00:15:30 (timestamp)",
            "Chapter 5, Section 2.1",
        ]

        for location in locations:
            citation = Citation(source="test", location=location, confidence=0.8)
            assert citation.location == location

    def test_citation_metadata_flexibility(self):
        """Test that metadata accepts arbitrary key-value pairs."""
        citation = Citation(
            source="test",
            confidence=0.9,
            metadata={
                "author": "Jane Doe",
                "publication_date": "2024-01-15",
                "citation_type": "academic_paper",
                "peer_reviewed": True,
                "impact_factor": 4.5,
                "keywords": ["machine learning", "optimization"],
                "nested": {"journal": "Nature", "volume": 123, "issue": 4},
            },
        )

        assert citation.metadata["author"] == "Jane Doe"
        assert citation.metadata["peer_reviewed"] is True
        assert citation.metadata["impact_factor"] == 4.5
        assert len(citation.metadata["keywords"]) == 2
        assert citation.metadata["nested"]["journal"] == "Nature"

    def test_citation_serialization(self):
        """Test citation serialization to dict."""
        citation = Citation(
            source="https://example.com",
            location="Page 10",
            confidence=0.85,
            snippet="Important finding here",
            metadata={"author": "John Doe"},
        )

        data = citation.model_dump()

        assert data["source"] == "https://example.com"
        assert data["location"] == "Page 10"
        assert data["confidence"] == 0.85
        assert data["snippet"] == "Important finding here"
        assert data["metadata"]["author"] == "John Doe"

    def test_citation_json_serialization(self):
        """Test citation JSON serialization."""
        citation = Citation(
            source="test_source", confidence=0.75, snippet="Test snippet"
        )

        json_str = citation.model_dump_json()
        assert isinstance(json_str, str)

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert parsed["source"] == "test_source"
        assert parsed["confidence"] == 0.75
        assert parsed["snippet"] == "Test snippet"

    def test_citation_from_dict(self):
        """Test creating citation from dictionary."""
        data = {
            "source": "https://arxiv.org/abs/2401.12345",
            "location": "Section 3.2",
            "confidence": 0.95,
            "snippet": "Test snippet",
            "metadata": {"author": "Smith et al."},
        }

        citation = Citation(**data)

        assert citation.source == data["source"]
        assert citation.location == data["location"]
        assert citation.confidence == data["confidence"]
        assert citation.snippet == data["snippet"]
        assert citation.metadata == data["metadata"]

    def test_citation_json_roundtrip(self):
        """Test complete JSON serialization roundtrip."""
        original = Citation(
            source="https://example.com/research",
            location="Figure 7, Page 42",
            confidence=0.92,
            snippet="Detailed experimental results demonstrate...",
            metadata={"author": "Research Team", "year": 2024, "verified": True},
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        data = json.loads(json_str)
        restored = Citation(**data)

        assert restored.source == original.source
        assert restored.location == original.location
        assert restored.confidence == original.confidence
        assert restored.snippet == original.snippet
        assert restored.metadata == original.metadata


class TestCitationMap:
    """Test suite for CitationMap model."""

    def test_citation_map_creation(self):
        """Test basic citation map creation."""
        citations = [
            Citation(
                source="https://arxiv.org/abs/2401.12345",
                location="Section 3.2",
                confidence=0.95,
                snippet="Our experiments show a 23% improvement...",
            ),
            Citation(
                source="paper2.pdf",
                location="Figure 4",
                confidence=0.85,
                snippet="Results demonstrate significant gains...",
            ),
        ]

        citation_map = CitationMap(
            claim_id="claim-001",
            claim_text="Machine learning models improve accuracy by 23%",
            citations=citations,
            strength=0.9,
            metadata={
                "argument_type": "empirical",
                "verification_status": "verified",
                "citation_count": 2,
            },
        )

        assert citation_map.claim_id == "claim-001"
        assert (
            citation_map.claim_text == "Machine learning models improve accuracy by 23%"
        )
        assert len(citation_map.citations) == 2
        assert citation_map.strength == 0.9
        assert citation_map.metadata["argument_type"] == "empirical"
        assert citation_map.metadata["verification_status"] == "verified"
        assert citation_map.metadata["citation_count"] == 2

    def test_citation_map_minimal_creation(self):
        """Test citation map with only required fields."""
        citation_map = CitationMap(
            claim_id="claim-001", claim_text="Test claim", strength=0.7
        )

        assert citation_map.claim_id == "claim-001"
        assert citation_map.claim_text == "Test claim"
        assert citation_map.citations == []  # Default empty list
        assert citation_map.strength == 0.7
        assert citation_map.metadata == {}  # Default empty dict

    def test_citation_map_empty_claim_id(self):
        """Test that empty claim_id fails validation."""
        with pytest.raises(ValidationError):
            CitationMap(
                claim_id="", claim_text="Test", strength=0.5
            )  # min_length=1 constraint

    def test_citation_map_empty_claim_text(self):
        """Test that empty claim_text fails validation."""
        with pytest.raises(ValidationError):
            CitationMap(
                claim_id="claim-001",
                claim_text="",
                strength=0.5,  # min_length=1 constraint
            )

    def test_citation_map_strength_bounds(self):
        """Test strength value bounds (0.0-1.0)."""
        # Valid boundary values
        map_zero = CitationMap(claim_id="test", claim_text="Test claim", strength=0.0)
        assert map_zero.strength == 0.0

        map_one = CitationMap(claim_id="test", claim_text="Test claim", strength=1.0)
        assert map_one.strength == 1.0

        # Invalid: strength < 0
        with pytest.raises(ValidationError):
            CitationMap(claim_id="test", claim_text="Test claim", strength=-0.1)

        # Invalid: strength > 1
        with pytest.raises(ValidationError):
            CitationMap(claim_id="test", claim_text="Test claim", strength=1.1)

    def test_citation_map_single_citation(self):
        """Test citation map with a single citation."""
        citation = Citation(source="https://example.com", confidence=0.9)

        citation_map = CitationMap(
            claim_id="claim-single",
            claim_text="Single citation claim",
            citations=[citation],
            strength=0.9,
        )

        assert len(citation_map.citations) == 1
        assert citation_map.citations[0].source == "https://example.com"

    def test_citation_map_multiple_citations(self):
        """Test citation map with multiple supporting citations."""
        citations = [
            Citation(source="source1", confidence=0.9),
            Citation(source="source2", confidence=0.85),
            Citation(source="source3", confidence=0.95),
            Citation(source="source4", confidence=0.8),
            Citation(source="source5", confidence=0.9),
        ]

        citation_map = CitationMap(
            claim_id="claim-multi",
            claim_text="Well-supported claim with multiple sources",
            citations=citations,
            strength=0.88,
        )

        assert len(citation_map.citations) == 5
        assert all(isinstance(c, Citation) for c in citation_map.citations)

        # Verify all sources
        sources = [c.source for c in citation_map.citations]
        assert sources == ["source1", "source2", "source3", "source4", "source5"]

    def test_citation_map_metadata_flexibility(self):
        """Test that metadata accepts arbitrary fields."""
        citation_map = CitationMap(
            claim_id="claim-meta",
            claim_text="Test claim",
            strength=0.8,
            metadata={
                "argument_type": "empirical",
                "verification_status": "verified",
                "citation_count": 3,
                "reviewer": "Dr. Smith",
                "review_date": "2024-01-15",
                "confidence_level": "high",
                "tags": ["ml", "optimization", "performance"],
                "cross_references": ["claim-002", "claim-005"],
            },
        )

        assert citation_map.metadata["argument_type"] == "empirical"
        assert citation_map.metadata["citation_count"] == 3
        assert len(citation_map.metadata["tags"]) == 3
        assert len(citation_map.metadata["cross_references"]) == 2

    def test_citation_map_serialization(self):
        """Test citation map serialization to dict."""
        citation = Citation(source="test", confidence=0.9)

        citation_map = CitationMap(
            claim_id="claim-serialize",
            claim_text="Serialization test claim",
            citations=[citation],
            strength=0.85,
            metadata={"test_key": "test_value"},
        )

        data = citation_map.model_dump()

        assert data["claim_id"] == "claim-serialize"
        assert data["claim_text"] == "Serialization test claim"
        assert len(data["citations"]) == 1
        assert data["strength"] == 0.85
        assert data["metadata"]["test_key"] == "test_value"

    def test_citation_map_json_serialization(self):
        """Test citation map JSON serialization."""
        citation_map = CitationMap(
            claim_id="claim-json", claim_text="JSON test claim", strength=0.75
        )

        json_str = citation_map.model_dump_json()
        assert isinstance(json_str, str)

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert parsed["claim_id"] == "claim-json"
        assert parsed["claim_text"] == "JSON test claim"
        assert parsed["strength"] == 0.75

    def test_citation_map_json_roundtrip(self):
        """Test complete JSON serialization roundtrip with nested citations."""
        original = CitationMap(
            claim_id="claim-roundtrip",
            claim_text="Complex roundtrip test",
            citations=[
                Citation(
                    source="source1",
                    location="page 10",
                    confidence=0.9,
                    snippet="Evidence 1",
                ),
                Citation(
                    source="source2", confidence=0.85, metadata={"author": "Jane Doe"}
                ),
            ],
            strength=0.87,
            metadata={"argument_type": "empirical", "verified": True},
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        data = json.loads(json_str)
        restored = CitationMap(**data)

        # Verify claim fields
        assert restored.claim_id == original.claim_id
        assert restored.claim_text == original.claim_text
        assert restored.strength == original.strength

        # Verify citations
        assert len(restored.citations) == 2
        assert restored.citations[0].source == "source1"
        assert restored.citations[0].location == "page 10"
        assert restored.citations[0].confidence == 0.9
        assert restored.citations[1].source == "source2"
        assert restored.citations[1].metadata["author"] == "Jane Doe"

        # Verify metadata
        assert restored.metadata == original.metadata

    def test_citation_map_nested_validation(self):
        """Test that nested citation validation works correctly."""
        # Invalid citation (empty source) in citation map
        with pytest.raises(ValidationError):
            CitationMap(
                claim_id="test",
                claim_text="Test",
                citations=[
                    Citation(source="valid", confidence=0.9),
                    Citation(source="", confidence=0.8),  # Empty source
                ],
                strength=0.85,
            )

        # Invalid citation (confidence out of bounds)
        with pytest.raises(ValidationError):
            CitationMap(
                claim_id="test",
                claim_text="Test",
                citations=[Citation(source="test", confidence=1.5)],  # > 1.0
                strength=0.85,
            )


class TestCitationIntegration:
    """Test integration scenarios for citation tracking."""

    def test_claim_evidence_mapping(self):
        """Test mapping a claim to its supporting evidence."""
        # Create citations from different sources
        academic_paper = Citation(
            source="https://arxiv.org/abs/2401.12345",
            location="Section 3.2, Table 1",
            confidence=0.95,
            snippet="Performance improved by 23% ± 2%",
            metadata={"paper_type": "peer_reviewed", "year": 2024},
        )

        industry_report = Citation(
            source="https://tech-report.com/ml-performance",
            location="Page 15",
            confidence=0.80,
            snippet="Industry benchmarks show 20-25% improvement",
            metadata={"source_type": "industry_report"},
        )

        technical_blog = Citation(
            source="https://engineering-blog.com/optimization",
            confidence=0.70,
            snippet="Our experiments yielded 22% better accuracy",
            metadata={"source_type": "blog", "verified": False},
        )

        # Map claim to all supporting citations
        claim_map = CitationMap(
            claim_id="claim-performance-001",
            claim_text="New ML optimization technique improves accuracy by ~23%",
            citations=[academic_paper, industry_report, technical_blog],
            strength=0.82,  # Average weighted by confidence
            metadata={
                "argument_type": "empirical",
                "verification_status": "verified",
                "citation_count": 3,
                "confidence_range": "0.70-0.95",
                "sources": ["academic", "industry", "blog"],
            },
        )

        # Verify the mapping
        assert len(claim_map.citations) == 3
        assert claim_map.strength == 0.82

        # Verify different confidence levels
        confidences = [c.confidence for c in claim_map.citations]
        assert max(confidences) == 0.95
        assert min(confidences) == 0.70

        # Verify citation types via metadata
        assert claim_map.citations[0].metadata["paper_type"] == "peer_reviewed"
        assert claim_map.citations[1].metadata["source_type"] == "industry_report"
        assert claim_map.citations[2].metadata["verified"] is False

    def test_multiple_claims_same_source(self):
        """Test using the same source for multiple different claims."""
        source = "https://arxiv.org/abs/2401.12345"

        # Different locations in same source
        citation1 = Citation(
            source=source,
            location="Section 2.1",
            confidence=0.9,
            snippet="Training time reduced by 40%",
        )

        citation2 = Citation(
            source=source,
            location="Section 3.3",
            confidence=0.85,
            snippet="Memory usage decreased by 25%",
        )

        citation3 = Citation(
            source=source,
            location="Section 4.2",
            confidence=0.92,
            snippet="Inference latency improved by 15%",
        )

        # Create separate claims from different sections
        claim1 = CitationMap(
            claim_id="claim-training",
            claim_text="Training time reduced by 40%",
            citations=[citation1],
            strength=0.9,
        )

        claim2 = CitationMap(
            claim_id="claim-memory",
            claim_text="Memory usage decreased by 25%",
            citations=[citation2],
            strength=0.85,
        )

        claim3 = CitationMap(
            claim_id="claim-latency",
            claim_text="Inference latency improved by 15%",
            citations=[citation3],
            strength=0.92,
        )

        # Verify all use same source but different locations
        assert claim1.citations[0].source == source
        assert claim2.citations[0].source == source
        assert claim3.citations[0].source == source

        assert claim1.citations[0].location != claim2.citations[0].location
        assert claim2.citations[0].location != claim3.citations[0].location

    def test_citation_strength_calculation(self):
        """Test calculating overall citation strength from multiple sources."""
        citations = [
            Citation(source="s1", confidence=0.95),
            Citation(source="s2", confidence=0.90),
            Citation(source="s3", confidence=0.85),
            Citation(source="s4", confidence=0.80),
        ]

        # Calculate average confidence
        avg_confidence = sum(c.confidence for c in citations) / len(citations)

        citation_map = CitationMap(
            claim_id="claim-strength",
            claim_text="Test claim",
            citations=citations,
            strength=avg_confidence,
        )

        assert citation_map.strength == 0.875
        assert len(citation_map.citations) == 4

    def test_citation_filtering_by_confidence(self):
        """Test filtering citations by confidence threshold."""
        all_citations = [
            Citation(source="high1", confidence=0.95),
            Citation(source="high2", confidence=0.90),
            Citation(source="medium1", confidence=0.75),
            Citation(source="medium2", confidence=0.70),
            Citation(source="low1", confidence=0.55),
        ]

        # Filter to high confidence citations (>= 0.80)
        high_confidence = [c for c in all_citations if c.confidence >= 0.80]

        citation_map = CitationMap(
            claim_id="claim-filtered",
            claim_text="High confidence claim",
            citations=high_confidence,
            strength=sum(c.confidence for c in high_confidence) / len(high_confidence),
            metadata={
                "total_citations": len(all_citations),
                "high_confidence_citations": len(high_confidence),
                "threshold": 0.80,
            },
        )

        assert len(citation_map.citations) == 2
        assert all(c.confidence >= 0.80 for c in citation_map.citations)
        assert citation_map.metadata["total_citations"] == 5
        assert citation_map.metadata["high_confidence_citations"] == 2

    def test_argument_workflow_citation_tracking(self):
        """Test citation tracking in ARGUMENT workflow context."""
        # Simulate debate with proponent and opponent claims
        proponent_citation = Citation(
            source="pro-study.pdf",
            location="Results section",
            confidence=0.90,
            snippet="TypeScript reduces runtime errors by 15%",
            metadata={"stance": "proponent"},
        )

        opponent_citation = Citation(
            source="counter-study.pdf",
            location="Discussion",
            confidence=0.85,
            snippet="No significant difference in error rates",
            metadata={"stance": "opponent"},
        )

        # Proponent claim map
        pro_claim = CitationMap(
            claim_id="pro-typescript-001",
            claim_text="TypeScript significantly reduces runtime errors",
            citations=[proponent_citation],
            strength=0.90,
            metadata={
                "argument_type": "empirical",
                "stance": "proponent",
                "verification_status": "verified",
            },
        )

        # Opponent claim map (contradicts proponent)
        opp_claim = CitationMap(
            claim_id="opp-typescript-001",
            claim_text="TypeScript does not significantly affect error rates",
            citations=[opponent_citation],
            strength=0.85,
            metadata={
                "argument_type": "empirical",
                "stance": "opponent",
                "verification_status": "verified",
                "contradicts": "pro-typescript-001",
            },
        )

        # Verify both claims are properly tracked
        assert pro_claim.strength == 0.90
        assert opp_claim.strength == 0.85
        assert pro_claim.metadata["stance"] == "proponent"
        assert opp_claim.metadata["stance"] == "opponent"
        assert opp_claim.metadata["contradicts"] == "pro-typescript-001"
