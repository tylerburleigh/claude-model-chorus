"""
Tests for gap analysis module.

Verifies that gap detection identifies missing evidence, logical gaps,
and unsupported claims accurately with appropriate severity levels.
"""

import pytest

from model_chorus.core.gap_analysis import (
    Gap,
    GapSeverity,
    GapType,
    assess_gap_severity,
    detect_gaps,
    detect_logical_gaps,
    detect_missing_evidence,
    detect_unsupported_claims,
    generate_gap_recommendation,
)


class TestGapModel:
    """Test Gap Pydantic model."""

    def test_valid_gap_creation(self):
        """Test creating a valid Gap instance."""
        gap = Gap(
            gap_id="gap-001",
            gap_type=GapType.EVIDENCE,
            severity=GapSeverity.MAJOR,
            claim_id="claim-1",
            claim_text="AI improves accuracy",
            description="Lacks empirical evidence",
            recommendation="Add peer-reviewed citations",
            confidence=0.85,
            metadata={"test": "value"},
        )

        assert gap.gap_id == "gap-001"
        assert gap.gap_type == GapType.EVIDENCE
        assert gap.severity == GapSeverity.MAJOR
        assert gap.confidence == 0.85

    def test_confidence_validation_in_range(self):
        """Test confidence must be in [0.0, 1.0] range."""
        gap = Gap(
            gap_id="gap-001",
            gap_type=GapType.LOGICAL,
            severity=GapSeverity.MINOR,
            claim_id="claim-1",
            claim_text="Test claim",
            description="Test description",
            recommendation="Test recommendation",
            confidence=0.5,
        )
        assert gap.confidence == 0.5

    def test_confidence_validation_too_high(self):
        """Test confidence > 1.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Gap(
                gap_id="gap-001",
                gap_type=GapType.EVIDENCE,
                severity=GapSeverity.MAJOR,
                claim_id="claim-1",
                claim_text="Test claim",
                description="Test description",
                recommendation="Test recommendation",
                confidence=1.5,
            )

    def test_confidence_validation_too_low(self):
        """Test confidence < 0.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Gap(
                gap_id="gap-001",
                gap_type=GapType.EVIDENCE,
                severity=GapSeverity.MAJOR,
                claim_id="claim-1",
                claim_text="Test claim",
                description="Test description",
                recommendation="Test recommendation",
                confidence=-0.5,
            )


class TestSeverityAssessment:
    """Test gap severity assessment logic."""

    def test_evidence_gap_critical_no_citations(self):
        """Test CRITICAL severity for evidence gap with no citations."""
        severity = assess_gap_severity(
            gap_type=GapType.EVIDENCE, citation_count=0, expected_citations=2
        )

        assert severity == GapSeverity.CRITICAL

    def test_evidence_gap_major_no_citations_low_expectation(self):
        """Test MAJOR severity for evidence gap with no citations but low expectation."""
        severity = assess_gap_severity(
            gap_type=GapType.EVIDENCE, citation_count=0, expected_citations=1
        )

        assert severity == GapSeverity.MAJOR

    def test_evidence_gap_moderate_insufficient_citations(self):
        """Test MODERATE severity for insufficient citations."""
        severity = assess_gap_severity(
            gap_type=GapType.EVIDENCE, citation_count=1, expected_citations=2
        )

        assert severity == GapSeverity.MODERATE

    def test_logical_gap_major_no_support(self):
        """Test MAJOR severity for logical gap without support."""
        severity = assess_gap_severity(
            gap_type=GapType.LOGICAL, has_supporting_logic=False
        )

        assert severity == GapSeverity.MAJOR

    def test_logical_gap_moderate_with_support(self):
        """Test MODERATE severity for logical gap with some support."""
        severity = assess_gap_severity(
            gap_type=GapType.LOGICAL, has_supporting_logic=True
        )

        assert severity == GapSeverity.MODERATE

    def test_support_gap_moderate(self):
        """Test MODERATE severity for support gaps."""
        severity = assess_gap_severity(gap_type=GapType.SUPPORT)

        assert severity == GapSeverity.MODERATE

    def test_assumption_gap_moderate_no_logic(self):
        """Test MODERATE severity for assumption gap without logic."""
        severity = assess_gap_severity(
            gap_type=GapType.ASSUMPTION, has_supporting_logic=False
        )

        assert severity == GapSeverity.MODERATE

    def test_assumption_gap_minor_with_logic(self):
        """Test MINOR severity for assumption gap with logic."""
        severity = assess_gap_severity(
            gap_type=GapType.ASSUMPTION, has_supporting_logic=True
        )

        assert severity == GapSeverity.MINOR


class TestGapRecommendations:
    """Test gap recommendation generation."""

    def test_evidence_recommendation_critical(self):
        """Test recommendation for CRITICAL evidence gap."""
        rec = generate_gap_recommendation(
            GapType.EVIDENCE, GapSeverity.CRITICAL, "AI improves accuracy"
        )

        assert "empirical evidence" in rec.lower()
        assert "citations" in rec.lower()

    def test_evidence_recommendation_minor(self):
        """Test recommendation for MINOR evidence gap."""
        rec = generate_gap_recommendation(
            GapType.EVIDENCE, GapSeverity.MINOR, "AI improves accuracy"
        )

        assert "additional citations" in rec.lower()

    def test_logical_recommendation(self):
        """Test recommendation for logical gap."""
        rec = generate_gap_recommendation(
            GapType.LOGICAL, GapSeverity.MAJOR, "Therefore X is true"
        )

        assert "logical connection" in rec.lower() or "reasoning" in rec.lower()

    def test_support_recommendation(self):
        """Test recommendation for support gap."""
        rec = generate_gap_recommendation(
            GapType.SUPPORT, GapSeverity.MODERATE, "Main claim"
        )

        assert "supporting arguments" in rec.lower() or "sub-claims" in rec.lower()

    def test_assumption_recommendation(self):
        """Test recommendation for assumption gap."""
        rec = generate_gap_recommendation(
            GapType.ASSUMPTION, GapSeverity.MODERATE, "X leads to Y"
        )

        assert "assumption" in rec.lower() or "premises" in rec.lower()


class TestMissingEvidenceDetection:
    """Test detection of missing evidence gaps."""

    def test_detect_missing_evidence_no_citations(self):
        """Test detecting evidence gap when no citations provided."""
        gap = detect_missing_evidence(
            claim_id="claim-1",
            claim_text="AI reduces diagnostic errors by 40%",
            citations=[],
            expected_citation_count=1,
        )

        assert gap is not None
        assert gap.gap_type == GapType.EVIDENCE
        assert gap.severity in [GapSeverity.MAJOR, GapSeverity.CRITICAL]
        assert gap.claim_id == "claim-1"
        assert "citation" in gap.description.lower()

    def test_detect_missing_evidence_insufficient_citations(self):
        """Test detecting evidence gap with insufficient citations."""
        gap = detect_missing_evidence(
            claim_id="claim-2",
            claim_text="Multiple studies show benefits",
            citations=["citation-1"],
            expected_citation_count=3,
        )

        assert gap is not None
        assert gap.gap_type == GapType.EVIDENCE
        assert gap.severity == GapSeverity.MODERATE

    def test_no_gap_sufficient_citations(self):
        """Test no gap detected when citations are sufficient."""
        gap = detect_missing_evidence(
            claim_id="claim-3",
            claim_text="Research demonstrates X",
            citations=["citation-1", "citation-2"],
            expected_citation_count=2,
        )

        assert gap is None

    def test_evidence_gap_metadata(self):
        """Test that evidence gap includes proper metadata."""
        gap = detect_missing_evidence(
            claim_id="claim-4",
            claim_text="Test claim",
            citations=[],
            expected_citation_count=2,
        )

        assert gap is not None
        assert "citation_count" in gap.metadata
        assert "expected_citations" in gap.metadata
        assert gap.metadata["citation_count"] == 0
        assert gap.metadata["expected_citations"] == 2


class TestLogicalGapDetection:
    """Test detection of logical gaps."""

    def test_detect_logical_gap_conclusion_without_support(self):
        """Test detecting logical gap for conclusion without premises."""
        gap = detect_logical_gaps(
            claim_id="claim-1",
            claim_text="Therefore, we should implement universal healthcare",
            supporting_claims=[],
        )

        assert gap is not None
        assert gap.gap_type == GapType.LOGICAL
        assert gap.severity == GapSeverity.MAJOR
        assert "conclusion" in gap.description.lower()

    def test_no_logical_gap_with_support(self):
        """Test no logical gap when supporting claims provided."""
        gap = detect_logical_gaps(
            claim_id="claim-2",
            claim_text="Thus, the policy would be effective",
            supporting_claims=["All citizens benefit", "Cost is manageable"],
        )

        assert gap is None

    def test_no_logical_gap_non_conclusion_claim(self):
        """Test no logical gap for non-conclusion claims."""
        gap = detect_logical_gaps(
            claim_id="claim-3",
            claim_text="The system processes data efficiently",
            supporting_claims=[],
        )

        assert gap is None

    def test_conclusion_indicators_detected(self):
        """Test various conclusion indicators are detected."""
        indicators = [
            "therefore X",
            "thus Y",
            "hence Z",
            "consequently A",
            "as a result B",
            "it follows that C",
        ]

        for claim_text in indicators:
            gap = detect_logical_gaps(
                claim_id="test", claim_text=claim_text, supporting_claims=[]
            )
            assert gap is not None, f"Failed to detect gap for: {claim_text}"


class TestUnsupportedClaimsDetection:
    """Test batch detection of unsupported claims."""

    def test_detect_multiple_unsupported_claims(self):
        """Test detecting multiple unsupported claims in batch."""
        claims = [
            ("claim-1", "AI improves accuracy", []),
            ("claim-2", "Studies show benefits", []),
            ("claim-3", "Well-documented phenomenon", ["citation-1", "citation-2"]),
        ]

        gaps = detect_unsupported_claims(claims, min_citations_per_claim=1)

        # Should find gaps for claim-1 and claim-2
        assert len(gaps) >= 2
        gap_claim_ids = {gap.claim_id for gap in gaps}
        assert "claim-1" in gap_claim_ids
        assert "claim-2" in gap_claim_ids
        assert "claim-3" not in gap_claim_ids

    def test_detect_no_gaps_all_supported(self):
        """Test no gaps when all claims have citations."""
        claims = [
            ("claim-1", "X is true", ["citation-1"]),
            ("claim-2", "Y is valid", ["citation-2", "citation-3"]),
        ]

        gaps = detect_unsupported_claims(claims, min_citations_per_claim=1)

        assert len(gaps) == 0

    def test_custom_minimum_citations(self):
        """Test using custom minimum citation requirement."""
        claims = [
            ("claim-1", "Test claim", ["citation-1"]),
        ]

        # With min=2, should detect gap
        gaps = detect_unsupported_claims(claims, min_citations_per_claim=2)
        assert len(gaps) == 1

        # With min=1, should not detect gap
        gaps = detect_unsupported_claims(claims, min_citations_per_claim=1)
        assert len(gaps) == 0


class TestComprehensiveGapDetection:
    """Test comprehensive gap detection combining all types."""

    def test_detect_multiple_gap_types(self):
        """Test detecting both evidence and logical gaps."""
        claims = [
            {
                "claim_id": "claim-1",
                "claim_text": "AI improves outcomes",
                "citations": [],
                "supporting_claims": [],
            },
            {
                "claim_id": "claim-2",
                "claim_text": "Therefore, we should adopt AI",
                "citations": [],
                "supporting_claims": [],
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=1)

        # Should detect evidence gaps for both + logical gap for claim-2
        assert len(gaps) >= 2

        gap_types = {gap.gap_type for gap in gaps}
        assert GapType.EVIDENCE in gap_types
        assert GapType.LOGICAL in gap_types

    def test_detect_no_gaps_complete_argument(self):
        """Test no gaps detected for well-supported argument."""
        claims = [
            {
                "claim_id": "claim-1",
                "claim_text": "Studies show X is effective",
                "citations": ["citation-1", "citation-2"],
                "supporting_claims": [],
            },
            {
                "claim_id": "claim-2",
                "claim_text": "Implementation is feasible",
                "citations": ["citation-3"],
                "supporting_claims": [],
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=1)

        assert len(gaps) == 0

    def test_gap_detection_with_mixed_quality(self):
        """Test gap detection with mixed quality claims."""
        claims = [
            {
                "claim_id": "good",
                "claim_text": "Research demonstrates benefits",
                "citations": ["c1", "c2"],
                "supporting_claims": ["sub-claim"],
            },
            {
                "claim_id": "bad-evidence",
                "claim_text": "X is clearly true",
                "citations": [],
                "supporting_claims": ["sub-claim"],
            },
            {
                "claim_id": "bad-logic",
                "claim_text": "Therefore, policy Y is optimal",
                "citations": ["c3"],
                "supporting_claims": [],
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=1)

        # Should find evidence gap for "bad-evidence" and logical gap for "bad-logic"
        assert len(gaps) >= 2

        evidence_gaps = [g for g in gaps if g.gap_type == GapType.EVIDENCE]
        logical_gaps = [g for g in gaps if g.gap_type == GapType.LOGICAL]

        assert len(evidence_gaps) >= 1
        assert len(logical_gaps) >= 1

        # Verify the good claim has no gaps
        good_claim_gaps = [g for g in gaps if g.claim_id == "good"]
        assert len(good_claim_gaps) == 0


# Integration tests with realistic scenarios
class TestRealisticScenarios:
    """Test realistic gap detection scenarios."""

    def test_policy_argument_with_gaps(self):
        """Test gap detection in policy argument."""
        claims = [
            {
                "claim_id": "claim-1",
                "claim_text": "Universal basic income reduces poverty rates significantly",
                "citations": [],  # Missing evidence!
                "supporting_claims": [],
            },
            {
                "claim_id": "claim-2",
                "claim_text": "Therefore, all countries should implement UBI immediately",
                "citations": [],  # Missing evidence!
                "supporting_claims": [],  # Missing logical support!
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=2)

        assert len(gaps) >= 2

        # Check for evidence gaps
        evidence_gaps = [g for g in gaps if g.gap_type == GapType.EVIDENCE]
        assert len(evidence_gaps) >= 1

        # Check for logical gaps
        logical_gaps = [g for g in gaps if g.gap_type == GapType.LOGICAL]
        assert len(logical_gaps) >= 1

    def test_well_supported_scientific_claim(self):
        """Test well-supported scientific claim has no gaps."""
        claims = [
            {
                "claim_id": "claim-1",
                "claim_text": "Machine learning models achieve 95% accuracy on ImageNet dataset",
                "citations": [
                    "ResNet paper",
                    "ImageNet challenge results",
                    "Validation study",
                ],
                "supporting_claims": [
                    "Multiple architectures tested",
                    "Reproducible results",
                ],
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=2)

        # Should have no gaps - well supported
        assert len(gaps) == 0

    def test_argumentative_essay_analysis(self):
        """Test gap detection in argumentative essay structure."""
        claims = [
            {
                "claim_id": "thesis",
                "claim_text": "Climate change requires immediate action",
                "citations": ["IPCC report", "NASA data"],
                "supporting_claims": [],
            },
            {
                "claim_id": "evidence-1",
                "claim_text": "Global temperatures have risen 1.1°C since pre-industrial times",
                "citations": ["NOAA data"],
                "supporting_claims": [],
            },
            {
                "claim_id": "weak-claim",
                "claim_text": "Everyone agrees we must act now",
                "citations": [],  # Unsupported generalization
                "supporting_claims": [],
            },
            {
                "claim_id": "conclusion",
                "claim_text": "Thus, carbon emissions must be reduced by 50% by 2030",
                "citations": [],  # Missing policy citations
                "supporting_claims": [],  # Missing intermediate steps
            },
        ]

        gaps = detect_gaps(claims, min_citations_per_claim=1)

        # Should find gaps in weak-claim and conclusion
        assert len(gaps) >= 2

        weak_gaps = [g for g in gaps if g.claim_id == "weak-claim"]
        conclusion_gaps = [g for g in gaps if g.claim_id == "conclusion"]

        assert len(weak_gaps) >= 1  # Evidence gap
        assert len(conclusion_gaps) >= 1  # Evidence or logical gap
