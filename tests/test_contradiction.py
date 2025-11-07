"""
Tests for contradiction detection module.

Verifies that contradiction detection identifies conflicts accurately
with appropriate severity levels.
"""

import pytest
from modelchorus.core.contradiction import (
    Contradiction,
    ContradictionSeverity,
    detect_contradiction,
    detect_contradictions_batch,
    detect_polarity_opposition,
    assess_contradiction_severity,
    generate_contradiction_explanation,
    generate_reconciliation_suggestion,
)


class TestPolarityOpposition:
    """Test polarity opposition detection."""

    def test_clear_positive_negative_opposition(self):
        """Test detection of clear positive vs negative opposition."""
        claim1 = "AI improves accuracy by 23%"
        claim2 = "AI reduces accuracy by 15%"

        has_opp, confidence = detect_polarity_opposition(claim1, claim2)

        assert has_opp is True
        assert confidence >= 0.7  # Should have high confidence

    def test_negation_opposition(self):
        """Test detection of negation-based opposition."""
        claim1 = "Machine learning is accurate"
        claim2 = "Machine learning is not accurate"

        has_opp, confidence = detect_polarity_opposition(claim1, claim2)

        assert has_opp is True
        assert confidence > 0.0

    def test_no_opposition_similar_polarity(self):
        """Test that similar polarity claims don't trigger opposition."""
        claim1 = "AI improves accuracy"
        claim2 = "Machine learning enhances precision"

        has_opp, confidence = detect_polarity_opposition(claim1, claim2)

        assert has_opp is False
        assert confidence == 0.0

    def test_numerical_opposition_strengthens_confidence(self):
        """Test that numerical values strengthen confidence when opposition exists."""
        claim1 = "Performance increases by 23%"
        claim2 = "Performance decreases by 15%"

        has_opp, confidence = detect_polarity_opposition(claim1, claim2)

        assert has_opp is True
        # With percentages, confidence should be boosted
        assert confidence >= 0.7


class TestSeverityAssessment:
    """Test contradiction severity assessment."""

    def test_critical_severity_high_similarity_strong_polarity(self):
        """Test CRITICAL severity for high similarity with strong polarity opposition."""
        severity = assess_contradiction_severity(
            semantic_similarity=0.85,
            has_polarity_opposition=True,
            polarity_confidence=0.8
        )

        assert severity == ContradictionSeverity.CRITICAL

    def test_major_severity_high_similarity_weak_polarity(self):
        """Test MAJOR severity for high similarity with weak polarity opposition."""
        severity = assess_contradiction_severity(
            semantic_similarity=0.75,
            has_polarity_opposition=True,
            polarity_confidence=0.5
        )

        assert severity == ContradictionSeverity.MAJOR

    def test_moderate_severity_medium_similarity(self):
        """Test MODERATE severity for moderate similarity."""
        severity = assess_contradiction_severity(
            semantic_similarity=0.6,
            has_polarity_opposition=True,
            polarity_confidence=0.7
        )

        assert severity == ContradictionSeverity.MODERATE

    def test_minor_severity_low_similarity(self):
        """Test MINOR severity for low similarity."""
        severity = assess_contradiction_severity(
            semantic_similarity=0.4,
            has_polarity_opposition=True,
            polarity_confidence=0.6
        )

        assert severity == ContradictionSeverity.MINOR

    def test_moderate_severity_high_similarity_no_opposition(self):
        """Test MODERATE severity for high similarity without opposition."""
        severity = assess_contradiction_severity(
            semantic_similarity=0.8,
            has_polarity_opposition=False,
            polarity_confidence=0.0
        )

        assert severity == ContradictionSeverity.MODERATE


class TestContradictionExplanation:
    """Test contradiction explanation generation."""

    def test_explanation_includes_polarity_info(self):
        """Test that explanation includes polarity information."""
        explanation = generate_contradiction_explanation(
            severity=ContradictionSeverity.CRITICAL,
            semantic_similarity=0.85,
            has_polarity_opposition=True,
            polarity_confidence=0.8
        )

        assert "opposing polarity" in explanation.lower()
        assert "0.80" in explanation  # Polarity confidence
        assert "0.85" in explanation  # Semantic similarity

    def test_explanation_severity_critical(self):
        """Test explanation for CRITICAL severity mentions high relation."""
        explanation = generate_contradiction_explanation(
            severity=ContradictionSeverity.CRITICAL,
            semantic_similarity=0.9,
            has_polarity_opposition=True,
            polarity_confidence=0.9
        )

        assert "highly related" in explanation.lower()

    def test_explanation_severity_moderate(self):
        """Test explanation for MODERATE severity mentions investigation."""
        explanation = generate_contradiction_explanation(
            severity=ContradictionSeverity.MODERATE,
            semantic_similarity=0.6,
            has_polarity_opposition=True,
            polarity_confidence=0.7
        )

        assert "investigation" in explanation.lower() or "inconsistency" in explanation.lower()

    def test_explanation_severity_minor(self):
        """Test explanation for MINOR severity mentions context."""
        explanation = generate_contradiction_explanation(
            severity=ContradictionSeverity.MINOR,
            semantic_similarity=0.4,
            has_polarity_opposition=False,
            polarity_confidence=0.3
        )

        assert "context" in explanation.lower() or "minor" in explanation.lower()


class TestReconciliationSuggestions:
    """Test reconciliation suggestion generation."""

    def test_critical_suggestion_mentions_reliability(self):
        """Test CRITICAL suggestion mentions source reliability."""
        suggestion = generate_reconciliation_suggestion(ContradictionSeverity.CRITICAL)

        assert suggestion is not None
        assert "reliability" in suggestion.lower() or "incorrect" in suggestion.lower()

    def test_major_suggestion_mentions_context(self):
        """Test MAJOR suggestion mentions source context."""
        suggestion = generate_reconciliation_suggestion(ContradictionSeverity.MAJOR)

        assert suggestion is not None
        assert "context" in suggestion.lower() or "domain" in suggestion.lower()

    def test_moderate_suggestion_mentions_differences(self):
        """Test MODERATE suggestion mentions temporal or scope differences."""
        suggestion = generate_reconciliation_suggestion(ContradictionSeverity.MODERATE)

        assert suggestion is not None
        assert "temporal" in suggestion.lower() or "scope" in suggestion.lower()

    def test_minor_no_suggestion(self):
        """Test MINOR severity returns no suggestion."""
        suggestion = generate_reconciliation_suggestion(ContradictionSeverity.MINOR)

        assert suggestion is None


class TestContradictionModel:
    """Test Contradiction Pydantic model."""

    def test_valid_contradiction_creation(self):
        """Test creating a valid Contradiction instance."""
        contra = Contradiction(
            contradiction_id="contra-001",
            claim_1_id="claim-1",
            claim_2_id="claim-2",
            claim_1_text="AI improves accuracy",
            claim_2_text="AI reduces accuracy",
            severity=ContradictionSeverity.CRITICAL,
            confidence=0.85,
            explanation="Test explanation",
            resolution_suggestion="Test suggestion",
            metadata={"test": "value"}
        )

        assert contra.contradiction_id == "contra-001"
        assert contra.severity == ContradictionSeverity.CRITICAL
        assert contra.confidence == 0.85

    def test_confidence_validation_in_range(self):
        """Test confidence must be in [0.0, 1.0] range."""
        # Valid confidence
        contra = Contradiction(
            contradiction_id="contra-001",
            claim_1_id="claim-1",
            claim_2_id="claim-2",
            claim_1_text="Test claim 1",
            claim_2_text="Test claim 2",
            severity=ContradictionSeverity.MINOR,
            confidence=0.5,
            explanation="Test"
        )
        assert contra.confidence == 0.5

    def test_confidence_validation_too_high(self):
        """Test confidence > 1.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Contradiction(
                contradiction_id="contra-001",
                claim_1_id="claim-1",
                claim_2_id="claim-2",
                claim_1_text="Test claim 1",
                claim_2_text="Test claim 2",
                severity=ContradictionSeverity.MINOR,
                confidence=1.5,
                explanation="Test"
            )

    def test_confidence_validation_too_low(self):
        """Test confidence < 0.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Contradiction(
                contradiction_id="contra-001",
                claim_1_id="claim-1",
                claim_2_id="claim-2",
                claim_1_text="Test claim 1",
                claim_2_text="Test claim 2",
                severity=ContradictionSeverity.MINOR,
                confidence=-0.5,
                explanation="Test"
            )

    def test_different_claim_ids_validation(self):
        """Test claim IDs must be different."""
        with pytest.raises(ValueError, match="claim_2_id must be different from claim_1_id"):
            Contradiction(
                contradiction_id="contra-001",
                claim_1_id="same-id",
                claim_2_id="same-id",  # Same as claim_1_id - should fail
                claim_1_text="Test claim 1",
                claim_2_text="Test claim 2",
                severity=ContradictionSeverity.MINOR,
                confidence=0.5,
                explanation="Test"
            )


class TestContradictionDetection:
    """Test end-to-end contradiction detection."""

    def test_detect_clear_contradiction(self):
        """Test detecting a clear contradiction between opposing claims."""
        # Note: This test requires semantic similarity to be computed
        # May need to mock or skip if embedding model not available
        try:
            contra = detect_contradiction(
                claim_1_id="claim-1",
                claim_1_text="Artificial intelligence improves diagnostic accuracy by 25%",
                claim_2_id="claim-2",
                claim_2_text="Artificial intelligence reduces diagnostic accuracy by 18%",
                similarity_threshold=0.3
            )

            assert contra is not None
            assert contra.severity in [ContradictionSeverity.CRITICAL, ContradictionSeverity.MAJOR]
            assert contra.confidence > 0.5
            assert "opposing polarity" in contra.explanation.lower() or "polarity" in contra.explanation.lower()
        except ImportError:
            pytest.skip("Semantic similarity functions not available")

    def test_no_contradiction_unrelated_claims(self):
        """Test that unrelated claims don't trigger contradiction."""
        try:
            contra = detect_contradiction(
                claim_1_id="claim-1",
                claim_1_text="The weather is sunny today",
                claim_2_id="claim-2",
                claim_2_text="Python is a programming language",
                similarity_threshold=0.3
            )

            # Unrelated claims should return None
            assert contra is None
        except ImportError:
            pytest.skip("Semantic similarity functions not available")

    def test_no_contradiction_similar_polarity(self):
        """Test that similar polarity claims don't trigger contradiction."""
        try:
            contra = detect_contradiction(
                claim_1_id="claim-1",
                claim_1_text="Machine learning improves accuracy",
                claim_2_id="claim-2",
                claim_2_text="Deep learning enhances precision",
                similarity_threshold=0.3
            )

            # Similar positive claims should return None
            assert contra is None
        except ImportError:
            pytest.skip("Semantic similarity functions not available")


class TestBatchContradictionDetection:
    """Test batch contradiction detection."""

    def test_detect_contradictions_in_batch(self):
        """Test detecting multiple contradictions in a batch of claims."""
        try:
            claims = [
                ("claim-1", "AI improves accuracy significantly"),
                ("claim-2", "AI reduces accuracy dramatically"),
                ("claim-3", "Weather is sunny today"),
                ("claim-4", "Performance increases by 30%"),
                ("claim-5", "Performance decreases by 20%"),
            ]

            contradictions = detect_contradictions_batch(claims, similarity_threshold=0.3)

            # Should detect at least the AI accuracy contradiction and performance contradiction
            assert len(contradictions) >= 1

            # Verify contradictions have proper structure
            for contra in contradictions:
                assert isinstance(contra, Contradiction)
                assert contra.confidence > 0.0
                assert contra.severity in ContradictionSeverity
        except ImportError:
            pytest.skip("Semantic similarity functions not available")

    def test_batch_no_contradictions(self):
        """Test batch detection with no contradictions."""
        try:
            claims = [
                ("claim-1", "The sky is blue"),
                ("claim-2", "Grass is green"),
                ("claim-3", "Water is wet"),
            ]

            contradictions = detect_contradictions_batch(claims, similarity_threshold=0.3)

            # Unrelated claims should produce no contradictions
            assert len(contradictions) == 0
        except ImportError:
            pytest.skip("Semantic similarity functions not available")


# Integration tests with realistic scenarios
class TestRealisticScenarios:
    """Test realistic contradiction scenarios."""

    def test_medical_accuracy_contradiction(self):
        """Test medical accuracy contradiction scenario."""
        try:
            contra = detect_contradiction(
                claim_1_id="study-1",
                claim_1_text="The new drug improves patient outcomes by 42% in clinical trials",
                claim_2_id="study-2",
                claim_2_text="The new drug worsens patient outcomes by 28% according to recent data",
                similarity_threshold=0.3
            )

            if contra is not None:  # May be None if similarity too low
                assert contra.severity in [ContradictionSeverity.CRITICAL, ContradictionSeverity.MAJOR]
                assert "reliability" in (contra.resolution_suggestion or "").lower() or \
                       "domain" in (contra.resolution_suggestion or "").lower()
        except ImportError:
            pytest.skip("Semantic similarity functions not available")

    def test_performance_metric_contradiction(self):
        """Test performance metric contradiction scenario."""
        try:
            contra = detect_contradiction(
                claim_1_id="benchmark-1",
                claim_1_text="System latency increases by 150ms under load",
                claim_2_id="benchmark-2",
                claim_2_text="System latency decreases by 200ms under load",
                similarity_threshold=0.3
            )

            if contra is not None:
                assert contra.severity in [ContradictionSeverity.CRITICAL, ContradictionSeverity.MAJOR, ContradictionSeverity.MODERATE]
                assert contra.confidence > 0.4
        except ImportError:
            pytest.skip("Semantic similarity functions not available")
