"""
Contradiction detection models and logic for ARGUMENT workflow.

Provides data models for tracking contradictions between claims across
sources, with severity classification and resolution tracking.

Used in ARGUMENT workflow to identify and analyze conflicting claims,
assess contradiction severity, and suggest resolutions.

Public API:
    - Contradiction: Pydantic model for contradiction representation
    - ContradictionSeverity: Enum for severity levels (MINOR/MODERATE/MAJOR/CRITICAL)
    - detect_contradiction: Main entry point for detecting contradictions between two claims
    - detect_contradictions_batch: Detect contradictions across multiple claims efficiently
    - generate_contradiction_explanation: Generate human-readable explanation for contradictions
    - generate_reconciliation_suggestion: Generate suggestions for resolving contradictions
    - assess_contradiction_severity: Assess severity based on semantic and polarity analysis
    - detect_polarity_opposition: Detect opposing polarity between claims
"""

from enum import Enum
from typing import Any, Dict, Optional, List, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re


class ContradictionSeverity(str, Enum):
    """
    Severity levels for contradictions between claims.

    Classifies contradictions by their importance and impact on
    argument validity. Higher severity indicates more significant
    conflicts requiring immediate attention.

    Values:
        MINOR: Slight inconsistency, may be due to different perspectives
               or temporal differences. Low impact on argument validity.
        MODERATE: Notable contradiction that should be investigated.
                 May indicate measurement differences or scope variations.
        MAJOR: Significant contradiction that undermines argument coherence.
              Requires careful analysis and resolution.
        CRITICAL: Direct, irreconcilable contradiction that invalidates
                 one or both claims. Immediate attention required.
    """

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class Contradiction(BaseModel):
    """
    Model for tracking contradictions between claims in ARGUMENT workflow.

    Represents a detected contradiction between two claims, including
    severity assessment, confidence in detection, and resolution suggestions.
    Used to identify conflicts in evidence and maintain argument coherence.

    Attributes:
        contradiction_id: Unique identifier for this contradiction
        claim_1_id: Identifier of the first conflicting claim
        claim_2_id: Identifier of the second conflicting claim
        claim_1_text: Full text of the first claim
        claim_2_text: Full text of the second claim
        severity: Severity level of the contradiction
        confidence: Confidence in contradiction detection (0.0-1.0)
        explanation: Detailed explanation of why claims contradict
        resolution_suggestion: Optional suggestion for resolving the contradiction
        metadata: Additional metadata (detection_method, timestamp, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contradiction_id": "contra-001",
                "claim_1_id": "claim-123",
                "claim_2_id": "claim-456",
                "claim_1_text": "Machine learning improves accuracy by 23%",
                "claim_2_text": "Machine learning reduces accuracy by 15%",
                "severity": "major",
                "confidence": 0.92,
                "explanation": "Claims present opposing effects on accuracy with different magnitudes",
                "resolution_suggestion": "Investigate experimental conditions and datasets - may be domain-specific",
                "metadata": {
                    "detection_method": "semantic_similarity",
                    "detected_at": "2025-11-06T17:30:00Z",
                    "semantic_similarity": 0.87,
                    "polarity_opposition": True,
                },
            }
        }
    )

    contradiction_id: str = Field(
        ...,
        description="Unique identifier for this contradiction",
        min_length=1,
    )

    claim_1_id: str = Field(
        ...,
        description="Identifier of the first conflicting claim",
        min_length=1,
    )

    claim_2_id: str = Field(
        ...,
        description="Identifier of the second conflicting claim",
        min_length=1,
    )

    claim_1_text: str = Field(
        ...,
        description="Full text of the first claim",
        min_length=1,
    )

    claim_2_text: str = Field(
        ...,
        description="Full text of the second claim",
        min_length=1,
    )

    severity: ContradictionSeverity = Field(
        ...,
        description="Severity level of the contradiction (minor/moderate/major/critical)",
    )

    confidence: float = Field(
        ...,
        description="Confidence in contradiction detection (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    explanation: str = Field(
        ...,
        description="Detailed explanation of why the claims contradict each other",
        min_length=1,
    )

    resolution_suggestion: Optional[str] = Field(
        default=None,
        description="Optional suggestion for resolving or investigating the contradiction",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (detection_method, timestamp, scores, etc.)",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """
        Validate confidence is in valid range.

        Ensures confidence score is between 0.0 and 1.0 inclusive.
        Pydantic's ge/le constraints handle this, but explicit validator
        provides clearer error messages.

        Args:
            v: Confidence value to validate

        Returns:
            Validated confidence value

        Raises:
            ValueError: If confidence is outside [0.0, 1.0] range
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("claim_2_id")
    @classmethod
    def validate_different_claims(cls, v: str, info) -> str:
        """
        Ensure claim IDs are different.

        A claim cannot contradict itself - the two claim IDs must be distinct.
        This validator checks that claim_1_id != claim_2_id when claim_2_id is set.

        Args:
            v: claim_2_id value
            info: Validation context with other field values

        Returns:
            Validated claim_2_id

        Raises:
            ValueError: If both claim IDs are identical
        """
        # Check if claim_1_id exists and matches claim_2_id
        claim_1_id = info.data.get("claim_1_id")
        if claim_1_id and v == claim_1_id:
            raise ValueError(f"claim_2_id must be different from claim_1_id (both are '{v}')")
        return v


# ============================================================================
# Contradiction Detection Logic
# ============================================================================

# Import semantic similarity functions (lazy import to avoid circular dependencies)
def _import_semantic_functions():
    """Import semantic similarity functions from workflows.argument.semantic."""
    try:
        from model_chorus.workflows.argument.semantic import (
            compute_claim_similarity,
            compute_embedding,
        )
        return compute_claim_similarity, compute_embedding
    except ImportError as e:
        raise ImportError(
            f"Cannot import semantic similarity functions: {e}. "
            "Ensure model_chorus.workflows.argument.semantic module is available."
        )


def _import_citation_map():
    """Import CitationMap model."""
    try:
        from model_chorus.core.models import CitationMap
        return CitationMap
    except ImportError as e:
        raise ImportError(f"Cannot import CitationMap model: {e}")


# Polarity keywords for detecting opposing claims
POSITIVE_KEYWORDS = [
    "improves", "increases", "enhances", "better", "higher", "more",
    "strengthens", "boosts", "raises", "grows", "gains", "benefits",
]

NEGATIVE_KEYWORDS = [
    "reduces", "decreases", "worsens", "worse", "lower", "less",
    "weakens", "diminishes", "drops", "declines", "losses", "harms",
]

NEGATION_KEYWORDS = [
    "not", "never", "no", "without", "neither", "nor", "cannot",
    "can't", "won't", "doesn't", "don't", "isn't", "aren't",
]


def detect_polarity_opposition(
    claim_text_1: str,
    claim_text_2: str,
) -> Tuple[bool, float]:
    """
    Detect if two claims have opposing polarity (positive vs negative).

    Uses keyword-based analysis to identify claims that make opposite
    assertions about the same topic.

    Args:
        claim_text_1: First claim text
        claim_text_2: Second claim text

    Returns:
        Tuple of (has_opposition, confidence)
        - has_opposition: True if claims have opposing polarity
        - confidence: Confidence in polarity detection (0.0-1.0)

    Example:
        >>> has_opp, conf = detect_polarity_opposition(
        ...     "AI improves accuracy by 23%",
        ...     "AI reduces accuracy by 15%"
        ... )
        >>> print(f"Opposition: {has_opp}, Confidence: {conf:.2f}")
        Opposition: True, Confidence: 0.80
    """
    text1_lower = claim_text_1.lower()
    text2_lower = claim_text_2.lower()

    # Check for positive keywords in claim 1
    has_positive_1 = any(kw in text1_lower for kw in POSITIVE_KEYWORDS)
    has_negative_1 = any(kw in text1_lower for kw in NEGATIVE_KEYWORDS)
    has_negation_1 = any(kw in text1_lower for kw in NEGATION_KEYWORDS)

    # Check for positive keywords in claim 2
    has_positive_2 = any(kw in text2_lower for kw in POSITIVE_KEYWORDS)
    has_negative_2 = any(kw in text2_lower for kw in NEGATIVE_KEYWORDS)
    has_negation_2 = any(kw in text2_lower for kw in NEGATION_KEYWORDS)

    # Detect polarity opposition patterns
    confidence = 0.0
    has_opposition = False

    # Pattern 1: One claim is positive, other is negative
    if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
        has_opposition = True
        confidence = 0.7

    # Pattern 2: One claim is negated, other is not
    if has_negation_1 != has_negation_2:
        # If one is negated and the other isn't, possible opposition
        if not has_opposition:
            has_opposition = True
            confidence = 0.5
        else:
            # Strengthen confidence if we already detected opposition
            confidence = min(confidence + 0.1, 0.9)

    # Pattern 3: Numerical opposition (e.g., "23% increase" vs "15% decrease")
    # Look for percentages or numbers with opposite directions
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    numbers_1 = re.findall(percent_pattern, text1_lower)
    numbers_2 = re.findall(percent_pattern, text2_lower)

    if numbers_1 and numbers_2 and has_opposition:
        # If we found numbers and already detected opposition, increase confidence
        confidence = min(confidence + 0.1, 0.95)

    return has_opposition, confidence


def assess_contradiction_severity(
    semantic_similarity: float,
    has_polarity_opposition: bool,
    polarity_confidence: float,
) -> ContradictionSeverity:
    """
    Assess the severity of a contradiction based on multiple factors.

    Combines semantic similarity and polarity opposition to classify
    contradiction severity.

    Args:
        semantic_similarity: Cosine similarity between claims (0.0-1.0)
        has_polarity_opposition: Whether claims have opposing polarity
        polarity_confidence: Confidence in polarity detection (0.0-1.0)

    Returns:
        ContradictionSeverity enum value

    Severity Rules:
        - High similarity (>0.7) + strong polarity opposition = CRITICAL
        - High similarity (>0.7) + weak polarity opposition = MAJOR
        - Moderate similarity (0.5-0.7) + polarity opposition = MODERATE
        - Low similarity (<0.5) + polarity opposition = MINOR

    Example:
        >>> severity = assess_contradiction_severity(
        ...     semantic_similarity=0.85,
        ...     has_polarity_opposition=True,
        ...     polarity_confidence=0.8
        ... )
        >>> print(severity)
        ContradictionSeverity.CRITICAL
    """
    if semantic_similarity >= 0.7:
        # High similarity - claims are about the same thing
        if has_polarity_opposition and polarity_confidence >= 0.7:
            return ContradictionSeverity.CRITICAL
        elif has_polarity_opposition:
            return ContradictionSeverity.MAJOR
        else:
            # High similarity but no clear opposition - might be subtle
            return ContradictionSeverity.MODERATE

    elif semantic_similarity >= 0.5:
        # Moderate similarity - related claims
        if has_polarity_opposition:
            return ContradictionSeverity.MODERATE
        else:
            return ContradictionSeverity.MINOR

    else:
        # Low similarity - possibly unrelated, but if opposition detected, still minor
        return ContradictionSeverity.MINOR


def generate_contradiction_explanation(
    severity: ContradictionSeverity,
    semantic_similarity: float,
    has_polarity_opposition: bool,
    polarity_confidence: float,
) -> str:
    """
    Generate human-readable explanation for a detected contradiction.

    Creates a detailed explanation describing why two claims contradict,
    including semantic similarity scores and polarity analysis.

    Args:
        severity: Assessed severity level of the contradiction
        semantic_similarity: Cosine similarity between claims (0.0-1.0)
        has_polarity_opposition: Whether claims have opposing polarity
        polarity_confidence: Confidence in polarity detection (0.0-1.0)

    Returns:
        Formatted explanation string describing the contradiction

    Example:
        >>> explanation = generate_contradiction_explanation(
        ...     severity=ContradictionSeverity.CRITICAL,
        ...     semantic_similarity=0.85,
        ...     has_polarity_opposition=True,
        ...     polarity_confidence=0.8
        ... )
        >>> print(explanation)
        Claims have opposing polarity (confidence: 0.80). Semantic similarity: 0.85. Claims are highly related but present contradictory assertions
    """
    explanation_parts = []

    if has_polarity_opposition:
        explanation_parts.append(
            f"Claims have opposing polarity (confidence: {polarity_confidence:.2f})"
        )

    explanation_parts.append(
        f"Semantic similarity: {semantic_similarity:.2f}"
    )

    if severity in [ContradictionSeverity.CRITICAL, ContradictionSeverity.MAJOR]:
        explanation_parts.append(
            "Claims are highly related but present contradictory assertions"
        )
    elif severity == ContradictionSeverity.MODERATE:
        explanation_parts.append(
            "Claims show notable inconsistency requiring investigation"
        )
    else:
        explanation_parts.append(
            "Claims show minor inconsistency, may be due to different contexts"
        )

    return ". ".join(explanation_parts)


def generate_reconciliation_suggestion(
    severity: ContradictionSeverity,
) -> Optional[str]:
    """
    Generate reconciliation suggestion based on contradiction severity.

    Provides actionable guidance for investigating and resolving contradictions.
    Higher severity contradictions receive more urgent recommendations.

    Args:
        severity: Severity level of the contradiction

    Returns:
        Suggestion string for CRITICAL/MAJOR/MODERATE, None for MINOR

    Example:
        >>> suggestion = generate_reconciliation_suggestion(ContradictionSeverity.CRITICAL)
        >>> print(suggestion)
        Investigate source reliability and experimental conditions. One or both claims may be incorrect.
    """
    if severity == ContradictionSeverity.CRITICAL:
        return (
            "Investigate source reliability and experimental conditions. "
            "One or both claims may be incorrect."
        )
    elif severity == ContradictionSeverity.MAJOR:
        return (
            "Review source contexts and methodologies. "
            "Claims may apply to different domains or conditions."
        )
    elif severity == ContradictionSeverity.MODERATE:
        return (
            "Check for temporal differences or scope variations between sources."
        )

    # No suggestion for MINOR severity
    return None


def detect_contradiction(
    claim_1_id: str,
    claim_1_text: str,
    claim_2_id: str,
    claim_2_text: str,
    similarity_threshold: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[Contradiction]:
    """
    Detect contradiction between two claims.

    Main entry point for contradiction detection. Analyzes semantic
    similarity and polarity opposition to determine if claims contradict.

    Args:
        claim_1_id: Identifier for first claim
        claim_1_text: Text of first claim
        claim_2_id: Identifier for second claim
        claim_2_text: Text of second claim
        similarity_threshold: Minimum similarity to consider (default: 0.3)
        model_name: Sentence transformer model to use

    Returns:
        Contradiction object if contradiction detected, None otherwise

    Detection Logic:
        1. Compute semantic similarity
        2. If similarity < threshold: likely unrelated, return None
        3. Detect polarity opposition
        4. If no polarity opposition and low similarity: return None
        5. Assess severity
        6. Generate explanation
        7. Return Contradiction object

    Example:
        >>> contra = detect_contradiction(
        ...     "claim-1", "AI improves accuracy by 23%",
        ...     "claim-2", "AI reduces accuracy by 15%"
        ... )
        >>> if contra:
        ...     print(f"Severity: {contra.severity}")
        ...     print(f"Confidence: {contra.confidence}")
        Severity: ContradictionSeverity.CRITICAL
        Confidence: 0.87
    """
    # Import semantic functions
    compute_claim_similarity, _ = _import_semantic_functions()

    # Compute semantic similarity
    similarity = compute_claim_similarity(claim_1_text, claim_2_text, model_name=model_name)

    # If very low similarity, likely unrelated
    if similarity < similarity_threshold:
        return None

    # Detect polarity opposition
    has_polarity, polarity_conf = detect_polarity_opposition(claim_1_text, claim_2_text)

    # If no polarity opposition detected, not a contradiction
    # (Supporting or related claims, not contradictory)
    if not has_polarity:
        return None

    # Assess severity
    severity = assess_contradiction_severity(similarity, has_polarity, polarity_conf)

    # Compute overall confidence
    # Weighted average: 60% semantic similarity, 40% polarity confidence
    confidence = (0.6 * similarity) + (0.4 * polarity_conf)

    # Generate explanation using dedicated function
    explanation = generate_contradiction_explanation(
        severity=severity,
        semantic_similarity=similarity,
        has_polarity_opposition=has_polarity,
        polarity_confidence=polarity_conf,
    )

    # Generate resolution suggestion using dedicated function
    resolution_suggestion = generate_reconciliation_suggestion(severity)

    # Create contradiction object
    contradiction = Contradiction(
        contradiction_id=f"contra-{claim_1_id}-{claim_2_id}",
        claim_1_id=claim_1_id,
        claim_2_id=claim_2_id,
        claim_1_text=claim_1_text,
        claim_2_text=claim_2_text,
        severity=severity,
        confidence=confidence,
        explanation=explanation,
        resolution_suggestion=resolution_suggestion,
        metadata={
            "semantic_similarity": round(similarity, 3),
            "has_polarity_opposition": has_polarity,
            "polarity_confidence": round(polarity_conf, 3),
            "detection_method": "semantic_similarity_with_polarity",
            "model_name": model_name,
        },
    )

    return contradiction


def detect_contradictions_batch(
    claims: List[Tuple[str, str]],
    similarity_threshold: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Contradiction]:
    """
    Detect contradictions in a batch of claims.

    Efficiently compares all pairs of claims to find contradictions.
    Useful for analyzing collections of claims from multiple sources.

    Args:
        claims: List of (claim_id, claim_text) tuples
        similarity_threshold: Minimum similarity to consider (default: 0.3)
        model_name: Sentence transformer model to use

    Returns:
        List of detected Contradiction objects

    Example:
        >>> claims = [
        ...     ("claim-1", "AI improves accuracy"),
        ...     ("claim-2", "AI reduces accuracy"),
        ...     ("claim-3", "Weather is sunny"),
        ... ]
        >>> contradictions = detect_contradictions_batch(claims)
        >>> print(f"Found {len(contradictions)} contradictions")
        Found 1 contradictions
    """
    contradictions = []

    # Compare all pairs
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            claim_1_id, claim_1_text = claims[i]
            claim_2_id, claim_2_text = claims[j]

            contradiction = detect_contradiction(
                claim_1_id, claim_1_text,
                claim_2_id, claim_2_text,
                similarity_threshold=similarity_threshold,
                model_name=model_name,
            )

            if contradiction is not None:
                contradictions.append(contradiction)

    return contradictions
