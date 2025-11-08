"""
Gap analysis models and logic for ARGUMENT workflow.

Provides data models for identifying and tracking gaps in arguments,
including missing evidence, unsupported claims, and logical gaps.

Used in ARGUMENT workflow to identify weaknesses, assess coverage,
and suggest improvements for argument completeness.

Public API:
    - Gap: Pydantic model for gap representation
    - GapType: Enum for gap categories (EVIDENCE/LOGICAL/SUPPORT/ASSUMPTION)
    - GapSeverity: Enum for severity levels (MINOR/MODERATE/MAJOR/CRITICAL)
    - detect_gaps: Main entry point for detecting gaps in a set of claims
    - detect_missing_evidence: Identify claims lacking evidential support
    - detect_logical_gaps: Identify logical inconsistencies or missing steps
    - detect_unsupported_claims: Find claims without adequate citations
    - assess_gap_severity: Assess severity based on gap type and context
    - generate_gap_recommendation: Generate suggestions for filling gaps
"""

from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class GapType(str, Enum):
    """
    Types of gaps that can be detected in arguments.

    Classifies gaps by their nature to help prioritize remediation
    and guide improvement strategies.

    Values:
        EVIDENCE: Claim lacks supporting evidence or citations
        LOGICAL: Missing logical steps or reasoning gaps
        SUPPORT: Insufficient supporting arguments for main claim
        ASSUMPTION: Unstated or unjustified assumptions
    """

    EVIDENCE = "evidence"
    LOGICAL = "logical"
    SUPPORT = "support"
    ASSUMPTION = "assumption"


class GapSeverity(str, Enum):
    """
    Severity levels for gaps in arguments.

    Classifies gaps by their impact on argument validity and persuasiveness.
    Higher severity indicates more critical gaps requiring immediate attention.

    Values:
        MINOR: Small gap with minimal impact on argument strength.
               May improve clarity but not essential.
        MODERATE: Notable gap that weakens the argument.
                 Should be addressed to improve persuasiveness.
        MAJOR: Significant gap that undermines argument validity.
              Must be addressed for credible argumentation.
        CRITICAL: Fundamental gap that invalidates the argument.
                 Immediate attention required.
    """

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class Gap(BaseModel):
    """
    Model for tracking gaps in arguments.

    Represents a detected gap in reasoning, evidence, or support,
    including severity assessment, confidence in detection, and
    recommendations for improvement.

    Attributes:
        gap_id: Unique identifier for this gap
        gap_type: Type of gap (evidence/logical/support/assumption)
        severity: Severity level of the gap
        claim_id: Identifier of the claim with the gap
        claim_text: Full text of the claim
        description: Detailed description of what's missing
        recommendation: Suggestion for addressing the gap
        confidence: Confidence in gap detection (0.0-1.0)
        metadata: Additional metadata (detection_method, context, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gap_id": "gap-001",
                "gap_type": "evidence",
                "severity": "major",
                "claim_id": "claim-123",
                "claim_text": "Universal basic income reduces poverty",
                "description": "Claim lacks empirical evidence or citations to support the assertion",
                "recommendation": "Add citations to studies showing UBI impact on poverty rates",
                "confidence": 0.85,
                "metadata": {
                    "detection_method": "citation_analysis",
                    "detected_at": "2025-11-06T18:00:00Z",
                    "citation_count": 0,
                    "expected_citations": 2,
                },
            }
        }
    )

    gap_id: str = Field(
        ...,
        description="Unique identifier for this gap",
        min_length=1,
    )

    gap_type: GapType = Field(
        ...,
        description="Type of gap (evidence/logical/support/assumption)",
    )

    severity: GapSeverity = Field(
        ...,
        description="Severity level of the gap (minor/moderate/major/critical)",
    )

    claim_id: str = Field(
        ...,
        description="Identifier of the claim with the gap",
        min_length=1,
    )

    claim_text: str = Field(
        ...,
        description="Full text of the claim",
        min_length=1,
    )

    description: str = Field(
        ...,
        description="Detailed description of what's missing or incomplete",
        min_length=1,
    )

    recommendation: str = Field(
        ...,
        description="Suggestion for addressing or filling the gap",
        min_length=1,
    )

    confidence: float = Field(
        ...,
        description="Confidence in gap detection (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (detection_method, context, scores, etc.)",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """
        Validate confidence is in valid range.

        Ensures confidence score is between 0.0 and 1.0 inclusive.

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


# ============================================================================
# Gap Detection Logic
# ============================================================================


def _import_citation_map():
    """Import CitationMap model."""
    try:
        from model_chorus.core.models import CitationMap
        return CitationMap
    except ImportError as e:
        raise ImportError(f"Cannot import CitationMap model: {e}")


def assess_gap_severity(
    gap_type: GapType,
    citation_count: int = 0,
    expected_citations: int = 2,
    has_supporting_logic: bool = True,
) -> GapSeverity:
    """
    Assess the severity of a gap based on type and context.

    Combines gap type, citation coverage, and logical support to
    classify gap severity.

    Args:
        gap_type: Type of gap detected
        citation_count: Number of citations supporting the claim
        expected_citations: Expected number of citations for this type of claim
        has_supporting_logic: Whether logical support/reasoning is present

    Returns:
        GapSeverity enum value

    Severity Rules:
        - EVIDENCE gap with 0 citations = MAJOR or CRITICAL
        - LOGICAL gap without supporting logic = MAJOR
        - SUPPORT gap = MODERATE
        - ASSUMPTION gap = MINOR to MODERATE

    Example:
        >>> severity = assess_gap_severity(
        ...     gap_type=GapType.EVIDENCE,
        ...     citation_count=0,
        ...     expected_citations=2
        ... )
        >>> print(severity)
        GapSeverity.MAJOR
    """
    if gap_type == GapType.EVIDENCE:
        if citation_count == 0:
            # No citations at all - very severe
            if expected_citations >= 2:
                return GapSeverity.CRITICAL
            else:
                return GapSeverity.MAJOR
        elif citation_count < expected_citations:
            # Insufficient citations
            return GapSeverity.MODERATE
        else:
            # Sufficient citations
            return GapSeverity.MINOR

    elif gap_type == GapType.LOGICAL:
        if not has_supporting_logic:
            return GapSeverity.MAJOR
        else:
            return GapSeverity.MODERATE

    elif gap_type == GapType.SUPPORT:
        # Missing supporting arguments
        return GapSeverity.MODERATE

    elif gap_type == GapType.ASSUMPTION:
        # Unstated assumptions - varies by context
        if not has_supporting_logic:
            return GapSeverity.MODERATE
        else:
            return GapSeverity.MINOR

    # Default to MODERATE
    return GapSeverity.MODERATE


def generate_gap_recommendation(
    gap_type: GapType,
    severity: GapSeverity,
    claim_text: str,
) -> str:
    """
    Generate actionable recommendation for addressing a gap.

    Provides specific guidance based on gap type and severity to help
    improve argument completeness and validity.

    Args:
        gap_type: Type of gap detected
        severity: Severity level of the gap
        claim_text: Text of the claim with the gap

    Returns:
        Recommendation string with actionable guidance

    Example:
        >>> rec = generate_gap_recommendation(
        ...     GapType.EVIDENCE,
        ...     GapSeverity.MAJOR,
        ...     "AI improves accuracy"
        ... )
        >>> print(rec)
        Add empirical evidence with specific citations. Find peer-reviewed studies or data supporting this claim.
    """
    if gap_type == GapType.EVIDENCE:
        if severity in [GapSeverity.CRITICAL, GapSeverity.MAJOR]:
            return (
                "Add empirical evidence with specific citations. Find peer-reviewed studies "
                "or data supporting this claim."
            )
        else:
            return (
                "Consider adding additional citations to strengthen the claim's evidence base."
            )

    elif gap_type == GapType.LOGICAL:
        return (
            "Explain the logical connection between premises and conclusion. "
            "Add intermediate reasoning steps if needed."
        )

    elif gap_type == GapType.SUPPORT:
        return (
            "Provide supporting arguments or sub-claims that build toward the main assertion. "
            "Consider adding examples or analogies."
        )

    elif gap_type == GapType.ASSUMPTION:
        return (
            "Make implicit assumptions explicit. State and justify any underlying premises "
            "that the argument depends on."
        )

    # Default recommendation
    return "Review and strengthen this aspect of the argument."


def detect_missing_evidence(
    claim_id: str,
    claim_text: str,
    citations: Optional[List[Any]] = None,
    expected_citation_count: int = 1,
) -> Optional[Gap]:
    """
    Detect if a claim lacks adequate evidential support.

    Analyzes citation coverage to identify claims that need more
    empirical support or references.

    Args:
        claim_id: Identifier for the claim
        claim_text: Text of the claim to analyze
        citations: Optional list of Citation objects supporting this claim
        expected_citation_count: Minimum expected citations for this claim type

    Returns:
        Gap object if evidence gap detected, None otherwise

    Example:
        >>> gap = detect_missing_evidence(
        ...     "claim-1",
        ...     "AI reduces diagnostic errors by 40%",
        ...     citations=[]
        ... )
        >>> if gap:
        ...     print(f"Gap type: {gap.gap_type}, Severity: {gap.severity}")
        Gap type: GapType.EVIDENCE, Severity: GapSeverity.MAJOR
    """
    citation_count = len(citations) if citations else 0

    # Check if claim lacks sufficient evidence
    if citation_count < expected_citation_count:
        severity = assess_gap_severity(
            gap_type=GapType.EVIDENCE,
            citation_count=citation_count,
            expected_citations=expected_citation_count,
        )

        # Only report gap if moderate or higher
        if severity in [GapSeverity.MODERATE, GapSeverity.MAJOR, GapSeverity.CRITICAL]:
            description = (
                f"Claim has {citation_count} citation(s) but expects at least "
                f"{expected_citation_count}. Lacks adequate empirical support."
            )

            recommendation = generate_gap_recommendation(
                GapType.EVIDENCE, severity, claim_text
            )

            # Calculate confidence based on citation deficit
            confidence = min(0.9, 0.6 + (0.3 * (expected_citation_count - citation_count) / expected_citation_count))

            gap = Gap(
                gap_id=f"gap-evidence-{claim_id}",
                gap_type=GapType.EVIDENCE,
                severity=severity,
                claim_id=claim_id,
                claim_text=claim_text,
                description=description,
                recommendation=recommendation,
                confidence=confidence,
                metadata={
                    "citation_count": citation_count,
                    "expected_citations": expected_citation_count,
                    "detection_method": "citation_analysis",
                },
            )

            return gap

    return None


def detect_logical_gaps(
    claim_id: str,
    claim_text: str,
    supporting_claims: Optional[List[str]] = None,
) -> Optional[Gap]:
    """
    Detect logical gaps or missing reasoning steps.

    Analyzes whether a claim has adequate logical support or if there
    are missing steps in the reasoning chain.

    Args:
        claim_id: Identifier for the claim
        claim_text: Text of the claim to analyze
        supporting_claims: Optional list of supporting claim texts

    Returns:
        Gap object if logical gap detected, None otherwise

    Example:
        >>> gap = detect_logical_gaps(
        ...     "claim-1",
        ...     "Therefore, we should implement policy X",
        ...     supporting_claims=[]
        ... )
        >>> if gap:
        ...     print(f"Detected: {gap.description}")
    """
    has_support = supporting_claims is not None and len(supporting_claims) > 0

    # Check for conclusion indicators without support
    conclusion_indicators = [
        "therefore", "thus", "hence", "consequently", "as a result",
        "it follows that", "we conclude that", "in conclusion"
    ]

    text_lower = claim_text.lower()
    has_conclusion_indicator = any(indicator in text_lower for indicator in conclusion_indicators)

    # If claim appears to be a conclusion but lacks support, flag it
    if has_conclusion_indicator and not has_support:
        severity = assess_gap_severity(
            gap_type=GapType.LOGICAL,
            has_supporting_logic=False,
        )

        description = (
            "Claim presents a conclusion without explicit supporting premises or reasoning. "
            "Logical steps from evidence to conclusion are missing."
        )

        recommendation = generate_gap_recommendation(
            GapType.LOGICAL, severity, claim_text
        )

        gap = Gap(
            gap_id=f"gap-logical-{claim_id}",
            gap_type=GapType.LOGICAL,
            severity=severity,
            claim_id=claim_id,
            claim_text=claim_text,
            description=description,
            recommendation=recommendation,
            confidence=0.75,
            metadata={
                "has_supporting_claims": has_support,
                "conclusion_indicator_found": has_conclusion_indicator,
                "detection_method": "logical_structure_analysis",
            },
        )

        return gap

    return None


def detect_unsupported_claims(
    claims: List[tuple[str, str, Optional[List[Any]]]],
    min_citations_per_claim: int = 1,
) -> List[Gap]:
    """
    Detect unsupported claims in a collection.

    Batch analysis to identify all claims lacking adequate citation support.

    Args:
        claims: List of (claim_id, claim_text, citations) tuples
        min_citations_per_claim: Minimum expected citations per claim

    Returns:
        List of Gap objects for unsupported claims

    Example:
        >>> claims = [
        ...     ("claim-1", "AI improves accuracy", []),
        ...     ("claim-2", "Studies show benefits", [citation1, citation2]),
        ... ]
        >>> gaps = detect_unsupported_claims(claims)
        >>> print(f"Found {len(gaps)} unsupported claims")
    """
    gaps = []

    for claim_id, claim_text, citations in claims:
        gap = detect_missing_evidence(
            claim_id=claim_id,
            claim_text=claim_text,
            citations=citations,
            expected_citation_count=min_citations_per_claim,
        )

        if gap is not None:
            gaps.append(gap)

    return gaps


def detect_gaps(
    claims: List[Dict[str, Any]],
    min_citations_per_claim: int = 1,
) -> List[Gap]:
    """
    Main entry point for comprehensive gap detection.

    Analyzes a collection of claims to identify evidence gaps, logical gaps,
    and other weaknesses in argument structure.

    Args:
        claims: List of claim dictionaries with keys:
            - claim_id: Unique identifier
            - claim_text: Claim text
            - citations: Optional list of citations
            - supporting_claims: Optional list of supporting claim texts
        min_citations_per_claim: Minimum expected citations per claim

    Returns:
        List of all detected Gap objects

    Example:
        >>> claims = [
        ...     {
        ...         "claim_id": "claim-1",
        ...         "claim_text": "Universal basic income reduces poverty",
        ...         "citations": [],
        ...         "supporting_claims": []
        ...     }
        ... ]
        >>> gaps = detect_gaps(claims)
        >>> for gap in gaps:
        ...     print(f"{gap.gap_type}: {gap.description}")
    """
    all_gaps = []

    for claim in claims:
        claim_id = claim.get("claim_id", "unknown")
        claim_text = claim.get("claim_text", "")
        citations = claim.get("citations", [])
        supporting_claims = claim.get("supporting_claims", [])

        # Detect evidence gaps
        evidence_gap = detect_missing_evidence(
            claim_id=claim_id,
            claim_text=claim_text,
            citations=citations,
            expected_citation_count=min_citations_per_claim,
        )
        if evidence_gap:
            all_gaps.append(evidence_gap)

        # Detect logical gaps
        logical_gap = detect_logical_gaps(
            claim_id=claim_id,
            claim_text=claim_text,
            supporting_claims=supporting_claims,
        )
        if logical_gap:
            all_gaps.append(logical_gap)

    return all_gaps
