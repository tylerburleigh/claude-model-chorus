"""
Contradiction detection models for ARGUMENT workflow.

Provides data models for tracking contradictions between claims across
sources, with severity classification and resolution tracking.

Used in ARGUMENT workflow to identify and analyze conflicting claims,
assess contradiction severity, and suggest resolutions.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


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
