"""
Context analysis skill for Study workflow persona consultation.

This module provides context analysis before persona invocation, determining
which persona to consult next based on current investigation state, findings,
and unresolved questions.

Skill Input Parameters:
    current_phase: Current investigation phase (discovery/validation/planning/complete)
    confidence: Current confidence level (0-100 or ConfidenceLevel enum value)
    findings: List of findings/insights discovered so far
    unresolved_questions: List of questions still needing investigation
    prior_persona: Previously consulted persona name (optional)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

from ...core.models import InvestigationPhase, ConfidenceLevel


class ContextAnalysisInput(BaseModel):
    """
    Input model for context analysis skill.

    Defines the investigation context needed to determine which persona
    to consult next based on current phase, confidence, and findings.

    Attributes:
        current_phase: Current investigation phase (discovery/validation/planning)
        confidence: Current confidence level (0-100 or ConfidenceLevel enum value)
        findings: List of findings/insights discovered so far in the investigation
        unresolved_questions: List of questions that still need investigation
        prior_persona: Name of the previously consulted persona (optional)
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "current_phase": "discovery",
                "confidence": "medium",
                "findings": [
                    "Authentication uses JWT tokens",
                    "Token expiration check missing in some endpoints"
                ],
                "unresolved_questions": [
                    "Which endpoints lack token validation?",
                    "Is there a centralized auth middleware?"
                ],
                "prior_persona": "Researcher"
            }
        }
    }

    current_phase: str = Field(
        ...,
        description="Current investigation phase (discovery/validation/planning/complete)",
    )

    confidence: str = Field(
        ...,
        description="Current confidence level (ConfidenceLevel enum value or 0-100)",
    )

    findings: List[str] = Field(
        default_factory=list,
        description="List of findings/insights discovered so far",
    )

    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="List of questions that still need investigation",
    )

    prior_persona: Optional[str] = Field(
        default=None,
        description="Name of the previously consulted persona (if any)",
    )

    @field_validator("current_phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        """
        Validate that current_phase is a valid InvestigationPhase value.

        Args:
            v: Phase value to validate

        Returns:
            Validated phase value

        Raises:
            ValueError: If phase is not a valid InvestigationPhase enum value
        """
        valid_phases = {phase.value for phase in InvestigationPhase}
        if v.lower() not in valid_phases:
            raise ValueError(
                f"Invalid phase '{v}'. Must be one of: {', '.join(valid_phases)}"
            )
        return v.lower()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        """
        Validate confidence value is either a ConfidenceLevel enum value or 0-100.

        Args:
            v: Confidence value to validate

        Returns:
            Validated confidence value

        Raises:
            ValueError: If confidence is neither a valid enum value nor 0-100 range
        """
        # Check if it's a valid ConfidenceLevel enum value
        valid_levels = {level.value for level in ConfidenceLevel}
        if v.lower() in valid_levels:
            return v.lower()

        # Check if it's a numeric value in 0-100 range
        try:
            numeric_value = int(v)
            if 0 <= numeric_value <= 100:
                return v
        except ValueError:
            pass

        raise ValueError(
            f"Invalid confidence '{v}'. Must be a ConfidenceLevel enum value "
            f"({', '.join(valid_levels)}) or a number 0-100"
        )


@dataclass
class ContextAnalysisResult:
    """
    Result of context analysis determining next persona to consult.

    Contains the recommended persona, reasoning for the selection,
    and any context-specific guidance for the persona invocation.

    Attributes:
        recommended_persona: Name of the persona to consult next
        reasoning: Explanation for why this persona was selected
        context_summary: Summary of current investigation context
        guidance: Specific guidance or focus areas for the persona
        metadata: Additional analysis metadata
    """

    recommended_persona: str
    reasoning: str
    context_summary: str
    guidance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def analyze_context(context_input: ContextAnalysisInput) -> ContextAnalysisResult:
    """
    Analyze investigation context and determine next persona to consult.

    This is the main entry point for the context analysis skill. Based on
    the current investigation phase, confidence level, findings, and
    unresolved questions, it selects the most appropriate persona to
    consult next.

    Args:
        context_input: Validated context analysis input

    Returns:
        ContextAnalysisResult with recommended persona and guidance

    Note:
        This is a placeholder implementation. The actual persona selection
        logic will be implemented in task-3-1-2 (Context Analysis Logic).
        For now, returns a basic result structure.
    """
    # TODO: Implement actual context analysis logic in task-3-1-2
    # This placeholder just demonstrates the interface structure

    return ContextAnalysisResult(
        recommended_persona="Researcher",  # Placeholder
        reasoning="Placeholder reasoning - actual logic to be implemented",
        context_summary=f"Phase: {context_input.current_phase}, "
                       f"Confidence: {context_input.confidence}, "
                       f"{len(context_input.findings)} findings, "
                       f"{len(context_input.unresolved_questions)} questions",
        guidance=["Focus on gathering initial information"],
        metadata={
            "phase": context_input.current_phase,
            "confidence": context_input.confidence,
            "has_prior_persona": context_input.prior_persona is not None,
        }
    )
