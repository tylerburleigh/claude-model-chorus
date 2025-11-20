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

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
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
                    "Token expiration check missing in some endpoints",
                ],
                "unresolved_questions": [
                    "Which endpoints lack token validation?",
                    "Is there a centralized auth middleware?",
                ],
                "prior_persona": "Researcher",
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
            raise ValueError(f"Invalid phase '{v}'. Must be one of: {', '.join(valid_phases)}")
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
        confidence: Current confidence level from context
        guidance: Specific guidance or focus areas for the persona
        metadata: Additional analysis metadata
    """

    recommended_persona: str
    reasoning: str
    context_summary: str
    confidence: str
    guidance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dictionary representation with all fields
        """
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert the result to a JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2, None for compact)

        Returns:
            JSON string representation

        Example:
            >>> result = ContextAnalysisResult(...)
            >>> print(result.to_json())
            {
              "recommended_persona": "Researcher",
              "reasoning": "In discovery phase...",
              "context_summary": "Phase: DISCOVERY...",
              "confidence": "medium",
              "guidance": ["Gather initial information"],
              "metadata": {...}
            }
        """
        return json.dumps(self.to_dict(), indent=indent)


def _select_persona_by_phase_and_state(
    phase: str, findings_count: int, has_questions: bool, prior_persona: Optional[str]
) -> tuple[str, str, List[str]]:
    """
    Select persona based on investigation phase and current state.

    Phase-Based Selection Strategy:
    - DISCOVERY: Start with Researcher for initial exploration
    - VALIDATION: Use Critic to challenge assumptions and findings
    - PLANNING: Use Planner to synthesize into actionable roadmap
    - COMPLETE: No further persona needed (investigation complete)

    Args:
        phase: Current investigation phase
        findings_count: Number of existing findings
        has_questions: Whether there are unresolved questions
        prior_persona: Previously consulted persona (to avoid repetition)

    Returns:
        Tuple of (persona_name, reasoning, guidance_list)
    """
    # Phase-based persona selection
    if phase == InvestigationPhase.DISCOVERY.value:
        # In discovery phase, alternate between Researcher and Critic
        if prior_persona == "Researcher" and findings_count > 0:
            # If Researcher just ran and we have findings, use Critic to challenge them
            return (
                "Critic",
                "After initial research findings, the Critic will challenge assumptions "
                "and identify potential gaps or edge cases.",
                [
                    "Challenge the initial findings",
                    "Look for edge cases or contradictions",
                    "Identify gaps in the current understanding",
                ],
            )
        else:
            # Default to Researcher for systematic exploration
            return (
                "Researcher",
                "In discovery phase, the Researcher will systematically gather information "
                "and build foundational understanding.",
                [
                    "Gather initial information systematically",
                    "Identify key patterns and relationships",
                    "Build comprehensive understanding",
                ],
            )

    elif phase == InvestigationPhase.VALIDATION.value:
        # In validation phase, primarily use Critic
        if prior_persona == "Critic":
            # If Critic just ran, use Researcher to fill identified gaps
            return (
                "Researcher",
                "After critical analysis, the Researcher will address identified gaps "
                "and strengthen findings with additional evidence.",
                [
                    "Address gaps identified by critical analysis",
                    "Gather supporting evidence for key findings",
                    "Validate assumptions with deeper investigation",
                ],
            )
        else:
            # Default to Critic for validation
            return (
                "Critic",
                "In validation phase, the Critic will rigorously test findings "
                "and ensure robustness before moving forward.",
                [
                    "Stress-test existing findings",
                    "Look for counterexamples or exceptions",
                    "Ensure conclusions are well-supported",
                ],
            )

    elif phase == InvestigationPhase.PLANNING.value:
        # In planning phase, primarily use Planner
        if prior_persona == "Planner" and has_questions:
            # If Planner just ran but questions remain, use Researcher or Critic
            if findings_count < 3:
                return (
                    "Researcher",
                    "Unresolved questions remain. The Researcher will gather additional "
                    "information before finalizing the plan.",
                    [
                        "Address remaining open questions",
                        "Gather any missing information",
                        "Complete the investigation picture",
                    ],
                )
            else:
                return (
                    "Critic",
                    "Before finalizing the plan, the Critic will ensure all aspects "
                    "have been thoroughly validated.",
                    [
                        "Validate the proposed plan",
                        "Identify potential risks or issues",
                        "Ensure nothing has been overlooked",
                    ],
                )
        else:
            # Default to Planner for synthesis
            return (
                "Planner",
                "In planning phase, the Planner will synthesize all findings "
                "into a coherent, actionable roadmap.",
                [
                    "Synthesize all findings into coherent plan",
                    "Define clear action items",
                    "Prioritize next steps by impact",
                ],
            )

    else:  # COMPLETE phase
        # Investigation is complete, no further persona needed
        return (
            "None",
            "Investigation is complete. No further persona consultation needed.",
            ["Review final results and close investigation"],
        )


def analyze_context(context_input: ContextAnalysisInput) -> ContextAnalysisResult:
    """
    Analyze investigation context and determine next persona to consult.

    This is the main entry point for the context analysis skill. Based on
    the current investigation phase, confidence level, findings, and
    unresolved questions, it selects the most appropriate persona to
    consult next.

    Selection Strategy:
    - DISCOVERY phase: Alternate between Researcher (exploration) and Critic (challenge)
    - VALIDATION phase: Primarily Critic (testing), with Researcher to fill gaps
    - PLANNING phase: Primarily Planner (synthesis), with fallback to Researcher/Critic if needed
    - COMPLETE phase: No further persona needed

    The logic also considers:
    - Prior persona to avoid repetition
    - Presence of findings (do we have material to work with?)
    - Presence of unresolved questions (do we need more investigation?)

    Args:
        context_input: Validated context analysis input

    Returns:
        ContextAnalysisResult with recommended persona and guidance
    """
    phase = context_input.current_phase
    findings_count = len(context_input.findings)
    has_questions = len(context_input.unresolved_questions) > 0
    prior_persona = context_input.prior_persona

    # Select persona based on phase and state
    persona, reasoning, guidance = _select_persona_by_phase_and_state(
        phase, findings_count, has_questions, prior_persona
    )

    # Build context summary
    context_summary = (
        f"Phase: {phase.upper()}, "
        f"Confidence: {context_input.confidence}, "
        f"{len(context_input.findings)} finding(s), "
        f"{len(context_input.unresolved_questions)} unresolved question(s)"
    )

    if prior_persona:
        context_summary += f", Prior: {prior_persona}"

    return ContextAnalysisResult(
        recommended_persona=persona,
        reasoning=reasoning,
        context_summary=context_summary,
        confidence=context_input.confidence,
        guidance=guidance,
        metadata={
            "phase": phase,
            "findings_count": findings_count,
            "has_questions": has_questions,
            "prior_persona": prior_persona,
            "selection_strategy": "phase_based_with_rotation",
        },
    )
