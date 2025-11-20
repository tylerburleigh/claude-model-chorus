"""
Persona router for Study workflow.

This module provides persona routing functionality, determining which persona
to consult next based on current investigation state using the context analysis skill.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...core.models import StudyState
from .context_analysis import (
    ContextAnalysisInput,
    ContextAnalysisResult,
    analyze_context,
)
from .persona_base import Persona, PersonaRegistry

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Result of persona routing decision.

    Contains the selected persona instance, reasoning for selection,
    and guidance for the persona invocation.

    Attributes:
        persona: The Persona instance to consult next (None if investigation complete)
        persona_name: Name of the selected persona
        reasoning: Explanation for why this persona was selected
        confidence: Current confidence level from investigation state
        guidance: Specific guidance or focus areas for the persona
        context_summary: Summary of the investigation context
        metadata: Additional routing metadata
        timestamp: When this routing decision was made (ISO format)
    """

    persona: Persona | None
    persona_name: str
    reasoning: str
    confidence: str
    guidance: list[str]
    context_summary: str
    metadata: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RoutingHistoryEntry:
    """
    Historical record of a routing decision.

    Tracks routing decisions over time for analysis and debugging.

    Attributes:
        timestamp: When the routing decision was made (ISO format)
        investigation_id: Investigation this routing was for
        phase: Investigation phase at time of routing
        confidence: Confidence level at time of routing
        findings_count: Number of findings at time of routing
        questions_count: Number of unresolved questions
        prior_persona: Previously consulted persona (if any)
        selected_persona: Persona selected by routing decision
        reasoning: Reasoning for the selection
        context_summary: Summary of investigation context
    """

    timestamp: str
    investigation_id: str
    phase: str
    confidence: str
    findings_count: int
    questions_count: int
    prior_persona: str | None
    selected_persona: str
    reasoning: str
    context_summary: str


class PersonaRouter:
    """
    Router for determining which persona to consult next in Study workflow.

    Uses context analysis skill to intelligently route to appropriate personas
    based on investigation phase, confidence level, findings, and prior consultations.

    The router integrates the context analysis logic with the persona registry,
    providing a complete routing solution from state analysis to persona retrieval.

    Attributes:
        registry: PersonaRegistry containing available personas
        routing_history: List of historical routing decisions for analysis
    """

    # Fallback mapping for when skill invocation fails
    # Maps investigation phase to appropriate persona
    FALLBACK_PERSONA_MAP = {
        "discovery": "Researcher",
        "validation": "Critic",
        "planning": "Planner",
        "complete": None,  # No persona needed when complete
    }

    def __init__(self, registry: PersonaRegistry):
        """
        Initialize PersonaRouter with a persona registry.

        Args:
            registry: PersonaRegistry containing available personas
        """
        self.registry = registry
        self.routing_history: list[RoutingHistoryEntry] = []
        logger.info("PersonaRouter initialized")

    def _get_fallback_persona(
        self, current_phase: str, confidence: str
    ) -> tuple[str | None, str, list[str]]:
        """
        Get fallback persona when context analysis skill fails.

        Uses simple rule-based mapping from investigation phase to persona:
        - DISCOVERY → researcher (for systematic exploration)
        - VALIDATION → critic (for rigorous testing)
        - PLANNING → planner (for synthesis and roadmap)
        - COMPLETE → None (investigation complete)

        For unknown phases, defaults to researcher as the safest option.

        Args:
            current_phase: Current investigation phase
            confidence: Current confidence level

        Returns:
            Tuple of (persona_name, reasoning, guidance_list)
        """
        # Normalize phase to lowercase for lookup
        phase_normalized = current_phase.lower()

        # Get fallback persona from map (default to Researcher for unknown phases)
        persona_name = self.FALLBACK_PERSONA_MAP.get(phase_normalized, "Researcher")

        # Build reasoning message
        reasoning = (
            f"Fallback routing used due to context analysis failure. "
            f"Selected '{persona_name}' based on phase '{current_phase}'. "
        )

        # Provide generic guidance based on persona
        if persona_name == "Researcher":
            guidance = [
                "Systematically gather information",
                "Identify key patterns and relationships",
                "Build comprehensive understanding",
            ]
            reasoning += (
                "Researcher is appropriate for exploration and information gathering."
            )
        elif persona_name == "Critic":
            guidance = [
                "Challenge assumptions and findings",
                "Look for edge cases or contradictions",
                "Ensure robustness of conclusions",
            ]
            reasoning += "Critic is appropriate for validation and testing."
        elif persona_name == "Planner":
            guidance = [
                "Synthesize findings into coherent plan",
                "Define clear action items",
                "Prioritize next steps",
            ]
            reasoning += "Planner is appropriate for synthesis and planning."
        elif persona_name is None:
            guidance = ["Investigation complete - no further action needed"]
            reasoning += "Investigation is complete."
        else:
            # Unknown persona (shouldn't happen with current map)
            guidance = ["Consult persona for investigation guidance"]

        logger.info(
            f"Fallback routing: phase='{current_phase}' → persona='{persona_name}'"
        )

        return (persona_name, reasoning, guidance)

    def route_next_persona(
        self, state: StudyState, unresolved_questions: list[str] | None = None
    ) -> RoutingDecision:
        """
        Determine next persona to consult based on investigation state.

        Analyzes the current investigation state and uses the context analysis
        skill to select the most appropriate persona for the next consultation.

        Args:
            state: Current StudyState containing investigation context
            unresolved_questions: Optional list of unresolved questions
                                 (extracted from state if not provided)

        Returns:
            RoutingDecision with selected persona and routing information

        Raises:
            ValueError: If routing decision selects unknown persona
        """
        # Extract current investigation context
        current_phase = state.current_phase
        confidence = state.confidence

        # Extract findings as simple strings
        # StudyState.findings is List[Dict[str, Any]] with structure:
        # {"persona": str, "finding": str, "evidence": List[str], "confidence": str}
        findings = [
            f"[{f.get('persona', 'Unknown')}] {f.get('finding', '')}"
            for f in state.findings
        ]

        # Determine prior persona (last persona consulted)
        prior_persona = None
        if state.personas_active:
            prior_persona = state.personas_active[-1]

        # Use unresolved_questions if provided, otherwise extract from state
        questions = unresolved_questions if unresolved_questions is not None else []

        logger.info(
            f"Routing decision requested: phase={current_phase}, "
            f"confidence={confidence}, findings={len(findings)}, "
            f"questions={len(questions)}, prior_persona={prior_persona}"
        )

        # Invoke context analysis skill with fallback handling
        analysis_result: ContextAnalysisResult | None = None
        used_fallback = False

        try:
            context_input = ContextAnalysisInput(
                current_phase=current_phase,
                confidence=confidence,
                findings=findings,
                unresolved_questions=questions,
                prior_persona=prior_persona,
            )

            analysis_result = analyze_context(context_input)

            # Log routing decision
            logger.info(
                f"Context analysis recommended: {analysis_result.recommended_persona}"
            )
            logger.debug(f"Reasoning: {analysis_result.reasoning}")

        except Exception as e:
            # Context analysis failed - use fallback routing
            logger.warning(
                f"Context analysis skill failed: {type(e).__name__}: {str(e)}. "
                f"Using fallback routing based on phase."
            )
            used_fallback = True

            # Get fallback persona based on current phase
            fallback_persona, fallback_reasoning, fallback_guidance = (
                self._get_fallback_persona(current_phase, confidence)
            )

            # Create synthetic analysis result from fallback
            analysis_result = ContextAnalysisResult(
                recommended_persona=fallback_persona or "None",
                reasoning=fallback_reasoning,
                context_summary=(
                    f"Phase: {current_phase.upper()}, "
                    f"Confidence: {confidence}, "
                    f"{len(findings)} finding(s), "
                    f"{len(questions)} unresolved question(s) "
                    f"(fallback routing)"
                ),
                confidence=confidence,
                guidance=fallback_guidance,
                metadata={
                    "fallback_used": True,
                    "fallback_reason": str(e),
                    "original_error_type": type(e).__name__,
                    "phase": current_phase,
                    "findings_count": len(findings),
                    "has_questions": len(questions) > 0,
                    "prior_persona": prior_persona,
                },
            )

            logger.info(
                f"Fallback routing selected: {analysis_result.recommended_persona}"
            )

        # Retrieve persona from registry
        persona_name = analysis_result.recommended_persona

        # Handle "None" case (investigation complete)
        if persona_name == "None":
            logger.info("Investigation complete - no further persona needed")
            return RoutingDecision(
                persona=None,
                persona_name="None",
                reasoning=analysis_result.reasoning,
                confidence=analysis_result.confidence,
                guidance=analysis_result.guidance,
                context_summary=analysis_result.context_summary,
                metadata=analysis_result.metadata,
            )

        # Retrieve persona instance
        persona = self.registry.get(persona_name)

        if persona is None:
            error_msg = (
                f"Routing decision selected persona '{persona_name}', "
                f"but persona not found in registry. "
                f"Available personas: {[p.name for p in self.registry.list_all()]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Persona '{persona_name}' retrieved from registry")

        decision = RoutingDecision(
            persona=persona,
            persona_name=persona_name,
            reasoning=analysis_result.reasoning,
            confidence=analysis_result.confidence,
            guidance=analysis_result.guidance,
            context_summary=analysis_result.context_summary,
            metadata=analysis_result.metadata,
        )

        # Record routing decision in history
        history_entry = RoutingHistoryEntry(
            timestamp=decision.timestamp,
            investigation_id=state.investigation_id,
            phase=current_phase,
            confidence=confidence,
            findings_count=len(findings),
            questions_count=len(questions),
            prior_persona=prior_persona,
            selected_persona=persona_name,
            reasoning=analysis_result.reasoning,
            context_summary=analysis_result.context_summary,
        )
        self.routing_history.append(history_entry)

        logger.info(
            f"Routing decision recorded: {persona_name} (history: {len(self.routing_history)} entries)"
        )

        return decision

    def get_available_personas(self) -> list[str]:
        """
        Get list of all available persona names in the registry.

        Returns:
            List of persona names
        """
        return [persona.name for persona in self.registry.list_all()]

    def get_routing_history(
        self, investigation_id: str | None = None, limit: int | None = None
    ) -> list[RoutingHistoryEntry]:
        """
        Get routing history, optionally filtered by investigation.

        Args:
            investigation_id: Filter by specific investigation (None for all)
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of RoutingHistoryEntry records
        """
        history = self.routing_history

        # Filter by investigation if specified
        if investigation_id:
            history = [
                entry for entry in history if entry.investigation_id == investigation_id
            ]

        # Reverse to get most recent first
        history = list(reversed(history))

        # Apply limit if specified
        if limit:
            history = history[:limit]

        return history

    def clear_routing_history(self, investigation_id: str | None = None) -> int:
        """
        Clear routing history, optionally for a specific investigation.

        Args:
            investigation_id: Clear only this investigation (None for all)

        Returns:
            Number of entries cleared
        """
        if investigation_id:
            initial_count = len(self.routing_history)
            self.routing_history = [
                entry
                for entry in self.routing_history
                if entry.investigation_id != investigation_id
            ]
            cleared = initial_count - len(self.routing_history)
        else:
            cleared = len(self.routing_history)
            self.routing_history = []

        logger.info(f"Cleared {cleared} routing history entries")
        return cleared
