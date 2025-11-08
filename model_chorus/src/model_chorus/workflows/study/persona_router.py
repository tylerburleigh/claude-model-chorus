"""
Persona router for Study workflow.

This module provides persona routing functionality, determining which persona
to consult next based on current investigation state using the context analysis skill.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ...core.models import StudyState
from .persona_base import Persona, PersonaRegistry
from .context_analysis import ContextAnalysisInput, ContextAnalysisResult, analyze_context

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
    """

    persona: Optional[Persona]
    persona_name: str
    reasoning: str
    confidence: str
    guidance: List[str]
    context_summary: str
    metadata: Dict[str, Any]


class PersonaRouter:
    """
    Router for determining which persona to consult next in Study workflow.

    Uses context analysis skill to intelligently route to appropriate personas
    based on investigation phase, confidence level, findings, and prior consultations.

    The router integrates the context analysis logic with the persona registry,
    providing a complete routing solution from state analysis to persona retrieval.

    Attributes:
        registry: PersonaRegistry containing available personas
    """

    def __init__(self, registry: PersonaRegistry):
        """
        Initialize PersonaRouter with a persona registry.

        Args:
            registry: PersonaRegistry containing available personas
        """
        self.registry = registry
        logger.info("PersonaRouter initialized")

    def route_next_persona(
        self,
        state: StudyState,
        unresolved_questions: Optional[List[str]] = None
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

        # Invoke context analysis skill
        context_input = ContextAnalysisInput(
            current_phase=current_phase,
            confidence=confidence,
            findings=findings,
            unresolved_questions=questions,
            prior_persona=prior_persona
        )

        analysis_result: ContextAnalysisResult = analyze_context(context_input)

        # Log routing decision
        logger.info(
            f"Context analysis recommended: {analysis_result.recommended_persona}"
        )
        logger.debug(f"Reasoning: {analysis_result.reasoning}")

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
                metadata=analysis_result.metadata
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

        return RoutingDecision(
            persona=persona,
            persona_name=persona_name,
            reasoning=analysis_result.reasoning,
            confidence=analysis_result.confidence,
            guidance=analysis_result.guidance,
            context_summary=analysis_result.context_summary,
            metadata=analysis_result.metadata
        )

    def get_available_personas(self) -> List[str]:
        """
        Get list of all available persona names in the registry.

        Returns:
            List of persona names
        """
        return [persona.name for persona in self.registry.list_all()]
