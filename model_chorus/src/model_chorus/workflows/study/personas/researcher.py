"""
Researcher persona implementation for STUDY workflow.

The Researcher persona focuses on systematic investigation, deep analysis,
and comprehensive exploration of topics.
"""

from typing import Dict, Any, List
from ..persona_base import Persona, PersonaResponse


class ResearcherPersona(Persona):
    """
    Researcher persona with deep analysis focus.

    This persona specializes in:
    - Systematic investigation and methodical exploration
    - Deep dive analysis of complex topics
    - Identifying patterns and connections
    - Building comprehensive understanding
    - Evidence-based reasoning

    The Researcher persona approaches investigations with rigor and thoroughness,
    seeking to uncover underlying principles and detailed insights.
    """

    def __init__(self, temperature: float = 0.7, max_tokens: int = 4096):
        """
        Initialize the Researcher persona.

        Args:
            temperature: Generation temperature (default: 0.7 for balanced exploration)
            max_tokens: Maximum tokens per response (default: 4096)
        """
        prompt_template = """You are a systematic Researcher conducting a thorough investigation.

Your role:
- Approach topics methodically and comprehensively
- Dig deep into details and underlying principles
- Identify patterns, connections, and relationships
- Build structured understanding from evidence
- Ask probing questions to uncover insights

Investigation approach:
1. Analyze the topic systematically
2. Break down complex concepts into components
3. Explore relationships and dependencies
4. Identify key insights and findings
5. Assess confidence based on evidence quality

Provide your findings in a clear, structured format with supporting reasoning."""

        super().__init__(
            name="Researcher",
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, context: Dict[str, Any]) -> PersonaResponse:
        """
        Invoke the Researcher persona with investigation context.

        The Researcher conducts systematic analysis and returns detailed findings
        with confidence assessments based on evidence quality.

        Args:
            context: Investigation context containing:
                - prompt: Research question or topic
                - history: Conversation history
                - phase: Current investigation phase
                - state: Current investigation state

        Returns:
            PersonaResponse with research findings and confidence update

        Note:
            This is a placeholder implementation. Full implementation will
            integrate with providers for actual LLM-powered research.
        """
        # TODO: Implement actual LLM provider integration
        # For now, return structured placeholder

        prompt = context.get("prompt", "")
        phase = context.get("phase", "discovery")

        # Placeholder findings based on investigation phase
        findings = [
            f"[Researcher] Systematic analysis of: {prompt}",
            f"[Researcher] Investigation phase: {phase}",
            "[Researcher] Placeholder: Deep analysis would occur here",
        ]

        # Placeholder confidence assessment
        # In real implementation, this would be based on actual findings
        confidence_update = None
        if phase == "discovery":
            confidence_update = "medium"  # Initial exploration confidence
        elif phase == "validation":
            confidence_update = "high"  # Validated findings confidence

        return PersonaResponse(
            findings=findings,
            confidence_update=confidence_update,
            metadata={"persona": self.name, "phase": phase, "approach": "systematic_analysis"},
        )


# Factory function for creating Researcher persona instances
def create_researcher(temperature: float = 0.7, max_tokens: int = 4096) -> ResearcherPersona:
    """
    Factory function to create a Researcher persona instance.

    Args:
        temperature: Generation temperature (default: 0.7)
        max_tokens: Maximum tokens per response (default: 4096)

    Returns:
        Configured ResearcherPersona instance
    """
    return ResearcherPersona(temperature=temperature, max_tokens=max_tokens)
