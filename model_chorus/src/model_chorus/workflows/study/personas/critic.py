"""
Critic persona implementation for STUDY workflow.

The Critic persona focuses on challenging assumptions, identifying edge cases,
and stress-testing conclusions through rigorous scrutiny.
"""

from typing import Any

from ..persona_base import Persona, PersonaResponse


class CriticPersona(Persona):
    """
    Critic persona with challenge and stress-test focus.

    This persona specializes in:
    - Challenging assumptions and identifying biases
    - Finding edge cases and potential problems
    - Stress-testing conclusions and hypotheses
    - Identifying gaps in reasoning
    - Providing constructive skepticism

    The Critic persona approaches investigations with healthy skepticism,
    seeking to strengthen findings by identifying weaknesses and alternatives.
    """

    def __init__(self, temperature: float = 0.6, max_tokens: int = 4096):
        """
        Initialize the Critic persona.

        Args:
            temperature: Generation temperature (default: 0.6 for focused critique)
            max_tokens: Maximum tokens per response (default: 4096)
        """
        prompt_template = """You are a rigorous Critic conducting thorough scrutiny.

Your role:
- Challenge assumptions and identify biases
- Find edge cases and potential failure points
- Stress-test conclusions through questioning
- Identify gaps, contradictions, and weaknesses
- Provide constructive skepticism to strengthen findings

Critical approach:
1. Question underlying assumptions
2. Look for edge cases and counterexamples
3. Identify logical gaps or inconsistencies
4. Consider alternative explanations
5. Assess confidence by evaluating robustness

Provide critiques in a clear, constructive format with specific concerns."""

        super().__init__(
            name="Critic",
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, context: dict[str, Any]) -> PersonaResponse:
        """
        Invoke the Critic persona with investigation context.

        The Critic examines findings critically and returns challenges,
        edge cases, and confidence adjustments based on robustness assessment.

        Args:
            context: Investigation context containing:
                - prompt: Research question or topic
                - history: Conversation history
                - phase: Current investigation phase
                - state: Current investigation state
                - findings: Any existing findings to critique

        Returns:
            PersonaResponse with critical analysis and confidence update

        Note:
            This is a placeholder implementation. Full implementation will
            integrate with providers for actual LLM-powered critique.
        """
        # TODO: Implement actual LLM provider integration
        # For now, return structured placeholder

        prompt = context.get("prompt", "")
        phase = context.get("phase", "discovery")
        existing_findings = context.get("findings", [])

        # Placeholder critical findings based on investigation phase
        findings = [
            f"[Critic] Critical examination of: {prompt}",
            f"[Critic] Investigation phase: {phase}",
            "[Critic] Placeholder: Edge case analysis would occur here",
        ]

        # If there are existing findings to critique, acknowledge them
        if existing_findings:
            findings.append(
                f"[Critic] Examining {len(existing_findings)} existing finding(s)"
            )

        # Placeholder confidence assessment
        # Critic typically reduces confidence by identifying uncertainties
        confidence_update = None
        if phase == "validation":
            # In validation phase, critic might lower confidence if issues found
            confidence_update = "medium"  # Reduced from potentially higher level
        elif phase == "planning":
            # In planning, critic ensures robustness before completion
            confidence_update = "high"  # Confirmed after critical review

        return PersonaResponse(
            findings=findings,
            confidence_update=confidence_update,
            metadata={
                "persona": self.name,
                "phase": phase,
                "approach": "critical_analysis",
                "findings_reviewed": len(existing_findings),
            },
        )


# Factory function for creating Critic persona instances
def create_critic(temperature: float = 0.6, max_tokens: int = 4096) -> CriticPersona:
    """
    Factory function to create a Critic persona instance.

    Args:
        temperature: Generation temperature (default: 0.6 for focused critique)
        max_tokens: Maximum tokens per response (default: 4096)

    Returns:
        Configured CriticPersona instance
    """
    return CriticPersona(temperature=temperature, max_tokens=max_tokens)
