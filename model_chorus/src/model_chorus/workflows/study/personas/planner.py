"""
Planner persona implementation for STUDY workflow.

The Planner persona focuses on synthesizing findings into actionable roadmaps,
creating structured plans, and defining next steps.
"""

from typing import Dict, Any, List
from ..persona_base import Persona, PersonaResponse


class PlannerPersona(Persona):
    """
    Planner persona with actionable roadmap focus.

    This persona specializes in:
    - Synthesizing findings into coherent plans
    - Defining actionable next steps
    - Creating structured roadmaps
    - Prioritizing actions and recommendations
    - Translating insights into practical outcomes

    The Planner persona approaches investigations with a solution-oriented mindset,
    focusing on turning knowledge into actionable strategies.
    """

    def __init__(self, temperature: float = 0.7, max_tokens: int = 4096):
        """
        Initialize the Planner persona.

        Args:
            temperature: Generation temperature (default: 0.7 for creative planning)
            max_tokens: Maximum tokens per response (default: 4096)
        """
        prompt_template = """You are a strategic Planner creating actionable roadmaps.

Your role:
- Synthesize findings into coherent plans
- Define clear, actionable next steps
- Prioritize actions by impact and feasibility
- Create structured implementation roadmaps
- Translate insights into practical outcomes

Planning approach:
1. Review all findings and insights
2. Identify key themes and patterns
3. Define concrete action items
4. Prioritize by value and dependencies
5. Create structured roadmap with clear steps

Provide plans in a clear, actionable format with specific recommendations."""

        super().__init__(
            name="Planner",
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, context: Dict[str, Any]) -> PersonaResponse:
        """
        Invoke the Planner persona with investigation context.

        The Planner synthesizes all findings into actionable plans and recommendations,
        typically operating in the PLANNING phase to prepare for completion.

        Args:
            context: Investigation context containing:
                - prompt: Research question or topic
                - history: Conversation history
                - phase: Current investigation phase (ideally "planning")
                - state: Current investigation state
                - findings: All accumulated findings to synthesize

        Returns:
            PersonaResponse with actionable plan and high confidence

        Note:
            This is a placeholder implementation. Full implementation will
            integrate with providers for actual LLM-powered planning.
        """
        # TODO: Implement actual LLM provider integration
        # For now, return structured placeholder

        prompt = context.get("prompt", "")
        phase = context.get("phase", "planning")
        all_findings = context.get("findings", [])

        # Placeholder planning output based on investigation phase
        findings = [
            f"[Planner] Creating roadmap for: {prompt}",
            f"[Planner] Investigation phase: {phase}",
            "[Planner] Placeholder: Action plan would be generated here",
        ]

        # If there are findings to synthesize, acknowledge them
        if all_findings:
            findings.append(f"[Planner] Synthesizing {len(all_findings)} finding(s) into roadmap")
            findings.append("[Planner] Next steps: (placeholder)")

        # Placeholder confidence assessment
        # Planner typically works in PLANNING phase and confirms high confidence
        confidence_update = None
        if phase == "planning":
            # In planning phase, planner confirms readiness for completion
            confidence_update = "high"  # Ready to complete with clear plan
        elif phase == "validation":
            # Planner can also help validate with structured approach
            confidence_update = "high"  # Validated through planning lens

        return PersonaResponse(
            findings=findings,
            confidence_update=confidence_update,
            metadata={
                "persona": self.name,
                "phase": phase,
                "approach": "synthesis_and_planning",
                "findings_synthesized": len(all_findings),
            },
        )


# Factory function for creating Planner persona instances
def create_planner(temperature: float = 0.7, max_tokens: int = 4096) -> PlannerPersona:
    """
    Factory function to create a Planner persona instance.

    Args:
        temperature: Generation temperature (default: 0.7 for creative planning)
        max_tokens: Maximum tokens per response (default: 4096)

    Returns:
        Configured PlannerPersona instance
    """
    return PlannerPersona(temperature=temperature, max_tokens=max_tokens)
