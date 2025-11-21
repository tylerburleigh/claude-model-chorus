"""
Configuration dataclasses for STUDY workflow.

This module defines configuration options and defaults for persona-based
collaborative research investigations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonaConfig:
    """
    Configuration for a single persona in the study workflow.

    Attributes:
        name: Persona name (e.g., "Researcher", "Critic")
        expertise: Domain expertise description
        role: Role in investigation (e.g., "primary investigator", "critical reviewer")
        system_prompt: Optional custom system prompt for this persona
        temperature: Optional temperature override for this persona
        metadata: Additional persona-specific metadata
    """

    name: str
    expertise: str
    role: str
    system_prompt: str | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyConfig:
    """
    Configuration for STUDY workflow execution.

    Defines default settings and behavior for persona-based investigations,
    including persona configurations, iteration limits, and confidence thresholds.

    Attributes:
        personas: List of persona configurations to use
        max_iterations: Maximum number of investigation iterations
        min_confidence_for_completion: Minimum confidence level to complete investigation
        enable_cross_persona_dialogue: Whether personas should interact with each other
        temperature: Default temperature for generation (can be overridden per persona)
        max_tokens: Maximum tokens per persona response
        enable_file_analysis: Whether to enable file examination capabilities
        collaboration_rounds: Number of rounds of cross-persona collaboration
        metadata: Additional workflow-specific metadata
    """

    personas: list[PersonaConfig] = field(
        default_factory=lambda: [
            PersonaConfig(
                name="Researcher",
                expertise="systematic investigation and analysis",
                role="primary investigator",
                temperature=0.7,
            ),
            PersonaConfig(
                name="Critic",
                expertise="identifying assumptions and edge cases",
                role="critical reviewer",
                temperature=0.6,
            ),
        ]
    )

    max_iterations: int = 5
    min_confidence_for_completion: str = "high"  # ConfidenceLevel value
    enable_cross_persona_dialogue: bool = True
    temperature: float = 0.7
    max_tokens: int = 4096
    enable_file_analysis: bool = True
    collaboration_rounds: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationResult:
    """
    Result from a study workflow investigation.

    Contains all findings, persona contributions, and investigation metadata
    from a completed investigation session.

    Attributes:
        investigation_id: Unique identifier for this investigation
        final_phase: Final investigation phase reached
        final_confidence: Final confidence level achieved
        iteration_count: Number of iterations completed
        persona_findings: Dict mapping persona names to their findings
        synthesis: Final synthesis of all findings
        relevant_files: Files examined during investigation
        metadata: Additional result metadata
    """

    investigation_id: str
    final_phase: str
    final_confidence: str
    iteration_count: int
    persona_findings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    synthesis: str | None = None
    relevant_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
