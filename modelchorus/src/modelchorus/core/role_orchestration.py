"""
Role-based orchestration framework for sequential and parallel model execution.

Provides infrastructure for coordinating multiple AI models in different roles,
enabling workflows like ARGUMENT (multi-perspective debate), IDEATE (creative
brainstorming with diverse viewpoints), and RESEARCH (multi-source investigation).

This module defines the core abstractions for role assignment, stance configuration,
and orchestration patterns that enable sophisticated multi-model workflows.

Public API:
    - ModelRole: Data class defining a model's role, stance, and prompt customization
    - RoleOrchestrator: Coordinator for executing models in assigned roles
    - OrchestrationPattern: Enum for execution patterns (sequential/parallel/hybrid)
"""

from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class OrchestrationPattern(str, Enum):
    """
    Execution patterns for multi-model orchestration.

    Defines how multiple models with assigned roles are coordinated
    during workflow execution. Different patterns enable different
    collaboration strategies.

    Values:
        SEQUENTIAL: Execute models one at a time in defined order
                   (e.g., analyst → critic → synthesizer)
        PARALLEL: Execute all models concurrently, then aggregate
                 (e.g., multiple experts providing simultaneous input)
        HYBRID: Mix of sequential and parallel phases
               (e.g., parallel research → sequential debate → parallel voting)
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class ModelRole(BaseModel):
    """
    Data class defining a model's role, stance, and prompt customization.

    Represents a specific role assignment for an AI model in a multi-model
    workflow. Includes the role name, optional stance (for/against/neutral),
    and prompt customization to guide the model's behavior.

    Used in workflows like ARGUMENT (models with different stances),
    IDEATE (models with different creative perspectives), and RESEARCH
    (models focusing on different aspects of investigation).

    Attributes:
        role: Descriptive name for this role (e.g., "proponent", "critic", "synthesizer")
        model: Model identifier to assign to this role (e.g., "gpt-5", "gemini-2.5-pro")
        stance: Optional stance for debate-style workflows ("for", "against", "neutral")
        stance_prompt: Optional additional prompt text to reinforce the stance
        system_prompt: Optional system-level prompt for this role
        temperature: Optional temperature override for this role (0.0-1.0)
        max_tokens: Optional max tokens override for this role
        metadata: Additional metadata for this role (tags, priority, etc.)

    Example:
        >>> proponent = ModelRole(
        ...     role="proponent",
        ...     model="gpt-5",
        ...     stance="for",
        ...     stance_prompt="You are advocating FOR the proposal. Present strong supporting arguments."
        ... )
        >>> critic = ModelRole(
        ...     role="critic",
        ...     model="gemini-2.5-pro",
        ...     stance="against",
        ...     stance_prompt="You are critically analyzing AGAINST the proposal. Identify weaknesses and risks."
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "proponent",
                "model": "gpt-5",
                "stance": "for",
                "stance_prompt": "You are advocating FOR the proposal. Present strong supporting arguments.",
                "system_prompt": "You are an expert debater focused on building compelling cases.",
                "temperature": 0.8,
                "max_tokens": 4000,
                "metadata": {
                    "priority": 1,
                    "tags": ["debate", "advocacy"],
                },
            }
        }
    )

    role: str = Field(
        ...,
        description="Descriptive name for this role (e.g., 'proponent', 'critic', 'synthesizer')",
        min_length=1,
        max_length=100,
    )

    model: str = Field(
        ...,
        description="Model identifier to assign to this role (e.g., 'gpt-5', 'gemini-2.5-pro')",
        min_length=1,
    )

    stance: Optional[str] = Field(
        default=None,
        description="Optional stance for debate-style workflows ('for', 'against', 'neutral')",
    )

    stance_prompt: Optional[str] = Field(
        default=None,
        description="Optional additional prompt text to reinforce the stance",
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-level prompt for this role",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Optional temperature override for this role (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Optional max tokens override for this role",
        gt=0,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this role (tags, priority, constraints, etc.)",
    )

    @field_validator("stance")
    @classmethod
    def validate_stance(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate stance is one of the allowed values.

        Ensures stance, if provided, is one of the standard values
        for consistency across workflows.

        Args:
            v: Stance value to validate

        Returns:
            Validated stance value (lowercase)

        Raises:
            ValueError: If stance is not one of the allowed values
        """
        if v is None:
            return v

        allowed_stances = {"for", "against", "neutral"}
        stance_lower = v.lower()

        if stance_lower not in allowed_stances:
            raise ValueError(
                f"Stance must be one of {allowed_stances}, got '{v}'"
            )

        return stance_lower

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        """
        Validate temperature is in valid range.

        Ensures temperature, if provided, is between 0.0 and 1.0 inclusive.
        Pydantic's ge/le constraints handle this, but explicit validator
        provides clearer error messages.

        Args:
            v: Temperature value to validate

        Returns:
            Validated temperature value

        Raises:
            ValueError: If temperature is outside [0.0, 1.0] range
        """
        if v is None:
            return v

        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {v}")

        return v

    def get_full_prompt(self, base_prompt: str) -> str:
        """
        Construct full prompt by combining base prompt with role customizations.

        Merges the workflow's base prompt with this role's stance prompt
        and system prompt to create the complete prompt sent to the model.

        Args:
            base_prompt: The base prompt from the workflow

        Returns:
            Complete prompt string with role customizations applied

        Example:
            >>> role = ModelRole(
            ...     role="critic",
            ...     model="gpt-5",
            ...     stance="against",
            ...     stance_prompt="Identify weaknesses and risks.",
            ...     system_prompt="You are a critical analyst."
            ... )
            >>> full_prompt = role.get_full_prompt("Analyze this proposal: ...")
            >>> print(full_prompt)
            You are a critical analyst.

            Identify weaknesses and risks.

            Analyze this proposal: ...
        """
        prompt_parts = []

        # Add system prompt if provided
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)

        # Add stance prompt if provided
        if self.stance_prompt:
            prompt_parts.append(self.stance_prompt)

        # Add base prompt
        prompt_parts.append(base_prompt)

        return "\n\n".join(prompt_parts)
