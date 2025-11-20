"""
Persona implementations for STUDY workflow.

This module defines the Persona dataclass and related persona management
for persona-based collaborative research.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class PersonaResponse:
    """
    Response from a persona invocation.

    Contains the persona's findings and any confidence level updates
    based on the investigation.

    Attributes:
        findings: List of findings or insights from the persona
        confidence_update: Optional confidence level change based on findings
        metadata: Additional response metadata
    """

    findings: List[str] = field(default_factory=list)
    confidence_update: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Persona:
    """
    Represents a persona in the STUDY workflow.

    A persona is a specialized investigator with specific expertise and
    characteristics that guide its contributions to the research process.

    Attributes:
        name: The persona's name (e.g., "Researcher", "Critic")
        prompt_template: Template for prompting this persona
        temperature: Temperature setting for generation (controls randomness)
        max_tokens: Maximum tokens in persona's responses
    """

    name: str
    prompt_template: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096

    def invoke(self, context: Dict[str, Any]) -> PersonaResponse:
        """
        Invoke the persona with the given context.

        This method executes the persona's investigation logic based on the
        provided context, returning findings and potential confidence updates.

        Args:
            context: Investigation context including prompt, history, etc.

        Returns:
            PersonaResponse containing findings and confidence updates

        Note:
            This is a placeholder implementation. Actual persona invocation
            logic will be implemented in subsequent tasks.
        """
        # Placeholder implementation
        # TODO: Implement actual persona invocation with provider
        return PersonaResponse(
            findings=[f"{self.name} placeholder finding"],
            confidence_update=None,
            metadata={"persona": self.name, "invoked": True},
        )


class PersonaRegistry:
    """
    Registry for managing available personas in the STUDY workflow.

    Provides centralized management of persona definitions, allowing
    registration, retrieval, and listing of available personas.
    """

    def __init__(self):
        """Initialize an empty persona registry."""
        self._personas: Dict[str, Persona] = {}

    def register(self, persona: Persona) -> None:
        """
        Register a new persona in the registry.

        Args:
            persona: The Persona instance to register

        Raises:
            ValueError: If a persona with the same name already exists
        """
        if persona.name in self._personas:
            raise ValueError(f"Persona '{persona.name}' is already registered")
        self._personas[persona.name] = persona

    def get(self, name: str) -> Optional[Persona]:
        """
        Retrieve a persona by name.

        Args:
            name: The name of the persona to retrieve

        Returns:
            The Persona instance if found, None otherwise
        """
        return self._personas.get(name)

    def list_all(self) -> List[Persona]:
        """
        List all registered personas.

        Returns:
            A list of all registered Persona instances
        """
        return list(self._personas.values())
