"""
Persona implementations for STUDY workflow.

This module defines the Persona dataclass and related persona management
for persona-based collaborative research.
"""

from dataclasses import dataclass
from typing import Optional


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
