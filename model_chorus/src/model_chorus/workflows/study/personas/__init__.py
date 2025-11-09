"""
Persona implementations for STUDY workflow.

This package contains concrete persona implementations for the study workflow,
including Researcher, Critic, Planner, and other specialized investigation personas.
"""

from ..persona_base import PersonaRegistry
from .researcher import ResearcherPersona, create_researcher
from .critic import CriticPersona, create_critic
from .planner import PlannerPersona, create_planner

__all__ = [
    "ResearcherPersona",
    "create_researcher",
    "CriticPersona",
    "create_critic",
    "PlannerPersona",
    "create_planner",
    "get_default_registry",
    "create_default_personas",
]


def create_default_personas() -> list:
    """
    Create the default set of personas for STUDY workflow.

    Returns:
        List of default persona instances (Researcher, Critic, Planner)
    """
    return [
        create_researcher(),
        create_critic(),
        create_planner(),
    ]


def get_default_registry() -> PersonaRegistry:
    """
    Get a registry pre-populated with default personas.

    Returns:
        PersonaRegistry with Researcher, Critic, and Planner registered
    """
    registry = PersonaRegistry()
    for persona in create_default_personas():
        registry.register(persona)
    return registry
