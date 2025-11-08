"""
Persona implementations for STUDY workflow.

This package contains concrete persona implementations for the study workflow,
including Researcher, Critic, and other specialized investigation personas.
"""

from .researcher import ResearcherPersona, create_researcher

__all__ = [
    "ResearcherPersona",
    "create_researcher",
]
