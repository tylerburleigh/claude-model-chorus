"""
Workflow implementations for ModelChorus.

This module contains specific workflow types such as thinkdeep, debug,
consensus, codereview, precommit, and planner.
"""

from .argument import ArgumentWorkflow
from .chat import ChatWorkflow
from .consensus import (
    ConsensusWorkflow,
    ConsensusStrategy,
    ConsensusResult,
    ProviderConfig,
)
from .ideate import IdeateWorkflow
from .study import StudyWorkflow
from .thinkdeep import ThinkDeepWorkflow
from ..core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    InvestigationPhase,
    StudyState,
    ThinkDeepState,
)

__all__ = [
    "ArgumentWorkflow",
    "ChatWorkflow",
    "ConsensusWorkflow",
    "ConsensusStrategy",
    "ConsensusResult",
    "ProviderConfig",
    "IdeateWorkflow",
    "StudyWorkflow",
    "ThinkDeepWorkflow",
    "ConfidenceLevel",
    "Hypothesis",
    "InvestigationStep",
    "InvestigationPhase",
    "StudyState",
    "ThinkDeepState",
]
