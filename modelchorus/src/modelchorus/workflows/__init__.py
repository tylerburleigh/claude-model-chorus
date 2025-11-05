"""
Workflow implementations for ModelChorus.

This module contains specific workflow types such as thinkdeep, debug,
consensus, codereview, precommit, and planner.
"""

from .consensus import (
    ConsensusWorkflow,
    ConsensusStrategy,
    ConsensusResult,
    ProviderConfig,
)

__all__ = [
    "ConsensusWorkflow",
    "ConsensusStrategy",
    "ConsensusResult",
    "ProviderConfig",
]
