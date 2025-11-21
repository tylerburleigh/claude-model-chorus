"""
Core orchestration logic for ModelChorus workflows.

This module contains the main workflow engine and model orchestration
components that power multi-model AI workflows.
"""

from .base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from .contradiction import (
    Contradiction,
    ContradictionSeverity,
    assess_contradiction_severity,
    detect_contradiction,
    detect_contradictions_batch,
    detect_polarity_opposition,
    generate_contradiction_explanation,
    generate_reconciliation_suggestion,
)
from .conversation import ConversationMemory
from .gap_analysis import (
    Gap,
    GapSeverity,
    GapType,
    assess_gap_severity,
    detect_gaps,
    detect_logical_gaps,
    detect_missing_evidence,
    detect_unsupported_claims,
    generate_gap_recommendation,
)
from .models import (
    ConfidenceLevel,
    ConsensusConfig,
    ConversationMessage,
    ConversationState,
    ConversationThread,
    Hypothesis,
    InvestigationStep,
    ModelResponse,
    ModelSelection,
    ThinkDeepState,
    WorkflowRequest,
    WorkflowResponse,
)
from .models import (
    WorkflowStep as WorkflowStepModel,
)
from .prompts import get_read_only_system_prompt, prepend_system_constraints
from .registry import WorkflowRegistry

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowStep",
    "WorkflowRegistry",
    "get_read_only_system_prompt",
    "prepend_system_constraints",
    "ConfidenceLevel",
    "WorkflowRequest",
    "WorkflowResponse",
    "ModelSelection",
    "WorkflowStepModel",
    "ModelResponse",
    "ConsensusConfig",
    "ConversationMemory",
    "ConversationMessage",
    "ConversationThread",
    "ConversationState",
    "Hypothesis",
    "InvestigationStep",
    "ThinkDeepState",
    # Gap Analysis
    "Gap",
    "GapType",
    "GapSeverity",
    "detect_gaps",
    "detect_missing_evidence",
    "detect_logical_gaps",
    "detect_unsupported_claims",
    "assess_gap_severity",
    "generate_gap_recommendation",
    # Contradiction Detection
    "Contradiction",
    "ContradictionSeverity",
    "detect_contradiction",
    "detect_contradictions_batch",
    "detect_polarity_opposition",
    "assess_contradiction_severity",
    "generate_contradiction_explanation",
    "generate_reconciliation_suggestion",
]
