"""
Core orchestration logic for ModelChorus workflows.

This module contains the main workflow engine and model orchestration
components that power multi-model AI workflows.
"""

from .base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from .registry import WorkflowRegistry
from .models import (
    ConfidenceLevel,
    WorkflowRequest,
    WorkflowResponse,
    ModelSelection,
    WorkflowStep as WorkflowStepModel,
    ModelResponse,
    ConsensusConfig,
    ConversationMessage,
    ConversationThread,
    ConversationState,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)
from .conversation import ConversationMemory
from .gap_analysis import (
    Gap,
    GapType,
    GapSeverity,
    detect_gaps,
    detect_missing_evidence,
    detect_logical_gaps,
    detect_unsupported_claims,
    assess_gap_severity,
    generate_gap_recommendation,
)
from .contradiction import (
    Contradiction,
    ContradictionSeverity,
    detect_contradiction,
    detect_contradictions_batch,
    detect_polarity_opposition,
    assess_contradiction_severity,
    generate_contradiction_explanation,
    generate_reconciliation_suggestion,
)

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowStep",
    "WorkflowRegistry",
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
