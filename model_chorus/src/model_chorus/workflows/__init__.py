"""
Workflow implementations for ModelChorus.

This module contains specific workflow types such as thinkdeep, debug,
consensus, codereview, precommit, and planner.
"""

from ..core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationPhase,
    InvestigationStep,
    StudyState,
    ThinkDeepState,
)
from ..core.registry import WorkflowRegistry
from .argument import ArgumentWorkflow
from .chat import ChatWorkflow
from .consensus import (
    ConsensusResult,
    ConsensusStrategy,
    ConsensusWorkflow,
    ProviderConfig,
)
from .ideate import IdeateWorkflow
from .study import StudyWorkflow
from .thinkdeep import ThinkDeepWorkflow

# Register workflows and metadata
# Note: ArgumentWorkflow, IdeateWorkflow, and StudyWorkflow are already registered
# using the @WorkflowRegistry.register decorator in their respective files.
# We only need to register metadata for those, and both workflow + metadata for
# ChatWorkflow and ThinkDeepWorkflow.

# Register ChatWorkflow (not using decorator)
WorkflowRegistry.register_workflow("chat", ChatWorkflow)
WorkflowRegistry.register_metadata(
    "chat",
    description="Single-model peer consultation with conversation threading",
    version="1.0.0",
    author="ModelChorus Team",
    category="consultation",
    parameters=["prompt", "continuation_id", "system_prompt"],
    examples=["model-chorus chat 'What is quantum computing?'"],
)

# Register ConsensusWorkflow (not using decorator)
WorkflowRegistry.register_workflow("consensus", ConsensusWorkflow)
WorkflowRegistry.register_metadata(
    "consensus",
    description="Multi-model consultation with parallel execution and configurable synthesis strategies",
    version="1.0.0",
    author="ModelChorus Team",
    category="consultation",
    parameters=["prompt", "strategy", "num_to_consult"],
    examples=["model-chorus consensus 'Explain quantum computing' --num-to-consult 2"],
)

# ArgumentWorkflow is already registered via decorator, just add metadata
WorkflowRegistry.register_metadata(
    "argument",
    description="Structured dialectical reasoning through three-role analysis",
    version="1.0.0",
    author="ModelChorus Team",
    category="reasoning",
    parameters=["prompt", "continuation_id", "temperature"],
    examples=[
        "model-chorus argument 'Universal basic income would reduce poverty'"
    ],
)

# Register ThinkDeepWorkflow (not using decorator)
WorkflowRegistry.register_workflow("thinkdeep", ThinkDeepWorkflow)
WorkflowRegistry.register_metadata(
    "thinkdeep",
    description="Extended reasoning with systematic investigation and hypothesis tracking",
    version="2.0.0",
    author="ModelChorus Team",
    category="reasoning",
    parameters=[
        "step",
        "step_number",
        "findings",
        "hypothesis",
        "confidence",
        "thinking_mode",
    ],
    examples=[
        "model-chorus thinkdeep --step 'Investigate API latency' --step-number 1 --total-steps 3 --findings 'Examining logs' --confidence exploring"
    ],
)

# IdeateWorkflow is already registered via decorator, just add metadata
WorkflowRegistry.register_metadata(
    "ideate",
    description="Creative brainstorming and idea generation with structured output",
    version="1.0.0",
    author="ModelChorus Team",
    category="creativity",
    parameters=["prompt", "num_ideas", "temperature"],
    examples=["model-chorus ideate 'New features for a task management app' -n 10"],
)

# StudyWorkflow is already registered via decorator, just add metadata
WorkflowRegistry.register_metadata(
    "study",
    description="Persona-based collaborative research with intelligent role orchestration",
    version="1.0.0",
    author="ModelChorus Team",
    category="research",
    parameters=["prompt", "personas", "depth"],
    examples=["model-chorus study 'Research the impact of AI on education'"],
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
