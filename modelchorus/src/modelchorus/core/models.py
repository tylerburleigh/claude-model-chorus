"""
Pydantic models for ModelChorus workflow requests and responses.

This module defines the data models used for workflow communication,
providing validation and serialization for workflow inputs and outputs.

Additionally defines conversation infrastructure models for multi-turn
conversations with continuation support, adapted from Zen MCP patterns.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class ConfidenceLevel(str, Enum):
    """
    Confidence level enum for investigation workflows.

    Used in Thinkdeep workflow to track the investigator's confidence
    in their hypothesis as evidence accumulates. Levels progress from
    initial exploration through to complete certainty.

    Values:
        EXPLORING: Just starting investigation, no clear hypothesis yet
        LOW: Early investigation with initial hypothesis forming
        MEDIUM: Some supporting evidence found
        HIGH: Strong evidence supporting hypothesis
        VERY_HIGH: Very strong evidence, high confidence
        ALMOST_CERTAIN: Near complete confidence, comprehensive evidence
        CERTAIN: 100% confidence, hypothesis validated beyond reasonable doubt
    """

    EXPLORING = "exploring"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    ALMOST_CERTAIN = "almost_certain"
    CERTAIN = "certain"


class WorkflowRequest(BaseModel):
    """
    Request model for workflow execution.

    Encapsulates all the information needed to execute a workflow,
    including the prompt, model specifications, and configuration.

    Attributes:
        prompt: The main input prompt/task for the workflow
        models: List of model identifiers to use (e.g., ["gpt-4", "claude-3"])
        config: Optional workflow-specific configuration dictionary
        system_prompt: Optional system prompt for model context
        temperature: Optional temperature setting (0.0-1.0)
        max_tokens: Optional maximum tokens for generation
        images: Optional list of image paths or URLs for vision-capable models
        metadata: Additional metadata for the workflow execution
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Analyze this code for potential issues",
                "models": ["claude-3-opus", "gpt-4"],
                "config": {"thinking_mode": "max", "thoroughness": "high"},
                "temperature": 0.7,
            }
        }
    )

    prompt: str = Field(
        ...,
        description="The main input prompt or task for the workflow",
        min_length=1,
    )

    models: List[str] = Field(
        default_factory=list,
        description="List of model identifiers to use in the workflow",
    )

    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific configuration parameters",
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to provide context to models",
    )

    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temperature for model generation (0.0-1.0)",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens for model generation",
    )

    images: Optional[List[str]] = Field(
        default=None,
        description="Optional list of image paths or URLs for vision models",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for workflow execution",
    )


class WorkflowResponse(BaseModel):
    """
    Response model for workflow execution.

    Contains the results of a workflow execution along with metadata
    about the execution process.

    Attributes:
        result: The main result/output from the workflow
        success: Whether the workflow executed successfully
        workflow_name: Name of the workflow that was executed
        steps: Number of steps executed in the workflow
        models_used: List of models that were used
        error: Error message if the workflow failed
        metadata: Additional metadata about the execution
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": "Analysis complete: Found 3 potential issues...",
                "success": True,
                "workflow_name": "codereview",
                "steps": 5,
                "models_used": ["claude-3-opus", "gpt-4"],
                "metadata": {"duration_seconds": 12.5, "tokens_used": 2500},
            }
        }
    )

    result: str = Field(
        ...,
        description="The main result or output from the workflow",
    )

    success: bool = Field(
        ...,
        description="Whether the workflow executed successfully",
    )

    workflow_name: str = Field(
        ...,
        description="Name of the workflow that was executed",
    )

    steps: int = Field(
        default=0,
        ge=0,
        description="Number of steps executed in the workflow",
    )

    models_used: List[str] = Field(
        default_factory=list,
        description="List of models that were actually used",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if the workflow failed",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata (duration, tokens, etc.)",
    )


class ModelSelection(BaseModel):
    """
    Model for specifying model selection criteria.

    Used to configure which models should be used for specific
    workflow steps or roles.

    Attributes:
        model_id: The model identifier (e.g., "gpt-4", "claude-3-opus")
        role: Optional role for this model (e.g., "analyzer", "synthesizer")
        config: Optional model-specific configuration
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "claude-3-opus",
                "role": "primary_analyzer",
                "config": {"temperature": 0.3, "thinking_mode": "high"},
            }
        }
    )

    model_id: str = Field(
        ...,
        description="The model identifier",
        min_length=1,
    )

    role: Optional[str] = Field(
        default=None,
        description="Optional role for this model in the workflow",
    )

    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific configuration parameters",
    )


class WorkflowStep(BaseModel):
    """
    Model for a single workflow execution step.

    Represents one step in a multi-step workflow, capturing what
    was done, which model was used, and the results.

    Attributes:
        step_number: Sequential step number (1-indexed)
        description: Human-readable description of this step
        model: Model that executed this step
        prompt: The prompt used for this step
        response: The response from this step
        metadata: Additional step metadata
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step_number": 1,
                "description": "Initial code analysis",
                "model": "claude-3-opus",
                "prompt": "Analyze this code...",
                "response": "Found 3 issues...",
                "metadata": {"duration_ms": 1200, "tokens": 450},
            }
        }
    )

    step_number: int = Field(
        ...,
        ge=1,
        description="Sequential step number",
    )

    description: str = Field(
        ...,
        description="Human-readable description of this step",
        min_length=1,
    )

    model: str = Field(
        ...,
        description="Model identifier that executed this step",
    )

    prompt: str = Field(
        ...,
        description="The prompt used for this step",
    )

    response: str = Field(
        ...,
        description="The response from this step",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this step",
    )


class ModelResponse(BaseModel):
    """
    Model for a response from a single model.

    Represents the output from querying a specific model, used in
    multi-model workflows to track individual model contributions.

    Attributes:
        model: Identifier of the model that generated this response
        content: The response content/text
        role: Optional role this model played (e.g., "for", "against", "neutral")
        metadata: Additional response metadata (tokens, latency, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "claude-3-opus",
                "content": "I recommend approach A because...",
                "role": "advocate",
                "metadata": {"tokens": 350, "latency_ms": 1200},
            }
        }
    )

    model: str = Field(
        ...,
        description="Model identifier that generated this response",
        min_length=1,
    )

    content: str = Field(
        ...,
        description="The response content from the model",
        min_length=1,
    )

    role: Optional[str] = Field(
        default=None,
        description="Optional role this model played in the workflow",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens, latency, cost, etc.)",
    )


class ConsensusConfig(BaseModel):
    """
    Configuration for consensus-building workflows.

    Specifies how multiple models should be consulted and how their
    responses should be synthesized into a consensus.

    Attributes:
        mode: Consensus mode ("debate", "vote", "synthesis")
        stances: Optional list of stances to assign to models (e.g., ["for", "against", "neutral"])
        temperature: Temperature for model responses
        min_agreement: Minimum agreement threshold (0.0-1.0) for consensus
        synthesis_model: Optional model to use for final synthesis
        max_rounds: Maximum number of consensus rounds
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "debate",
                "stances": ["for", "against", "neutral"],
                "temperature": 0.7,
                "min_agreement": 0.6,
                "synthesis_model": "claude-3-opus",
                "max_rounds": 3,
            }
        }
    )

    mode: str = Field(
        ...,
        description="Consensus mode: 'debate', 'vote', or 'synthesis'",
        pattern="^(debate|vote|synthesis)$",
    )

    stances: Optional[List[str]] = Field(
        default=None,
        description="Optional stances to assign to models",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for model responses",
    )

    min_agreement: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum agreement threshold for consensus (0.0-1.0)",
    )

    synthesis_model: Optional[str] = Field(
        default=None,
        description="Optional model to use for final synthesis",
    )

    max_rounds: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of consensus rounds",
    )


# ============================================================================
# Conversation Infrastructure Models
# ============================================================================
# Based on Zen MCP patterns, adapted for CLI-based orchestration


class ConversationMessage(BaseModel):
    """
    Single message in a conversation thread.

    Based on Zen MCP's ConversationTurn but adapted for CLI-based architecture.
    Tracks who said what, when, and with what context (files, models, workflow).

    Attributes:
        role: Message role - 'user' or 'assistant'
        content: The actual message text/content
        timestamp: ISO format timestamp of when message was created
        files: Optional list of file paths referenced in this message
        workflow_name: Optional workflow that generated this message (for assistant messages)
        model_provider: Optional provider type used (cli, api, mcp)
        model_name: Optional specific model identifier (e.g., claude-3-opus, gpt-5)
        metadata: Additional message metadata (tokens, latency, cost, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "Analyze this code for potential security issues",
                "timestamp": "2025-11-05T12:00:00Z",
                "files": ["src/auth.py", "src/crypto.py"],
                "metadata": {"user_id": "dev-123", "session_id": "abc-456"},
            }
        }
    )

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
    )

    content: str = Field(
        ...,
        description="The message content/text",
        min_length=1,
    )

    timestamp: str = Field(
        ...,
        description="ISO format timestamp of message creation",
    )

    files: Optional[List[str]] = Field(
        default=None,
        description="Files referenced in this message",
    )

    workflow_name: Optional[str] = Field(
        default=None,
        description="Workflow that generated this message (if assistant)",
    )

    model_provider: Optional[str] = Field(
        default=None,
        description="Provider type: cli, api, or mcp",
    )

    model_name: Optional[str] = Field(
        default=None,
        description="Specific model used (e.g., claude-3-opus, gpt-5, gemini-2.5-pro)",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata (tokens, latency, cost, etc.)",
    )


class ConversationThread(BaseModel):
    """
    Complete conversation context for a thread.

    Adapted from Zen MCP's ThreadContext with enhancements for CLI orchestration:
    - Provider-agnostic design supporting multiple CLI providers
    - Explicit lifecycle management (status field)
    - Support for conversation branching (future enhancement)
    - Workflow-specific state persistence

    Attributes:
        thread_id: UUID identifying this conversation thread
        parent_thread_id: Optional parent thread ID for conversation chains
        created_at: ISO timestamp of thread creation
        last_updated_at: ISO timestamp of last update
        workflow_name: Workflow that created this thread
        messages: All messages in chronological order
        state: Workflow-specific state data (persisted across turns)
        initial_context: Original request parameters
        status: Thread lifecycle status (active, completed, archived)
        branch_point: Optional message ID where branch occurred
        sibling_threads: Other thread IDs branched from same point
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": "550e8400-e29b-41d4-a716-446655440000",
                "parent_thread_id": None,
                "created_at": "2025-11-05T12:00:00Z",
                "last_updated_at": "2025-11-05T12:15:30Z",
                "workflow_name": "consensus",
                "messages": [
                    {
                        "role": "user",
                        "content": "Analyze this code",
                        "timestamp": "2025-11-05T12:00:00Z",
                    },
                    {
                        "role": "assistant",
                        "content": "Found 3 issues...",
                        "timestamp": "2025-11-05T12:01:00Z",
                        "model_provider": "cli",
                        "model_name": "claude-3-opus",
                    },
                ],
                "state": {"models_consulted": ["claude", "gpt-5", "gemini"]},
                "initial_context": {"prompt": "Analyze this code", "models": ["claude", "gpt-5"]},
                "status": "active",
            }
        }
    )

    thread_id: str = Field(
        ...,
        description="UUID identifying this conversation thread",
    )

    parent_thread_id: Optional[str] = Field(
        default=None,
        description="Parent thread ID for conversation chains",
    )

    created_at: str = Field(
        ...,
        description="ISO timestamp of thread creation",
    )

    last_updated_at: str = Field(
        ...,
        description="ISO timestamp of last update",
    )

    workflow_name: str = Field(
        ...,
        description="Workflow that created this thread",
        min_length=1,
    )

    messages: List[ConversationMessage] = Field(
        default_factory=list,
        description="All messages in chronological order",
    )

    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific state data (persisted across turns)",
    )

    initial_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original request parameters",
    )

    status: Literal["active", "completed", "archived"] = Field(
        default="active",
        description="Thread lifecycle status",
    )

    branch_point: Optional[str] = Field(
        default=None,
        description="Message ID where branch occurred (if branched)",
    )

    sibling_threads: List[str] = Field(
        default_factory=list,
        description="Other thread IDs branched from same point",
    )


class Hypothesis(BaseModel):
    """
    Model for tracking hypotheses in investigation workflows.

    Used in Thinkdeep workflow to track hypothesis evolution during
    systematic investigation, including the hypothesis text, supporting
    evidence, and current validation status.

    Attributes:
        hypothesis: The hypothesis text/statement being investigated
        evidence: List of evidence items supporting or refuting this hypothesis
        status: Current validation status (active, disproven, validated)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hypothesis": "API uses async/await pattern instead of callbacks",
                "evidence": [
                    "Found async def in auth.py line 45",
                    "Tests use asyncio.run() in test_auth.py",
                    "No callback patterns found in service layer"
                ],
                "status": "validated"
            }
        }
    )

    hypothesis: str = Field(
        ...,
        description="The hypothesis text/statement being investigated",
        min_length=1,
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence items supporting or refuting this hypothesis",
    )

    status: Literal["active", "disproven", "validated"] = Field(
        default="active",
        description="Current validation status of the hypothesis",
    )


class InvestigationStep(BaseModel):
    """
    Model for a single investigation step in Thinkdeep workflow.

    Captures the details of one step in a systematic investigation,
    including what was found, which files were examined, and the
    current confidence level in the hypothesis.

    Attributes:
        step_number: Sequential step number (1-indexed)
        findings: Key findings and insights discovered in this step
        files_checked: List of files examined during this step
        confidence: Current confidence level after this step
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step_number": 1,
                "findings": "Found async/await pattern in auth service. No callbacks detected in user-facing API.",
                "files_checked": [
                    "src/services/auth.py",
                    "src/api/users.py",
                    "tests/test_auth.py"
                ],
                "confidence": "high"
            }
        }
    )

    step_number: int = Field(
        ...,
        ge=1,
        description="Sequential step number in the investigation",
    )

    findings: str = Field(
        ...,
        description="Key findings and insights discovered in this step",
        min_length=1,
    )

    files_checked: List[str] = Field(
        default_factory=list,
        description="List of files examined during this step",
    )

    confidence: str = Field(
        ...,
        description="Current confidence level after this step (ConfidenceLevel value)",
        min_length=1,
    )


class ThinkDeepState(BaseModel):
    """
    State model for Thinkdeep workflow multi-turn conversations.

    Maintains the complete investigation state across conversation turns,
    tracking hypothesis evolution, investigation steps, confidence progression,
    and files examined.

    Attributes:
        hypotheses: List of all hypotheses tracked during investigation
        steps: List of all investigation steps completed
        current_confidence: Current overall confidence level
        relevant_files: All files identified as relevant to the investigation
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hypotheses": [
                    {
                        "hypothesis": "API uses async/await pattern",
                        "evidence": ["Found async def in auth.py", "Tests use asyncio"],
                        "status": "validated"
                    },
                    {
                        "hypothesis": "Error handling uses custom exceptions",
                        "evidence": ["Found CustomError class", "Used in auth service"],
                        "status": "active"
                    }
                ],
                "steps": [
                    {
                        "step_number": 1,
                        "findings": "Examined auth service implementation",
                        "files_checked": ["src/auth.py"],
                        "confidence": "medium"
                    },
                    {
                        "step_number": 2,
                        "findings": "Validated hypothesis with tests",
                        "files_checked": ["tests/test_auth.py"],
                        "confidence": "high"
                    }
                ],
                "current_confidence": "high",
                "relevant_files": [
                    "src/services/auth.py",
                    "src/api/users.py",
                    "tests/test_auth.py"
                ]
            }
        }
    )

    hypotheses: List[Hypothesis] = Field(
        default_factory=list,
        description="List of all hypotheses tracked during investigation",
    )

    steps: List[InvestigationStep] = Field(
        default_factory=list,
        description="List of all investigation steps completed",
    )

    current_confidence: str = Field(
        default="exploring",
        description="Current overall confidence level (ConfidenceLevel value)",
    )

    relevant_files: List[str] = Field(
        default_factory=list,
        description="All files identified as relevant to the investigation",
    )


class ConversationState(BaseModel):
    """
    Generic state container for workflow-specific conversation data.

    Provides type-safe structure for storing arbitrary workflow state
    while maintaining serializability for file-based persistence.
    Includes versioning to support schema evolution.

    Attributes:
        workflow_name: Workflow this state belongs to
        data: Arbitrary workflow-specific state data
        schema_version: State schema version for compatibility
        created_at: ISO timestamp of state creation
        updated_at: ISO timestamp of last state update
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_name": "consensus",
                "data": {
                    "current_model_index": 2,
                    "models_consulted": ["gpt-5", "claude", "gemini"],
                    "consensus_reached": False,
                    "confidence_level": "medium",
                },
                "schema_version": "1.0",
                "created_at": "2025-11-05T12:00:00Z",
                "updated_at": "2025-11-05T12:15:00Z",
            }
        }
    )

    workflow_name: str = Field(
        ...,
        description="Workflow this state belongs to",
        min_length=1,
    )

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary workflow-specific state data",
    )

    schema_version: str = Field(
        default="1.0",
        description="State schema version for compatibility",
    )

    created_at: str = Field(
        ...,
        description="ISO timestamp of state creation",
    )

    updated_at: str = Field(
        ...,
        description="ISO timestamp of last state update",
    )


class Citation(BaseModel):
    """
    Citation model for tracking sources in ARGUMENT workflow.

    Tracks the source of information, its location, and confidence level
    for evidence-based argumentation and research workflows.

    Attributes:
        source: The source identifier (URL, file path, document ID, etc.)
        location: Specific location within source (page, line, section, timestamp)
        confidence: Confidence level in the citation accuracy (0.0-1.0)
        snippet: Optional text snippet from the source
        metadata: Additional citation metadata (author, date, context, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "https://arxiv.org/abs/2401.12345",
                "location": "Section 3.2, Figure 4",
                "confidence": 0.95,
                "snippet": "Our experiments show a 23% improvement in accuracy...",
                "metadata": {
                    "author": "Smith et al.",
                    "publication_date": "2024-01-15",
                    "citation_type": "academic_paper",
                },
            }
        }
    )

    source: str = Field(
        ...,
        description="Source identifier (URL, file path, document ID, etc.)",
        min_length=1,
    )

    location: Optional[str] = Field(
        default=None,
        description="Specific location within source (page, line, section, timestamp)",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in citation accuracy (0.0-1.0)",
    )

    snippet: Optional[str] = Field(
        default=None,
        description="Optional text snippet from the source",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional citation metadata (author, date, context, etc.)",
    )
