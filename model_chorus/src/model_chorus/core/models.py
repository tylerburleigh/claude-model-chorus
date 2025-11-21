"""
Pydantic models for ModelChorus workflow requests and responses.

This module defines the data models used for workflow communication,
providing validation and serialization for workflow inputs and outputs.

Additionally defines conversation infrastructure models for multi-turn
conversations with continuation support, adapted from Zen MCP patterns.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class InvestigationPhase(str, Enum):
    """
    Investigation phase enum for persona-based research workflows.

    Used in Study workflow to track the current phase of collaborative
    investigation. Phases progress from initial discovery through to
    completion with systematic exploration and validation.

    Values:
        DISCOVERY: Initial exploration phase where personas gather information
        VALIDATION: Critical examination phase where findings are validated
        PLANNING: Synthesis phase where insights are organized into actionable plans
        COMPLETE: Investigation concluded with comprehensive findings
    """

    DISCOVERY = "discovery"
    VALIDATION = "validation"
    PLANNING = "planning"
    COMPLETE = "complete"


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

    models: list[str] = Field(
        default_factory=list,
        description="List of model identifiers to use in the workflow",
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific configuration parameters",
    )

    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt to provide context to models",
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temperature for model generation (0.0-1.0)",
    )

    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens for model generation",
    )

    images: list[str] | None = Field(
        default=None,
        description="Optional list of image paths or URLs for vision models",
    )

    metadata: dict[str, Any] = Field(
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

    models_used: list[str] = Field(
        default_factory=list,
        description="List of models that were actually used",
    )

    error: str | None = Field(
        default=None,
        description="Error message if the workflow failed",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata (duration, tokens, etc.)",
    )


class WorkflowMetadata(BaseModel):
    """
    Metadata about a registered workflow for discovery and documentation.

    Provides descriptive information about workflows registered in the
    WorkflowRegistry, enabling workflow listing, documentation generation,
    and user discovery of available workflows.

    Attributes:
        name: Unique workflow identifier (e.g., "consensus", "thinkdeep")
        description: Human-readable description of what the workflow does
        version: Semantic version string (e.g., "1.0.0")
        author: Workflow author or maintainer
        category: Optional workflow category for organization
        parameters: Optional list of workflow-specific parameter names
        examples: Optional list of usage examples
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "consensus",
                "description": "Multi-model consultation with configurable synthesis",
                "version": "2.0.0",
                "author": "ModelChorus Team",
                "category": "consultation",
                "parameters": ["strategy", "providers", "temperature"],
                "examples": [
                    "model-chorus consensus 'Analyze this code' --strategy synthesize"
                ],
            }
        }
    )

    name: str = Field(
        ...,
        description="Unique workflow identifier",
        min_length=1,
    )

    description: str = Field(
        ...,
        description="Human-readable description of workflow functionality",
        min_length=1,
    )

    version: str = Field(
        default="1.0.0",
        description="Semantic version string",
        pattern=r"^\d+\.\d+\.\d+$",
    )

    author: str = Field(
        default="Unknown",
        description="Workflow author or maintainer",
    )

    category: str | None = Field(
        default=None,
        description="Optional category for workflow organization",
    )

    parameters: list[str] = Field(
        default_factory=list,
        description="List of workflow-specific parameter names",
    )

    examples: list[str] = Field(
        default_factory=list,
        description="Usage examples for documentation",
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

    role: str | None = Field(
        default=None,
        description="Optional role for this model in the workflow",
    )

    config: dict[str, Any] = Field(
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

    metadata: dict[str, Any] = Field(
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

    role: str | None = Field(
        default=None,
        description="Optional role this model played in the workflow",
    )

    metadata: dict[str, Any] = Field(
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

    stances: list[str] | None = Field(
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

    synthesis_model: str | None = Field(
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

    files: list[str] | None = Field(
        default=None,
        description="Files referenced in this message",
    )

    workflow_name: str | None = Field(
        default=None,
        description="Workflow that generated this message (if assistant)",
    )

    model_provider: str | None = Field(
        default=None,
        description="Provider type: cli, api, or mcp",
    )

    model_name: str | None = Field(
        default=None,
        description="Specific model used (e.g., claude-3-opus, gpt-5, gemini-2.5-pro)",
    )

    metadata: dict[str, Any] = Field(
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
                "initial_context": {
                    "prompt": "Analyze this code",
                    "models": ["claude", "gpt-5"],
                },
                "status": "active",
            }
        }
    )

    thread_id: str = Field(
        ...,
        description="UUID identifying this conversation thread",
    )

    parent_thread_id: str | None = Field(
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

    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="All messages in chronological order",
    )

    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific state data (persisted across turns)",
    )

    initial_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Original request parameters",
    )

    status: Literal["active", "completed", "archived"] = Field(
        default="active",
        description="Thread lifecycle status",
    )

    branch_point: str | None = Field(
        default=None,
        description="Message ID where branch occurred (if branched)",
    )

    sibling_threads: list[str] = Field(
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
                    "No callback patterns found in service layer",
                ],
                "status": "validated",
            }
        }
    )

    hypothesis: str = Field(
        ...,
        description="The hypothesis text/statement being investigated",
        min_length=1,
    )

    evidence: list[str] = Field(
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
                    "tests/test_auth.py",
                ],
                "confidence": "high",
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

    files_checked: list[str] = Field(
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
                        "status": "validated",
                    },
                    {
                        "hypothesis": "Error handling uses custom exceptions",
                        "evidence": ["Found CustomError class", "Used in auth service"],
                        "status": "active",
                    },
                ],
                "steps": [
                    {
                        "step_number": 1,
                        "findings": "Examined auth service implementation",
                        "files_checked": ["src/auth.py"],
                        "confidence": "medium",
                    },
                    {
                        "step_number": 2,
                        "findings": "Validated hypothesis with tests",
                        "files_checked": ["tests/test_auth.py"],
                        "confidence": "high",
                    },
                ],
                "current_confidence": "high",
                "relevant_files": [
                    "src/services/auth.py",
                    "src/api/users.py",
                    "tests/test_auth.py",
                ],
            }
        }
    )

    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="List of all hypotheses tracked during investigation",
    )

    steps: list[InvestigationStep] = Field(
        default_factory=list,
        description="List of all investigation steps completed",
    )

    current_confidence: str = Field(
        default="exploring",
        description="Current overall confidence level (ConfidenceLevel value)",
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="All files identified as relevant to the investigation",
    )


class StudyState(BaseModel):
    """
    State model for Study workflow multi-persona investigations.

    Maintains the complete investigation state across conversation turns,
    tracking investigation phase, persona findings, confidence levels,
    and collaborative exploration progress.

    Attributes:
        investigation_id: Unique identifier for this investigation session
        session_id: Session/thread identifier for conversation continuity
        current_phase: Current phase of investigation (InvestigationPhase)
        confidence: Overall confidence level in findings (ConfidenceLevel)
        iteration_count: Number of investigation iterations completed
        findings: List of all findings from persona investigations
        personas_active: List of personas currently participating
        relevant_files: Files examined during investigation
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "investigation_id": "inv_2025-11-08_001",
                "session_id": "thread_abc123",
                "current_phase": "discovery",
                "confidence": "medium",
                "iteration_count": 3,
                "findings": [
                    {
                        "persona": "Researcher",
                        "finding": "Authentication uses JWT tokens",
                        "evidence": ["Found JWT library in dependencies"],
                        "confidence": "high",
                    },
                    {
                        "persona": "Critic",
                        "finding": "Token expiration not consistently checked",
                        "evidence": ["Missing validation in 3 endpoints"],
                        "confidence": "medium",
                    },
                ],
                "personas_active": ["Researcher", "Critic"],
                "relevant_files": ["src/auth.py", "src/middleware/jwt.py"],
            }
        }
    )

    investigation_id: str = Field(
        ...,
        description="Unique identifier for this investigation session",
    )

    session_id: str = Field(
        ...,
        description="Session/thread identifier for conversation continuity",
    )

    current_phase: str = Field(
        default="discovery",
        description="Current investigation phase (InvestigationPhase value)",
    )

    confidence: str = Field(
        default="exploring",
        description="Overall confidence level in findings (ConfidenceLevel value)",
    )

    iteration_count: int = Field(
        default=0,
        description="Number of investigation iterations completed",
    )

    findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of all findings from persona investigations",
    )

    personas_active: list[str] = Field(
        default_factory=list,
        description="List of personas currently participating in investigation",
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="Files examined during investigation",
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

    data: dict[str, Any] = Field(
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

    location: str | None = Field(
        default=None,
        description="Specific location within source (page, line, section, timestamp)",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in citation accuracy (0.0-1.0)",
    )

    snippet: str | None = Field(
        default=None,
        description="Optional text snippet from the source",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional citation metadata (author, date, context, etc.)",
    )


class CitationMap(BaseModel):
    """
    Maps claims to their supporting citations for evidence tracking.

    Used in ARGUMENT workflow to maintain bidirectional mapping between
    claims/arguments and their source citations, enabling verification
    and citation analysis.

    Attributes:
        claim_id: Unique identifier for the claim being supported
        claim_text: The actual claim or argument text
        citations: List of Citation objects supporting this claim
        strength: Overall strength of citation support (0.0-1.0)
        metadata: Additional mapping metadata (argument_type, verification_status, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "claim_id": "claim-001",
                "claim_text": "Machine learning models improve accuracy by 23%",
                "citations": [
                    {
                        "source": "https://arxiv.org/abs/2401.12345",
                        "location": "Section 3.2",
                        "confidence": 0.95,
                        "snippet": "Our experiments show a 23% improvement...",
                    },
                    {
                        "source": "paper2.pdf",
                        "location": "Figure 4",
                        "confidence": 0.85,
                        "snippet": "Results demonstrate significant gains...",
                    },
                ],
                "strength": 0.9,
                "metadata": {
                    "argument_type": "empirical",
                    "verification_status": "verified",
                    "citation_count": 2,
                },
            }
        }
    )

    claim_id: str = Field(
        ...,
        description="Unique identifier for the claim being supported",
        min_length=1,
    )

    claim_text: str = Field(
        ...,
        description="The actual claim or argument text",
        min_length=1,
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description="List of Citation objects supporting this claim",
    )

    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall strength of citation support (0.0-1.0)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional mapping metadata (argument_type, verification_status, etc.)",
    )


class Claim(BaseModel):
    """
    Represents a factual or arguable statement extracted from model output.

    Claims are fundamental units in argument analysis and research workflows.
    Each claim captures a specific assertion along with metadata about its source,
    location, and confidence level.

    Used in workflows like:
    - ARGUMENT: Extracting claims from debate responses for contradiction detection
    - RESEARCH: Identifying factual claims across multiple sources
    - IDEATE: Capturing key assertions from diverse perspectives

    Attributes:
        content: The actual claim text (the statement being made)
        source_id: Identifier for the source (role name, document ID, model ID, etc.)
        location: Optional location within source (line number, section, paragraph, etc.)
        confidence: Confidence score for this claim (0.0 = low, 1.0 = high)

    Example:
        >>> claim = Claim(
        ...     content="TypeScript reduces runtime errors by 15%",
        ...     source_id="proponent",
        ...     location="paragraph 2",
        ...     confidence=0.8
        ... )
        >>> print(claim)
        [proponent@paragraph 2] (0.80): TypeScript reduces runtime errors by 15%
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "TypeScript reduces runtime errors by 15%",
                "source_id": "proponent",
                "location": "paragraph 2",
                "confidence": 0.8,
            }
        }
    )

    content: str = Field(
        ...,
        description="The actual claim text (the statement being made)",
        min_length=1,
    )

    source_id: str = Field(
        ...,
        description="Identifier for the source (role name, document ID, model ID, etc.)",
        min_length=1,
    )

    location: str | None = Field(
        default=None,
        description="Optional location within source (line number, section, paragraph, etc.)",
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence score for this claim (0.0 = low confidence, 1.0 = high confidence)",
        ge=0.0,
        le=1.0,
    )

    def __str__(self) -> str:
        """
        Return human-readable string representation of the claim.

        Format: [source_id@location] (confidence): content
        If location is not specified, format is: [source_id] (confidence): content

        Returns:
            Human-readable claim string

        Example:
            >>> claim = Claim(content="Test claim", source_id="analyst", confidence=0.9)
            >>> str(claim)
            '[analyst] (0.90): Test claim'
        """
        location_str = f"@{self.location}" if self.location else ""
        return (
            f"[{self.source_id}{location_str}] ({self.confidence:.2f}): {self.content}"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert claim to dictionary for serialization.

        Returns:
            Dictionary representation of the claim

        Example:
            >>> claim = Claim(content="Test", source_id="test", confidence=0.5)
            >>> claim.to_dict()
            {'content': 'Test', 'source_id': 'test', 'location': None, 'confidence': 0.5}
        """
        return {
            "content": self.content,
            "source_id": self.source_id,
            "location": self.location,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Claim":
        """
        Create a Claim from a dictionary.

        Args:
            data: Dictionary containing claim fields

        Returns:
            New Claim instance

        Example:
            >>> data = {'content': 'Test', 'source_id': 'test', 'confidence': 0.5}
            >>> claim = Claim.from_dict(data)
            >>> claim.content
            'Test'
        """
        return cls(**data)


class Evidence(BaseModel):
    """
    Represents supporting or refuting evidence for claims and hypotheses.

    Evidence is a fundamental building block in investigation, research, and
    argument workflows. Each piece of evidence captures specific information
    from a source along with its type, strength, and relevance.

    Used in workflows like:
    - THINKDEEP: Tracking evidence that supports or refutes hypotheses
    - ARGUMENT: Recording factual evidence for debate claims
    - RESEARCH: Documenting findings from multiple sources

    Attributes:
        content: The actual evidence text (the observation or fact)
        source_id: Identifier for the source (file path, model ID, document ID, etc.)
        location: Optional location within source (line number, section, page, etc.)
        evidence_type: Type of evidence (supporting, refuting, neutral, contextual)
        strength: Strength/weight of this evidence (0.0 = weak, 1.0 = strong)
        timestamp: Optional ISO timestamp when evidence was collected
        metadata: Additional evidence metadata (tags, categories, analysis, etc.)

    Example:
        >>> evidence = Evidence(
        ...     content="Found async def pattern in auth.py line 45",
        ...     source_id="src/services/auth.py",
        ...     location="line 45",
        ...     evidence_type="supporting",
        ...     strength=0.9
        ... )
        >>> print(evidence)
        [src/services/auth.py@line 45] (supporting, 0.90): Found async def pattern in auth.py line 45
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Found async def pattern in auth.py line 45",
                "source_id": "src/services/auth.py",
                "location": "line 45",
                "evidence_type": "supporting",
                "strength": 0.9,
                "timestamp": "2025-11-06T12:00:00Z",
                "metadata": {
                    "tags": ["async", "authentication"],
                    "category": "code_analysis",
                    "verified": True,
                },
            }
        }
    )

    content: str = Field(
        ...,
        description="The actual evidence text (the observation or fact)",
        min_length=1,
    )

    source_id: str = Field(
        ...,
        description="Identifier for the source (file path, model ID, document ID, etc.)",
        min_length=1,
    )

    location: str | None = Field(
        default=None,
        description="Optional location within source (line number, section, page, etc.)",
    )

    evidence_type: Literal["supporting", "refuting", "neutral", "contextual"] = Field(
        default="supporting",
        description="Type of evidence: supporting, refuting, neutral, or contextual",
    )

    strength: float = Field(
        default=1.0,
        description="Strength/weight of this evidence (0.0 = weak, 1.0 = strong)",
        ge=0.0,
        le=1.0,
    )

    timestamp: str | None = Field(
        default=None,
        description="Optional ISO timestamp when evidence was collected",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evidence metadata (tags, categories, analysis, etc.)",
    )

    def __str__(self) -> str:
        """
        Return human-readable string representation of the evidence.

        Format: [source_id@location] (evidence_type, strength): content
        If location is not specified, format is: [source_id] (evidence_type, strength): content

        Returns:
            Human-readable evidence string

        Example:
            >>> evidence = Evidence(
            ...     content="Test finding",
            ...     source_id="test.py",
            ...     evidence_type="supporting",
            ...     strength=0.8
            ... )
            >>> str(evidence)
            '[test.py] (supporting, 0.80): Test finding'
        """
        location_str = f"@{self.location}" if self.location else ""
        return f"[{self.source_id}{location_str}] ({self.evidence_type}, {self.strength:.2f}): {self.content}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert evidence to dictionary for serialization.

        Returns:
            Dictionary representation of the evidence

        Example:
            >>> evidence = Evidence(
            ...     content="Test",
            ...     source_id="test",
            ...     evidence_type="supporting",
            ...     strength=0.5
            ... )
            >>> evidence.to_dict()
            {
                'content': 'Test',
                'source_id': 'test',
                'location': None,
                'evidence_type': 'supporting',
                'strength': 0.5,
                'timestamp': None,
                'metadata': {}
            }
        """
        return {
            "content": self.content,
            "source_id": self.source_id,
            "location": self.location,
            "evidence_type": self.evidence_type,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """
        Create an Evidence from a dictionary.

        Args:
            data: Dictionary containing evidence fields

        Returns:
            New Evidence instance

        Example:
            >>> data = {
            ...     'content': 'Test',
            ...     'source_id': 'test',
            ...     'evidence_type': 'supporting',
            ...     'strength': 0.5
            ... }
            >>> evidence = Evidence.from_dict(data)
            >>> evidence.content
            'Test'
        """
        return cls(**data)


class ArgumentPerspective(BaseModel):
    """
    Represents a single perspective in an argument analysis.

    Each perspective captures one role's analysis (Creator, Skeptic, or Moderator)
    with their stance, reasoning, and key points.

    Attributes:
        role: The role name (creator, skeptic, moderator)
        stance: The perspective's stance (for, against, neutral)
        content: Full response content from this perspective
        key_points: List of key points or arguments
        model: Model used for this perspective
        metadata: Additional perspective metadata

    Example:
        >>> perspective = ArgumentPerspective(
        ...     role="creator",
        ...     stance="for",
        ...     content="Universal basic income would reduce poverty...",
        ...     key_points=["Ensures basic needs", "Reduces wealth gap"],
        ...     model="claude-sonnet-4"
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "creator",
                "stance": "for",
                "content": "Universal basic income would reduce poverty...",
                "key_points": ["Ensures basic needs", "Reduces wealth gap"],
                "model": "claude-sonnet-4",
                "metadata": {},
            }
        }
    )

    role: Literal["creator", "skeptic", "moderator"] = Field(
        ..., description="The role name (creator, skeptic, moderator)"
    )

    stance: Literal["for", "against", "neutral"] = Field(
        ..., description="The perspective's stance (for, against, neutral)"
    )

    content: str = Field(
        ..., description="Full response content from this perspective", min_length=1
    )

    key_points: list[str] = Field(
        default_factory=list, description="List of key points or arguments"
    )

    model: str = Field(..., description="Model used for this perspective", min_length=1)

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional perspective metadata"
    )


class ArgumentMap(BaseModel):
    """
    Structured output from ARGUMENT workflow containing all perspectives.

    ArgumentMap provides a comprehensive view of the dialectical analysis,
    including the thesis (Creator), rebuttal (Skeptic), and synthesis (Moderator).
    This structured format enables programmatic access to different perspectives
    and supports downstream analysis, visualization, and decision-making.

    Attributes:
        topic: The argument topic or claim being analyzed
        perspectives: List of perspectives (Creator, Skeptic, Moderator)
        synthesis: Final balanced synthesis from Moderator
        metadata: Additional workflow metadata (thread_id, model, timestamps, etc.)

    Example:
        >>> arg_map = ArgumentMap(
        ...     topic="Universal basic income would reduce poverty",
        ...     perspectives=[creator_perspective, skeptic_perspective, moderator_perspective],
        ...     synthesis="After examining both perspectives...",
        ...     metadata={"thread_id": "abc123", "model": "claude-sonnet-4"}
        ... )
        >>> print(arg_map.perspectives[0].role)  # 'creator'
        >>> print(arg_map.perspectives[1].role)  # 'skeptic'
        >>> print(arg_map.perspectives[2].role)  # 'moderator'
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "Universal basic income would reduce poverty",
                "perspectives": [
                    {
                        "role": "creator",
                        "stance": "for",
                        "content": "Universal basic income would reduce poverty...",
                        "key_points": ["Ensures basic needs", "Reduces wealth gap"],
                        "model": "claude-sonnet-4",
                        "metadata": {},
                    }
                ],
                "synthesis": "After examining both perspectives...",
                "metadata": {"thread_id": "abc123"},
            }
        }
    )

    topic: str = Field(
        ..., description="The argument topic or claim being analyzed", min_length=1
    )

    perspectives: list[ArgumentPerspective] = Field(
        ...,
        description="List of perspectives (Creator, Skeptic, Moderator)",
        min_length=1,
    )

    synthesis: str = Field(
        ..., description="Final balanced synthesis from Moderator", min_length=1
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional workflow metadata (thread_id, model, timestamps, etc.)",
    )

    def get_perspective(self, role: str) -> ArgumentPerspective | None:
        """
        Get a specific perspective by role name.

        Args:
            role: Role name (creator, skeptic, or moderator)

        Returns:
            ArgumentPerspective if found, None otherwise

        Example:
            >>> creator = arg_map.get_perspective("creator")
            >>> print(creator.stance)  # 'for'
        """
        for perspective in self.perspectives:
            if perspective.role == role:
                return perspective
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ArgumentMap to dictionary for serialization.

        Returns:
            Dictionary representation of the argument map

        Example:
            >>> arg_map.to_dict()
            {
                'topic': 'Universal basic income...',
                'perspectives': [...],
                'synthesis': '...',
                'metadata': {...}
            }
        """
        return {
            "topic": self.topic,
            "perspectives": [p.model_dump() for p in self.perspectives],
            "synthesis": self.synthesis,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArgumentMap":
        """
        Create an ArgumentMap from a dictionary.

        Args:
            data: Dictionary containing argument map fields

        Returns:
            New ArgumentMap instance

        Example:
            >>> data = {
            ...     'topic': 'Test topic',
            ...     'perspectives': [{'role': 'creator', ...}],
            ...     'synthesis': 'Test synthesis',
            ...     'metadata': {}
            ... }
            >>> arg_map = ArgumentMap.from_dict(data)
        """
        # Convert perspective dicts to ArgumentPerspective objects
        perspectives = [
            ArgumentPerspective(**p) if isinstance(p, dict) else p
            for p in data["perspectives"]
        ]
        return cls(
            topic=data["topic"],
            perspectives=perspectives,
            synthesis=data["synthesis"],
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# IDEATE Workflow Models
# ============================================================================


class Idea(BaseModel):
    """
    Represents a single idea extracted from brainstorming.

    Used in the IDEATE workflow to track individual creative ideas
    that are extracted from multiple perspective-based brainstorming sessions.

    Attributes:
        id: Unique identifier for the idea (e.g., "idea-1")
        label: Brief descriptive label for the idea (1-2 words)
        description: Full description of the idea (1-2 sentences)
        perspective: The perspective this idea originated from (practical, innovative, etc.)
        source_model: Model that generated this idea
        metadata: Additional metadata about the idea

    Example:
        >>> idea = Idea(
        ...     id="idea-1",
        ...     label="Gamification System",
        ...     description="Add game mechanics like points and badges to improve engagement",
        ...     perspective="innovative",
        ...     source_model="claude"
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "idea-1",
                "label": "AI-Powered Search",
                "description": "Implement semantic search using AI embeddings for better discovery",
                "perspective": "technical",
                "source_model": "gpt-4",
                "metadata": {"tokens": 150, "temperature": 0.9},
            }
        }
    )

    id: str = Field(
        ...,
        description="Unique identifier for the idea (e.g., 'idea-1')",
        pattern=r"^idea-\d+$",
    )

    label: str = Field(
        ...,
        description="Brief descriptive label (1-5 words)",
        min_length=1,
        max_length=100,
    )

    description: str = Field(
        ..., description="Full description of the idea (1-3 sentences)", min_length=1
    )

    perspective: str = Field(
        ...,
        description="Perspective this idea originated from (practical, innovative, user-focused, etc.)",
    )

    source_model: str | None = Field(
        default=None, description="Model that generated this idea"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the idea"
    )


class IdeaCluster(BaseModel):
    """
    Represents a themed cluster of related ideas.

    Used in the IDEATE workflow to group similar ideas into coherent
    themes after convergent analysis. Each cluster represents a distinct
    approach or solution category.

    Attributes:
        id: Unique identifier for the cluster (e.g., "cluster-1")
        theme: Theme name that describes this cluster
        description: Detailed description of the cluster theme
        idea_ids: List of idea IDs belonging to this cluster
        ideas: Optional list of full Idea objects in this cluster
        scores: Evaluation scores for this cluster (feasibility, impact, etc.)
        overall_score: Average score across all criteria (0.0-5.0)
        recommendation: Priority recommendation (High/Medium/Low Priority)
        metadata: Additional metadata about the cluster

    Example:
        >>> cluster = IdeaCluster(
        ...     id="cluster-1",
        ...     theme="User Experience Improvements",
        ...     description="Ideas focused on enhancing user interface and usability",
        ...     idea_ids=["idea-1", "idea-3", "idea-5"],
        ...     scores={"feasibility": 4.5, "impact": 4.0, "novelty": 3.5},
        ...     overall_score=4.0,
        ...     recommendation="High Priority"
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "cluster-1",
                "theme": "AI-Enhanced Features",
                "description": "Ideas leveraging AI to improve core functionality",
                "idea_ids": ["idea-1", "idea-2", "idea-4"],
                "scores": {"feasibility": 4.0, "impact": 4.5, "novelty": 4.0},
                "overall_score": 4.17,
                "recommendation": "High Priority",
                "metadata": {"num_ideas": 3},
            }
        }
    )

    id: str = Field(
        ...,
        description="Unique identifier for the cluster (e.g., 'cluster-1')",
        pattern=r"^cluster-\d+$",
    )

    theme: str = Field(
        ...,
        description="Theme name describing this cluster",
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        default="", description="Detailed description of the cluster theme"
    )

    idea_ids: list[str] = Field(
        default_factory=list, description="List of idea IDs in this cluster"
    )

    ideas: list[Idea] | None = Field(
        default=None, description="Optional full Idea objects in this cluster"
    )

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation scores (e.g., feasibility: 4.5, impact: 4.0)",
    )

    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Average score across all criteria (0.0-5.0)",
    )

    recommendation: str = Field(
        default="Medium Priority",
        description="Priority recommendation (High/Medium/Low Priority)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the cluster"
    )

    def add_idea(self, idea: Idea) -> None:
        """
        Add an idea to this cluster.

        Args:
            idea: Idea object to add

        Example:
            >>> cluster.add_idea(idea_obj)
        """
        if idea.id not in self.idea_ids:
            self.idea_ids.append(idea.id)

        if self.ideas is None:
            self.ideas = []

        if idea not in self.ideas:
            self.ideas.append(idea)

    def get_idea_count(self) -> int:
        """
        Get number of ideas in this cluster.

        Returns:
            Number of ideas

        Example:
            >>> count = cluster.get_idea_count()
        """
        return len(self.idea_ids)


class IdeationState(BaseModel):
    """
    Represents the complete state of an ideation workflow session.

    Tracks the full lifecycle of an IDEATE workflow execution, from initial
    brainstorming through convergent analysis, selection, and elaboration.

    Attributes:
        session_id: Unique identifier for this ideation session
        topic: The topic or problem being ideated on
        perspectives: List of perspectives used in brainstorming
        ideas: All extracted ideas from brainstorming
        clusters: Thematic clusters of related ideas
        selected_cluster_ids: IDs of clusters selected for elaboration
        elaborations: Detailed outlines for selected clusters
        scoring_criteria: Criteria used for evaluation (feasibility, impact, etc.)
        workflow_metadata: Metadata about workflow execution
        created_at: Timestamp when ideation session started
        updated_at: Timestamp of last update

    Example:
        >>> state = IdeationState(
        ...     session_id="ideation-2024-01-15-001",
        ...     topic="How can we improve our API documentation?",
        ...     perspectives=["practical", "innovative", "user-focused"],
        ...     ideas=[idea1, idea2, idea3],
        ...     clusters=[cluster1, cluster2],
        ...     selected_cluster_ids=["cluster-1"],
        ...     scoring_criteria=["feasibility", "impact", "user_value"]
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "ideation-2024-01-15-001",
                "topic": "Improve user onboarding flow",
                "perspectives": ["practical", "innovative", "user-focused"],
                "ideas": [],
                "clusters": [],
                "selected_cluster_ids": [],
                "elaborations": {},
                "scoring_criteria": ["feasibility", "impact", "novelty"],
                "workflow_metadata": {"models_used": ["claude", "gpt-4"]},
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:45:00Z",
            }
        }
    )

    session_id: str = Field(
        ..., description="Unique identifier for this ideation session", min_length=1
    )

    topic: str = Field(
        ..., description="The topic or problem being ideated on", min_length=1
    )

    perspectives: list[str] = Field(
        default_factory=list,
        description="Perspectives used in brainstorming (practical, innovative, etc.)",
    )

    ideas: list[Idea] = Field(
        default_factory=list, description="All extracted ideas from brainstorming"
    )

    clusters: list[IdeaCluster] = Field(
        default_factory=list, description="Thematic clusters of related ideas"
    )

    selected_cluster_ids: list[str] = Field(
        default_factory=list, description="IDs of clusters selected for elaboration"
    )

    elaborations: dict[str, str] = Field(
        default_factory=dict,
        description="Detailed outlines for selected clusters (cluster_id -> outline)",
    )

    scoring_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria used for evaluation (feasibility, impact, etc.)",
    )

    workflow_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata about workflow execution"
    )

    created_at: str | None = Field(
        default=None, description="ISO timestamp when ideation session started"
    )

    updated_at: str | None = Field(
        default=None, description="ISO timestamp of last update"
    )

    def add_idea(self, idea: Idea) -> None:
        """
        Add an idea to the ideation state.

        Args:
            idea: Idea object to add

        Example:
            >>> state.add_idea(new_idea)
        """
        if idea not in self.ideas:
            self.ideas.append(idea)

    def add_cluster(self, cluster: IdeaCluster) -> None:
        """
        Add a cluster to the ideation state.

        Args:
            cluster: IdeaCluster object to add

        Example:
            >>> state.add_cluster(new_cluster)
        """
        if cluster not in self.clusters:
            self.clusters.append(cluster)

    def get_cluster_by_id(self, cluster_id: str) -> IdeaCluster | None:
        """
        Get a cluster by its ID.

        Args:
            cluster_id: Cluster ID to look up

        Returns:
            IdeaCluster if found, None otherwise

        Example:
            >>> cluster = state.get_cluster_by_id("cluster-1")
        """
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def get_selected_clusters(self) -> list[IdeaCluster]:
        """
        Get all selected clusters.

        Returns:
            List of IdeaCluster objects that were selected

        Example:
            >>> selected = state.get_selected_clusters()
            >>> print(f"Selected {len(selected)} clusters")
        """
        return [
            cluster
            for cluster in self.clusters
            if cluster.id in self.selected_cluster_ids
        ]

    def get_idea_count(self) -> int:
        """
        Get total number of ideas.

        Returns:
            Total number of ideas

        Example:
            >>> count = state.get_idea_count()
        """
        return len(self.ideas)

    def get_cluster_count(self) -> int:
        """
        Get total number of clusters.

        Returns:
            Total number of clusters

        Example:
            >>> count = state.get_cluster_count()
        """
        return len(self.clusters)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert IdeationState to dictionary for serialization.

        Returns:
            Dictionary representation of the ideation state

        Example:
            >>> state_dict = state.to_dict()
        """
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "perspectives": self.perspectives,
            "ideas": [idea.model_dump() for idea in self.ideas],
            "clusters": [cluster.model_dump() for cluster in self.clusters],
            "selected_cluster_ids": self.selected_cluster_ids,
            "elaborations": self.elaborations,
            "scoring_criteria": self.scoring_criteria,
            "workflow_metadata": self.workflow_metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdeationState":
        """
        Create an IdeationState from a dictionary.

        Args:
            data: Dictionary containing ideation state fields

        Returns:
            New IdeationState instance

        Example:
            >>> state = IdeationState.from_dict(state_data)
        """
        # Convert idea dicts to Idea objects
        ideas = [Idea(**i) if isinstance(i, dict) else i for i in data.get("ideas", [])]

        # Convert cluster dicts to IdeaCluster objects
        clusters = [
            IdeaCluster(**c) if isinstance(c, dict) else c
            for c in data.get("clusters", [])
        ]

        return cls(
            session_id=data["session_id"],
            topic=data["topic"],
            perspectives=data.get("perspectives", []),
            ideas=ideas,
            clusters=clusters,
            selected_cluster_ids=data.get("selected_cluster_ids", []),
            elaborations=data.get("elaborations", {}),
            scoring_criteria=data.get("scoring_criteria", []),
            workflow_metadata=data.get("workflow_metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ============================================================================
# RESEARCH Workflow Models
# ============================================================================


class Source(BaseModel):
    """
    Represents a research source with metadata and validation.

    Used in the RESEARCH workflow to track sources that provide evidence
    for research findings. Each source includes credibility assessment,
    type classification, and validation status.

    Attributes:
        source_id: Unique identifier for this source
        title: Title or description of the source
        url: Optional URL or reference to the source
        source_type: Type of source (article, paper, book, website, etc.)
        credibility: Credibility assessment (high, medium, low, unassessed)
        tags: Optional list of tags for categorization
        validated: Whether source has been validated
        validation_score: Numeric validation score if validated
        validation_notes: List of validation findings
        ingested_at: ISO timestamp when source was added
        metadata: Additional source metadata

    Example:
        >>> source = Source(
        ...     source_id="src-001",
        ...     title="Machine Learning Best Practices",
        ...     url="https://example.com/ml-practices",
        ...     source_type="article",
        ...     credibility="high",
        ...     tags=["machine-learning", "best-practices"]
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_id": "src-001",
                "title": "Machine Learning Best Practices",
                "url": "https://example.com/ml-practices",
                "source_type": "article",
                "credibility": "high",
                "tags": ["machine-learning", "best-practices"],
                "validated": True,
                "validation_score": 8,
                "validation_notes": ["URL provided", "Credible type: article"],
                "ingested_at": "2025-11-06T12:00:00Z",
                "metadata": {},
            }
        }
    )

    source_id: str = Field(
        ..., description="Unique identifier for this source", min_length=1
    )

    title: str = Field(
        ..., description="Title or description of the source", min_length=1
    )

    url: str | None = Field(
        default=None, description="Optional URL or reference to the source"
    )

    source_type: str = Field(
        default="unknown",
        description="Type of source (article, paper, book, website, etc.)",
    )

    credibility: Literal["high", "medium", "low", "unassessed"] = Field(
        default="unassessed", description="Credibility assessment"
    )

    tags: list[str] = Field(
        default_factory=list, description="Optional list of tags for categorization"
    )

    validated: bool = Field(
        default=False, description="Whether source has been validated"
    )

    validation_score: int = Field(
        default=0, ge=0, description="Numeric validation score if validated"
    )

    validation_notes: list[str] = Field(
        default_factory=list, description="List of validation findings"
    )

    ingested_at: str = Field(..., description="ISO timestamp when source was added")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional source metadata"
    )
