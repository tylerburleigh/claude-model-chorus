"""
Pydantic models for ModelChorus workflow requests and responses.

This module defines the data models used for workflow communication,
providing validation and serialization for workflow inputs and outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


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
