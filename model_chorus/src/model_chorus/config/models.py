"""
Pydantic models for ModelChorus configuration.

This module provides comprehensive typed configuration models with validation
for all configuration sections including providers, workflows, and generation defaults.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GenerationDefaults(BaseModel):
    """
    Default generation parameters that apply globally.

    These defaults can be overridden by workflow-specific settings or
    runtime parameters passed to CLI commands.
    """

    timeout: float | None = Field(
        None,
        gt=0,
        description="Default timeout in seconds for AI model requests",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Default temperature for response generation (0.0-2.0)",
    )
    max_tokens: int | None = Field(
        None, gt=0, description="Default maximum tokens for responses"
    )
    system_prompt: str | None = Field(
        None, description="Default system prompt for all workflows"
    )


class ProviderConfig(BaseModel):
    """
    Configuration for a specific AI provider.

    Allows customization of provider-specific settings like model selection,
    API endpoints, and provider-specific parameters.
    """

    enabled: bool = Field(
        True, description="Whether this provider is enabled for use"
    )
    model: str | None = Field(
        None, description="Specific model to use for this provider"
    )
    api_base: str | None = Field(
        None, description="Custom API base URL (for compatible providers)"
    )
    timeout: float | None = Field(
        None, gt=0, description="Provider-specific timeout override"
    )
    max_retries: int | None = Field(
        None, ge=0, description="Maximum number of retry attempts"
    )


class ConsensusWorkflowConfig(BaseModel):
    """Configuration specific to consensus workflow."""

    strategy: Literal["all_responses", "synthesize", "vote"] | None = Field(
        None, description="Strategy for combining multiple model responses"
    )
    num_to_consult: int | None = Field(
        None,
        ge=2,
        description="Number of models to consult (if not using all available)",
    )
    include_individual_responses: bool = Field(
        True, description="Include individual responses in output"
    )


class ThinkDeepWorkflowConfig(BaseModel):
    """Configuration specific to thinkdeep workflow."""

    thinking_mode: Literal["low", "medium", "high"] | None = Field(
        None, description="Depth of reasoning and analysis"
    )
    max_iterations: int | None = Field(
        None, ge=1, le=10, description="Maximum reasoning iterations"
    )
    confidence_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level to stop reasoning",
    )


class IdeateWorkflowConfig(BaseModel):
    """Configuration specific to ideate workflow."""

    num_ideas: int | None = Field(
        None, ge=1, description="Target number of ideas to generate"
    )
    creativity_level: Literal["conservative", "balanced", "creative"] | None = (
        Field(None, description="Level of creativity in idea generation")
    )
    diversity_weight: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Weight given to idea diversity vs quality",
    )


class ArgumentWorkflowConfig(BaseModel):
    """Configuration specific to argument workflow."""

    depth: Literal["quick", "thorough", "comprehensive"] | None = Field(
        None, description="Depth of argumentative analysis"
    )
    include_counterarguments: bool = Field(
        True, description="Generate counterarguments for each position"
    )


class ChatWorkflowConfig(BaseModel):
    """Configuration specific to chat workflow."""

    memory_enabled: bool = Field(
        True, description="Enable conversation memory/context"
    )
    max_history_messages: int | None = Field(
        None, ge=0, description="Maximum messages to keep in history"
    )


class WorkflowConfig(BaseModel):
    """
    Configuration for a specific workflow.

    Provides workflow-specific overrides for providers, generation parameters,
    and workflow-specific settings.
    """

    # Provider configuration
    provider: str | None = Field(
        None, description="Default provider for this workflow"
    )
    providers: list[str] | None = Field(
        None, description="List of providers to use (for multi-provider workflows)"
    )
    fallback_providers: list[str] | None = Field(
        None, description="Fallback providers if primary fails"
    )

    # Generation parameters (override global defaults)
    timeout: float | None = Field(None, gt=0, description="Workflow timeout")
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Workflow temperature"
    )
    max_tokens: int | None = Field(None, gt=0, description="Workflow max tokens")
    system_prompt: str | None = Field(
        None, description="Workflow system prompt"
    )

    # Workflow-specific configs (embedded)
    consensus: ConsensusWorkflowConfig | None = Field(
        None, description="Consensus workflow settings"
    )
    thinkdeep: ThinkDeepWorkflowConfig | None = Field(
        None, description="ThinkDeep workflow settings"
    )
    ideate: IdeateWorkflowConfig | None = Field(
        None, description="Ideate workflow settings"
    )
    argument: ArgumentWorkflowConfig | None = Field(
        None, description="Argument workflow settings"
    )
    chat: ChatWorkflowConfig | None = Field(
        None, description="Chat workflow settings"
    )

    # Legacy fields for backward compatibility
    strategy: str | None = Field(
        None, description="Legacy: use consensus.strategy instead"
    )
    thinking_mode: str | None = Field(
        None, description="Legacy: use thinkdeep.thinking_mode instead"
    )
    citation_style: str | None = Field(None, description="Citation format style")
    depth: str | None = Field(None, description="Legacy: use argument.depth instead")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str | None) -> str | None:
        """Validate provider name."""
        if v and v.lower() not in ["claude", "gemini", "codex", "cursor-agent"]:
            raise ValueError(
                f"Invalid provider: {v}. Must be one of: claude, gemini, codex, cursor-agent"
            )
        return v.lower() if v else None

    @field_validator("providers", "fallback_providers")
    @classmethod
    def validate_providers(cls, v: list[str] | None) -> list[str] | None:
        """Validate provider list."""
        if v:
            valid_providers = ["claude", "gemini", "codex", "cursor-agent"]
            for provider in v:
                if provider.lower() not in valid_providers:
                    raise ValueError(
                        f"Invalid provider: {provider}. Must be one of: {', '.join(valid_providers)}"
                    )
            return [p.lower() for p in v]
        return None


class ModelChorusConfig(BaseModel):
    """
    Root configuration model for ModelChorus.

    This is the top-level configuration object that encompasses all settings
    including global defaults, provider configurations, and workflow-specific
    settings.
    """

    # Global defaults
    default_provider: str | None = Field(
        None, description="Default provider when none specified"
    )
    generation: GenerationDefaults | None = Field(
        None, description="Global generation parameter defaults"
    )

    # Provider configurations
    providers: dict[str, ProviderConfig] | None = Field(
        None, description="Per-provider configuration settings"
    )

    # Workflow configurations
    workflows: dict[str, WorkflowConfig] | None = Field(
        None, description="Per-workflow configuration settings"
    )

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str | None) -> str | None:
        """Validate default provider."""
        if v and v.lower() not in ["claude", "gemini", "codex", "cursor-agent"]:
            raise ValueError(
                f"Invalid default_provider: {v}. Must be one of: claude, gemini, codex, cursor-agent"
            )
        return v.lower() if v else None

    @field_validator("providers")
    @classmethod
    def validate_provider_names(cls, v: dict[str, ProviderConfig] | None) -> dict[str, ProviderConfig] | None:
        """Validate provider names in providers config."""
        if v:
            valid_providers = ["claude", "gemini", "codex", "cursor-agent"]
            for provider_name in v.keys():
                if provider_name.lower() not in valid_providers:
                    raise ValueError(
                        f"Invalid provider name: {provider_name}. Must be one of: {', '.join(valid_providers)}"
                    )
        return v

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig object (default if not configured)
        """
        if self.providers and provider_name in self.providers:
            return self.providers[provider_name]
        return ProviderConfig()

    def get_workflow_config(self, workflow_name: str) -> WorkflowConfig:
        """
        Get configuration for a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            WorkflowConfig object (default if not configured)
        """
        if self.workflows and workflow_name in self.workflows:
            return self.workflows[workflow_name]
        return WorkflowConfig()

    def is_provider_enabled(self, provider_name: str) -> bool:
        """
        Check if a provider is enabled.

        Args:
            provider_name: Name of the provider

        Returns:
            True if enabled (default), False if explicitly disabled
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.enabled
