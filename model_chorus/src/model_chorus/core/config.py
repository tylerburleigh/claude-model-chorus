"""Configuration management for ModelChorus.

This module provides support for project-level configuration via .model-chorusrc files.
Config files can be in YAML or JSON format and support workflow-specific defaults.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class GenerationDefaults(BaseModel):
    """Default generation parameters."""

    timeout: float | None = Field(None, gt=0)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    system_prompt: str | None = None


class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""

    model: str | None = None


class WorkflowConfig(BaseModel):
    """Configuration for a specific workflow."""

    provider: str | None = None
    providers: list[str] | None = None
    fallback_providers: list[str] | None = None
    timeout: float | None = Field(None, gt=0)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    system_prompt: str | None = None

    # Consensus-specific
    strategy: str | None = Field(None, pattern=r"^(all_responses|synthesize|vote)$")

    # ThinkDeep-specific
    thinking_mode: str | None = Field(None, pattern=r"^(low|medium|high)$")

    # Research-specific
    citation_style: str | None = Field(None, pattern=r"^(informal|academic|apa|mla)$")
    depth: str | None = Field(None, pattern=r"^(quick|thorough|comprehensive)$")

    @field_validator("provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if v and v.lower() not in ["claude", "gemini", "codex", "cursor-agent"]:
            raise ValueError(
                f"Invalid provider: {v}. Must be one of: claude, gemini, codex, cursor-agent"
            )
        return v.lower() if v else None

    @field_validator("providers")
    def validate_providers(cls, v):
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
    """Root configuration model for ModelChorus."""

    default_provider: str | None = None
    providers: dict[str, ProviderConfig] | None = None
    generation: GenerationDefaults | None = None
    workflows: dict[str, WorkflowConfig] | None = None

    @field_validator("default_provider")
    def validate_default_provider(cls, v):
        """Validate default provider."""
        if v and v.lower() not in ["claude", "gemini", "codex", "cursor-agent"]:
            raise ValueError(
                f"Invalid default_provider: {v}. Must be one of: claude, gemini, codex, cursor-agent"
            )
        return v.lower() if v else None

    @field_validator("providers")
    def validate_provider_names(cls, v):
        """Validate provider names in providers config."""
        if v:
            valid_providers = ["claude", "gemini", "codex", "cursor-agent"]
            for provider_name in v.keys():
                if provider_name.lower() not in valid_providers:
                    raise ValueError(
                        f"Invalid provider: {provider_name}. Must be one of: {', '.join(valid_providers)}"
                    )
        return v

    @field_validator("workflows")
    def validate_workflow_names(cls, v):
        """Validate workflow names."""
        if v:
            valid_workflows = [
                "chat",
                "consensus",
                "thinkdeep",
                "argument",
                "ideate",
                "research",
            ]
            for workflow_name in v.keys():
                if workflow_name.lower() not in valid_workflows:
                    raise ValueError(
                        f"Invalid workflow: {workflow_name}. Must be one of: {', '.join(valid_workflows)}"
                    )
        return v


class ConfigLoader:
    """Loads and manages ModelChorus configuration."""

    CONFIG_FILENAMES = [
        ".model-chorusrc",
        ".model-chorusrc.yaml",
        ".model-chorusrc.yml",
        ".model-chorusrc.json",
    ]

    def __init__(self):
        self._config: ModelChorusConfig | None = None
        self._config_path: Path | None = None

    def find_config_file(self, start_path: Path | None = None) -> Path | None:
        """Find .model-chorusrc file by searching up from start_path.

        Args:
            start_path: Directory to start search from (defaults to cwd)

        Returns:
            Path to config file if found, None otherwise
        """
        search_path = start_path or Path.cwd()

        # Search from current directory up to root
        for parent in [search_path] + list(search_path.parents):
            for filename in self.CONFIG_FILENAMES:
                config_file = parent / filename
                if config_file.exists() and config_file.is_file():
                    return config_file

        return None

    def load_config(self, config_path: Path | None = None) -> ModelChorusConfig:
        """Load configuration from file.

        Args:
            config_path: Explicit path to config file (if None, searches for it)

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If explicit config_path provided but not found
            ValueError: If config file is invalid
        """
        if config_path is None:
            config_path = self.find_config_file()
            if config_path is None:
                # No config file found, return empty config
                self._config = ModelChorusConfig()
                return self._config
        elif not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self._config_path = config_path

        # Load and parse config file
        with open(config_path) as f:
            content = f.read()

        # Try to parse as YAML first, then JSON
        config_data = self._parse_config_content(content, config_path)

        # Validate with Pydantic
        try:
            self._config = ModelChorusConfig(**config_data)
            return self._config
        except Exception as e:
            raise ValueError(f"Invalid config file {config_path}: {e}")

    def _parse_config_content(self, content: str, config_path: Path) -> dict[str, Any]:
        """Parse config file content as YAML or JSON.

        Args:
            content: File content
            config_path: Path to config file (for error messages)

        Returns:
            Parsed configuration dict

        Raises:
            ValueError: If content cannot be parsed
        """
        # Try YAML first if available
        if YAML_AVAILABLE:
            try:
                return yaml.safe_load(content) or {}
            except yaml.YAMLError as e:
                # If it looks like JSON, try JSON parser
                if content.strip().startswith("{"):
                    pass  # Fall through to JSON parser
                else:
                    raise ValueError(f"Invalid YAML in {config_path}: {e}")

        # Try JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            if not YAML_AVAILABLE:
                raise ValueError(
                    f"Invalid JSON in {config_path}: {e}\n"
                    "Install PyYAML for YAML support: pip install pyyaml"
                )
            raise ValueError(
                f"Invalid config file {config_path}: Could not parse as YAML or JSON"
            )

    def get_config(self) -> ModelChorusConfig:
        """Get the loaded configuration.

        Returns:
            Current configuration (loads if not already loaded)
        """
        if self._config is None:
            try:
                self.load_config()
            except Exception:
                # If loading fails, return empty config
                self._config = ModelChorusConfig()
        assert self._config is not None
        return self._config

    def get_workflow_default(
        self, workflow: str, key: str, fallback: Any = None
    ) -> Any:
        """Get a default value for a specific workflow.

        Precedence: workflow-specific > global generation defaults > fallback

        Args:
            workflow: Workflow name (chat, consensus, etc.)
            key: Configuration key
            fallback: Fallback value if not found in config

        Returns:
            Configuration value
        """
        config = self.get_config()

        # Check workflow-specific config first
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            value = getattr(workflow_config, key, None)
            if value is not None:
                return value

        # Check global generation defaults
        if config.generation and hasattr(config.generation, key):
            value = getattr(config.generation, key)
            if value is not None:
                return value

        # Return fallback
        return fallback

    def get_default_provider(self, workflow: str, fallback: str = "claude") -> str:
        """Get default provider for a workflow.

        Args:
            workflow: Workflow name
            fallback: Fallback provider if not configured

        Returns:
            Provider name
        """
        config = self.get_config()

        # Check workflow-specific provider
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.provider:
                return workflow_config.provider

        # Check global default provider
        if config.default_provider:
            return config.default_provider

        return fallback

    def get_default_providers(
        self, workflow: str, fallback: list[str] | None = None
    ) -> list[str]:
        """Get default providers list for multi-provider workflows.

        Args:
            workflow: Workflow name
            fallback: Fallback providers if not configured

        Returns:
            List of provider names
        """
        if fallback is None:
            fallback = ["claude", "gemini"]

        config = self.get_config()

        # Check workflow-specific providers
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.providers:
                return workflow_config.providers

        return fallback

    def get_fallback_providers(self, workflow: str) -> list[str] | None:
        """Get fallback providers for a workflow.

        Args:
            workflow: Workflow name

        Returns:
            List of fallback provider names, or None if not configured
        """
        config = self.get_config()

        # Check workflow-specific fallback providers
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.fallback_providers:
                return workflow_config.fallback_providers

        return None

    def get_fallback_providers_excluding(
        self, workflow: str, exclude_provider: str
    ) -> list[str] | None:
        """Get fallback providers for a workflow, excluding a specific provider.

        This is useful when the primary provider is specified via CLI and should not
        also appear in the fallback list, preventing duplicate provider checks.

        Args:
            workflow: Workflow name
            exclude_provider: Provider name to exclude from fallback list

        Returns:
            List of fallback provider names (excluding the specified provider),
            or None if no fallback providers configured

        Example:
            >>> config = Config()
            >>> # If fallback_providers = ['gemini', 'claude', 'codex']
            >>> config.get_fallback_providers_excluding('thinkdeep', 'gemini')
            ['claude', 'codex']
        """
        fallbacks = self.get_fallback_providers(workflow)
        if fallbacks is None:
            return None

        # Filter out the excluded provider
        filtered = [p for p in fallbacks if p != exclude_provider]

        # Return None if list is now empty, otherwise return filtered list
        return filtered if filtered else None

    def get_provider_model(
        self, provider: str, fallback: str | None = None
    ) -> str | None:
        """Get the configured model for a specific provider.

        Args:
            provider: Provider name (claude, gemini, codex, cursor-agent)
            fallback: Fallback model if not configured

        Returns:
            Configured model name, or fallback if not configured
        """
        config = self.get_config()

        # Check provider-specific model configuration
        if config.providers and provider in config.providers:
            provider_config = config.providers[provider]
            if provider_config.model:
                return provider_config.model

        return fallback

    @property
    def config_path(self) -> Path | None:
        """Get the path to the loaded config file."""
        return self._config_path


# Global config loader instance
_config_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_path: Path | None = None) -> ModelChorusConfig:
    """Load configuration (convenience function).

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Loaded configuration
    """
    loader = get_config_loader()
    return loader.load_config(config_path)


def get_config() -> ModelChorusConfig:
    """Get the current configuration (convenience function).

    Returns:
        Current configuration
    """
    loader = get_config_loader()
    return loader.get_config()


# =============================================================================
# New .claude/model_chorus_config.yaml Configuration System
# =============================================================================


class GenerationDefaultsV2(BaseModel):
    """Default generation parameters for .claude/model_chorus_config.yaml."""

    timeout: float | None = Field(None, gt=0)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)


class ProviderConfigV2(BaseModel):
    """Configuration for a provider in .claude/model_chorus_config.yaml."""

    enabled: bool = True
    default_model: str | None = None
    timeout: float | None = Field(None, gt=0)


class WorkflowConfigV2(BaseModel):
    """Workflow configuration for .claude/model_chorus_config.yaml."""

    default_provider: str | None = None
    fallback_providers: list[str] | None = None
    timeout: float | None = Field(None, gt=0)

    # Priority-based provider selection for consensus workflow
    provider_priority: list[str] | None = None  # Ordered list of providers to try
    num_to_consult: int | None = Field(
        None, gt=0
    )  # Number of successful responses needed

    @field_validator("default_provider")
    def validate_provider(cls, v):
        """Validate provider name."""
        if v and v.lower() not in ["claude", "gemini", "codex", "cursor-agent"]:
            raise ValueError(
                f"Invalid provider: {v}. Must be one of: claude, gemini, codex, cursor-agent"
            )
        return v.lower() if v else None

    @field_validator("provider_priority", "fallback_providers")
    def validate_providers(cls, v):
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


class ModelChorusConfigV2(BaseModel):
    """Root configuration for .claude/model_chorus_config.yaml."""

    providers: dict[str, ProviderConfigV2] = Field(default_factory=dict)
    generation: GenerationDefaultsV2 | None = None
    workflows: dict[str, WorkflowConfigV2] | None = None

    @field_validator("providers")
    def validate_provider_names(cls, v):
        """Validate provider names."""
        valid_providers = ["claude", "gemini", "codex", "cursor-agent"]
        for provider_name in v.keys():
            if provider_name.lower() not in valid_providers:
                raise ValueError(
                    f"Invalid provider: {provider_name}. Must be one of: {', '.join(valid_providers)}"
                )
        return v

    @field_validator("workflows")
    def validate_workflow_names(cls, v):
        """Validate workflow names."""
        if v:
            valid_workflows = [
                "chat",
                "consensus",
                "thinkdeep",
                "argument",
                "ideate",
                "study",
            ]
            for workflow_name in v.keys():
                if workflow_name.lower() not in valid_workflows:
                    raise ValueError(
                        f"Invalid workflow: {workflow_name}. Must be one of: {', '.join(valid_workflows)}"
                    )
        return v


class ClaudeConfigLoader:
    """Loads and manages configuration from .claude/model_chorus_config.yaml."""

    CONFIG_FILENAME = "model_chorus_config.yaml"
    CONFIG_DIR = ".claude"

    def __init__(self):
        self._config: ModelChorusConfigV2 | None = None
        self._config_path: Path | None = None

    def find_config_file(self, start_path: Path | None = None) -> Path | None:
        """Find .claude/model_chorus_config.yaml by searching up from start_path.

        Args:
            start_path: Directory to start search from (defaults to cwd)

        Returns:
            Path to config file if found, None otherwise
        """
        search_path = start_path or Path.cwd()

        # Search from current directory up to root
        for parent in [search_path] + list(search_path.parents):
            config_file = parent / self.CONFIG_DIR / self.CONFIG_FILENAME
            if config_file.exists() and config_file.is_file():
                return config_file

        return None

    def load_config(self, config_path: Path | None = None) -> ModelChorusConfigV2:
        """Load configuration from .claude/model_chorus_config.yaml.

        Args:
            config_path: Explicit path to config file (if None, searches for it)

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If explicit config_path provided but not found
            ValueError: If config file is invalid
        """
        if config_path is None:
            config_path = self.find_config_file()
            if config_path is None:
                # No config file found, return default config with all providers disabled
                self._config = ModelChorusConfigV2(
                    providers={
                        "claude": ProviderConfigV2(enabled=False, timeout=None),
                        "gemini": ProviderConfigV2(enabled=False, timeout=None),
                        "codex": ProviderConfigV2(enabled=False, timeout=None),
                        "cursor-agent": ProviderConfigV2(enabled=False, timeout=None),
                    }
                )
                return self._config
        elif not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self._config_path = config_path

        # Load and parse config file
        with open(config_path) as f:
            content = f.read()

        # Parse as YAML
        if not YAML_AVAILABLE:
            raise ValueError(
                "PyYAML is required to load .claude/model_chorus_config.yaml. "
                "Install it with: pip install pyyaml"
            )

        try:
            config_data = yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")

        # Validate with Pydantic
        try:
            self._config = ModelChorusConfigV2(**config_data)
            return self._config
        except Exception as e:
            raise ValueError(f"Invalid config file {config_path}: {e}")

    def get_config(self) -> ModelChorusConfigV2:
        """Get the loaded configuration.

        Returns:
            Current configuration (loads if not already loaded)
        """
        if self._config is None:
            try:
                self.load_config()
            except Exception:
                # If loading fails, return default config with all providers disabled
                self._config = ModelChorusConfigV2(
                    providers={
                        "claude": ProviderConfigV2(enabled=False, timeout=None),
                        "gemini": ProviderConfigV2(enabled=False, timeout=None),
                        "codex": ProviderConfigV2(enabled=False, timeout=None),
                        "cursor-agent": ProviderConfigV2(enabled=False, timeout=None),
                    }
                )
        assert self._config is not None
        return self._config

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled.

        Args:
            provider: Provider name (claude, gemini, codex, cursor-agent)

        Returns:
            True if provider is enabled, False otherwise
        """
        config = self.get_config()
        provider = provider.lower()

        if provider in config.providers:
            return config.providers[provider].enabled

        # If provider not in config, assume disabled
        return False

    def get_enabled_providers(self) -> list[str]:
        """Get list of all enabled providers.

        Returns:
            List of enabled provider names
        """
        config = self.get_config()
        return [
            provider_name
            for provider_name, provider_config in config.providers.items()
            if provider_config.enabled
        ]

    def get_workflow_providers(
        self, workflow: str, fallback: list[str] | None = None
    ) -> list[str]:
        """Get providers for a workflow, filtered by enabled status.

        Args:
            workflow: Workflow name
            fallback: Fallback providers if not configured

        Returns:
            List of enabled provider names for the workflow
        """
        config = self.get_config()
        providers = fallback or []

        # Check workflow-specific configuration
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            # Check for new priority-based config first
            if workflow_config.provider_priority:
                providers = workflow_config.provider_priority
            elif workflow_config.fallback_providers:
                providers = workflow_config.fallback_providers

        # Filter by enabled status
        return [p for p in providers if self.is_provider_enabled(p)]

    def get_workflow_provider_priority(
        self, workflow: str, fallback: list[str] | None = None
    ) -> list[str]:
        """Get priority-ordered provider list for a workflow.

        Args:
            workflow: Workflow name
            fallback: Fallback providers if not configured

        Returns:
            List of enabled provider names in priority order
        """
        if fallback is None:
            fallback = ["gemini", "cursor-agent", "claude", "codex"]

        config = self.get_config()

        # Check workflow-specific priority configuration
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.provider_priority:
                providers = workflow_config.provider_priority
                # Filter by enabled status
                return [p for p in providers if self.is_provider_enabled(p)]

        # Use fallback and filter by enabled status
        return [p for p in fallback if self.is_provider_enabled(p)]

    def get_workflow_num_to_consult(self, workflow: str, fallback: int = 2) -> int:
        """Get number of providers to consult for a workflow.

        Args:
            workflow: Workflow name
            fallback: Fallback count if not configured (default: 2)

        Returns:
            Number of providers to consult
        """
        config = self.get_config()

        # Check workflow-specific configuration
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.num_to_consult is not None:
                return workflow_config.num_to_consult

        return fallback

    def get_default_providers(
        self, workflow: str, fallback: list[str] | None = None
    ) -> list[str]:
        """Get default providers list for multi-provider workflows.

        This is an alias for get_workflow_providers for compatibility with ConfigLoader.

        Args:
            workflow: Workflow name
            fallback: Fallback providers if not configured

        Returns:
            List of enabled provider names for the workflow
        """
        if fallback is None:
            fallback = ["claude", "gemini"]

        return self.get_workflow_providers(workflow, fallback)

    def get_workflow_default(
        self, workflow: str, key: str, fallback: Any = None
    ) -> Any:
        """Get a default value for a specific workflow.

        Precedence: workflow-specific > global generation defaults > fallback

        Args:
            workflow: Workflow name (chat, consensus, etc.)
            key: Configuration key
            fallback: Fallback value if not found in config

        Returns:
            Configuration value
        """
        config = self.get_config()

        # Check workflow-specific config first
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            value = getattr(workflow_config, key, None)
            if value is not None:
                return value

        # Check global generation defaults
        if config.generation and hasattr(config.generation, key):
            value = getattr(config.generation, key)
            if value is not None:
                return value

        # Return fallback
        return fallback

    def get_workflow_default_provider(
        self, workflow: str, fallback: str = "claude"
    ) -> str | None:
        """Get default provider for a workflow, if enabled.

        Args:
            workflow: Workflow name
            fallback: Fallback provider if not configured

        Returns:
            Provider name if enabled, None if disabled
        """
        config = self.get_config()
        provider = fallback

        # Check workflow-specific configuration
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.default_provider:
                provider = workflow_config.default_provider

        # Return provider only if enabled
        return provider if self.is_provider_enabled(provider) else None

    def get_workflow_fallback_providers(
        self, workflow: str, exclude_provider: str | None = None
    ) -> list[str]:
        """Get fallback providers for a workflow, filtered by enabled status.

        Args:
            workflow: Workflow name
            exclude_provider: Provider to exclude from fallback list

        Returns:
            List of enabled fallback provider names
        """
        config = self.get_config()
        fallback_providers = []

        # Check workflow-specific configuration
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.fallback_providers:
                fallback_providers = workflow_config.fallback_providers

        # Filter by enabled status and exclude specified provider
        if exclude_provider:
            fallback_providers = [
                p for p in fallback_providers if p != exclude_provider
            ]

        return [p for p in fallback_providers if self.is_provider_enabled(p)]

    def get_provider_model(self, provider: str) -> str | None:
        """Get the configured default model for a provider.

        Args:
            provider: Provider name

        Returns:
            Configured model name, or None if not configured
        """
        config = self.get_config()
        provider = provider.lower()

        if provider in config.providers:
            return config.providers[provider].default_model

        return None

    @property
    def config_path(self) -> Path | None:
        """Get the path to the loaded config file."""
        return self._config_path


# Global Claude config loader instance
_claude_config_loader: ClaudeConfigLoader | None = None


def get_claude_config_loader() -> ClaudeConfigLoader:
    """Get the global Claude config loader instance."""
    global _claude_config_loader
    if _claude_config_loader is None:
        _claude_config_loader = ClaudeConfigLoader()
    return _claude_config_loader


def get_claude_config() -> ModelChorusConfigV2:
    """Get the current Claude configuration (convenience function).

    Returns:
        Current configuration from .claude/model_chorus_config.yaml
    """
    loader = get_claude_config_loader()
    return loader.get_config()
