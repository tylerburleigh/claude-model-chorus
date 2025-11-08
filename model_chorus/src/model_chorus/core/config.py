"""Configuration management for ModelChorus.

This module provides support for project-level configuration via .model-chorusrc files.
Config files can be in YAML or JSON format and support workflow-specific defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class GenerationDefaults(BaseModel):
    """Default generation parameters."""

    timeout: Optional[float] = Field(None, gt=0)


class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""

    model: Optional[str] = None


class WorkflowConfig(BaseModel):
    """Configuration for a specific workflow."""

    provider: Optional[str] = None
    providers: Optional[List[str]] = None
    fallback_providers: Optional[List[str]] = None
    timeout: Optional[float] = Field(None, gt=0)

    # Consensus-specific
    strategy: Optional[str] = Field(None, pattern=r"^(all_responses|synthesize|vote)$")

    # ThinkDeep-specific
    thinking_mode: Optional[str] = Field(None, pattern=r"^(low|medium|high)$")

    # Research-specific
    citation_style: Optional[str] = Field(None, pattern=r"^(informal|academic|apa|mla)$")
    depth: Optional[str] = Field(None, pattern=r"^(quick|thorough|comprehensive)$")

    @field_validator('provider')
    def validate_provider(cls, v):
        """Validate provider name."""
        if v and v.lower() not in ['claude', 'gemini', 'codex', 'cursor-agent']:
            raise ValueError(f"Invalid provider: {v}. Must be one of: claude, gemini, codex, cursor-agent")
        return v.lower() if v else None

    @field_validator('providers')
    def validate_providers(cls, v):
        """Validate provider list."""
        if v:
            valid_providers = ['claude', 'gemini', 'codex', 'cursor-agent']
            for provider in v:
                if provider.lower() not in valid_providers:
                    raise ValueError(f"Invalid provider: {provider}. Must be one of: {', '.join(valid_providers)}")
            return [p.lower() for p in v]
        return None


class ModelChorusConfig(BaseModel):
    """Root configuration model for ModelChorus."""

    default_provider: Optional[str] = None
    providers: Optional[Dict[str, ProviderConfig]] = None
    generation: Optional[GenerationDefaults] = None
    workflows: Optional[Dict[str, WorkflowConfig]] = None

    @field_validator('default_provider')
    def validate_default_provider(cls, v):
        """Validate default provider."""
        if v and v.lower() not in ['claude', 'gemini', 'codex', 'cursor-agent']:
            raise ValueError(f"Invalid default_provider: {v}. Must be one of: claude, gemini, codex, cursor-agent")
        return v.lower() if v else None

    @field_validator('providers')
    def validate_provider_names(cls, v):
        """Validate provider names in providers config."""
        if v:
            valid_providers = ['claude', 'gemini', 'codex', 'cursor-agent']
            for provider_name in v.keys():
                if provider_name.lower() not in valid_providers:
                    raise ValueError(f"Invalid provider: {provider_name}. Must be one of: {', '.join(valid_providers)}")
        return v

    @field_validator('workflows')
    def validate_workflow_names(cls, v):
        """Validate workflow names."""
        if v:
            valid_workflows = ['chat', 'consensus', 'thinkdeep', 'argument', 'ideate', 'research']
            for workflow_name in v.keys():
                if workflow_name.lower() not in valid_workflows:
                    raise ValueError(f"Invalid workflow: {workflow_name}. Must be one of: {', '.join(valid_workflows)}")
        return v


class ConfigLoader:
    """Loads and manages ModelChorus configuration."""

    CONFIG_FILENAMES = ['.model-chorusrc', '.model-chorusrc.yaml', '.model-chorusrc.yml', '.model-chorusrc.json']

    def __init__(self):
        self._config: Optional[ModelChorusConfig] = None
        self._config_path: Optional[Path] = None

    def find_config_file(self, start_path: Optional[Path] = None) -> Optional[Path]:
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

    def load_config(self, config_path: Optional[Path] = None) -> ModelChorusConfig:
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
        with open(config_path, 'r') as f:
            content = f.read()

        # Try to parse as YAML first, then JSON
        config_data = self._parse_config_content(content, config_path)

        # Validate with Pydantic
        try:
            self._config = ModelChorusConfig(**config_data)
            return self._config
        except Exception as e:
            raise ValueError(f"Invalid config file {config_path}: {e}")

    def _parse_config_content(self, content: str, config_path: Path) -> Dict[str, Any]:
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
                if content.strip().startswith('{'):
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
            raise ValueError(f"Invalid config file {config_path}: Could not parse as YAML or JSON")

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
        return self._config

    def get_workflow_default(self, workflow: str, key: str, fallback: Any = None) -> Any:
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

    def get_default_providers(self, workflow: str, fallback: List[str] = None) -> List[str]:
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

    def get_fallback_providers(self, workflow: str) -> Optional[List[str]]:
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

    def get_provider_model(self, provider: str, fallback: Optional[str] = None) -> Optional[str]:
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
    def config_path(self) -> Optional[Path]:
        """Get the path to the loaded config file."""
        return self._config_path


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_path: Optional[Path] = None) -> ModelChorusConfig:
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
