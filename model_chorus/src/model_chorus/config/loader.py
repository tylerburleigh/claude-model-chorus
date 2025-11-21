"""
Configuration loader for ModelChorus.

This module provides ConfigLoader class for loading and managing configuration
from .model-chorusrc files (YAML or JSON format).
"""

import json
from pathlib import Path
from typing import Any

from .models import ModelChorusConfig, ProviderConfig, WorkflowConfig

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigLoader:
    """
    Loads and manages ModelChorus configuration.

    ConfigLoader handles:
    - Finding config files in project directories (.model-chorusrc)
    - Parsing YAML and JSON config formats
    - Validating configuration with Pydantic models
    - Providing convenient access to config values with fallback logic
    """

    CONFIG_FILENAMES = [
        ".model-chorusrc",
        ".model-chorusrc.yaml",
        ".model-chorusrc.yml",
        ".model-chorusrc.json",
    ]

    def __init__(self):
        """Initialize ConfigLoader with empty state."""
        self._config: ModelChorusConfig | None = None
        self._config_path: Path | None = None

    def find_config_file(self, start_path: Path | None = None) -> Path | None:
        """
        Find .model-chorusrc file by searching up from start_path.

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
        """
        Load configuration from file.

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

    def _parse_config_content(
        self, content: str, config_path: Path
    ) -> dict[str, Any]:
        """
        Parse config file content as YAML or JSON.

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
        """
        Get the loaded configuration.

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
        """
        Get a default value for a specific workflow.

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
        """
        Get default provider for a workflow, checking if it's enabled.

        Args:
            workflow: Workflow name
            fallback: Fallback provider if not configured

        Returns:
            Provider name if enabled, None if disabled
        """
        config = self.get_config()

        # Check workflow-specific provider
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.provider:
                # Check if provider is enabled
                if config.is_provider_enabled(workflow_config.provider):
                    return workflow_config.provider
                else:
                    # Provider is disabled, return None
                    return None

        # Check global default provider
        if config.default_provider:
            if config.is_provider_enabled(config.default_provider):
                return config.default_provider
            else:
                return None

        # Use fallback if enabled
        if config.is_provider_enabled(fallback):
            return fallback

        return None

    def get_workflow_default_providers(
        self, workflow: str, fallback: list[str] | None = None
    ) -> list[str]:
        """
        Get enabled providers list for multi-provider workflows.

        Args:
            workflow: Workflow name
            fallback: Fallback providers if not configured

        Returns:
            List of enabled provider names
        """
        if fallback is None:
            fallback = ["claude", "gemini"]

        config = self.get_config()

        # Check workflow-specific providers
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.providers:
                # Filter to only enabled providers
                return [
                    p
                    for p in workflow_config.providers
                    if config.is_provider_enabled(p)
                ]

        # Use fallback, filtered to enabled
        return [p for p in fallback if config.is_provider_enabled(p)]

    def get_workflow_fallback_providers(
        self, workflow: str, exclude_provider: str | None = None
    ) -> list[str]:
        """
        Get enabled fallback providers for a workflow.

        Args:
            workflow: Workflow name
            exclude_provider: Provider to exclude (e.g., primary provider)

        Returns:
            List of enabled fallback provider names
        """
        config = self.get_config()
        fallback_providers = []

        # Check workflow-specific fallback providers
        if config.workflows and workflow in config.workflows:
            workflow_config = config.workflows[workflow]
            if workflow_config.fallback_providers:
                fallback_providers = workflow_config.fallback_providers

        # Filter to enabled providers and exclude specified provider
        result = [
            p
            for p in fallback_providers
            if config.is_provider_enabled(p)
            and (exclude_provider is None or p != exclude_provider)
        ]

        return result

    def get_provider_model(
        self, provider: str, fallback: str | None = None
    ) -> str | None:
        """
        Get the configured model for a specific provider.

        Args:
            provider: Provider name (claude, gemini, codex, cursor-agent)
            fallback: Fallback model if not configured

        Returns:
            Model name, or fallback if not configured
        """
        config = self.get_config()

        # Check if provider has custom model configured
        if config.providers and provider in config.providers:
            provider_config = config.providers[provider]
            if provider_config.model:
                return provider_config.model

        return fallback


# Global config loader instance
_config_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """
    Get the global ConfigLoader instance.

    Returns:
        Singleton ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
