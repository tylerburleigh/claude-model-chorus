"""
Unit tests for configuration module.

Tests verify configuration loading, validation, and default behavior including:
- Config file discovery and loading (YAML and JSON)
- Configuration validation with Pydantic models
- Default value precedence (workflow-specific > global > fallback)
- Provider configuration (single and multi-provider)
- Config management CLI commands
"""

import json
import pytest
from pathlib import Path

from model_chorus.core.config import (
    ConfigLoader,
    GenerationDefaults,
    WorkflowConfig,
    ModelChorusConfig,
    get_config_loader,
)


class TestConfigLoader:
    """Test suite for ConfigLoader class."""

    # ========================================================================
    # Config File Discovery Tests
    # ========================================================================

    def test_find_config_file_in_current_dir(self, tmp_path):
        """Test finding .model-chorusrc in current directory."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("default_provider: claude")

        loader = ConfigLoader()
        found_path = loader.find_config_file(tmp_path)

        assert found_path == config_file

    def test_find_config_file_in_parent_dir(self, tmp_path):
        """Test finding .model-chorusrc in parent directory."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("default_provider: claude")

        # Create subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        loader = ConfigLoader()
        found_path = loader.find_config_file(subdir)

        assert found_path == config_file

    def test_find_config_file_yaml_extension(self, tmp_path):
        """Test finding .model-chorusrc.yaml file."""
        config_file = tmp_path / ".model-chorusrc.yaml"
        config_file.write_text("default_provider: gemini")

        loader = ConfigLoader()
        found_path = loader.find_config_file(tmp_path)

        assert found_path == config_file

    def test_find_config_file_json_extension(self, tmp_path):
        """Test finding .model-chorusrc.json file."""
        config_file = tmp_path / ".model-chorusrc.json"
        config_file.write_text('{"default_provider": "gemini"}')

        loader = ConfigLoader()
        found_path = loader.find_config_file(tmp_path)

        assert found_path == config_file

    def test_find_config_file_not_found(self, tmp_path):
        """Test when no config file exists."""
        loader = ConfigLoader()
        found_path = loader.find_config_file(tmp_path)

        assert found_path is None

    # ========================================================================
    # Config Loading Tests (YAML)
    # ========================================================================

    def test_load_yaml_config_global_defaults(self, tmp_path):
        """Test loading YAML config with global defaults."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
default_provider: gemini
generation:
  temperature: 0.8
  max_tokens: 1500
  timeout: 180.0
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        assert config.default_provider == "gemini"
        assert config.generation.temperature == 0.8
        assert config.generation.max_tokens == 1500
        assert config.generation.timeout == 180.0

    def test_load_yaml_config_workflow_specific(self, tmp_path):
        """Test loading YAML config with workflow-specific settings."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
workflows:
  chat:
    provider: claude
    temperature: 0.7
  consensus:
    providers:
      - claude
      - gemini
    strategy: synthesize
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        assert "chat" in config.workflows
        assert config.workflows["chat"].provider == "claude"
        assert config.workflows["chat"].temperature == 0.7

        assert "consensus" in config.workflows
        assert config.workflows["consensus"].providers == ["claude", "gemini"]
        assert config.workflows["consensus"].strategy == "synthesize"

    # ========================================================================
    # Config Loading Tests (JSON)
    # ========================================================================

    def test_load_json_config(self, tmp_path):
        """Test loading JSON config file."""
        config_file = tmp_path / ".model-chorusrc.json"
        config_data = {
            "default_provider": "codex",
            "generation": {"temperature": 0.9, "max_tokens": 2500},
        }
        config_file.write_text(json.dumps(config_data))

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        assert config.default_provider == "codex"
        assert config.generation.temperature == 0.9
        assert config.generation.max_tokens == 2500

    # ========================================================================
    # Config Validation Tests
    # ========================================================================

    def test_invalid_provider_raises_error(self, tmp_path):
        """Test that invalid provider name raises validation error."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("default_provider: invalid_provider")

        loader = ConfigLoader()

        with pytest.raises(ValueError, match="Invalid default_provider"):
            loader.load_config(config_file)

    def test_invalid_temperature_raises_error(self, tmp_path):
        """Test that invalid temperature raises validation error."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
generation:
  temperature: 5.0
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()

        with pytest.raises(ValueError):
            loader.load_config(config_file)

    def test_invalid_workflow_name_raises_error(self, tmp_path):
        """Test that invalid workflow name raises validation error."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
workflows:
  invalid_workflow:
    provider: claude
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()

        with pytest.raises(ValueError, match="Invalid workflow"):
            loader.load_config(config_file)

    def test_invalid_consensus_strategy_raises_error(self, tmp_path):
        """Test that invalid consensus strategy raises validation error."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
workflows:
  consensus:
    strategy: invalid_strategy
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()

        with pytest.raises(ValueError):
            loader.load_config(config_file)

    # ========================================================================
    # Default Value Resolution Tests
    # ========================================================================

    def test_get_default_provider_workflow_specific(self, tmp_path):
        """Test getting workflow-specific default provider."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
default_provider: gemini
workflows:
  chat:
    provider: claude
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        loader.load_config(config_file)

        # Chat should use workflow-specific provider
        assert loader.get_default_provider("chat", "fallback") == "claude"

        # Other workflows should use global default
        assert loader.get_default_provider("thinkdeep", "fallback") == "gemini"

    def test_get_default_provider_fallback(self, tmp_path, monkeypatch):
        """Test getting fallback provider when no config exists."""
        # Change to tmp directory to avoid loading project config
        monkeypatch.chdir(tmp_path)
        loader = ConfigLoader()
        loader.load_config()  # No config file

        assert loader.get_default_provider("chat", "claude") == "claude"

    def test_get_workflow_default_temperature(self, tmp_path):
        """Test getting workflow-specific temperature default."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
generation:
  temperature: 0.7
workflows:
  thinkdeep:
    temperature: 0.6
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        loader.load_config(config_file)

        # ThinkDeep should use workflow-specific temperature
        assert loader.get_workflow_default("thinkdeep", "temperature", 0.5) == 0.6

        # Chat should use global temperature
        assert loader.get_workflow_default("chat", "temperature", 0.5) == 0.7

    def test_get_default_providers_multi(self, tmp_path):
        """Test getting multiple providers for consensus workflows."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
workflows:
  consensus:
    providers:
      - claude
      - gemini
      - codex
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        loader.load_config(config_file)

        providers = loader.get_default_providers("consensus", ["fallback"])
        assert providers == ["claude", "gemini", "codex"]

    # ========================================================================
    # Empty/Minimal Config Tests
    # ========================================================================

    def test_load_empty_config(self, tmp_path):
        """Test loading empty config file returns empty config."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("")

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        assert config.default_provider is None
        assert config.generation is None
        assert config.workflows is None

    def test_load_minimal_config(self, tmp_path):
        """Test loading minimal config with only one setting."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("default_provider: claude")

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        assert config.default_provider == "claude"
        assert config.generation is None
        assert config.workflows is None

    # ========================================================================
    # Pydantic Model Tests
    # ========================================================================

    def test_generation_defaults_validation(self):
        """Test GenerationDefaults model validation."""
        defaults = GenerationDefaults(temperature=0.7, max_tokens=2000, timeout=120.0)

        assert defaults.temperature == 0.7
        assert defaults.max_tokens == 2000
        assert defaults.timeout == 120.0

    def test_workflow_config_validation(self):
        """Test WorkflowConfig model validation."""
        config = WorkflowConfig(provider="claude", temperature=0.7, strategy="synthesize")

        assert config.provider == "claude"
        assert config.temperature == 0.7
        assert config.strategy == "synthesize"

    def test_workflow_config_provider_normalization(self):
        """Test that provider names are normalized to lowercase."""
        config = WorkflowConfig(provider="CLAUDE")
        assert config.provider == "claude"

    def test_model_chorus_config_complete(self):
        """Test complete ModelChorusConfig with all fields."""
        config = ModelChorusConfig(
            default_provider="claude",
            generation=GenerationDefaults(temperature=0.7),
            workflows={
                "chat": WorkflowConfig(provider="gemini"),
                "consensus": WorkflowConfig(providers=["claude", "gemini"]),
            },
        )

        assert config.default_provider == "claude"
        assert config.generation.temperature == 0.7
        assert config.workflows["chat"].provider == "gemini"
        assert config.workflows["consensus"].providers == ["claude", "gemini"]

    # ========================================================================
    # Global Config Loader Tests
    # ========================================================================

    def test_get_config_loader_singleton(self):
        """Test that get_config_loader returns singleton instance."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()

        assert loader1 is loader2

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Test loading non-existent file raises FileNotFoundError."""
        config_file = tmp_path / "nonexistent.yaml"

        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_config(config_file)

    def test_load_invalid_yaml_raises_error(self, tmp_path):
        """Test loading invalid YAML raises ValueError."""
        config_file = tmp_path / ".model-chorusrc"
        config_file.write_text("invalid: yaml: content: [")

        loader = ConfigLoader()

        with pytest.raises(ValueError, match="Invalid"):
            loader.load_config(config_file)

    def test_load_invalid_json_raises_error(self, tmp_path):
        """Test loading invalid JSON raises ValueError."""
        config_file = tmp_path / ".model-chorusrc.json"
        config_file.write_text('{"invalid": json')

        loader = ConfigLoader()

        with pytest.raises(ValueError, match="Invalid"):
            loader.load_config(config_file)


class TestWorkflowConfigValidation:
    """Test suite for workflow-specific configuration validation."""

    def test_thinkdeep_thinking_mode_validation(self):
        """Test ThinkDeep thinking_mode validation."""
        # Valid thinking mode
        config = WorkflowConfig(thinking_mode="medium")
        assert config.thinking_mode == "medium"

        # Invalid thinking mode should raise error
        with pytest.raises(ValueError):
            WorkflowConfig(thinking_mode="invalid")

    def test_research_citation_style_validation(self):
        """Test Research citation_style validation."""
        # Valid citation styles
        for style in ["informal", "academic", "apa", "mla"]:
            config = WorkflowConfig(citation_style=style)
            assert config.citation_style == style

        # Invalid citation style should raise error
        with pytest.raises(ValueError):
            WorkflowConfig(citation_style="invalid")

    def test_research_depth_validation(self):
        """Test Research depth validation."""
        # Valid depths
        for depth in ["quick", "thorough", "comprehensive"]:
            config = WorkflowConfig(depth=depth)
            assert config.depth == depth

        # Invalid depth should raise error
        with pytest.raises(ValueError):
            WorkflowConfig(depth="invalid")

    def test_consensus_strategy_validation(self):
        """Test Consensus strategy validation."""
        # Valid strategies
        for strategy in ["all_responses", "synthesize", "vote"]:
            config = WorkflowConfig(strategy=strategy)
            assert config.strategy == strategy

        # Invalid strategy should raise error
        with pytest.raises(ValueError):
            WorkflowConfig(strategy="invalid")


class TestComplexConfigScenarios:
    """Test suite for complex real-world configuration scenarios."""

    def test_full_config_all_workflows(self, tmp_path):
        """Test loading comprehensive config with all workflows configured."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
default_provider: claude
generation:
  temperature: 0.7
  max_tokens: 2000

workflows:
  chat:
    provider: claude
    temperature: 0.7
  consensus:
    providers: [claude, gemini]
    strategy: synthesize
  thinkdeep:
    provider: claude
    thinking_mode: high
  argument:
    provider: claude
    temperature: 0.8
  ideate:
    providers: [claude, gemini]
    temperature: 0.9
  research:
    providers: [claude, gemini, codex]
    citation_style: academic
    depth: comprehensive
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        config = loader.load_config(config_file)

        # Verify all workflows are loaded correctly
        assert len(config.workflows) == 6
        assert config.workflows["chat"].provider == "claude"
        assert config.workflows["consensus"].providers == ["claude", "gemini"]
        assert config.workflows["thinkdeep"].thinking_mode == "high"
        assert config.workflows["research"].citation_style == "academic"

    def test_precedence_workflow_overrides_global(self, tmp_path):
        """Test that workflow-specific settings override global settings."""
        config_file = tmp_path / ".model-chorusrc"
        config_content = """
default_provider: gemini
generation:
  temperature: 0.7
  max_tokens: 2000

workflows:
  chat:
    provider: claude
    temperature: 0.9
    max_tokens: 3000
"""
        config_file.write_text(config_content)

        loader = ConfigLoader()
        loader.load_config(config_file)

        # Workflow-specific settings should override global
        assert loader.get_default_provider("chat") == "claude"
        assert loader.get_workflow_default("chat", "temperature", 0.5) == 0.9
        assert loader.get_workflow_default("chat", "max_tokens", 1000) == 3000

        # Other workflows should use global settings
        assert loader.get_default_provider("thinkdeep") == "gemini"
        assert loader.get_workflow_default("thinkdeep", "temperature", 0.5) == 0.7
