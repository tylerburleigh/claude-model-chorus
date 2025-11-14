"""
Shared test utilities and provider availability flags.

This module provides helper functions and flags that are used by both
conftest.py and individual test files.
"""

import os
import shutil
import subprocess
import yaml
from pathlib import Path


def load_ai_config():
    """
    Load ai_config.yaml to determine which providers are enabled.

    Returns:
        dict: The 'tools' section from ai_config.yaml, or empty dict if not found.
    """
    # Try multiple possible locations for ai_config.yaml
    possible_paths = [
        Path(__file__).parent.parent.parent / ".claude" / "ai_config.yaml",
        Path.cwd() / ".claude" / "ai_config.yaml",
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                return config.get("tools", {})
            except Exception as e:
                print(f"Warning: Failed to load {config_path}: {e}")
                continue

    # No config found - all providers considered enabled by default
    return {}


def is_provider_enabled_in_config(provider_name: str, ai_config: dict) -> bool:
    """
    Check if a provider is enabled in ai_config.yaml.

    Args:
        provider_name: Name of the provider (e.g., 'claude', 'gemini', 'codex')
        ai_config: The loaded ai_config tools section

    Returns:
        bool: True if enabled in config or if no config exists (default: enabled)
    """
    if not ai_config:
        # No config means all providers are enabled by default
        return True

    provider_config = ai_config.get(provider_name, {})
    return provider_config.get("enabled", False)


def is_cli_available(cli_command: str) -> bool:
    """
    Check if a CLI command is available in PATH and working.

    Args:
        cli_command: The CLI command to check (e.g., 'claude', 'gemini')

    Returns:
        bool: True if CLI is available and responds to --version
    """
    try:
        # First check if command exists in PATH
        if not shutil.which(cli_command):
            return False

        # Try running --version to verify it works
        result = subprocess.run(
            [cli_command, "--version"],
            capture_output=True,
            timeout=5,
            text=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False
    except Exception:
        return False


def is_provider_available(provider_name: str, cli_command: str, ai_config: dict) -> bool:
    """
    Combined check: provider must be enabled in config AND have CLI available.

    Args:
        provider_name: Name of the provider (e.g., 'claude', 'gemini', 'codex')
        cli_command: CLI command for the provider
        ai_config: The loaded ai_config tools section

    Returns:
        bool: True if both config-enabled and CLI available
    """
    config_enabled = is_provider_enabled_in_config(provider_name, ai_config)
    cli_available = is_cli_available(cli_command)

    return config_enabled and cli_available


# Load config once at module level
_AI_CONFIG = load_ai_config()

# Provider availability flags (cached at module load)
CLAUDE_AVAILABLE = is_provider_available("claude", "claude", _AI_CONFIG)
GEMINI_AVAILABLE = is_provider_available("gemini", "gemini", _AI_CONFIG)
CODEX_AVAILABLE = is_provider_available("codex", "codex", _AI_CONFIG)
CURSOR_AGENT_AVAILABLE = is_provider_available("cursor-agent", "cursor-agent", _AI_CONFIG)
ANY_PROVIDER_AVAILABLE = CLAUDE_AVAILABLE or GEMINI_AVAILABLE or CODEX_AVAILABLE or CURSOR_AGENT_AVAILABLE
