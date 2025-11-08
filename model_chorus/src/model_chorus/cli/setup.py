"""
Setup helper commands for ModelChorus.

Provides commands for /model-chorus-setup slash command.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List


# Default models for each provider
DEFAULT_MODELS = {
    'claude': 'sonnet',
    'gemini': 'gemini-2.5-pro',
    'codex': 'gpt-5-codex',
    'cursor-agent': 'composer-1'
}


def check_package_installed() -> Dict[str, Any]:
    """Check if model-chorus package is installed.

    Returns:
        Dict with installation status and details
    """
    try:
        result = subprocess.run(
            ['pip', 'show', 'model-chorus'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            # Package is installed
            lines = result.stdout.strip().split('\n')
            info = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip().lower()] = value.strip()

            return {
                "installed": True,
                "version": info.get('version', 'unknown'),
                "location": info.get('location', 'unknown'),
            }
        else:
            return {
                "installed": False,
                "message": "Package not found"
            }
    except Exception as e:
        return {
            "installed": False,
            "error": str(e)
        }


def install_package(dev_mode: bool = False) -> Dict[str, Any]:
    """Install model-chorus package.

    Args:
        dev_mode: If True, install in editable mode with -e flag

    Returns:
        Dict with installation result
    """
    try:
        if dev_mode:
            # Install in editable mode from current directory
            cmd = ['pip', 'install', '-e', '.']
        else:
            # Install from PyPI
            cmd = ['pip', 'install', 'model-chorus']

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            return {
                "success": True,
                "message": f"Successfully installed model-chorus {'(dev mode)' if dev_mode else ''}",
                "output": result.stdout
            }
        else:
            return {
                "success": False,
                "message": "Installation failed",
                "error": result.stderr
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_version_compatibility() -> Dict[str, Any]:
    """Check if installed package version matches plugin version.

    Compares the version from .claude-plugin/plugin.json with the installed
    package version. If package version is lower, recommends reinstall.

    Returns:
        Dict with compatibility status and version details
    """
    try:
        # Get plugin version from .claude-plugin/plugin.json
        plugin_json_path = Path.cwd() / '.claude-plugin' / 'plugin.json'
        plugin_version = None

        if plugin_json_path.exists():
            with open(plugin_json_path, 'r') as f:
                plugin_data = json.load(f)
                plugin_version = plugin_data.get('version', 'unknown')
        else:
            return {
                "compatible": True,
                "message": "Plugin manifest not found, skipping version check"
            }

        # Get installed package version
        package_info = check_package_installed()

        if not package_info.get('installed'):
            return {
                "compatible": True,
                "message": "Package not installed yet"
            }

        package_version = package_info.get('version', 'unknown')

        # Compare versions (simple string comparison for now)
        # For semantic versioning, we'd use packaging.version
        if package_version == 'unknown' or plugin_version == 'unknown':
            return {
                "compatible": True,
                "plugin_version": plugin_version,
                "package_version": package_version,
                "message": "Cannot determine version compatibility"
            }

        # Parse versions for comparison
        def parse_version(v: str) -> tuple:
            """Parse version string into comparable tuple."""
            try:
                return tuple(map(int, v.split('.')))
            except:
                return (0, 0, 0)

        plugin_ver_tuple = parse_version(plugin_version)
        package_ver_tuple = parse_version(package_version)

        if package_ver_tuple < plugin_ver_tuple:
            return {
                "compatible": False,
                "plugin_version": plugin_version,
                "package_version": package_version,
                "message": f"Package version ({package_version}) is older than plugin version ({plugin_version})",
                "recommendation": "reinstall"
            }
        elif package_ver_tuple == plugin_ver_tuple:
            return {
                "compatible": True,
                "plugin_version": plugin_version,
                "package_version": package_version,
                "message": f"Versions match ({package_version})"
            }
        else:
            return {
                "compatible": True,
                "plugin_version": plugin_version,
                "package_version": package_version,
                "message": f"Package version ({package_version}) is newer than plugin version ({plugin_version})",
                "warning": "Plugin may need updating"
            }

    except Exception as e:
        return {
            "compatible": True,
            "error": str(e),
            "message": "Error checking version compatibility"
        }


def check_available_providers() -> Dict[str, Any]:
    """Check which CLI providers are available on the system.

    Returns:
        Dict with available provider names and details
    """
    try:
        # Import providers
        from ..providers.claude_provider import ClaudeProvider
        from ..providers.gemini_provider import GeminiProvider
        from ..providers.codex_provider import CodexProvider
        from ..providers.cursor_agent_provider import CursorAgentProvider

        providers = {
            "claude": ClaudeProvider(),
            "gemini": GeminiProvider(),
            "codex": CodexProvider(),
            "cursor-agent": CursorAgentProvider(),
        }

        available = []
        unavailable = []

        async def check_all():
            """Check all providers concurrently."""
            tasks = []
            for name, provider in providers.items():
                tasks.append(check_provider(name, provider))
            return await asyncio.gather(*tasks)

        async def check_provider(name: str, provider):
            """Check a single provider."""
            is_available, error = await provider.check_availability()
            return (name, is_available, error)

        # Run checks
        results = asyncio.run(check_all())

        for name, is_available, error in results:
            if is_available:
                available.append(name)
            else:
                unavailable.append({"name": name, "error": error})

        return {
            "available": available,
            "unavailable": unavailable,
            "count": len(available)
        }

    except Exception as e:
        return {
            "available": [],
            "unavailable": [],
            "count": 0,
            "error": str(e)
        }


def check_config_exists(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Check if .model-chorusrc config file exists.

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with config file status
    """
    if project_root is None:
        project_root = Path.cwd()

    config_files = [
        '.model-chorusrc',
        '.model-chorusrc.yaml',
        '.model-chorusrc.yml',
        '.model-chorusrc.json'
    ]

    for filename in config_files:
        config_path = project_root / filename
        if config_path.exists():
            return {
                "exists": True,
                "path": str(config_path),
                "filename": filename
            }

    return {
        "exists": False,
        "message": "No config file found"
    }


def create_config_file(
    project_root: Optional[Path] = None,
    default_provider: str = "claude",
    timeout: float = 600.0,
    available_providers: Optional[List[str]] = None,
    workflows: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create .model-chorusrc configuration file.

    Args:
        project_root: Project root directory (defaults to cwd)
        default_provider: Default AI provider
        timeout: Default timeout in seconds
        available_providers: List of available providers for model config
        workflows: Workflow-specific configurations (optional)

    Returns:
        Dict with creation result
    """
    if project_root is None:
        project_root = Path.cwd()

    config_path = project_root / '.model-chorusrc'

    # Check if file already exists
    if config_path.exists():
        return {
            "success": False,
            "message": f"Config file already exists: {config_path}",
            "path": str(config_path)
        }

    # Build provider model configuration
    providers_config = {}
    if available_providers:
        for provider_name in available_providers:
            if provider_name in DEFAULT_MODELS:
                providers_config[provider_name] = {
                    "model": DEFAULT_MODELS[provider_name]
                }

    # Build config structure
    config = {
        "default_provider": default_provider,
        "generation": {
            "timeout": timeout
        }
    }

    if providers_config:
        config["providers"] = providers_config

    if workflows:
        config["workflows"] = workflows

    # Create YAML content
    yaml_content = f"""# ModelChorus Configuration
# This file was generated by /model-chorus-setup

# Default provider for all workflows
default_provider: {default_provider}

"""

    # Add provider model configuration
    if providers_config:
        yaml_content += "# Provider-specific model configuration\nproviders:\n"
        for provider_name, provider_config in providers_config.items():
            yaml_content += f"  {provider_name}:\n"
            yaml_content += f"    model: {provider_config['model']}\n"
        yaml_content += "\n"

    # Add generation parameters
    yaml_content += f"""# Global generation parameters
generation:
  timeout: {timeout}
"""

    if workflows:
        yaml_content += "\n# Workflow-specific overrides\nworkflows:\n"
        for workflow_name, workflow_config in workflows.items():
            yaml_content += f"  {workflow_name}:\n"
            for key, value in workflow_config.items():
                if isinstance(value, list):
                    yaml_content += f"    {key}:\n"
                    for item in value:
                        yaml_content += f"      - {item}\n"
                else:
                    yaml_content += f"    {key}: {value}\n"

    try:
        with open(config_path, 'w') as f:
            f.write(yaml_content)

        return {
            "success": True,
            "message": f"Created config file: {config_path}",
            "path": str(config_path),
            "config": config
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_express_config(
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Create Express (zero-question) .model-chorusrc configuration.

    Auto-detects available providers and configures with smart defaults:
    - Primary provider: first available (claude → gemini → codex → cursor-agent)
    - Fallbacks: all other available providers
    - Default models: configured for each available provider
    - Timeout: 120s (standard)
    - All workflows configured with balanced settings

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with creation result
    """
    if project_root is None:
        project_root = Path.cwd()

    # Check available providers
    provider_check = check_available_providers()
    available_providers = provider_check.get("available", [])

    if not available_providers:
        return {
            "success": False,
            "error": "No providers available. Please install at least one provider CLI.",
            "suggestion": "Install Claude CLI, Gemini CLI, Codex CLI, or Cursor Agent"
        }

    # Select primary provider (first available in priority order)
    # Deprioritize Claude since users are typically running from Claude Code
    priority_order = ['gemini', 'codex', 'cursor-agent', 'claude']
    primary_provider = None
    for provider in priority_order:
        if provider in available_providers:
            primary_provider = provider
            break

    if not primary_provider:
        primary_provider = available_providers[0]

    # Compute fallback providers
    fallback_providers = [p for p in available_providers if p != primary_provider]

    # Build workflows configuration
    workflows = {}

    # Add base workflows with fallback providers (all tiers)
    if fallback_providers:
        # Chat workflow
        workflows["chat"] = {
            "fallback_providers": fallback_providers
        }

        # Argument workflow
        workflows["argument"] = {
            "fallback_providers": fallback_providers
        }

        # ThinkDeep workflow
        workflows["thinkdeep"] = {
            "thinking_mode": "medium",
            "fallback_providers": fallback_providers
        }

        # Ideate workflow
        workflows["ideate"] = {
            "fallback_providers": fallback_providers
        }

    # Consensus workflow (use first 2-3 available providers)
    if len(available_providers) >= 2:
        workflows["consensus"] = {
            "providers": available_providers[:3],  # Use up to 3 providers
            "strategy": "all_responses"
        }

    # Use the existing create_config_file function with workflows
    return create_config_file(
        project_root=project_root,
        default_provider=primary_provider,
        timeout=600.0,
        available_providers=available_providers,
        workflows=workflows if workflows else None
    )


def create_tiered_config(
    project_root: Optional[Path] = None,
    tier: str = "quick",
    default_provider: str = "claude",
    # Standard tier options
    consensus_providers: Optional[list] = None,
    consensus_strategy: str = "all_responses",
    thinkdeep_thinking_mode: str = "medium",
    ideate_providers: Optional[list] = None,
    # Advanced tier options
    workflow_overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Create tiered .model-chorusrc configuration file.

    Args:
        project_root: Project root directory (defaults to cwd)
        tier: Configuration tier (quick, standard, advanced)
        default_provider: Default AI provider
        consensus_providers: Providers for consensus workflow (standard+)
        consensus_strategy: Strategy for consensus workflow (standard+)
        thinkdeep_thinking_mode: Thinking mode for thinkdeep (standard+)
        ideate_providers: Providers for ideate workflow (standard+)
        workflow_overrides: Additional workflow overrides (advanced)

    Returns:
        Dict with creation result
    """
    if project_root is None:
        project_root = Path.cwd()

    config_path = project_root / '.model-chorusrc'

    # Check if file already exists
    if config_path.exists():
        return {
            "success": False,
            "message": f"Config file already exists: {config_path}",
            "path": str(config_path)
        }

    # Check available providers and compute fallbacks
    provider_check = check_available_providers()
    available_providers = provider_check.get("available", [])

    # Compute fallback providers: all available except the primary
    fallback_providers = [p for p in available_providers if p != default_provider]

    # Build workflows config based on tier
    workflows = {}

    # Add base workflows with fallback providers (all tiers)
    if fallback_providers:
        # Chat workflow
        workflows["chat"] = {
            "fallback_providers": fallback_providers
        }

        # Argument workflow
        workflows["argument"] = {
            "fallback_providers": fallback_providers
        }

    if tier in ["standard", "advanced"]:
        # Add consensus workflow (multi-provider, no fallback)
        if consensus_providers:
            workflows["consensus"] = {
                "providers": consensus_providers,
                "strategy": consensus_strategy
            }

        # Add thinkdeep workflow (single-provider)
        workflows["thinkdeep"] = {
            "thinking_mode": thinkdeep_thinking_mode
        }
        if fallback_providers:
            workflows["thinkdeep"]["fallback_providers"] = fallback_providers

        # Add ideate workflow
        if ideate_providers:
            # Multi-provider ideate, no fallback
            workflows["ideate"] = {
                "providers": ideate_providers
            }
        elif fallback_providers:
            # Single-provider ideate, add fallback
            workflows["ideate"] = {
                "fallback_providers": fallback_providers
            }

    if tier == "advanced" and workflow_overrides:
        # Merge in additional workflow overrides
        for workflow_name, config in workflow_overrides.items():
            if workflow_name in workflows:
                workflows[workflow_name].update(config)
            else:
                workflows[workflow_name] = config

    # Use the existing create_config_file function with workflows
    return create_config_file(
        project_root=project_root,
        default_provider=default_provider,
        timeout=600.0,
        available_providers=available_providers,
        workflows=workflows if workflows else None
    )


def validate_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Validate .model-chorusrc configuration file.

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with validation result
    """
    if project_root is None:
        project_root = Path.cwd()

    # Import config loader
    try:
        from ..core.config import ConfigLoader
    except ImportError:
        return {
            "valid": False,
            "error": "Could not import ConfigLoader"
        }

    loader = ConfigLoader()
    config_path = loader.find_config_file(project_root)

    if not config_path:
        return {
            "valid": False,
            "message": "No config file found"
        }

    try:
        loader.load_config(config_path)
        return {
            "valid": True,
            "message": "Configuration is valid",
            "path": str(config_path)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "path": str(config_path)
        }


def check_permissions(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Check if Claude Code permissions are configured.

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with permissions status
    """
    if project_root is None:
        project_root = Path.cwd()

    settings_file = project_root / ".claude" / "settings.local.json"

    if not settings_file.exists():
        return {
            "configured": False,
            "message": "No .claude/settings.local.json file found"
        }

    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)

        permissions = settings.get('permissions', {}).get('allow', [])

        # Check for key model-chorus permissions
        has_model_chorus = any('model-chorus' in p for p in permissions)

        return {
            "configured": has_model_chorus,
            "permissions_count": len(permissions),
            "has_model_chorus_permissions": has_model_chorus
        }
    except Exception as e:
        return {
            "configured": False,
            "error": str(e)
        }


def add_to_gitignore(
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Add .model-chorusrc to project .gitignore.

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with result
    """
    if project_root is None:
        project_root = Path.cwd()

    gitignore_path = project_root / '.gitignore'

    # Entries to add
    entries_to_add = [
        '.model-chorusrc',
        '.model-chorusrc.yaml',
        '.model-chorusrc.yml',
        '.model-chorusrc.json'
    ]

    # Read existing gitignore or create new
    if gitignore_path.exists():
        try:
            with open(gitignore_path, 'r') as f:
                existing_content = f.read()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read .gitignore: {e}"
            }
    else:
        existing_content = ""

    # Check which entries are missing
    existing_lines = set(line.strip() for line in existing_content.split('\n'))
    entries_needed = [entry for entry in entries_to_add if entry not in existing_lines]

    if not entries_needed:
        return {
            "success": True,
            "message": "All entries already in .gitignore",
            "added_entries": []
        }

    # Add entries
    try:
        with open(gitignore_path, 'a') as f:
            # Add section header if file exists and doesn't end with newline
            if existing_content and not existing_content.endswith('\n'):
                f.write('\n')

            if existing_content:
                f.write('\n')

            f.write('# ModelChorus configuration\n')
            for entry in entries_needed:
                f.write(f'{entry}\n')

        return {
            "success": True,
            "message": f"Added {len(entries_needed)} entries to .gitignore",
            "added_entries": entries_needed,
            "path": str(gitignore_path)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write .gitignore: {e}"
        }


def add_permissions(
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Add ModelChorus permissions to .claude/settings.local.json.

    Args:
        project_root: Project root directory (defaults to cwd)

    Returns:
        Dict with result
    """
    if project_root is None:
        project_root = Path.cwd()

    settings_file = project_root / ".claude" / "settings.local.json"

    # Define permissions to add
    permissions_to_add = [
        "Bash(model-chorus:*)",
        "Skill(model-chorus:*)",
    ]

    # Create .claude directory if needed
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings or create new
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read settings file: {e}"
            }
    else:
        settings = {
            "permissions": {
                "allow": []
            }
        }

    # Ensure permissions structure exists
    if 'permissions' not in settings:
        settings['permissions'] = {}
    if 'allow' not in settings['permissions']:
        settings['permissions']['allow'] = []

    # Add new permissions (avoiding duplicates)
    existing = set(settings['permissions']['allow'])
    added = []

    for perm in permissions_to_add:
        if perm not in existing:
            settings['permissions']['allow'].append(perm)
            added.append(perm)

    # Write updated settings
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
            f.write('\n')

        return {
            "success": True,
            "message": f"Added {len(added)} permissions",
            "added_permissions": added,
            "path": str(settings_file)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write settings file: {e}"
        }


def main():
    """CLI entry point for setup commands."""
    import argparse

    parser = argparse.ArgumentParser(description="ModelChorus setup helper")
    subparsers = parser.add_subparsers(dest='command', help='Setup command')

    # check-install command
    subparsers.add_parser('check-install', help='Check if model-chorus is installed')

    # check-version command
    subparsers.add_parser('check-version', help='Check version compatibility')

    # check-available-providers command
    subparsers.add_parser('check-available-providers', help='Check which CLI providers are available')

    # install command
    install_parser = subparsers.add_parser('install', help='Install model-chorus')
    install_parser.add_argument('--dev', action='store_true', help='Install in development mode')

    # check-config command
    config_parser = subparsers.add_parser('check-config', help='Check if config exists')
    config_parser.add_argument('--project', default=None, help='Project root directory')

    # create-config command
    create_parser = subparsers.add_parser('create-config', help='Create config file')
    create_parser.add_argument('--project', default=None, help='Project root directory')
    create_parser.add_argument('--provider', default='claude', help='Default provider')

    # create-express-config command
    express_parser = subparsers.add_parser('create-express-config', help='Create Express (zero-question) config')
    express_parser.add_argument('--project', default=None, help='Project root directory')

    # create-tiered-config command
    tiered_parser = subparsers.add_parser('create-tiered-config', help='Create tiered config file')
    tiered_parser.add_argument('--project', default=None, help='Project root directory')
    tiered_parser.add_argument('--tier', default='quick', choices=['quick', 'standard', 'advanced'], help='Configuration tier')
    tiered_parser.add_argument('--provider', default='claude', help='Default provider')
    # Standard tier options
    tiered_parser.add_argument('--consensus-providers', nargs='+', help='Providers for consensus workflow')
    tiered_parser.add_argument('--consensus-strategy', default='all_responses', help='Consensus strategy')
    tiered_parser.add_argument('--thinkdeep-mode', default='medium', help='ThinkDeep thinking mode')
    tiered_parser.add_argument('--ideate-providers', nargs='+', help='Providers for ideate workflow')

    # validate-config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate config file')
    validate_parser.add_argument('--project', default=None, help='Project root directory')

    # check-permissions command
    perm_check_parser = subparsers.add_parser('check-permissions', help='Check permissions')
    perm_check_parser.add_argument('--project', default=None, help='Project root directory')

    # add-permissions command
    perm_add_parser = subparsers.add_parser('add-permissions', help='Add permissions')
    perm_add_parser.add_argument('--project', default=None, help='Project root directory')

    # add-to-gitignore command
    gitignore_parser = subparsers.add_parser('add-to-gitignore', help='Add .model-chorusrc to .gitignore')
    gitignore_parser.add_argument('--project', default=None, help='Project root directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    project = Path(args.project) if hasattr(args, 'project') and args.project else None

    # Execute command
    if args.command == 'check-install':
        result = check_package_installed()
    elif args.command == 'check-version':
        result = check_version_compatibility()
    elif args.command == 'check-available-providers':
        result = check_available_providers()
    elif args.command == 'install':
        result = install_package(dev_mode=args.dev)
    elif args.command == 'check-config':
        result = check_config_exists(project)
    elif args.command == 'create-config':
        result = create_config_file(
            project,
            default_provider=args.provider
        )
    elif args.command == 'create-express-config':
        result = create_express_config(project)
    elif args.command == 'create-tiered-config':
        result = create_tiered_config(
            project,
            tier=args.tier,
            default_provider=args.provider,
            consensus_providers=args.consensus_providers,
            consensus_strategy=args.consensus_strategy,
            thinkdeep_thinking_mode=args.thinkdeep_mode,
            ideate_providers=args.ideate_providers
        )
    elif args.command == 'validate-config':
        result = validate_config(project)
    elif args.command == 'check-permissions':
        result = check_permissions(project)
    elif args.command == 'add-permissions':
        result = add_permissions(project)
    elif args.command == 'add-to-gitignore':
        result = add_to_gitignore(project)
    else:
        print(json.dumps({"error": "Unknown command"}))
        sys.exit(1)

    # Output result as JSON
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    if 'success' in result:
        sys.exit(0 if result['success'] else 1)
    elif 'valid' in result:
        sys.exit(0 if result['valid'] else 1)
    elif 'installed' in result:
        sys.exit(0 if result['installed'] else 1)
    elif 'configured' in result:
        sys.exit(0 if result['configured'] else 1)
    elif 'compatible' in result:
        sys.exit(0 if result['compatible'] else 1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
