# ModelChorus Testing Guide

This document explains how the ModelChorus test suite works, particularly the provider-based test skipping infrastructure.

## Overview

The ModelChorus test suite includes:
- **Unit tests**: Test provider logic with mocks (always run)
- **Integration tests**: Test with real provider CLIs (skip if provider unavailable)

Tests automatically skip based on two conditions:
1. Provider is enabled in `ai_config.yaml`
2. Provider CLI is installed and working

## Provider Detection

Provider availability is determined by checking **both**:
- Configuration: Is the provider enabled in `.claude/ai_config.yaml`?
- Runtime: Is the CLI command available and working (`<cli> --version`)?

### Configuration File

Edit `.claude/ai_config.yaml` to enable/disable providers:

```yaml
tools:
  gemini:
    command: gemini
    enabled: true      # ← Set to true/false
  claude:
    command: claude
    enabled: false     # ← Disabled
  codex:
    command: codex
    enabled: true      # ← Enabled
```

### CLI Installation

Ensure provider CLIs are installed and in your PATH:
- **Claude**: `claude --version`
- **Gemini**: `gemini --version`
- **Codex**: `codex --version`

## Running Tests

### Run all tests
```bash
cd model_chorus
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific provider tests
```bash
# Only Gemini integration tests
pytest -v -m requires_gemini

# Only tests that require any provider
pytest -v -m requires_any_provider

# Skip provider-specific tests
pytest -v -m "not requires_gemini"
```

### Run specific test files
```bash
# Integration tests for chat workflow
pytest tests/test_chat_integration.py -v

# Gemini integration tests
pytest tests/test_gemini_integration.py -v

# Unit tests (always run, use mocks)
pytest tests/test_claude_provider.py -v
pytest tests/test_codex_provider.py -v
```

### List available markers
```bash
pytest --markers
```

## Mock Providers (Development)

For development without real provider CLIs, use mock providers:

```bash
# Run tests with mocks instead of real CLIs
USE_MOCK_PROVIDERS=true pytest
```

Mock providers simulate real provider behavior for testing:
- Understand conversation context
- Extract information from prompts
- Generate contextually appropriate responses
- Handle file context

## Test Markers

The test suite uses these pytest markers:

- `@pytest.mark.requires_claude`: Test requires Claude (config + CLI)
- `@pytest.mark.requires_gemini`: Test requires Gemini (config + CLI)
- `@pytest.mark.requires_codex`: Test requires Codex (config + CLI)
- `@pytest.mark.requires_any_provider`: Test requires at least one provider
- `@pytest.mark.integration`: Integration test (may be slow, uses real APIs)
- `@pytest.mark.unit`: Unit test (fast, uses mocks)
- `@pytest.mark.slow`: Slow-running test

## Test Organization

```
tests/
├── conftest.py                    # Pytest config, fixtures, provider detection
├── pytest.ini                     # Pytest configuration
├── README_TESTING.md             # This file
│
├── test_claude_provider.py       # Claude unit tests (use mocks, always run)
├── test_codex_provider.py        # Codex unit tests (use mocks, always run)
├── test_gemini_integration.py    # Gemini integration tests (skip if unavailable)
├── test_chat_integration.py      # Multi-provider chat tests (parameterized)
│
└── ... (other test files)
```

## Parameterized Provider Tests

Tests can automatically run across all available providers using the `provider_name` fixture:

```python
def test_something(provider_name, provider):
    """This test runs once for each available provider."""
    # provider_name will be "claude", "gemini", or "codex"
    # Tests auto-skip for unavailable providers
    pass
```

## Troubleshooting

### Tests are being skipped unexpectedly

**Check provider status:**
```bash
# Verify CLI availability
claude --version
gemini --version
codex --version

# Check ai_config.yaml
cat .claude/ai_config.yaml | grep -A2 "claude:"
```

**Common issues:**
1. Provider disabled in `ai_config.yaml` (set `enabled: true`)
2. CLI not installed (install provider CLI)
3. CLI not in PATH (check `which <cli>` or `where <cli>`)
4. CLI installed but not working (check `<cli> --version`)

### Provider detection status

The availability flags are printed when conftest.py loads:
```bash
pytest --collect-only
```

Look for skip reasons in the output:
```
<Module test_chat_integration.py>
  <Class TestMultiProviderChat>
    <Function test_basic_conversation[claude]> SKIPPED (Claude not available...)
    <Function test_basic_conversation[gemini]>
    <Function test_basic_conversation[codex]>
```

### Using mocks for development

If you don't have provider CLIs installed or want faster tests:

```bash
# All tests use mocks (no real CLI calls)
USE_MOCK_PROVIDERS=true pytest -v

# Specific test file with mocks
USE_MOCK_PROVIDERS=true pytest tests/test_chat_integration.py -v
```

## CI/CD Integration

For CI/CD pipelines, you can:

1. **Run with mocks only** (fast, no API costs):
   ```bash
   USE_MOCK_PROVIDERS=true pytest
   ```

2. **Run with selective providers** (control which providers to test):
   ```bash
   # Only test enabled providers in ai_config.yaml
   pytest -v

   # Skip all provider integration tests
   pytest -v -m "not (requires_claude or requires_gemini or requires_codex)"
   ```

3. **Matrix testing** (test each provider separately):
   ```bash
   # In GitHub Actions, GitLab CI, etc.
   pytest -v -m requires_gemini  # Gemini job
   pytest -v -m requires_claude  # Claude job
   pytest -v -m requires_codex   # Codex job
   ```

## Adding New Provider Tests

### Integration Test (requires real CLI)

```python
from conftest import MYPROVIDER_AVAILABLE

@pytest.mark.requires_myprovider
@pytest.mark.skipif(not MYPROVIDER_AVAILABLE, reason="MyProvider not available")
class TestMyProviderIntegration:
    """Integration tests for MyProvider."""

    @pytest.mark.asyncio
    async def test_real_api_call(self):
        # Makes actual CLI calls
        pass
```

### Unit Test (uses mocks, always runs)

```python
class TestMyProvider:
    """Unit tests for MyProvider."""

    def test_command_building(self):
        # Tests logic without real CLI
        # No markers needed - always runs
        pass
```

## Best Practices

1. **Use mocks for development**: Set `USE_MOCK_PROVIDERS=true` for fast iteration
2. **Use real CLIs for pre-commit**: Verify actual integration before committing
3. **Keep tests fast**: Use fast models (haiku, flash, mini) in integration tests
4. **Isolate tests**: Each test should be independent and use temp directories
5. **Document behavior**: Add clear docstrings explaining what each test verifies

## More Information

- See `conftest.py` for provider detection implementation
- See `pytest.ini` for pytest configuration
- See individual test files for examples
