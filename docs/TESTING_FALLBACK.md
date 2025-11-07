# Provider Fallback Testing Guide

This guide provides manual testing procedures for the provider fallback and availability checking features in ModelChorus.

## Prerequisites

- ModelChorus installed
- At least one provider CLI installed (claude, gemini, codex, or cursor-agent)
- `.modelchorusrc` configured with fallback providers

## Test Scenarios

### Scenario 1: All Providers Available

**Setup**: Ensure all configured providers are installed and working.

```bash
# Verify all providers are available
modelchorus list-providers --check
```

**Test**: Run a research workflow
```bash
modelchorus research "What is quantum computing?" --verbose
```

**Expected Result**:
- Workflow uses primary provider (claude by default)
- No warnings about unavailable providers
- Research completes successfully

---

### Scenario 2: Primary Unavailable, Fallback Available

**Setup**: Temporarily rename claude CLI to simulate unavailability

```bash
# macOS/Linux
which claude  # Note the path
sudo mv /path/to/claude /path/to/claude.bak

# Verify claude is unavailable
modelchorus list-providers --check
```

**Test**: Run research workflow with verbose output
```bash
modelchorus research "What is quantum computing?" --verbose
```

**Expected Result**:
- Warning message: "Some providers unavailable: ['claude']"
- Info message: "Will use available providers: ['gemini', ...]"
- Workflow succeeds using gemini (first fallback)
- Research completes successfully

**Cleanup**:
```bash
sudo mv /path/to/claude.bak /path/to/claude
```

---

### Scenario 3: All Providers Unavailable

**Setup**: Temporarily rename all provider CLIs

```bash
# Rename all CLIs
sudo mv /path/to/claude /path/to/claude.bak
sudo mv /path/to/gemini /path/to/gemini.bak
# ... etc for all providers
```

**Test**: Run research workflow
```bash
modelchorus research "What is quantum computing?"
```

**Expected Result**:
- Clear error message: "No providers available for research:"
- List of unavailable providers with reasons
- Helpful suggestions:
  - "Check installations: modelchorus list-providers --check"
  - "Install missing providers or update .modelchorusrc"
- Non-zero exit code

**Cleanup**:
```bash
sudo mv /path/to/claude.bak /path/to/claude
sudo mv /path/to/gemini.bak /path/to/gemini
# ... etc
```

---

### Scenario 4: Skip Provider Check Flag

**Test**: Run workflow with `--skip-provider-check`
```bash
time modelchorus research "topic" --skip-provider-check --verbose
```

**Expected Result**:
- No provider availability check performed
- Faster startup time (no async provider checks)
- Workflow proceeds directly to execution
- May fail during execution if provider is unavailable

**Comparison**: Run without the flag
```bash
time modelchorus research "topic" --verbose
```

**Expected Result**:
- Provider availability check performed
- Slightly slower startup (minimal overhead)
- Fails fast if no providers available

---

### Scenario 5: Fallback During Execution

**Setup**: Configure a provider that will fail during execution (e.g., invalid API key)

```bash
# Edit .modelchorusrc to use a provider with invalid credentials
# This tests runtime fallback, not availability check
```

**Test**: Run research workflow
```bash
modelchorus research "topic" --verbose
```

**Expected Result**:
- Primary provider fails during execution
- Warning logged: "Providers failed before success: ['claude']"
- Workflow automatically tries first fallback
- Research completes successfully with fallback provider

---

### Scenario 6: Multiple Workflows

Test fallback across different workflows:

**Chat**:
```bash
modelchorus chat "Hello" --verbose
```

**Argument**:
```bash
modelchorus argument "AI is beneficial" --verbose
```

**Ideate**:
```bash
modelchorus ideate "New app features" --verbose
```

**ThinkDeep**:
```bash
modelchorus thinkdeep \
  --step "Investigate problem" \
  --step-number 1 \
  --total-steps 1 \
  --next-step-required false \
  --findings "Initial analysis" \
  --verbose
```

**Expected Result**:
- All workflows support fallback consistently
- Same error messages and behavior
- `--skip-provider-check` works for all workflows

---

### Scenario 7: Configuration Loading

**Test**: Verify fallback providers are loaded from config

```bash
# Check your .modelchorusrc has fallback_providers defined
cat .modelchorusrc

# Run with verbose to see provider initialization
modelchorus research "topic" --verbose
```

**Expected Result**:
- Primary provider initialized
- Fallback providers initialized (shown in verbose output)
- If verbose: "Initializing fallback providers: gemini, codex, cursor-agent"
- If verbose and fallback unavailable: "⚠ Could not initialize fallback X"

---

### Scenario 8: Consensus Workflow (Special Case)

**Setup**: Configure consensus with 3 providers, make 1 unavailable

```bash
# Temporarily rename one CLI
sudo mv /path/to/codex /path/to/codex.bak
```

**Test**: Run consensus workflow
```bash
modelchorus consensus "Best programming language?" --verbose
```

**Expected Result**:
- Consensus continues with remaining 2 providers
- Metadata includes:
  - `providers_succeeded`: ['claude', 'gemini']
  - `providers_failed`: ['codex']
- Consensus completes successfully with partial results

**Cleanup**:
```bash
sudo mv /path/to/codex.bak /path/to/codex
```

---

## Automated Test Commands

Quick test suite for basic functionality:

```bash
# 1. Check provider availability
modelchorus list-providers --check

# 2. Test each workflow with verbose output
modelchorus chat "Test" --verbose
modelchorus research "Test topic" --verbose
modelchorus argument "Test claim" --verbose
modelchorus ideate "Test ideas" --verbose

# 3. Test skip-provider-check flag
modelchorus research "Test" --skip-provider-check --verbose

# 4. Test with output files
modelchorus research "Test" -o /tmp/research-test.json
cat /tmp/research-test.json | jq '.metadata'
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
export MODELCHORUS_LOG_LEVEL=DEBUG
modelchorus research "topic" --verbose
```

### Check Provider CLIs Directly

```bash
# Test each provider CLI individually
claude --version
gemini --version
codex --version
cursor-agent --version
```

### Verify Configuration

```bash
# Check config file
cat ~/.modelchorusrc  # or project .modelchorusrc

# Check which config is being used
modelchorus research "test" --verbose 2>&1 | grep -i "config"
```

### Provider Not Found Issues

```bash
# Check PATH
echo $PATH

# Find provider CLIs
which claude
which gemini
which codex
which cursor-agent

# Reinstall if needed
curl -fsSL https://claude.ai/install.sh | bash
npm install -g @google/gemini-cli
```

---

## Success Criteria

✅ All tests pass with expected results
✅ Error messages are clear and actionable
✅ Fallback works automatically without user intervention
✅ `--skip-provider-check` improves startup time
✅ Workflows fail gracefully when all providers unavailable
✅ Configuration properly loads fallback_providers
✅ Verbose output shows provider selection and failures

---

## Known Limitations

1. **Provider check adds minimal overhead** (~100-500ms depending on number of providers)
   - Use `--skip-provider-check` for time-critical operations

2. **Fallback only happens on provider errors**
   - Model errors (invalid prompts, content policy) don't trigger fallback
   - Authentication errors trigger fallback
   - CLI not found triggers fallback

3. **Consensus workflow doesn't use fallback_providers**
   - Consensus has its own multi-provider configuration
   - Already handles partial failures gracefully

---

## Reporting Issues

If you encounter issues during testing:

1. Collect logs with verbose output:
   ```bash
   modelchorus research "topic" --verbose 2>&1 | tee debug.log
   ```

2. Check provider availability:
   ```bash
   modelchorus list-providers --check 2>&1 | tee providers.log
   ```

3. Include in bug report:
   - ModelChorus version
   - Provider versions
   - Operating system
   - Full error message
   - Steps to reproduce
