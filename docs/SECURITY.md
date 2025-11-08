# Security Model

## Overview

ModelChorus implements a **read-only security model** when invoking external CLI agents (Claude CLI, Gemini CLI, Codex CLI, Cursor Agent). This ensures that workflows can gather information and generate insights without modifying system state or files.

## Read-Only Architecture

### Design Principle

When ModelChorus invokes external CLI agents as providers, those agents operate in a restricted mode that:

1. **Allows information gathering** - Read operations, searches, and web queries
2. **Blocks write operations** - File modifications, shell commands, and state changes
3. **Provides safe orchestration** - Multiple models can collaborate without risk of unintended modifications

### Rationale

Multi-model workflows should focus on reasoning, analysis, and synthesis rather than direct system modifications. By restricting providers to read-only operations:

- **Safety**: No unintended file modifications or command executions
- **Predictability**: Workflows produce insights, not side effects
- **Auditability**: All operations are inspectable and traceable
- **Composability**: Multiple workflows can run concurrently without conflicts

## Provider-Specific Security Controls

### Claude CLI Provider

**Security Mechanism**: Tool whitelisting and blacklisting via CLI flags

**Implementation**:
```bash
claude --print --output-format json \
  --allowedTools "Read,Grep,Glob,WebSearch,WebFetch,Task" \
  --disallowedTools "Write,Edit,Bash" \
  "your prompt"
```

**Allowed Tools** (Read-Only Operations):
- `Read` - Read file contents
- `Grep` - Search file contents with regex
- `Glob` - Find files by pattern
- `WebSearch` - Search the web for information
- `WebFetch` - Fetch web content
- `Task` - Launch sub-agents for exploration/research

**Blocked Tools** (Write Operations):
- `Write` - Create or overwrite files
- `Edit` - Modify existing files
- `Bash` - Execute shell commands

**Documentation**: See `model_chorus/src/model_chorus/providers/claude_provider.py:116-161`

---

### Codex CLI Provider

**Security Mechanism**: Sandbox modes with filesystem access controls

**Implementation**:
```bash
codex exec --json \
  --sandbox read-only \
  --ask-for-approval never \
  "your prompt"
```

**Security Features**:
- `--sandbox read-only` - Enforces read-only filesystem access
- `--ask-for-approval never` - Disables approval prompts for automated operation

**Capabilities**:
- ✅ Read files and directories
- ✅ Search and inspect code
- ✅ Analyze project structure
- ❌ Write or modify files
- ❌ Execute commands that modify state

**Documentation**: See `model_chorus/src/model_chorus/providers/codex_provider.py:114-155`

---

### Gemini CLI Provider

**Security Mechanism**: Non-interactive mode with default read-only tools

**Implementation**:
```bash
gemini "your prompt" -m pro --output-format json
```

**Security Features**:
- Non-interactive mode defaults to read-only tools automatically
- No `--yolo` flag (which would enable all tools without confirmation)

**Capabilities**:
- ✅ Read-only operations (file reading, searching)
- ✅ Information retrieval
- ❌ File modifications (without `--yolo` flag)

**Documentation**: See `model_chorus/src/model_chorus/providers/gemini_provider.py:118-158`

---

### Cursor Agent Provider

**Security Mechanism**: Propose-only mode without force flag

**Implementation**:
```bash
cursor-agent -p --json "your prompt"
```

**Security Features**:
- Print mode (`-p`) enables non-interactive scripting
- No `--force` flag means changes are only proposed, not applied

**Capabilities**:
- ✅ Code analysis and suggestions
- ✅ Read-only file inspection
- ✅ Propose changes (without applying them)
- ❌ Direct file modifications (without `--force`)

**Documentation**: See `model_chorus/src/model_chorus/providers/cursor_agent_provider.py:114-160`

---

## Available Operations

### Safe Operations (Allowed)

| Operation | Description | Use Cases |
|-----------|-------------|-----------|
| **Read** | Read file contents | Code review, analysis, context gathering |
| **Grep** | Search with regex | Finding patterns, code discovery |
| **Glob** | File pattern matching | Locating files by type or pattern |
| **WebSearch** | Search the web | Information retrieval, research |
| **WebFetch** | Fetch web content | Documentation lookup, API reference |
| **Task** | Launch sub-agents | Exploration, multi-step research |

### Blocked Operations (Security Risk)

| Operation | Risk | Blocked In |
|-----------|------|------------|
| **Write** | File creation/overwriting | All providers |
| **Edit** | File modification | All providers |
| **Bash** | Shell command execution | All providers |
| **File operations** | mv, cp, rm, mkdir, etc. | All providers |

---

## Security Testing

### Test Coverage

Read-only mode enforcement is verified through unit tests:

1. **Claude Provider Tests** (`model_chorus/tests/test_claude_provider.py:168-203`)
   - `test_read_only_mode_allowed_tools` - Verifies allowed tools flag
   - `test_read_only_mode_disallowed_tools` - Verifies blocked tools flag

2. **Codex Provider Tests** (`model_chorus/tests/test_codex_provider.py:182-210`)
   - `test_read_only_sandbox_mode` - Verifies read-only sandbox flag
   - `test_non_interactive_approval_mode` - Verifies approval bypass

### Running Security Tests

```bash
cd model_chorus
pytest tests/test_claude_provider.py::TestClaudeProvider::test_read_only_mode_allowed_tools
pytest tests/test_claude_provider.py::TestClaudeProvider::test_read_only_mode_disallowed_tools
pytest tests/test_codex_provider.py::TestCodexProvider::test_read_only_sandbox_mode
pytest tests/test_codex_provider.py::TestCodexProvider::test_non_interactive_approval_mode
```

---

## Known Limitations and Mitigations

### Claude CLI

**Issue**: Non-interactive mode may still request permissions (Bug #581)
**Mitigation**: Explicit `--allowedTools` and `--disallowedTools` flags override permission prompts

**Impact**: Minimal - flags provide defense in depth

---

### Codex CLI

**Issue**: May ignore read-only mode when using MCP edit tools (Bug #4152)
**Mitigation**: MCP tools are not used in ModelChorus workflows
**Impact**: Low - affected functionality is not utilized

---

### Gemini CLI

**Limitation**: No granular tool control - binary choice between read-only and `--yolo`
**Mitigation**: Never use `--yolo` flag, rely on default read-only tools
**Impact**: None - default behavior is sufficient

---

### Cursor Agent

**Limitation**: MCP tools in headless mode require interactive workspace trust
**Mitigation**: Standard tools work correctly; MCP tools not required
**Impact**: Low - core functionality unaffected

---

## Security Recommendations

### For Users

1. **Trust ModelChorus workflows** - Providers cannot modify your files
2. **Run workflows freely** - No risk of unintended side effects
3. **Review outputs** - Workflows produce analysis and insights you can act on
4. **Keep CLI tools updated** - Security fixes and improvements

### For Developers

1. **Never add write-enabling flags** - Don't use `--yolo`, `--force`, or remove sandbox modes
2. **Test read-only enforcement** - Add tests for any new providers
3. **Document security changes** - Update this file when modifying provider security
4. **Principle of least privilege** - Only enable capabilities required for the use case

---

## Threat Model

### In Scope

- Accidental file modifications by workflows
- Unintended command executions via shell tools
- Filesystem corruption from concurrent operations
- State changes that affect reproducibility

### Out of Scope

- Malicious CLI tool implementations (trust the tool vendors)
- Network-based attacks (web fetching is allowed and necessary)
- Information disclosure via read operations (read access is intentional)
- Resource exhaustion (handled by CLI tool rate limits and timeouts)

---

## Compliance and Auditing

### Verification

To verify read-only mode is active across all providers:

```bash
# Run all security-related tests
pytest model_chorus/tests/test_*_provider.py -k "read_only or sandbox or approval"

# Check provider implementations
grep -r "allowedTools\|disallowedTools" model_chorus/src/model_chorus/providers/
grep -r "sandbox.*read-only" model_chorus/src/model_chorus/providers/
grep -r "yolo\|force" model_chorus/src/model_chorus/providers/
```

### Audit Log

All provider invocations are logged with full command details:

```python
logger.debug(f"Built {provider} command: {' '.join(command)}")
```

Check logs at `DEBUG` level to audit all CLI invocations.

---

## Future Enhancements

### Planned Features

1. **Configurable security levels** - Allow users to opt into write modes for specific use cases
2. **Per-request security controls** - Override read-only mode on a per-request basis
3. **Security policy files** - Define allowed operations via configuration
4. **Sandboxed execution environments** - Run providers in containerized environments
5. **Permission auditing** - Track and log all security-relevant operations

### Non-Goals

- **Full isolation** - Providers can still read arbitrary files (intentional for context gathering)
- **Network restrictions** - Web access is necessary for research workflows
- **Resource quotas** - Handled by underlying CLI tools and OS

---

## Security Contact

For security concerns or to report vulnerabilities:

1. **GitHub Issues** - For general security questions
2. **Private disclosure** - For vulnerabilities, contact repository maintainers directly

---

## Change Log

### 2025-11-08 - Initial Security Model

- Implemented read-only mode across all CLI providers
- Added tool restrictions (Claude), sandbox modes (Codex), and safety defaults (Gemini, Cursor Agent)
- Created test coverage for security controls
- Documented security architecture and threat model
