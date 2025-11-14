# ModelChorus - Multi-Model Consensus

Multi-model AI consensus building. Orchestrate responses from multiple AI providers (Claude, Gemini, Codex, Cursor Agent) to get robust, well-reasoned answers.

## Overview

ModelChorus is both a **Python package** for multi-model AI orchestration and a **Claude Code plugin** for seamless consensus building within your development workflow.

**Key Features:**
- **Six Powerful Workflows** - CHAT, CONSENSUS, THINKDEEP, ARGUMENT, IDEATE, and STUDY for different use cases
- **Multi-Provider Support** - Coordinate Claude, Gemini, OpenAI Codex, and Cursor Agent
- **Provider Fallback & Resilience** - Automatic fallback to alternative providers when primary fails
- **Conversation Continuity** - Multi-turn conversations with state persistence
- **Systematic Investigation** - Hypothesis tracking, confidence progression, and dialectical reasoning
- **CLI & Python API** - Use via command-line or programmatically
- **Async Execution** - Parallel provider calls for speed
- **Rich Output** - Beautiful terminal output with detailed results
- **🔒 Read-Only Security Model** - Providers restricted to safe operations (no file modifications)

---

## Security

**ModelChorus operates with a read-only security model.** When workflows invoke external CLI agents, those agents are restricted to safe, read-only operations:

✅ **Allowed Operations:**
- Reading files and searching code
- Web searches and fetching documentation
- Analyzing and generating insights
- Launching sub-agents for research

❌ **Blocked Operations:**
- Writing or modifying files
- Executing shell commands
- Any operations that change system state

This ensures workflows can gather context and generate analysis without risk of unintended modifications. All providers (Claude, Gemini, Codex, Cursor Agent) enforce read-only mode automatically.

**Learn more:** See [docs/SECURITY.md](docs/SECURITY.md) for detailed security architecture and controls.

---

## Core Workflows

ModelChorus provides five powerful workflows for different scenarios:

### CHAT - Simple Conversation

**Single-model conversation with continuity**

Quick consultations, iterative refinement, and building on previous responses.

**CLI Example:**
```bash
# Start conversation
model-chorus chat "What is quantum computing?" -p claude
# Returns: Thread ID: abc-123

# Continue conversation
model-chorus chat "Give me an example" --continue abc-123
```

**Python Example:**
```python
from model_chorus.workflows import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ChatWorkflow(provider, conversation_memory=memory)

# First message
result1 = await workflow.run("What is quantum computing?")
thread_id = result1.metadata.get('thread_id')

# Follow-up
result2 = await workflow.run(
    "How does it differ from classical?",
    continuation_id=thread_id
)
```

**When to use:** Quick Q&A, iterative design, code reviews, learning conversations

---

### THINKDEEP - Systematic Investigation

**Extended reasoning with hypothesis tracking and confidence progression**

Complex debugging, security analysis, and problems requiring methodical investigation.

**CLI Example:**
```bash
# Start investigation
model-chorus thinkdeep "Why is authentication failing?" \
  -f src/auth.py -p claude -e gemini
# Returns: Thread ID: def-456

# Continue investigation
model-chorus thinkdeep "Check async patterns" --continue def-456

# Check progress
model-chorus thinkdeep-status def-456 --steps
```

**Python Example:**
```python
from model_chorus.workflows import ThinkDeepWorkflow
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
expert = GeminiProvider()  # Optional
memory = ConversationMemory()
workflow = ThinkDeepWorkflow(
    provider,
    expert_provider=expert,
    conversation_memory=memory
)

# Start investigation
result1 = await workflow.run(
    "Authentication failing intermittently",
    files=["src/auth.py", "tests/test_auth.py"]
)
thread_id = result1.metadata.get('thread_id')

# Continue with new findings
result2 = await workflow.run(
    "Found race condition in token validation",
    continuation_id=thread_id
)

# Check investigation state
state = workflow.get_investigation_state(thread_id)
print(f"Confidence: {state.current_confidence}")
print(f"Hypotheses: {len(state.hypotheses)}")
```

**Key Features:**
- Hypothesis tracking and evolution
- Confidence progression (exploring → certain)
- Optional expert validation from second model
- File examination tracking
- State persistence across sessions

**When to use:** Complex bugs, security analysis, performance issues, systematic problem-solving

---

### CONSENSUS - Multi-Model Perspectives

**Coordinate multiple AI models for robust answers**

Architecture decisions, technology evaluations, and reducing single-model bias.

**CLI Example:**
```bash
# Get multiple perspectives
model-chorus consensus "REST vs GraphQL for our API?" \
  -p claude -p gemini -p codex -s synthesize
```

**Python Example:**
```python
from model_chorus.workflows import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider

providers = [ClaudeProvider(), GeminiProvider()]
workflow = ConsensusWorkflow(
    providers=providers,
    strategy=ConsensusStrategy.SYNTHESIZE
)

request = GenerationRequest(
    prompt="Explain trade-offs between REST and GraphQL",
    temperature=0.7
)

result = await workflow.execute(request)
print(f"Consensus: {result.consensus_response}")
```

**When to use:** Important decisions, multiple expert perspectives, comparing approaches

---

### ARGUMENT - Dialectical Reasoning

**Structured multi-perspective debate for analyzing claims and proposals**

Policy debates, technology decisions, and balanced argument analysis.

**CLI Example:**
```bash
# Analyze an argument from multiple perspectives
model-chorus argument "Universal healthcare should be implemented" \
  -p claude
```

**Python Example:**
```python
from model_chorus.workflows import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ArgumentWorkflow(provider, conversation_memory=memory)

# Analyze argument with pro, con, and synthesis perspectives
result = await workflow.run("Universal healthcare should be implemented")
thread_id = result.metadata.get('thread_id')
```

**Key Features:**
- Three distinct roles: Creator (pro), Skeptic (con), Moderator (synthesis)
- Balanced analysis avoiding single-perspective bias
- Identifies trade-offs and common ground
- Conversation continuity for iterative refinement

**When to use:** Policy analysis, technology evaluations, balanced decision-making, argument critique

---

### IDEATE - Collaborative Brainstorming

**Multi-perspective idea generation and exploration**

Creative problem-solving, feature brainstorming, and solution exploration.

**CLI Example:**
```bash
# Generate ideas from multiple perspectives
model-chorus ideate "Ways to improve user onboarding" \
  -p claude -p gemini
```

**Python Example:**
```python
from model_chorus.workflows import IdeateWorkflow
from model_chorus.providers import ClaudeProvider, GeminiProvider

providers = [ClaudeProvider(), GeminiProvider()]
workflow = IdeateWorkflow(providers=providers)

# Generate and synthesize ideas from multiple models
result = await workflow.run("Ways to improve user onboarding")
```

**Key Features:**
- Multi-model collaborative brainstorming
- Idea clustering and categorization
- Synthesis of complementary perspectives
- Handles model disagreements constructively

**When to use:** Feature brainstorming, creative problem-solving, exploring alternatives, innovation sessions

---

### STUDY - Persona-Based Collaborative Research

**Multi-persona investigation with role-based orchestration**

Complex research, codebase analysis, and collaborative exploration with specialized personas.

**CLI Example:**
```bash
# Start new investigation
model-chorus study start --scenario "Explore authentication system patterns"

# Continue investigation
model-chorus study start --scenario "Deep dive into OAuth 2.0" --continue thread-id-123

# Include files for context
model-chorus study start --scenario "Analyze this codebase" \
  -f src/auth.py -f tests/test_auth.py

# Use specific personas
model-chorus study start --scenario "Security analysis" \
  --persona SecurityExpert --persona Architect

# Continue existing investigation
model-chorus study next --investigation thread-id-123

# View investigation memory
model-chorus study view --investigation thread-id-123 --show-all
```

**Python Example:**
```python
from model_chorus.workflows import StudyWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = StudyWorkflow(
    provider,
    conversation_memory=memory,
    config={'personas': [
        {'name': 'Researcher', 'role': 'investigator'},
        {'name': 'Architect', 'role': 'investigator'}
    ]}
)

# Start investigation
result = await workflow.run(
    prompt="Explore authentication patterns in codebase",
    files=["src/auth.py", "src/middleware/auth.ts"]
)
thread_id = result.metadata.get('thread_id')

# Continue investigation
result2 = await workflow.run(
    prompt="Analyze OAuth 2.0 implementation",
    continuation_id=thread_id
)
```

**Key Features:**
- Multi-persona collaborative investigation
- Role-based orchestration with specialized expertise
- Conversation continuity across investigation steps
- File context integration
- Memory viewing for investigation review

**When to use:** Complex research, codebase analysis, multi-perspective investigation, collaborative exploration

---

**See [docs/workflows/](docs/workflows/) for detailed workflow guides:**
- [ARGUMENT.md](docs/workflows/ARGUMENT.md) - Dialectical reasoning
- [IDEATE.md](docs/workflows/IDEATE.md) - Collaborative brainstorming

## Installation

### As a Claude Code Plugin

Install directly from Claude Code:

```
/plugin add https://github.com/tylerburleigh/claude-model-chorus
```

Or add locally for development:

```
/plugin add /path/to/claude-model-chorus
```

### Post-Install Setup

After installing the plugin, install the Python package:

```bash
cd ~/.claude/plugins/model-chorus/model_chorus
pip install -e .
```

### As a Python Package (Standalone)

```bash
# From source
git clone https://github.com/tylerburleigh/claude-model-chorus.git
cd claude-model-chorus/model_chorus
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Via CLI

**CHAT - Simple conversation:**
```bash
model-chorus chat "Explain quantum computing" -p claude
# Returns: Thread ID: abc-123

model-chorus chat "Give me an example" --continue abc-123
```

**THINKDEEP - Systematic investigation:**
```bash
model-chorus thinkdeep "Debug authentication issue" \
  -f src/auth.py -p claude -e gemini

model-chorus thinkdeep-status thread-id-here --steps
```

**CONSENSUS - Multi-model perspectives:**
```bash
model-chorus consensus "REST vs GraphQL?" \
  -p claude -p gemini -s synthesize
```

**ARGUMENT - Dialectical reasoning:**
```bash
model-chorus argument "Universal healthcare should be implemented" -p claude
```

**IDEATE - Collaborative brainstorming:**
```bash
model-chorus ideate "Ways to improve user onboarding" -p claude -p gemini
```

**STUDY - Persona-based research:**
```bash
# Start new investigation
model-chorus study start --scenario "Explore authentication patterns"

# Continue investigation
model-chorus study next --investigation thread-id-123

# View investigation memory
model-chorus study view --investigation thread-id-123
```

### Via Python API

**CHAT workflow:**
```python
from model_chorus.workflows import ChatWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ChatWorkflow(provider, conversation_memory=memory)

result = await workflow.run("Explain quantum computing")
thread_id = result.metadata.get('thread_id')

# Continue conversation
result2 = await workflow.run(
    "Give me an example",
    continuation_id=thread_id
)
```

**THINKDEEP workflow:**
```python
from model_chorus.workflows import ThinkDeepWorkflow
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
expert = GeminiProvider()
memory = ConversationMemory()
workflow = ThinkDeepWorkflow(provider, expert_provider=expert, conversation_memory=memory)

result = await workflow.run(
    "Debug authentication issue",
    files=["src/auth.py"]
)
state = workflow.get_investigation_state(result.metadata['thread_id'])
```

**CONSENSUS workflow:**
```python
from model_chorus.workflows import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider, GenerationRequest

providers = [ClaudeProvider(), GeminiProvider()]
workflow = ConsensusWorkflow(providers=providers, strategy=ConsensusStrategy.SYNTHESIZE)

request = GenerationRequest(prompt="REST vs GraphQL trade-offs", temperature=0.7)
result = await workflow.execute(request)
```

**ARGUMENT workflow:**
```python
from model_chorus.workflows import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ArgumentWorkflow(provider, conversation_memory=memory)

result = await workflow.run("Universal healthcare should be implemented")
```

**IDEATE workflow:**
```python
from model_chorus.workflows import IdeateWorkflow
from model_chorus.providers import ClaudeProvider, GeminiProvider

providers = [ClaudeProvider(), GeminiProvider()]
workflow = IdeateWorkflow(providers=providers)

result = await workflow.run("Ways to improve user onboarding")
```

### Via Claude Code Skill

```bash
# CHAT workflow
model-chorus chat "How do I implement JWT auth?" -p claude

# THINKDEEP workflow
model-chorus thinkdeep "Investigate memory leak" -f src/cache.py -p claude -e gemini

# CONSENSUS workflow
model-chorus consensus "TypeScript vs JavaScript?" -p claude -p gemini -s synthesize

# ARGUMENT workflow
model-chorus argument "Universal healthcare should be implemented" -p claude

# IDEATE workflow
model-chorus ideate "Ways to improve user onboarding" -p claude -p gemini
```

## Consensus Strategies

### 1. all_responses (default)
Returns all responses from all providers. Use when you want to see every perspective.

### 2. first_valid
Returns the first successful response. Use for quick answers.

### 3. majority
Returns the most common response. Use when you want agreement.

### 4. weighted
Weights responses by confidence scores. Use to favor higher-confidence answers.

### 5. synthesize
Combines all responses into a comprehensive answer. **Recommended for complex questions.**

## Supported Providers

- **Claude** - Anthropic Claude (via CLI)
- **Gemini** - Google Gemini (via CLI)
- **Codex** - OpenAI Codex (via CLI)
- **Cursor Agent** - Cursor Agent (via CLI)

### Provider Setup

Each provider requires its CLI tool and API key:

**Claude:**
```bash
pip install anthropic-cli
export ANTHROPIC_API_KEY="your-key"
```

**Gemini:**
```bash
pip install google-generativeai
export GOOGLE_API_KEY="your-key"
```

**Codex:**
```bash
pip install openai-cli
export OPENAI_API_KEY="your-key"
```

**Cursor Agent:**
```bash
# Cursor CLI (usually installed with Cursor IDE)
```

## Configuration

ModelChorus uses `.claude/model_chorus_config.yaml` to manage provider availability and workflow defaults.

### Creating Configuration

**Auto-detect available providers (recommended):**
```bash
python -m model_chorus.cli.setup create-claude-config
```

**Manual provider selection:**
```bash
python -m model_chorus.cli.setup create-claude-config \
  --enabled-providers claude gemini --no-auto-detect
```

### Configuration File Structure

```yaml
# .claude/model_chorus_config.yaml
providers:
  claude:
    enabled: true
    default_model: sonnet

  gemini:
    enabled: true
    default_model: gemini-2.5-flash

  codex:
    enabled: false
    default_model: gpt-5-codex

  cursor-agent:
    enabled: false
    default_model: composer-1

workflows:
  chat:
    default_provider: claude
    fallback_providers:
      - gemini

  consensus:
    providers:
      - claude
      - gemini

  thinkdeep:
    default_provider: claude
    fallback_providers:
      - gemini
```

### Enabling/Disabling Providers

Edit `.claude/model_chorus_config.yaml` and set `enabled: true/false`:

```yaml
providers:
  claude:
    enabled: true  # ✓ Enabled
  gemini:
    enabled: false  # ✗ Disabled - won't be used in workflows
```

Disabled providers are automatically skipped by all workflows. This is useful when:
- A provider's API key is unavailable
- You want to reduce API costs
- Testing specific provider combinations
- A provider CLI is not installed

### Provider Fallback & Resilience

ModelChorus workflows automatically fallback to alternative providers if the primary fails. Only **enabled** providers are used in fallback chains:

```yaml
workflows:
  chat:
    default_provider: claude
    fallback_providers:  # Tries in order if primary fails (only if enabled)
      - gemini
      - codex
      - cursor-agent
```

**Example**: If Claude is unavailable, the workflow automatically uses Gemini:

```bash
$ model-chorus chat "quantum computing" --verbose

⚠ Some providers unavailable:
  ✗ claude: CLI command 'claude' not found in PATH

✓ Will use available providers: gemini

[Chat completes successfully using gemini]
```

**Check Provider Availability**:
```bash
# Verify all providers are installed and working
model-chorus list-providers --check

# Output shows status for each provider:
● claude
  Status: ✓ Installed and working
  Provider: Claude
  CLI Command: claude

● gemini
  Status: ✗ Not available
  Issue: CLI command 'gemini' not found in PATH
  Install: npm install -g @google/gemini-cli
```

**Skip Provider Check** (faster startup):
```bash
# Skip availability check for time-critical operations
model-chorus chat "topic" --skip-provider-check
```

## CLI Commands

```bash
# CHAT workflow
model-chorus chat "prompt" [options]

# THINKDEEP workflow
model-chorus thinkdeep "prompt" [options]
model-chorus thinkdeep-status THREAD_ID [options]

# CONSENSUS workflow
model-chorus consensus "prompt" [options]

# ARGUMENT workflow
model-chorus argument "prompt" [options]

# IDEATE workflow
model-chorus ideate "prompt" [options]

# List available providers and models
model-chorus list-providers

# Show version
model-chorus version

# Help
model-chorus --help
model-chorus chat --help
model-chorus thinkdeep --help
model-chorus consensus --help
model-chorus argument --help
model-chorus ideate --help
```

## CLI Options

### CHAT Command
```
model-chorus chat [PROMPT]

Arguments:
  PROMPT                    Your question or message [required]

Options:
  -p, --provider TEXT       Provider to use [default: claude]
  --continue TEXT          Thread ID to continue conversation
  -f, --files TEXT         Files to include (repeatable)
  --system TEXT            System prompt for context
  -t, --temperature FLOAT   Temperature (0.0-1.0) [default: 0.7]
  --max-tokens INTEGER      Maximum tokens to generate
  -o, --output PATH        Save results to JSON file
  -v, --verbose            Show detailed execution info
```

### THINKDEEP Command
```
model-chorus thinkdeep [PROMPT]

Arguments:
  PROMPT                    Investigation question or task [required]

Options:
  -p, --provider TEXT       Primary provider [default: claude]
  -e, --expert TEXT         Expert provider for validation (optional)
  --continue TEXT          Thread ID to continue investigation
  -f, --files TEXT         Files to examine (repeatable)
  --system TEXT            System prompt for context
  -t, --temperature FLOAT   Temperature (0.0-1.0) [default: 0.7]
  --max-tokens INTEGER      Maximum tokens to generate
  --disable-expert         Disable expert validation
  -o, --output PATH        Save results to JSON file
  -v, --verbose            Show detailed execution info

model-chorus thinkdeep-status [THREAD_ID]

Arguments:
  THREAD_ID                Investigation thread to inspect [required]

Options:
  --steps                  Show all investigation steps
  --files                  Show examined files
  -v, --verbose            Show detailed information
```

### CONSENSUS Command
```
model-chorus consensus [PROMPT]

Arguments:
  PROMPT                    Question or task for all models [required]

Options:
  -p, --provider TEXT       Provider to use (repeatable) [default: claude, gemini]
  -s, --strategy TEXT       Consensus strategy [default: all_responses]
  --system TEXT            System prompt for context
  -t, --temperature FLOAT   Temperature (0.0-1.0) [default: 0.7]
  --max-tokens INTEGER      Maximum tokens to generate
  --timeout FLOAT          Timeout per provider (seconds) [default: 120.0]
  -o, --output PATH        Save results to JSON file
  -v, --verbose            Show detailed execution info
```

### ARGUMENT Command
```
model-chorus argument [PROMPT]

Arguments:
  PROMPT                    Claim or argument to analyze [required]

Options:
  -p, --provider TEXT       Provider to use [default: claude]
  --continue TEXT          Thread ID to continue conversation
  -f, --files TEXT         Files to include (repeatable)
  --system TEXT            System prompt for context
  -t, --temperature FLOAT   Temperature (0.0-1.0) [default: 0.7]
  --max-tokens INTEGER      Maximum tokens to generate
  -o, --output PATH        Save results to JSON file
  -v, --verbose            Show detailed execution info
```

### IDEATE Command
```
model-chorus ideate [PROMPT]

Arguments:
  PROMPT                    Topic or problem for brainstorming [required]

Options:
  -p, --provider TEXT       Provider to use (repeatable) [default: claude, gemini]
  -f, --files TEXT         Files to include (repeatable)
  --system TEXT            System prompt for context
  -t, --temperature FLOAT   Temperature (0.0-1.0) [default: 0.7]
  --max-tokens INTEGER      Maximum tokens to generate
  --timeout FLOAT          Timeout per provider (seconds) [default: 120.0]
  -o, --output PATH        Save results to JSON file
  -v, --verbose            Show detailed execution info
```

## Examples

### Example 1: Iterative Code Review (CHAT)

```bash
# Start code review
model-chorus chat "Review this authentication function" \
  -f src/auth.py -p claude
# Returns: Thread ID: abc-123

# Follow-up questions
model-chorus chat "How would you refactor the token validation?" --continue abc-123
model-chorus chat "Add error handling examples" --continue abc-123
```

**Result:** Multi-turn conversation with full context of previous messages.

### Example 2: Complex Bug Investigation (THINKDEEP)

```bash
# Start investigation
model-chorus thinkdeep "Users report intermittent 500 errors" \
  -f src/api/users.py -f logs/error.log \
  -p claude -e gemini
# Returns: Thread ID: def-456

# Continue with findings
model-chorus thinkdeep "Found race condition in async handler" --continue def-456

# Check investigation progress
model-chorus thinkdeep-status def-456 --steps --files
```

**Result:** Systematic investigation with hypothesis tracking and confidence progression.

### Example 3: Architecture Decision (CONSENSUS)

```bash
model-chorus consensus \
  "Should we use REST or GraphQL for our API?" \
  -p claude -p gemini -p codex \
  -s synthesize \
  --output decision.json
```

**Result:** Synthesized recommendation from multiple AI perspectives.

## Output Format

**Terminal:**
```
Executing consensus workflow...
Prompt: What's the best caching strategy?
Providers: 2
Strategy: synthesize

✓ Workflow completed

┌──────────┬─────────┬─────────────────┐
│ Provider │ Status  │ Response Length │
├──────────┼─────────┼─────────────────┤
│ claude   │ ✓ Success│ 450 chars      │
│ gemini   │ ✓ Success│ 523 chars      │
└──────────┴─────────┴─────────────────┘

Consensus Response:
[Combined answer from both models...]
```

**JSON (with --output):**
```json
{
  "prompt": "...",
  "strategy": "synthesize",
  "providers": ["claude", "gemini"],
  "consensus_response": "...",
  "responses": {
    "claude": {"content": "...", "model": "...", "usage": {...}},
    "gemini": {"content": "...", "model": "...", "usage": {...}}
  },
  "failed_providers": [],
  "metadata": {...}
}
```

## Architecture

```
claude-model-chorus/
├── .claude-plugin/          # Plugin configuration
│   ├── plugin.json          # Plugin manifest
│   └── marketplace.json     # Marketplace distribution
├── skills/                  # Skill definitions
│   └── consensus/
│       └── SKILL.md         # Consensus skill documentation
└── model_chorus/            # Python package
    ├── src/model_chorus/
    │   ├── core/           # Base workflow abstractions
    │   ├── providers/      # AI provider implementations
    │   ├── workflows/      # Consensus workflow
    │   ├── cli/            # CLI interface
    │   └── utils/          # Utilities
    ├── tests/              # Test suite
    └── README.md           # Python package docs
```

## Development

### Running Tests

```bash
cd model_chorus/
pytest
```

### Code Quality

```bash
# Format
black .

# Lint
ruff check .

# Type check
mypy model_chorus
```

### Local Plugin Development

```bash
# Install plugin locally
/plugin add /path/to/claude-model-chorus

# Test the skill
Use Skill(model-chorus:consensus): model-chorus consensus "test prompt" -p claude -p gemini
```

## Python Package

For detailed Python API documentation, see [`model_chorus/README.md`](model_chorus/README.md).

The package provides:
- **Workflow abstractions** - Base classes for building workflows
- **Provider system** - Unified interface for AI providers
- **Type-safe models** - Pydantic models for requests/responses
- **Async architecture** - Built with async/await patterns
- **Extensible** - Easy to add new workflows and providers

## Performance

- **Parallel execution**: All providers run concurrently
- **Async I/O**: Non-blocking architecture
- **Configurable timeouts**: Per-provider control
- **Typical latency**: 2-10 seconds (depends on providers and response length)

## Limitations

- Requires provider CLI tools installed
- Requires valid API keys for each provider
- Network connectivity required
- Subject to provider rate limits
- API costs apply per provider call

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Troubleshooting

**Plugin not loading?**
- Check `.claude/plugins/model-chorus/` exists
- Verify `plugin.json` is valid JSON
- Restart Claude Code

**Consensus skill not working?**
- Ensure Python package is installed: `cd ~/.claude/plugins/model-chorus/model_chorus && pip install -e .`
- Check provider CLI tools are installed
- Verify API keys are configured

**Provider failures?**
- Check API key environment variables
- Verify CLI tool is in PATH
- Check network connectivity
- Review provider-specific errors with `--verbose`

## Links

- **GitHub**: https://github.com/tylerburleigh/claude-model-chorus
- **Issues**: https://github.com/tylerburleigh/claude-model-chorus/issues
- **Python Package Docs**: [`model_chorus/README.md`](model_chorus/README.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

ModelChorus leverages the power of multiple AI models to deliver robust, well-reasoned results through consensus building.
