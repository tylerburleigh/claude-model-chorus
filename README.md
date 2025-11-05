# ModelChorus - Multi-Model Consensus

Multi-model AI consensus building for Claude Code. Orchestrate responses from multiple AI providers (Claude, Gemini, Codex, Cursor Agent) to get robust, well-reasoned answers.

## Overview

ModelChorus is both a **Python package** for multi-model AI orchestration and a **Claude Code plugin** for seamless consensus building within your development workflow.

**Key Features:**
- **Multi-Provider Support** - Coordinate Claude, Gemini, OpenAI Codex, and Cursor Agent
- **Flexible Consensus Strategies** - Choose from 5 strategies: all_responses, first_valid, majority, weighted, synthesize
- **CLI & Python API** - Use via command-line or programmatically
- **Async Execution** - Parallel provider calls for speed
- **Rich Output** - Beautiful terminal output with detailed results

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
cd ~/.claude/plugins/model-chorus/modelchorus
pip install -e .
```

### As a Python Package (Standalone)

```bash
# From source
git clone https://github.com/tylerburleigh/claude-model-chorus.git
cd claude-model-chorus/modelchorus
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Via CLI

```bash
# Basic consensus with 2 providers
modelchorus consensus "Explain quantum computing" \
  --provider claude \
  --provider gemini

# With synthesis strategy
modelchorus consensus "What's the best caching strategy?" \
  -p claude -p gemini -p codex \
  -s synthesize \
  --verbose

# Save results to JSON
modelchorus consensus "Design a microservices architecture" \
  -p claude -p gemini \
  -s synthesize \
  --output results.json
```

### Via Python API

```python
import asyncio
from modelchorus.workflows import ConsensusWorkflow, ConsensusStrategy
from modelchorus.providers import ClaudeProvider, GeminiProvider, GenerationRequest

async def main():
    # Create providers
    providers = [ClaudeProvider(), GeminiProvider()]

    # Create workflow
    workflow = ConsensusWorkflow(
        providers=providers,
        strategy=ConsensusStrategy.SYNTHESIZE,
        default_timeout=120.0
    )

    # Create request
    request = GenerationRequest(
        prompt="Explain the trade-offs between REST and GraphQL",
        system_prompt="You are a senior software architect",
        temperature=0.7
    )

    # Execute
    result = await workflow.execute(request, strategy=ConsensusStrategy.SYNTHESIZE)

    print(f"Consensus: {result.consensus_response}")
    print(f"Responses from {len(result.provider_results)} providers")

asyncio.run(main())
```

### Via Claude Code Skill

```
Use Skill(model-chorus:consensus) by running:

modelchorus consensus "Should we use TypeScript or JavaScript?" \
  --provider claude \
  --provider gemini \
  --strategy synthesize
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

## CLI Commands

```bash
# Run consensus
modelchorus consensus "prompt" [options]

# List available providers and models
modelchorus list-providers

# Show version
modelchorus version

# Help
modelchorus --help
modelchorus consensus --help
```

## CLI Options

```
modelchorus consensus [PROMPT]

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
  --help                   Show help message
```

## Examples

### Example 1: Technical Decision

```bash
modelchorus consensus \
  "Should we use REST or GraphQL for our API?" \
  -p claude -p gemini -p codex \
  -s all_responses \
  --output decision.json
```

Result: See all three perspectives, save for team review.

### Example 2: Architecture Design

```bash
modelchorus consensus \
  "Design a caching strategy for 100k concurrent users" \
  -p claude -p gemini \
  -s synthesize \
  --system "You are a senior distributed systems architect" \
  --verbose
```

Result: Synthesized design incorporating both models' expertise.

### Example 3: Quick Answer

```bash
modelchorus consensus \
  "How do I reverse a list in Python?" \
  -p claude -p codex \
  -s first_valid
```

Result: Fast answer from whichever provider responds first.

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
└── modelchorus/            # Python package
    ├── src/modelchorus/
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
cd modelchorus/
pytest
```

### Code Quality

```bash
# Format
black .

# Lint
ruff check .

# Type check
mypy modelchorus
```

### Local Plugin Development

```bash
# Install plugin locally
/plugin add /path/to/claude-model-chorus

# Test the skill
Use Skill(model-chorus:consensus): modelchorus consensus "test prompt" -p claude -p gemini
```

## Python Package

For detailed Python API documentation, see [`modelchorus/README.md`](modelchorus/README.md).

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
- Ensure Python package is installed: `cd ~/.claude/plugins/model-chorus/modelchorus && pip install -e .`
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
- **Python Package Docs**: [`modelchorus/README.md`](modelchorus/README.md)
- **Skill Documentation**: [`skills/consensus/SKILL.md`](skills/consensus/SKILL.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

ModelChorus leverages the power of multiple AI models to deliver robust, well-reasoned results through consensus building.
