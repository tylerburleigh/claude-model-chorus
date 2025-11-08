# ModelChorus

A flexible orchestration engine for multi-model AI workflows including thinking, debugging, consensus building, code review, and more.

## Overview

ModelChorus provides a unified framework for orchestrating complex multi-model AI workflows. Whether you need deep analytical thinking, systematic debugging, consensus from multiple perspectives, or comprehensive code review, ModelChorus coordinates multiple AI models to deliver robust, well-reasoned results.

## Features

- **Multi-Model Orchestration**: Coordinate multiple AI providers (Anthropic, OpenAI, Google) in a single workflow
- **Flexible Workflow System**: Built-in workflows for common patterns (thinking, debugging, consensus, code review)
- **Plugin Architecture**: Easy-to-extend registry system for custom workflows
- **Type-Safe**: Comprehensive Pydantic models for requests, responses, and configurations
- **Provider Abstraction**: Unified interface across different AI providers
- **Async-First**: Built with modern async/await patterns for optimal performance

## Installation

### From PyPI (when published)

```bash
pip install modelchorus
```

### From Source

```bash
git clone https://github.com/yourusername/modelchorus.git
cd modelchorus
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/modelchorus.git
cd modelchorus
pip install -e ".[dev]"
```

## Quick Start

```python
from modelchorus.core import WorkflowRegistry, WorkflowRequest
from modelchorus.workflows import ThinkDeepWorkflow

# Register your workflow
@WorkflowRegistry.register("thinkdeep")
class ThinkDeepWorkflow(BaseWorkflow):
    async def run(self, prompt: str, **kwargs):
        # Implementation
        pass

# Create a workflow request
request = WorkflowRequest(
    prompt="Analyze the trade-offs between microservices and monolithic architecture",
    models=["claude-3-opus", "gpt-4"],
    config={"thinking_mode": "deep", "steps": 5}
)

# Execute the workflow
workflow_class = WorkflowRegistry.get("thinkdeep")
workflow = workflow_class("Deep Thinking", "Multi-step analytical workflow")
result = await workflow.run(request.prompt, models=request.models, **request.config)

print(f"Result: {result.synthesis}")
```

## Supported Workflows

- **Chat**: Simple single-model peer consultation with conversation threading
- **ThinkDeep**: Multi-step analytical reasoning with progressive refinement
- **Consensus**: Multi-model consensus building through structured debate or voting

### Chat Workflow

The Chat workflow provides straightforward peer consultation with a single AI model, supporting conversation threading for multi-turn interactions. Unlike multi-model workflows, Chat focuses on simplicity and conversational flow.

**Key Features:**
- Single-model interactions (no multi-model overhead)
- Conversation threading via continuation_id
- Automatic conversation history management
- Simple request/response pattern
- Ideal for quick consultations and iterative conversations

**Use Cases:**
- Quick second opinions from an AI model
- Iterative brainstorming and idea refinement
- Simple consultations without orchestration complexity
- Building conversational applications

**Example:**

```python
from modelchorus.providers import ClaudeProvider
from modelchorus.workflows import ChatWorkflow
from modelchorus.core.conversation import ConversationMemory

# Setup
provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ChatWorkflow(provider, conversation_memory=memory)

# First message (creates new conversation)
result1 = await workflow.run("What is quantum computing?")
thread_id = result1.metadata.get('thread_id')
print(result1.synthesis)

# Follow-up message (continues conversation)
result2 = await workflow.run(
    "How does it differ from classical computing?",
    continuation_id=thread_id
)
print(result2.synthesis)

# Check conversation history
thread = workflow.get_thread(thread_id)
print(f"Total messages: {len(thread.messages)}")
```

### Other Workflows

- **ThinkDeep**: Multi-step analytical reasoning with progressive refinement
- **Debug**: Systematic debugging with hypothesis testing and root cause analysis
- **Consensus**: Multi-model consensus building through structured debate or voting
- **CodeReview**: Comprehensive code review covering quality, security, and performance
- **PreCommit**: Pre-commit validation and change impact assessment
- **Planner**: Strategic planning with iterative refinement

## Supported Providers

- **Anthropic** (Claude 3 Opus, Sonnet, Haiku)
- **OpenAI** (GPT-4, GPT-3.5)
- **Google** (Gemini Pro, Gemini Flash)

### CLI Provider Compatibility

ModelChorus supports integration with AI model CLIs. Each CLI has different argument support:

| Provider | CLI Command | Temperature | Max Tokens | System Prompt | Output Format | Notes |
|----------|-------------|-------------|------------|---------------|---------------|-------|
| **Claude** | `claude` | ❌ | ❌ | ✅ | ✅ `--output-format json` | Uses config for model params |
| **Gemini** | `gemini` | ❌ | ❌ | ❌ | ✅ `-o json` | Positional prompt, limited CLI options |
| **Codex** | `codex` | ❌ | ❌ | ✅ | ✅ `--json` | Uses config for model params |
| **Cursor Agent** | `cursor-agent` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Untested (CLI not available) |

**Legend:**
- ✅ Supported via CLI arguments
- ❌ Not supported via CLI (may use config files)
- ⚠️ Unknown/untested

**Important Notes:**
- **Gemini CLI** uses positional arguments for prompts, not `--prompt` flag
- **Gemini CLI** doesn't support temperature, max_tokens, system_prompt, or images via CLI
- **Claude CLI** and **Codex CLI** use configuration files for model parameters
- All providers support JSON output for structured parsing

## Configuration

ModelChorus supports multiple configuration methods to customize default behavior for workflows and providers.

### Configuration Precedence

Configuration is applied in the following order (highest to lowest priority):

1. **CLI Arguments** - Explicitly provided command-line arguments
2. **Workflow-Specific Config** - Settings in `.modelchorusrc` for specific workflows
3. **Global Config** - Global defaults in `.modelchorusrc`
4. **Hardcoded Defaults** - Built-in default values

### Project Configuration File

Create a `.modelchorusrc` file in your project root to set default providers, generation parameters, and workflow-specific settings.

#### Creating a Config File

```bash
# Generate a sample configuration file
modelchorus config init

# Validate your configuration
modelchorus config validate

# View current effective configuration
modelchorus config show
```

#### Config File Format

ModelChorus supports both YAML and JSON formats. Use any of these filenames:
- `.modelchorusrc` (YAML)
- `.modelchorusrc.yaml`
- `.modelchorusrc.yml`
- `.modelchorusrc.json`

#### Example Configuration

```yaml
# Global default provider (applies to all workflows unless overridden)
default_provider: claude

# Global generation parameters
generation:
  temperature: 0.7
  max_tokens: 2000
  timeout: 120.0

# Workflow-specific overrides
workflows:
  chat:
    provider: claude
    temperature: 0.7

  consensus:
    providers:
      - claude
      - gemini
    strategy: synthesize
    temperature: 0.7

  thinkdeep:
    provider: claude
    thinking_mode: medium
    temperature: 0.6

  argument:
    provider: claude
    temperature: 0.8

  ideate:
    providers:
      - claude
      - gemini
    temperature: 0.9
```

#### Configurable Settings

**Global Settings:**
- `default_provider` - Default AI provider (claude, gemini, codex, cursor-agent)
- `generation.temperature` - Default temperature for generation (0.0-2.0)
- `generation.max_tokens` - Default maximum tokens
- `generation.timeout` - Default timeout in seconds
- `generation.system_prompt` - Default system prompt

**Workflow-Specific Settings:**

*All Workflows:*
- `provider` - Single provider for the workflow
- `providers` - Multiple providers (for consensus/ideate)
- `temperature` - Creativity level
- `max_tokens` - Maximum tokens to generate
- `system_prompt` - Custom system prompt

*Consensus Workflow:*
- `strategy` - Consensus strategy (all_responses, synthesize, vote)
- `timeout` - Per-provider timeout

*ThinkDeep Workflow:*
- `thinking_mode` - Reasoning depth (low, medium, high)

#### CLI Commands

```bash
# Show current configuration
modelchorus config show

# Show detailed configuration (includes effective defaults)
modelchorus config show --verbose

# Validate configuration file
modelchorus config validate

# Create sample configuration
modelchorus config init
```

### Environment Variables

```bash
# Provider API Keys
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

### Programmatic Configuration

```python
from modelchorus.providers import AnthropicProvider

provider = AnthropicProvider(
    provider_name="anthropic",
    api_key="your-key-here"
)
```

## Architecture

ModelChorus follows a modular architecture:

- **Core**: Base workflow classes, registry, and data models
- **Workflows**: Specific workflow implementations
- **Providers**: AI provider integrations and abstractions
- **CLI**: Command-line interface for running workflows
- **Utils**: Shared utilities and helpers

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy modelchorus
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Documentation**: https://modelchorus.readthedocs.io
- **PyPI**: https://pypi.org/project/modelchorus
- **GitHub**: https://github.com/yourusername/modelchorus
- **Issues**: https://github.com/yourusername/modelchorus/issues

## Acknowledgments

ModelChorus is inspired by the growing need for sophisticated multi-model AI workflows that leverage the strengths of different models and providers.
