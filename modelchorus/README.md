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

## Configuration

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
