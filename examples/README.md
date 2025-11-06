# ModelChorus Workflow Examples

This directory contains comprehensive examples demonstrating how to use ModelChorus workflows.

## Quick Start

Run all examples:
```bash
python examples/workflow_examples.py
```

Run a specific example:
```bash
python examples/workflow_examples.py argument_basic
python examples/workflow_examples.py ideate_creative
python examples/workflow_examples.py research_sources
```

## Available Examples

### ARGUMENT Workflow (Dialectical Reasoning)

| Example | Description |
|---------|-------------|
| `argument_basic` | Basic argument analysis with Creator/Skeptic/Moderator roles |
| `argument_files` | Analysis with supporting document context |
| `argument_continuation` | Threading conversations for follow-up analysis |
| `argument_config` | Custom configuration (temperature, system prompts) |

**Use cases**: Policy analysis, technical decisions, evaluating proposals

### IDEATE Workflow (Creative Brainstorming)

| Example | Description |
|---------|-------------|
| `ideate_basic` | Simple creative idea generation |
| `ideate_creative` | High-creativity mode (temperature=1.0) |
| `ideate_constraints` | Ideation with specific constraints/criteria |
| `ideate_refine` | Drilling down into specific ideas via continuation |

**Use cases**: Product features, marketing campaigns, problem-solving

### RESEARCH Workflow (Systematic Investigation)

| Example | Description |
|---------|-------------|
| `research_basic` | Basic research with moderate depth |
| `research_sources` | Research with source files and citations |
| `research_depths` | Comparison of shallow/moderate/thorough depths |
| `research_citations` | Different citation styles (informal/academic/technical) |

**Use cases**: Literature review, competitive analysis, fact-checking

### Cross-Workflow Patterns

| Example | Description |
|---------|-------------|
| `error_handling` | Best practices for handling failures |
| `output_management` | Saving/loading workflow results |
| `provider_comparison` | Comparing outputs across different providers |

## Prerequisites

1. **Install ModelChorus**:
   ```bash
   pip install modelchorus
   ```

2. **Configure Provider CLIs**:
   - Install provider CLI tools (claude, gemini, codex)
   - Set API keys in environment variables
   - See [Provider Documentation](../docs/providers.md) for details

3. **Python 3.8+** required

## Example Structure

Each example follows this pattern:

```python
async def example_name():
    # 1. Initialize provider and memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # 2. Create workflow
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # 3. Run workflow
    result = await workflow.run(
        prompt="Your prompt here",
        # ... additional parameters
    )

    # 4. Handle results
    if result.success:
        print(result.synthesis)
    else:
        print(f"Error: {result.error}")
```

## Common Parameters

### All Workflows

- `prompt`: The main query/question/topic (required)
- `temperature`: Control randomness (0.0-1.0, default varies by workflow)
- `max_tokens`: Maximum output length (optional)
- `system_prompt`: Additional context/instructions (optional)
- `continuation_id`: Thread ID for continuing conversations (optional)
- `files`: List of file paths for context (optional)

### ARGUMENT Workflow

No workflow-specific parameters. Uses standard parameters above.

### IDEATE Workflow

- `num_ideas`: Number of ideas to generate (default: 5)
- Higher `temperature` (0.9-1.0) recommended for creativity

### RESEARCH Workflow

- `citation_style`: "informal", "academic", or "technical" (default: "informal")
- `research_depth`: "shallow", "moderate", "thorough", or "comprehensive" (default: "thorough")
- Lower `temperature` (0.4-0.6) recommended for factual accuracy

## Tips for Effective Use

### ARGUMENT Workflow

- Provide clear, specific claims/arguments
- Use file context for data-heavy analyses
- Continue conversations for deeper exploration
- Lower temperature (0.5-0.7) for balanced analysis

### IDEATE Workflow

- Start broad, then refine via continuation
- Use constraints in system_prompt for focused ideation
- High temperature (0.9-1.0) for unconventional ideas
- Request more ideas (10-15) for diverse options

### RESEARCH Workflow

- Ingest source files for citation-based research
- Match depth to complexity (shallow for overviews, comprehensive for deep dives)
- Use academic citations for formal research
- Lower temperature (0.4-0.6) for accuracy

## Troubleshooting

### "Provider not found" error
- Ensure provider CLI tool is installed (e.g., `claude --version`)
- Check API keys are set in environment

### "Import error"
- Verify ModelChorus is installed: `pip install modelchorus`
- Check Python path if running from different directory

### Slow execution
- Research workflow with `comprehensive` depth can be slow
- Use `shallow` or `moderate` depth for faster results
- Consider caching results for repeated queries

## Next Steps

- See [CLI Documentation](../docs/cli.md) for command-line usage
- Check [Workflow API Reference](../docs/workflows.md) for detailed parameters
- Explore [Advanced Patterns](../docs/advanced.md) for complex use cases

## Contributing

To add new examples:
1. Add example function to `workflow_examples.py`
2. Register in `run_specific_example()` mapping
3. Update this README with description
4. Ensure example includes clear comments and error handling
