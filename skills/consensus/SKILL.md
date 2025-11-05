---
name: model-chorus:consensus
description: Multi-model consensus building by orchestrating multiple AI providers through CLI-based execution
---

# ModelChorus: Consensus

Build consensus across multiple AI models by coordinating responses from different providers (Claude, Gemini, Codex, Cursor Agent) using the ModelChorus CLI.

## When to Use

Use this skill when you need:
- **Multiple perspectives** - Get opinions from different AI models
- **Consensus building** - Synthesize insights from multiple models
- **Model comparison** - See how different models approach the same problem
- **Robust answers** - Reduce single-model bias through multi-model consultation
- **Parallel execution** - Run multiple models concurrently for speed

## How It Works

The consensus skill runs the `modelchorus consensus` CLI command, which:

1. **Sends prompt to multiple providers** - Executes in parallel
2. **Collects all responses** - Gathers results from each model
3. **Applies consensus strategy** - Processes responses based on strategy
4. **Returns synthesized result** - Provides combined output

**Supported Providers:**
- `claude` - Anthropic Claude (via CLI)
- `gemini` - Google Gemini (via CLI)
- `codex` - OpenAI Codex (via CLI)
- `cursor-agent` - Cursor Agent (via CLI)

## Usage

To use this skill, invoke the `modelchorus consensus` command:

```bash
modelchorus consensus "Your prompt here" \
  --provider claude \
  --provider gemini \
  --strategy synthesize \
  --verbose
```

## Parameters

### Required
- **prompt** (positional argument): The question or task to send to all models

### Optional
- `--provider, -p`: Provider to use (can specify multiple times)
  - Default: `["claude", "gemini"]`
  - Options: `claude`, `gemini`, `codex`, `cursor-agent`

- `--strategy, -s`: Consensus strategy to use
  - Default: `all_responses`
  - Options:
    - `all_responses` - Return all model responses
    - `first_valid` - Return first successful response
    - `majority` - Return most common response
    - `weighted` - Weight responses by model confidence
    - `synthesize` - Combine all responses into synthesis

- `--system`: System prompt for context (optional)

- `--temperature, -t`: Temperature for generation (0.0-1.0)
  - Default: `0.7`

- `--max-tokens`: Maximum tokens to generate per model

- `--timeout`: Timeout per provider in seconds
  - Default: `120.0`

- `--output, -o`: Save results to JSON file

- `--verbose, -v`: Show detailed execution information

## Consensus Strategies

### 1. all_responses
Returns all responses from all providers.

**Use when:** You want to see every model's perspective

```bash
modelchorus consensus "Explain quantum computing" \
  -p claude -p gemini -p codex \
  -s all_responses
```

### 2. first_valid
Returns the first successful response.

**Use when:** You want a quick answer, don't care which model provides it

```bash
modelchorus consensus "What is 2+2?" \
  -p claude -p gemini \
  -s first_valid
```

### 3. majority
Returns the most common response (by similarity).

**Use when:** You want the answer most models agree on

```bash
modelchorus consensus "Is Python better than JavaScript?" \
  -p claude -p gemini -p codex \
  -s majority
```

### 4. weighted
Weights responses by model confidence scores.

**Use when:** You want to favor higher-confidence answers

```bash
modelchorus consensus "Estimate project timeline" \
  -p claude -p gemini \
  -s weighted
```

### 5. synthesize
Combines all responses into a synthesized answer.

**Use when:** You want a comprehensive answer incorporating all perspectives

```bash
modelchorus consensus "Design a caching strategy" \
  -p claude -p gemini -p codex \
  -s synthesize
```

## Examples

### Example 1: Technical Question with Synthesis

```bash
modelchorus consensus \
  "What's the best way to handle authentication in a microservices architecture?" \
  --provider claude \
  --provider gemini \
  --strategy synthesize \
  --system "You are an expert in distributed systems" \
  --verbose
```

**What happens:**
1. Sends question to Claude and Gemini
2. Both models respond with their perspective
3. Responses are synthesized into comprehensive answer
4. Shows execution details (--verbose)

### Example 2: Quick Answer

```bash
modelchorus consensus \
  "How do I reverse a string in Python?" \
  -p claude -p codex \
  -s first_valid
```

**What happens:**
1. Sends to Claude and Codex
2. Returns first successful response
3. Fast result, no synthesis needed

### Example 3: Multiple Perspectives

```bash
modelchorus consensus \
  "Should we use REST or GraphQL for our new API?" \
  -p claude -p gemini -p codex \
  -s all_responses \
  --temperature 0.8 \
  --output api-decision.json
```

**What happens:**
1. All 3 models give their perspective
2. Results saved to api-decision.json
3. Higher temperature (0.8) for more creative responses
4. You can compare all opinions

### Example 4: With System Prompt

```bash
modelchorus consensus \
  "Estimate the complexity of implementing real-time notifications" \
  -p claude -p gemini \
  -s synthesize \
  --system "You are a senior software architect with 15 years experience. Focus on practical considerations and trade-offs." \
  --max-tokens 1000
```

**What happens:**
1. System prompt sets expertise level
2. Both models answer with that context
3. Responses synthesized
4. Limited to 1000 tokens per response

## Output Format

**Terminal Output:**
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
[Synthesized answer combining both perspectives...]
```

**JSON Output (with --output):**
```json
{
  "prompt": "What's the best caching strategy?",
  "strategy": "synthesize",
  "providers": ["claude", "gemini"],
  "consensus_response": "...",
  "responses": {
    "claude": {
      "content": "...",
      "model": "claude-3-opus",
      "usage": {...},
      "stop_reason": "end_turn"
    },
    "gemini": {
      "content": "...",
      "model": "gemini-pro",
      "usage": {...},
      "stop_reason": "stop"
    }
  },
  "failed_providers": [],
  "metadata": {...}
}
```

## Provider Configuration

Providers require CLI tools to be installed and configured:

**Claude:**
```bash
# Install Anthropic CLI
pip install anthropic-cli

# Configure API key
export ANTHROPIC_API_KEY="your-key"
```

**Gemini:**
```bash
# Install Google CLI
pip install google-generativeai

# Configure API key
export GOOGLE_API_KEY="your-key"
```

**Codex (OpenAI):**
```bash
# Install OpenAI CLI
pip install openai-cli

# Configure API key
export OPENAI_API_KEY="your-key"
```

**Cursor Agent:**
```bash
# Cursor CLI must be installed
# Usually available if Cursor IDE is installed
```

## Best Practices

**DO:**
- ✅ Use multiple providers for important decisions
- ✅ Choose appropriate consensus strategy
- ✅ Save results with `--output` for later reference
- ✅ Use `--verbose` to understand what's happening
- ✅ Adjust `--temperature` based on task (lower for factual, higher for creative)

**DON'T:**
- ❌ Use too many providers (2-3 is usually enough)
- ❌ Expect identical responses (models differ)
- ❌ Use `synthesize` for simple factual questions (overkill)
- ❌ Forget to configure provider API keys

## Error Handling

**If a provider fails:**
- Other providers continue execution
- Failed providers listed in `failed_providers`
- Consensus built from successful responses only
- Exit code 1 if ALL providers fail

**Common issues:**
- **Provider not found**: CLI tool not installed or not in PATH
- **API key missing**: Set environment variable for provider
- **Timeout**: Increase `--timeout` value
- **Rate limit**: Reduce concurrent providers or add delays

## Integration with ModelChorus

This skill uses the ModelChorus Python package and CLI:

**CLI Command:**
```bash
modelchorus consensus [options]
```

**Python API (advanced):**
```python
from modelchorus.workflows import ConsensusWorkflow
from modelchorus.providers import ClaudeProvider, GeminiProvider

providers = [ClaudeProvider(), GeminiProvider()]
workflow = ConsensusWorkflow(providers, strategy=ConsensusStrategy.SYNTHESIZE)
result = await workflow.execute(request)
```

## Advanced Usage

### Programmatic Execution

You can call the consensus workflow from Python code:

```python
import asyncio
from modelchorus.workflows import ConsensusWorkflow, ConsensusStrategy
from modelchorus.providers import ClaudeProvider, GeminiProvider, GenerationRequest

async def main():
    providers = [ClaudeProvider(), GeminiProvider()]
    workflow = ConsensusWorkflow(
        providers=providers,
        strategy=ConsensusStrategy.SYNTHESIZE,
        default_timeout=120.0
    )

    request = GenerationRequest(
        prompt="Explain quantum computing",
        system_prompt="You are a physics professor",
        temperature=0.7
    )

    result = await workflow.execute(request, strategy=ConsensusStrategy.SYNTHESIZE)
    print(f"Consensus: {result.consensus_response}")

asyncio.run(main())
```

### Custom Providers

List available providers:

```bash
modelchorus list-providers
```

Output shows all configured providers and their available models.

## Performance

- **Parallel execution**: All providers run concurrently
- **Async architecture**: Non-blocking I/O
- **Timeout handling**: Configurable per-provider timeout
- **Typical latency**: 2-10 seconds depending on providers and response length

## Limitations

- Requires provider CLI tools installed
- Requires valid API keys for each provider
- Network connectivity required
- Rate limits apply per provider
- Cost: Each provider call incurs API costs

## See Also

- **ModelChorus CLI**: Run `modelchorus --help` for all options
- **Provider Docs**: Check each provider's CLI documentation
- **Python Package**: See `modelchorus/README.md` for API details
