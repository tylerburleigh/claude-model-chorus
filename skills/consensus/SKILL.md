---
name: consensus
description: Multi-model consultation with parallel execution and configurable synthesis strategies
---

# CONSENSUS

## Overview

The CONSENSUS workflow enables querying multiple AI models simultaneously and synthesizing their responses for improved accuracy, diverse perspectives, and reliable decision-making. This workflow executes all providers in parallel and applies configurable strategies to combine or select from their outputs.

**Key Capabilities:**
- Parallel multi-model execution for fast responses
- Five consensus strategies for different use cases (all_responses, synthesize, majority, weighted, first_valid)
- Per-provider timeout and error handling
- Response weighting and prioritization
- Graceful degradation when providers fail

**Use Cases:**
- Cross-validation of facts or recommendations across multiple AI models
- Gathering diverse perspectives on complex decisions or architectural choices
- Improving answer reliability through multi-model consensus
- Comparing provider strengths and response quality
- Building robust systems that don't depend on a single model

## When to Use

Use the CONSENSUS workflow when you need to:

- **Multiple perspectives** - Get different viewpoints from various AI models to make better-informed decisions
- **Fact verification** - Cross-check facts, calculations, or recommendations across models for accuracy
- **Critical decisions** - Use multi-model consensus for high-stakes decisions where a single model's view isn't sufficient
- **Provider comparison** - Evaluate and compare how different models approach the same problem
- **Reliability improvement** - Increase confidence in answers through multi-model agreement or synthesis

## When NOT to Use

Avoid the CONSENSUS workflow when:

| Situation | Use Instead |
|-----------|-------------|
| You need a simple conversational interaction | **CHAT** - Single-model conversation with threading |
| You need deep investigation with hypothesis tracking | **THINKDEEP** - Systematic investigation with confidence progression |
| You need structured debate analysis | **ARGUMENT** - Three-role dialectical creator/skeptic/moderator |
| You need creative brainstorming | **IDEATE** - Structured idea generation |
| You need systematic research with citations | **RESEARCH** - Comprehensive information gathering |

## Consensus Strategies

CONSENSUS provides five strategies for combining or selecting model responses. Choose the strategy based on your specific needs:

### 1. all_responses (Default)

**What it does:** Returns all model responses separately with clear provider labels, separated by dividers.

**When to use:**
- You want to see each model's complete response independently
- You need to manually compare and evaluate different perspectives
- You want full visibility into how each model approached the question
- You're evaluating provider performance or quality

**Output format:**
```
## CLAUDE

[Claude's complete response]

---

## GEMINI

[Gemini's complete response]

---

## CODEX

[Codex's complete response]
```

**Example use case:** Comparing code review feedback from different models to understand diverse approaches.

---

### 2. synthesize

**What it does:** Combines all responses into a synthesized view with numbered perspectives from each model.

**When to use:**
- You want structured presentation of all perspectives
- You need to see how different models complement each other
- You want organized comparison of viewpoints
- You're building documentation from multiple sources

**Output format:**
```
# Synthesized Response from Multiple Models

## Perspective 1: claude

[Claude's response]

## Perspective 2: gemini

[Gemini's response]

## Perspective 3: codex

[Codex's response]
```

**Example use case:** Gathering architectural recommendations where each model provides complementary insights.

---

### 3. majority

**What it does:** Returns the most common response among all models (basic string matching).

**When to use:**
- You want the answer that most models agree on
- You need consensus-driven results for factual queries
- You're looking for the most widely agreed-upon perspective
- You want to filter out outlier responses

**Output format:** Single response string (the most common one)

**Example use case:** Verifying a factual answer or calculation where multiple models should converge on the same result.

**Note:** Current implementation uses simple string matching. For complex responses, consider using `all_responses` or `synthesize` instead.

---

### 4. weighted

**What it does:** Returns a weighted response, currently using response length as a heuristic (longer = more detailed).

**When to use:**
- You want the most comprehensive response
- You prefer detailed explanations over brief answers
- You're willing to accept a single "best" response based on heuristics
- You trust that length correlates with quality for your use case

**Output format:** Single response string (the longest/most detailed one)

**Example use case:** Technical explanations where detail and thoroughness are valued.

**Note:** Current implementation uses length as a proxy for quality. Future versions may incorporate provider-specific weights.

---

### 5. first_valid

**What it does:** Returns the first successful response from any provider, ignoring subsequent responses.

**When to use:**
- You want the fastest possible response time
- All providers are roughly equivalent for your task
- You need basic fallback behavior
- Response time matters more than consensus

**Output format:** Single response string (first successful one)

**Example use case:** Quick queries where any provider's answer is sufficient, but you want automatic failover if the primary provider fails.

---

## Strategy Selection Guide

**Decision tree:**

```
Need to see all perspectives independently? � all_responses
Need organized, numbered perspectives? � synthesize
Need consensus answer (factual query)? � majority
Need most detailed/comprehensive response? � weighted
Need fastest response with fallback? � first_valid
```

**By use case:**

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| Fact-checking | `majority` or `synthesize` | Verify agreement across models |
| Architecture decisions | `all_responses` or `synthesize` | Compare different approaches |
| Code review | `all_responses` | See diverse feedback independently |
| Quick queries | `first_valid` | Speed with failover |
| Technical explanations | `weighted` or `synthesize` | Get comprehensive details |
| Provider comparison | `all_responses` | Evaluate each model separately |

## Basic Usage

### Simple Example

```bash
modelchorus consensus "What is quantum computing?"
```

**Expected Output:**
The command executes on default providers (Claude and Gemini) in parallel using the `all_responses` strategy. Returns responses from both models with clear provider labels.

**Note:** CONSENSUS does NOT support conversation threading (`--continue`). Each invocation is stateless.

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `["claude", "gemini"]` | AI providers to use (repeatable for multiple) |
| `--strategy` | `-s` | `all_responses` | Consensus strategy (`all_responses`, `synthesize`, `majority`, `weighted`, `first_valid`) |
| `--file` | `-f` | None | File paths for context (repeatable) |
| `--system` | | None | Additional system prompt |
| `--temperature` | `-t` | `0.7` | Creativity level (0.0-1.0) |
| `--max-tokens` | | None | Maximum response length |
| `--timeout` | | `120.0` | Timeout per provider in seconds |
| `--output` | `-o` | None | Save result to JSON file |
| `--verbose` | `-v` | False | Show detailed execution info |

**Important:** CONSENSUS does NOT support `--continue` (no conversation threading).

## Technical Contract

### Parameters

**Required:**
- `prompt` (string): The question or statement to send to all selected AI models

**Optional:**
- `--provider, -p` (string, repeatable): AI providers to query - Valid values: `claude`, `gemini`, `codex`, `cursor-agent` - Default: `["claude", "gemini"]` - Can be specified multiple times to query additional providers
- `--strategy, -s` (string): Consensus synthesis strategy - Valid values: `all_responses`, `synthesize`, `majority`, `weighted`, `first_valid` - Default: `all_responses`
- `--file, -f` (string, repeatable): File paths to include as context for all models - Can be specified multiple times - Files must exist before execution
- `--system` (string): Additional system prompt to customize model behavior across all providers
- `--temperature, -t` (float): Response creativity level for all models - Range: 0.0-1.0 - Default: 0.7 - Lower values are more deterministic
- `--max-tokens` (integer): Maximum response length in tokens per provider - Provider-specific limits apply
- `--timeout` (float): Timeout per provider in seconds - Default: 120.0 - Prevents hanging on slow providers
- `--output, -o` (string): Path to save JSON output file - Creates or overwrites file at specified path
- `--verbose, -v` (boolean): Enable detailed execution information - Default: false - Shows per-provider timing and status

### Return Format

The CONSENSUS workflow returns a JSON object with the following structure:

```json
{
  "result": "Synthesized response text based on selected strategy...",
  "session_id": null,
  "metadata": {
    "strategy": "all_responses",
    "providers_queried": ["claude", "gemini"],
    "providers_succeeded": ["claude", "gemini"],
    "providers_failed": [],
    "execution_time_seconds": 3.45,
    "temperature": 0.7,
    "timestamp": "2025-11-07T10:30:00Z",
    "provider_details": {
      "claude": {
        "model": "claude-3-5-sonnet-20241022",
        "prompt_tokens": 150,
        "completion_tokens": 300,
        "total_tokens": 450,
        "response_time_seconds": 2.1
      },
      "gemini": {
        "model": "gemini-2.5-pro-latest",
        "prompt_tokens": 145,
        "completion_tokens": 280,
        "total_tokens": 425,
        "response_time_seconds": 3.2
      }
    }
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | The synthesized or combined response based on the selected strategy |
| `session_id` | null | Always null for CONSENSUS (no conversation threading support) |
| `metadata.strategy` | string | The consensus strategy used (`all_responses`, `synthesize`, `majority`, `weighted`, `first_valid`) |
| `metadata.providers_queried` | array[string] | List of all providers that were queried |
| `metadata.providers_succeeded` | array[string] | List of providers that returned successful responses |
| `metadata.providers_failed` | array[string] | List of providers that failed or timed out |
| `metadata.execution_time_seconds` | float | Total time for parallel execution (not sum of individual times) |
| `metadata.temperature` | float | Temperature setting used for all providers (0.0-1.0) |
| `metadata.timestamp` | string | ISO 8601 timestamp of when the request was processed |
| `metadata.provider_details` | object | Per-provider execution details including model, tokens, and timing |

**Usage Notes:**
- CONSENSUS executes all providers in parallel for fast results
- `providers_failed` will list any providers that timed out or returned errors
- Token counts and response times are tracked per provider for analysis
- The `result` format varies by strategy (see Consensus Strategies section)
- CONSENSUS does not support conversation threading - each invocation is independent

## Advanced Usage

### With Provider Selection

```bash
# Use two specific providers
modelchorus consensus "Explain neural networks" -p claude -p gemini

# Use three providers for more perspectives
modelchorus consensus "Review this architecture" -p claude -p gemini -p codex

# Use all available providers
modelchorus consensus "Fact-check this claim" -p claude -p gemini -p codex -p cursor-agent
```

**Provider Selection Tips:**
- **Multiple providers required:** Specify at least 2 providers with multiple `-p` flags
- **Order doesn't matter:** All providers execute in parallel simultaneously
- **Failures handled gracefully:** If one provider fails, others continue
- **Provider names:** `claude`, `gemini`, `codex`, `cursor-agent`

**Parallel Execution:**
- All providers execute simultaneously (not sequentially)
- Total execution time ≈ slowest provider's response time
- Timeouts are per-provider, not total
- Failed providers don't block successful ones

### With Strategy Selection

```bash
# Get all responses separately (default)
modelchorus consensus "Compare React vs Vue"

# Get synthesized structured view
modelchorus consensus "Best practices for API design" -s synthesize

# Get majority consensus
modelchorus consensus "What is 2 + 2?" -s majority

# Get most detailed response
modelchorus consensus "Explain microservices" -s weighted

# Get fastest response with failover
modelchorus consensus "Quick question about Python" -s first_valid
```

**Strategy Selection:**
- Strategy is applied AFTER all providers respond
- Doesn't affect parallel execution behavior
- See "Consensus Strategies" section for detailed strategy guidance

### With File Context

```bash
# Include file context for all providers
modelchorus consensus "Review this code" -f src/main.py

# Multiple files with synthesis
modelchorus consensus "Analyze these components" -f models.py -f services.py -f api.py -s synthesize
```

**File Handling:**
- All files are read and provided to ALL providers
- Each provider receives identical file context
- File context doesn't increase execution time (parallel processing)

### With Timeout Control

```bash
# Increase timeout for slower providers or complex queries
modelchorus consensus "Detailed analysis needed" --timeout 180

# Shorter timeout for quick queries
modelchorus consensus "Fast check" --timeout 60
```

**Timeout Behavior:**
- Timeout applies PER PROVIDER independently
- If provider times out, it's marked as failed but others continue
- Successful providers' responses are still returned and used
- Default timeout: 120 seconds (2 minutes)

### Adjusting Creativity

```bash
# Lower temperature for factual consensus
modelchorus consensus "What are the Python PEP 8 rules?" -t 0.3

# Higher temperature for creative perspectives
modelchorus consensus "Brainstorm app names" -t 0.9

# Default balanced setting
modelchorus consensus "Explain design patterns" -t 0.7
```

**Temperature applies to all providers uniformly.**

### Saving Results

```bash
# Save full consensus results to JSON
modelchorus consensus "Evaluate this proposal" -s synthesize --output evaluation.json
```

**Output file contains:**
- Consensus response (based on strategy)
- All individual provider responses
- Provider metadata and success/failure status
- Strategy used
- Execution timing and token usage

## Best Practices

1. **Choose appropriate strategies for use cases** - Use `all_responses` or `synthesize` for complex decisions, `majority` for factual verification, `weighted` for comprehensive answers, and `first_valid` for speed with failover.

2. **Use multiple providers for diversity** - Include at least 2-3 different providers (e.g., claude + gemini + codex) to get genuinely diverse perspectives, not just redundancy.

3. **Set appropriate timeouts** - Increase timeout (180-300s) for complex queries or slower providers; decrease (30-60s) for simple queries where speed matters.

4. **Include relevant file context** - When reviewing code or documents, use `-f` flags to provide identical context to all providers for fair comparison.

5. **Consider cost vs value** - Multi-model consensus costs more than single-model. Use CONSENSUS for decisions where multiple perspectives provide genuine value.

## Examples

### Example 1: Architecture Decision with Synthesized Perspectives

**Scenario:** You need to decide between microservices and monolithic architecture for a new project.

**Command:**
```bash
modelchorus consensus "Should I use microservices or monolithic architecture for a mid-size SaaS product with 5 developers?" -s synthesize --output architecture-decision.json
```

**Expected Outcome:** Structured synthesis showing each model's perspective on the architecture decision, allowing you to see complementary insights and make an informed choice.

---

### Example 2: Fact Verification with Majority Consensus

**Scenario:** You need to verify a technical claim or calculation.

**Command:**
```bash
modelchorus consensus "Is it true that Python's GIL prevents true multi-threading? Explain briefly." -s majority -t 0.4
```

**Expected Outcome:** The most common answer among the three models, with low temperature ensuring factual accuracy. If models agree, high confidence in the answer.

---

### Example 3: Code Review with Multiple Perspectives

**Scenario:** You want diverse code review feedback from multiple AI models.

**Command:**
```bash
modelchorus consensus "Review this code for bugs, performance issues, and best practices" -f src/auth.py -s all_responses
```

**Expected Outcome:** Three separate code reviews showing different perspectives - one model might focus on security, another on performance, another on readability.

---

### Example 4: Quick Query with Failover

**Scenario:** You need a fast answer and want automatic failover if primary provider is unavailable.

**Command:**
```bash
modelchorus consensus "What does the Python 'yield' keyword do?" -s first_valid --timeout 30
```

**Expected Outcome:** First successful response returns immediately. If Claude responds in 2 seconds, you get that answer without waiting for Gemini. If Claude times out, Gemini's response is used.

## Troubleshooting

### Issue: All providers failed

**Symptoms:** Error message indicating all providers timed out or failed

**Cause:** Network issues, provider outages, or timeout too short for complex queries

**Solution:**
```bash
# Increase timeout and try again
modelchorus consensus "Your prompt" --timeout 240

# Check provider availability
modelchorus list-providers

# Try with single provider to isolate issue
modelchorus chat "Your prompt" -p claude
```

---

### Issue: One provider consistently times out

**Symptoms:** One specific provider always fails while others succeed

**Cause:** That provider is slower, has quota limits, or is experiencing issues

**Solution:**
```bash
# Increase timeout for that specific use case
modelchorus consensus "Your prompt" -p claude -p gemini --timeout 180

# Or exclude the problematic provider
modelchorus consensus "Your prompt" -p claude -p codex
```

---

### Issue: Responses are too similar/not diverse

**Symptoms:** All models give nearly identical responses

**Cause:** Models are trained similarly, or query has objectively correct answer

**Solution:**
- This is expected for factual queries (use `majority` to verify consensus)
- For creative tasks, increase temperature: `--temperature 0.8`
- For diverse perspectives, ensure you're asking for opinions, not facts
- Consider using ARGUMENT workflow for dialectical analysis instead

---

### Issue: Strategy not producing expected output

**Symptoms:** `majority` or `weighted` returning unexpected results

**Cause:** Strategy implementation limitations or response format incompatibility

**Solution:**
```bash
# Use all_responses or synthesize to see raw responses first
modelchorus consensus "Your prompt" -p claude -p gemini -s all_responses

# Then manually evaluate which strategy is appropriate
# For complex responses, synthesize or all_responses work best
```

## Related Workflows

- **CHAT** - When you only need a single model's perspective with conversation threading
- **ARGUMENT** - When you need structured dialectical analysis with creator/skeptic/moderator roles rather than parallel independent perspectives

---

**See Also:**
- ModelChorus Documentation: `/docs/WORKFLOWS.md`
- Provider Information: `modelchorus list-providers`
- General CLI Help: `modelchorus --help`
