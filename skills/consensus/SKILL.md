---
name: model-chorus:consensus
description: Builds multi-model consensus through systematic analysis and structured debate for complex decisions, architectural choices, feature proposals, and technology evaluations
---

# ModelChorus: Consensus

Multi-model consensus building through systematic consultation and analysis. Gathers perspectives from multiple AI models with different stances to synthesize comprehensive recommendations.

## When to Use

Use this skill when you need:
- **Complex decisions** - Multiple valid approaches, need expert input
- **Architectural choices** - Technology selection, design patterns
- **Feature proposals** - Evaluating new feature designs
- **Technology evaluation** - Comparing frameworks, libraries, tools
- **Risk assessment** - Getting diverse perspectives on risks
- **Code design review** - Multiple opinions on implementation approaches

## How It Works

The consensus workflow:

1. **Your Analysis** - You analyze the problem first (not shared with models)
2. **Multi-Model Consultation** - Each model evaluates independently
3. **Stance Assignment** - Models can be "for", "against", or "neutral"
4. **Structured Debate** - Each model presents their perspective
5. **Synthesis** - Results are synthesized into comprehensive recommendation

This prevents groupthink and ensures diverse perspectives.

## Usage

Invoke Consensus with your decision or proposal:

```
Use Consensus to evaluate:
[Your proposal or decision to evaluate]

Models to consult:
- Model 1: [model name] - Stance: [for/against/neutral]
- Model 2: [model name] - Stance: [for/against/neutral]

Files: [relevant file paths]
```

## Parameters

- **step** (required): Current consultation step
- **step_number** (required): Current step (starts at 1)
- **total_steps** (required): Total consultation steps
- **next_step_required** (required): True if more models to consult
- **findings** (required): Your analysis (step 1) or summary of model response (steps 2+)

**Required (Step 1 only):**
- **models**: List of models with optional stance and stance_prompt
  - Each entry: `{model: "name", stance: "for/against/neutral", stance_prompt: "..."}`
  - Minimum 2 models
  - Each (model, stance) pair must be unique

**Optional:**
- **relevant_files**: Supporting files for context (absolute paths)
- **images**: Diagrams or screenshots
- **continuation_id**: Resume previous consensus session
- **current_model_index**: Internally managed, tracks progress
- **model_responses**: Internally managed, logs responses

## Stances Explained

**for** - Model argues in favor of the proposal
**against** - Model argues against the proposal
**neutral** - Model provides balanced analysis

**Example:**
```javascript
models: [
  {model: "gpt-5-pro", stance: "for", stance_prompt: "Argue why this is good"},
  {model: "gemini-2.5-pro", stance: "against", stance_prompt: "Identify risks"},
  {model: "gpt-5-codex", stance: "neutral"}
]
```

## Examples

### Example 1: Technology Selection

```
Consensus Evaluation:

Step 1: Your Independent Analysis

Proposal: "Should we migrate from REST to GraphQL for our API?"

Your analysis: "GraphQL offers better flexibility for mobile clients,
but adds complexity. Team has minimal GraphQL experience."

Models:
- gpt-5-pro (for): "Argue benefits of GraphQL migration"
- gemini-2.5-pro (against): "Identify risks and downsides"
- gpt-5 (neutral): "Provide balanced analysis"

Relevant files:
- /home/user/project/api/endpoints.py
- /home/user/project/docs/api_spec.yaml

Step: 1/4
Next: true
```

### Example 2: Architecture Decision

```
Consensus Evaluation:

Step 1: Analyze microservices vs monolith

Proposal: "Split our monolith into 5 microservices for scalability"

Your findings: "Current monolith handles 10k req/sec fine. Team size: 4 developers.
Microservices add operational overhead but enable independent scaling."

Models:
- gemini-2.5-pro (for)
- gpt-5-pro (against)
- gpt-5-codex (neutral)

Relevant files:
- /home/user/project/architecture.md
- /home/user/project/performance/benchmarks.py

Step: 1/4
Next: true
```

### Example 3: Feature Design

```
Consensus Evaluation:

Step 1: Evaluate new caching strategy

Proposal: "Implement read-through cache with Redis for all API responses"

Your analysis: "Would reduce database load by ~60% based on access patterns.
Adds Redis dependency and cache invalidation complexity."

Models:
- gpt-5-pro (for): "Focus on performance benefits"
- gemini-2.5-pro (neutral): "Balanced cost-benefit"
- gpt-5-codex (against): "Highlight complexity and risks"

Step: 1/4
Next: true
```

### Example 4: Code Design Review

```
Consensus Evaluation:

Step 1: Review authentication refactoring approach

Proposal: "Refactor auth middleware to use decorator pattern"

Your findings: "Current middleware is 300 lines, hard to test. Decorator
pattern would improve testability and reusability."

Models:
- gpt-5-codex (for)
- gemini-2.5-pro (neutral)

Relevant files:
- /home/user/project/auth/middleware.py
- /home/user/project/auth/decorators.py (proposed)

Step: 1/3
Next: true
```

## Workflow Steps

**Step 1: Your Independent Analysis**
- Write exact proposal/question for models
- Document your analysis (not shared with models)
- Define models and stances
- Gather relevant files

**Steps 2-N: Model Consultations**
- Each model evaluates independently
- You summarize their response in findings
- Track model_responses internally
- Continue until all models consulted

**Final Step: Synthesis**
- External model synthesizes all perspectives
- Provides comprehensive recommendation
- Balances different viewpoints
- Includes action items

## Best Practices

**DO:**
- ✅ Do your own analysis first (step 1)
- ✅ Use clear, specific proposals
- ✅ Assign diverse stances (for/against/neutral)
- ✅ Provide relevant code/docs
- ✅ Use minimum 2 models (3-4 recommended)
- ✅ Make each (model, stance) pair unique

**DON'T:**
- ❌ Share your analysis with models
- ❌ Use vague proposals
- ❌ Assign same stance to all models
- ❌ Skip your independent analysis
- ❌ Use more than 4 models (diminishing returns)
- ❌ Repeat (model, stance) combinations

## Model Selection

**Recommended models:**
- `gemini-2.5-pro` - Excellent for analysis
- `gpt-5-pro` - Strong reasoning capabilities
- `gpt-5-codex` - Best for code-specific decisions
- `gpt-5` - Balanced general analysis

Use listmodels tool for complete list.

## Stance Strategies

**Balanced** (Recommended):
- 1 for, 1 against, 1 neutral
- Ensures diverse perspectives

**Devil's Advocate**:
- 2 for, 1 against
- Use when you're leaning toward proposal

**Risk Assessment**:
- 1 for, 2 against
- Use for high-risk decisions

**Neutral Analysis**:
- All neutral
- Pure objective evaluation

## Expert Synthesis

Default: External model synthesizes responses. Set `use_assistant_model: false` to synthesize yourself.

## Integration with ModelChorus

Uses `mcp__zen__consensus` tool from Zen MCP server for structured multi-model consensus building.

Can also use CLI directly:
```bash
modelchorus consensus "Your prompt" -p claude -p gemini -s synthesize
```

## See Also

- **planner** - For breaking down implementation plans
- **codereview** - For code-specific reviews
- **thinkdeep** - For deep investigation
- **chat** - For single-model consultations
