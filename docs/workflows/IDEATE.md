# IDEATE Workflow

Creative brainstorming and idea generation with configurable creativity levels and structured output.

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Type** | Single-model creative ideation |
| **Models** | 1 provider (generates multiple ideas) |
| **Best For** | Brainstorming, feature ideas, creative solutions, innovation |
| **Conversation** | ✅ Yes (supports continuation) |
| **State** | ✅ Stateful (conversation threading) |
| **Output** | Multiple ideas + synthesis/summary |

---

## What It Does

The IDEATE workflow generates creative ideas through structured brainstorming. Unlike simple prompts, it systematically generates multiple distinct ideas with:

**Structured Generation:**
- Generates configurable number of ideas (default: 5)
- Each idea is distinct and fully developed
- Ideas explore different angles and approaches

**Creative Control:**
- Adjustable creativity via temperature parameter
- Can add constraints or criteria via system prompts
- Supports focused or exploratory ideation

**Synthesis:**
- Summarizes generated ideas
- Identifies best ideas and common themes
- Provides actionable recommendations

**Iterative Refinement:**
- Supports continuation for drilling into specific ideas
- Can expand on promising concepts
- Enables multi-turn ideation sessions

**Result:** Multiple creative ideas with analysis and recommendations, ready for evaluation or implementation.

---

## When to Use

Use IDEATE when you need:

### Product & Feature Brainstorming
```bash
# Generate feature ideas
model-chorus ideate "New features for a personal task management app" --num-ideas 7

# Explore product directions
model-chorus ideate "Revenue streams for a developer tools startup" --temperature 0.9
```

### Problem-Solving
```bash
# Find creative solutions
model-chorus ideate "Ways to reduce API latency by 50%"

# Explore alternatives
model-chorus ideate "Alternative approaches to user authentication that improve UX"
```

### Marketing & Business Strategy
```bash
# Marketing campaigns
model-chorus ideate "Creative marketing campaigns for sustainable fashion brand" --temperature 1.0

# Growth strategies
model-chorus ideate "User acquisition strategies for B2B SaaS startup"
```

### Technical Innovation
```bash
# Architecture ideas
model-chorus ideate "Innovative ways to handle real-time data synchronization"

# Performance optimizations
model-chorus ideate "Creative caching strategies for our API"
```

### Process Improvement
```bash
# Team workflows
model-chorus ideate "Ways to improve code review processes"

# Developer experience
model-chorus ideate "Ideas for better developer onboarding"
```

---

## When NOT to Use

**Don't use IDEATE when:**

| Situation | Use Instead | Reason |
|-----------|-------------|--------|
| Need to analyze existing argument | **ARGUMENT** | IDEATE generates new ideas, doesn't evaluate existing ones |
| Want multiple model perspectives | **CONSENSUS** | IDEATE uses single model |
| Need systematic investigation | **THINKDEEP** | IDEATE is creative, not investigative |
| Simple question or fact | **CHAT** | IDEATE adds overhead for idea generation |
| Already know what you want | Direct implementation | No need for brainstorming |

---

## CLI Usage

### Basic Invocation

**Simple ideation:**
```bash
model-chorus ideate "New features for task management app"
```

**Expected output:**
```
Generating ideas for: New features for task management app

--- Idea 1: Smart Task Prioritization ---
[Detailed idea description]

--- Idea 2: Collaborative Workspaces ---
[Detailed idea description]

--- Idea 3: AI-Powered Time Estimates ---
[Detailed idea description]

--- Idea 4: Focus Mode with Pomodoro ---
[Detailed idea description]

--- Idea 5: Task Dependencies Visualization ---
[Detailed idea description]

--- Summary & Recommendations ---
[Synthesis of ideas with recommendations]

Thread ID: ideate-thread-abc123
```

### Number of Ideas

**Control how many ideas are generated:**
```bash
# Generate 3 ideas (quick brainstorming)
model-chorus ideate "API versioning strategies" --num-ideas 3

# Generate 5 ideas (default, balanced)
model-chorus ideate "API versioning strategies"

# Generate 10 ideas (comprehensive exploration)
model-chorus ideate "API versioning strategies" --num-ideas 10
```

**Guidelines:**
- **3-4 ideas:** Quick brainstorming, focused exploration
- **5-7 ideas:** Standard brainstorming (recommended)
- **8-10 ideas:** Comprehensive, exploratory ideation

### Creativity Control (Temperature)

**Adjust creativity level:**
```bash
# Low creativity (0.3-0.5): Practical, conventional ideas
model-chorus ideate "Database optimization techniques" --temperature 0.5

# Medium creativity (0.6-0.8): Balanced creativity (default)
model-chorus ideate "User engagement strategies" --temperature 0.7

# High creativity (0.9-1.0): Bold, unconventional ideas
model-chorus ideate "Disruptive product features" --temperature 1.0
```

### With Provider Selection

**Specify which AI provider to use:**
```bash
# Use Claude (excellent for creative ideation)
model-chorus ideate "Marketing campaign ideas" --provider claude

# Use Gemini (strong analytical creativity)
model-chorus ideate "Technical architecture ideas" --provider gemini

# Use Codex (best for code-related ideas)
model-chorus ideate "Code refactoring approaches" --provider codex
```

### With Constraints

**Add specific constraints or criteria:**
```bash
# Use system prompt for constraints
model-chorus ideate "Revenue streams for startup" \
  --system "Must be implementable within 6 months, budget under $50k, target individual developers"

# Focus ideation with context
model-chorus ideate "API design improvements" \
  --system "Current API has 50k requests/day, RESTful, needs backward compatibility"
```

### With File Context

**Provide supporting documents:**
```bash
# Single file
model-chorus ideate "Improvements for this codebase" \
  --file src/architecture.md

# Multiple files
model-chorus ideate "Refactoring strategies for auth system" \
  --file src/auth/current_impl.py \
  --file docs/auth_requirements.md \
  --file docs/known_issues.md
```

### With Continuation (Threading)

**Continue brainstorming session:**
```bash
# Initial brainstorming
model-chorus ideate "Gamification features for learning platform"
# Returns: Thread ID: ideate-thread-xyz789

# Drill into specific idea
model-chorus ideate "Expand on idea #2 (achievement system). Provide implementation details." \
  --continue ideate-thread-xyz789

# Explore variations
model-chorus ideate "What if we combine ideas #1 and #3?" \
  --continue ideate-thread-xyz789
```

---

## Python API Usage

### Basic Usage

```python
import asyncio
from model_chorus.workflows import IdeateWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def generate_ideas():
    # Initialize provider and memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create workflow
    workflow = IdeateWorkflow(
        provider=provider,
        conversation_memory=memory
    )

    # Generate ideas
    result = await workflow.run(
        prompt="New features for personal task management app",
        num_ideas=5,
        temperature=0.9
    )

    if result.success:
        # Access individual ideas
        for i, step in enumerate(result.steps, 1):
            idea_name = step.metadata.get('name', f'Idea {i}')
            print(f"\n--- {idea_name} ---")
            print(step.content)

        # Access summary
        print("\n--- Summary ---")
        print(result.synthesis)

        # Get thread ID for continuation
        thread_id = result.metadata.get('thread_id')
        print(f"\nThread ID: {thread_id}")
    else:
        print(f"Ideation failed: {result.error}")

# Run
asyncio.run(generate_ideas())
```

### With Constraints

```python
async def constrained_ideation():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    constraints = """
    Constraints:
    - Must be implementable within 6 months
    - Budget under $50,000
    - Target audience: individual developers
    - Must align with open-source values
    """

    result = await workflow.run(
        prompt="Revenue streams for developer tools startup",
        num_ideas=6,
        system_prompt=constraints,
        temperature=0.85
    )

    if result.success:
        print(result.synthesis)
```

### With Continuation

```python
async def iterative_ideation():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Initial brainstorming
    result1 = await workflow.run(
        prompt="Gamification features for online learning platform",
        num_ideas=5
    )

    thread_id = result1.metadata['thread_id']

    # Refine specific idea
    result2 = await workflow.run(
        prompt="Expand on idea #2 (achievement system). Provide detailed implementation steps.",
        num_ideas=1,
        continuation_id=thread_id
    )

    print(result2.synthesis)
```

### High-Creativity Brainstorming

```python
async def creative_ideation():
    """Generate highly creative, unconventional ideas."""
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    result = await workflow.run(
        prompt="Unconventional marketing campaigns for sustainable fashion",
        num_ideas=7,
        temperature=1.0  # Maximum creativity
    )

    if result.success:
        for step in result.steps:
            print(f"\n{step.metadata.get('name')}")
            print(step.content)
```

### Accessing Result Components

```python
result = await workflow.run(prompt="Generate ideas", num_ideas=5)

if result.success:
    # Individual ideas
    ideas = [
        {
            'name': step.metadata.get('name', f'Idea {i}'),
            'content': step.content,
            'metadata': step.metadata
        }
        for i, step in enumerate(result.steps, 1)
    ]

    # Summary/synthesis
    summary = result.synthesis

    # Metadata
    thread_id = result.metadata['thread_id']
    model_used = result.metadata.get('model', 'unknown')
    ideas_count = len(result.steps)

    print(f"Generated {ideas_count} ideas using {model_used}")
    print(f"Thread ID: {thread_id}")
```

### Error Handling

```python
async def robust_ideation():
    try:
        provider = ClaudeProvider()
        memory = ConversationMemory()
        workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

        result = await workflow.run(
            prompt="Feature ideas for mobile app",
            num_ideas=5,
            temperature=0.8
        )

        if result.success:
            return result.steps, result.synthesis
        else:
            # Handle workflow failure
            print(f"Ideation failed: {result.error}")
            # Retry with different parameters, log error, etc.
            return None, None

    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        # Check API keys, provider availability, etc.
        return None, None
```

---

## Advanced Features

### Tiered Creativity Levels

Use temperature to control idea characteristics:

```python
# Conservative ideation (0.3-0.5)
# - Practical, proven approaches
# - Incremental improvements
# - Low-risk ideas
result = await workflow.run(
    prompt="Database optimization techniques",
    temperature=0.4,
    num_ideas=5
)

# Balanced ideation (0.6-0.8) [RECOMMENDED]
# - Mix of practical and creative
# - Some innovation, manageable risk
# - Good balance for most use cases
result = await workflow.run(
    prompt="User engagement strategies",
    temperature=0.7,
    num_ideas=6
)

# Highly creative ideation (0.9-1.0)
# - Bold, unconventional ideas
# - High innovation, higher risk
# - Good for breakthrough thinking
result = await workflow.run(
    prompt="Disruptive product features",
    temperature=1.0,
    num_ideas=8
)
```

### Idea Refinement Pattern

Start broad, then drill down:

```python
async def refine_ideas(provider, memory):
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Step 1: Generate broad ideas
    result1 = await workflow.run(
        prompt="Ways to improve developer productivity",
        num_ideas=7,
        temperature=0.8
    )

    thread_id = result1.metadata['thread_id']

    # Step 2: User selects promising idea (e.g., idea #3)
    # Now refine it
    result2 = await workflow.run(
        prompt="Expand on idea #3 (automated code review). Provide: implementation steps, required tools, estimated timeline, potential challenges.",
        num_ideas=1,
        continuation_id=thread_id,
        temperature=0.6  # Lower temperature for practical details
    )

    return result2.synthesis
```

### Constrained Brainstorming

Guide ideation with specific criteria:

```python
async def constrained_brainstorming():
    workflow = IdeateWorkflow(provider, memory)

    # Define constraints
    constraints = """
    Generate ideas considering:
    - Team size: 5 developers
    - Timeline: 3 months
    - Budget: $30,000
    - Tech stack: Python, React, PostgreSQL
    - Must integrate with existing systems
    - Focus on user-facing improvements
    """

    result = await workflow.run(
        prompt="Feature ideas for our customer portal",
        system_prompt=constraints,
        num_ideas=6,
        temperature=0.7
    )

    return result
```

### Idea Comparison

Generate ideas from multiple perspectives:

```python
async def multi_perspective_ideation(prompt):
    """Generate ideas using different creativity levels."""
    workflow = IdeateWorkflow(provider, memory)

    results = {}

    # Practical ideas
    results['practical'] = await workflow.run(
        prompt=prompt,
        num_ideas=5,
        temperature=0.5
    )

    # Balanced ideas
    results['balanced'] = await workflow.run(
        prompt=prompt,
        num_ideas=5,
        temperature=0.7
    )

    # Creative ideas
    results['creative'] = await workflow.run(
        prompt=prompt,
        num_ideas=5,
        temperature=0.95
    )

    return results
```

---

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Topic/area for idea generation |
| `num_ideas` | `int` | `5` | Number of ideas to generate |
| `temperature` | `float` | `0.7` | Creativity level (0.0-1.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `system_prompt` | `str` | `None` | Constraints, context, or criteria |
| `continuation_id` | `str` | `None` | Thread ID to continue session |
| `files` | `List[str]` | `[]` | File paths for context |

### Number of Ideas Guidelines

| Count | Use Case | Duration | Output |
|-------|----------|----------|--------|
| **3-4** | Quick exploration, focused brainstorming | Fast | Concise, targeted |
| **5-7** | Standard brainstorming, balanced coverage | Medium | Recommended for most use cases |
| **8-10** | Comprehensive exploration, diverse ideas | Slower | Extensive, thorough |
| **10+** | Exhaustive ideation, maximum diversity | Longest | Very comprehensive |

### Temperature Guidelines

| Range | Creativity | Use Case | Characteristics |
|-------|------------|----------|----------------|
| **0.0-0.4** | Low | Optimization, improvements | Practical, conservative, proven approaches |
| **0.5-0.7** | Medium | General brainstorming | Balanced mix of practical and creative |
| **0.8-0.9** | High | Innovation, new concepts | Bold ideas, some unconventional thinking |
| **0.95-1.0** | Maximum | Breakthrough thinking | Highly creative, disruptive, experimental |

### Provider Comparison

**Claude:**
- Excellent creative ideation
- Strong at nuanced, contextual ideas
- Recommended temperature: 0.7-0.9

**Gemini:**
- Analytical creativity
- Good at structured, logical ideas
- Recommended temperature: 0.6-0.8

**Codex:**
- Best for technical/code ideas
- Strong at practical implementations
- Recommended temperature: 0.5-0.7

---

## Best Practices

### 1. Match Temperature to Use Case

```python
# Infrastructure improvements - be practical
await workflow.run("Database optimization strategies", temperature=0.5)

# Product features - balanced
await workflow.run("New app features", temperature=0.7)

# Marketing campaigns - be creative
await workflow.run("Marketing campaign ideas", temperature=0.95)
```

### 2. Provide Context When Helpful

**Without context:**
```python
result = await workflow.run("Feature ideas")
```

**With helpful context:**
```python
result = await workflow.run(
    prompt="Feature ideas for our mobile app",
    system_prompt="Target users: 18-25 year olds, focus on social features, budget: $100k",
    files=["docs/user_research.md", "docs/current_features.md"]
)
```

### 3. Start Broad, Then Refine

```python
# Step 1: Broad exploration
result1 = await workflow.run("Ways to improve API performance", num_ideas=8)

# Step 2: Refine promising idea
result2 = await workflow.run(
    "Expand on idea #5 (caching strategy) with implementation details",
    continuation_id=thread_id,
    num_ideas=1,
    temperature=0.6  # Lower temperature for details
)
```

### 4. Use Constraints to Focus Ideas

```python
constraints = """
Generate ideas that:
- Can be implemented in < 3 months
- Require no additional headcount
- Use existing tech stack
- Have measurable impact on user engagement
"""

result = await workflow.run(
    prompt="Quick wins for user retention",
    system_prompt=constraints,
    num_ideas=5
)
```

### 5. Clean Up Threads

```python
# After ideation is complete
memory.archive_thread(thread_id)

# Or delete if no longer needed
memory.delete_thread(thread_id)
```

### 6. Save Valuable Ideas

```python
import json
from pathlib import Path

result = await workflow.run(prompt="Feature ideas", num_ideas=5)

if result.success:
    # Save to file
    ideas = {
        'prompt': "Feature ideas",
        'ideas': [
            {'name': s.metadata.get('name'), 'content': s.content}
            for s in result.steps
        ],
        'summary': result.synthesis
    }

    Path("brainstorming_results.json").write_text(json.dumps(ideas, indent=2))
```

---

## Common Use Patterns

### Pattern 1: Feature Brainstorming

```python
async def feature_brainstorm(context: dict):
    """Generate feature ideas with product context."""
    workflow = IdeateWorkflow(provider, memory)

    system_prompt = f"""
    Product context:
    - Target users: {context['users']}
    - Current features: {context['features']}
    - Goals: {context['goals']}
    - Constraints: {context['constraints']}
    """

    result = await workflow.run(
        prompt=context['brainstorm_topic'],
        system_prompt=system_prompt,
        num_ideas=7,
        temperature=0.8
    )

    # Extract and rank ideas
    ideas = [
        {
            'name': step.metadata.get('name'),
            'description': step.content,
            'index': i
        }
        for i, step in enumerate(result.steps, 1)
    ]

    return {
        'ideas': ideas,
        'recommendations': result.synthesis
    }
```

### Pattern 2: Problem-Solution Ideation

```python
async def solve_problem(problem: str, constraints: List[str]):
    """Generate creative solutions to a specific problem."""
    workflow = IdeateWorkflow(provider, memory)

    constraints_text = "\n".join(f"- {c}" for c in constraints)

    result = await workflow.run(
        prompt=f"Creative solutions for: {problem}",
        system_prompt=f"Constraints:\n{constraints_text}",
        num_ideas=6,
        temperature=0.85
    )

    return result.steps, result.synthesis
```

### Pattern 3: Tiered Exploration

```python
async def tiered_ideation(topic: str):
    """Generate ideas at different creativity levels."""
    workflow = IdeateWorkflow(provider, memory)

    # Conservative ideas
    practical = await workflow.run(
        prompt=f"Practical, proven approaches for: {topic}",
        num_ideas=4,
        temperature=0.5
    )

    # Balanced ideas
    balanced = await workflow.run(
        prompt=f"Innovative but feasible ideas for: {topic}",
        num_ideas=5,
        temperature=0.75
    )

    # Creative ideas
    creative = await workflow.run(
        prompt=f"Bold, unconventional ideas for: {topic}",
        num_ideas=4,
        temperature=0.95
    )

    return {
        'practical': practical.steps,
        'balanced': balanced.steps,
        'creative': creative.steps
    }
```

---

## Troubleshooting

### Issue: Ideas Too Similar

**Symptoms:** Generated ideas lack diversity, feel repetitive

**Solutions:**
```python
# Increase temperature for more variety
result = await workflow.run(prompt="...", temperature=0.9)

# Generate more ideas (more attempts = more diversity)
result = await workflow.run(prompt="...", num_ideas=10)

# Add instruction for diversity
result = await workflow.run(
    prompt="...",
    system_prompt="Generate highly diverse ideas that explore different angles and approaches"
)
```

### Issue: Ideas Too Generic

**Symptoms:** Ideas lack specificity or depth

**Solutions:**
```python
# Provide more context
result = await workflow.run(
    prompt="Feature ideas for our app",
    system_prompt="Context: Healthcare app for elderly users, focus on accessibility",
    files=["docs/user_research.md"]
)

# Request specific details
result = await workflow.run(
    prompt="Feature ideas with implementation details and user value propositions"
)

# Refine ideas through continuation
result2 = await workflow.run(
    prompt="Expand idea #3 with specific details",
    continuation_id=thread_id
)
```

### Issue: Ideas Too Impractical

**Symptoms:** Ideas are creative but not feasible

**Solutions:**
```python
# Lower temperature
result = await workflow.run(prompt="...", temperature=0.6)

# Add practical constraints
result = await workflow.run(
    prompt="...",
    system_prompt="Generate practical ideas that can be implemented within 3 months with current team"
)

# Request actionable ideas
result = await workflow.run(
    prompt="Actionable, implementable ideas for..."
)
```

### Issue: Lost Thread Context

**Symptoms:** Continuation doesn't reference previous ideas

**Solutions:**
```python
# Verify thread ID
print(f"Thread ID: {result.metadata['thread_id']}")

# Check thread exists
threads = memory.list_threads()
print(threads)

# Check conversation history
history = memory.get_thread_history(thread_id)
print(f"Messages: {len(history)}")
```

### Issue: Provider Errors

**Symptoms:** "Provider failed" or API errors

**Solutions:**
```bash
# Check API keys
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY

# Test provider CLI
claude --version
gemini --version

# Try different provider
model-chorus ideate "test ideas" --provider gemini
```

### Issue: Ideas Too Brief

**Symptoms:** Each idea is very short, lacking detail

**Solutions:**
```python
# Increase max_tokens
result = await workflow.run(prompt="...", max_tokens=4000)

# Request detailed ideas
result = await workflow.run(
    prompt="Detailed, comprehensive ideas for...",
    system_prompt="For each idea, provide: description, implementation approach, benefits, and challenges"
)

# Generate fewer ideas (more space per idea)
result = await workflow.run(prompt="...", num_ideas=3)
```

---

## Comparison with Other Workflows

### IDEATE vs CHAT

| Aspect | IDEATE | CHAT |
|--------|--------|------|
| **Purpose** | Generate multiple creative ideas | Simple conversation |
| **Output** | Multiple distinct ideas + synthesis | Single response |
| **Best for** | Brainstorming, exploration | Quick questions, iteration |
| **Structure** | Structured idea generation | Freeform dialogue |
| **Use when** | Need multiple options | Need single answer or discussion |

**Example:**
```bash
# Use IDEATE for brainstorming
model-chorus ideate "Feature ideas for mobile app"

# Use CHAT for discussion
model-chorus chat "What do users want in a mobile app?" -p claude
```

### IDEATE vs ARGUMENT

| Aspect | IDEATE | ARGUMENT |
|--------|--------|----------|
| **Purpose** | Generate new ideas | Analyze existing arguments |
| **Process** | Creative generation | Dialectical reasoning |
| **Output** | Multiple ideas | Pro/Con/Synthesis |
| **Best for** | Brainstorming | Evaluating claims |
| **Use when** | Need creative options | Need to analyze a position |

**Example:**
```bash
# Use IDEATE to generate options
model-chorus ideate "Alternative authentication methods"

# Use ARGUMENT to evaluate specific option
model-chorus argument "Passwordless auth improves security"
```

### IDEATE vs CONSENSUS

| Aspect | IDEATE | CONSENSUS |
|--------|--------|-----------|
| **Models** | Single model (multiple ideas) | Multiple models (multiple perspectives) |
| **Purpose** | Creative ideation | Multi-model comparison |
| **Best for** | Generating options | Validating approaches |
| **Threading** | Yes | No |
| **Use when** | Need creative ideas | Need multiple expert opinions |

**Example:**
```bash
# Use IDEATE for idea generation
model-chorus ideate "Marketing strategies" --num-ideas 7

# Use CONSENSUS to evaluate best strategy
model-chorus consensus "Which marketing strategy is best?" \
  -p claude -p gemini -s synthesize
```

---

## Real-World Examples

### Example 1: Product Feature Brainstorming

```bash
model-chorus ideate "New features for developer productivity tool" \
  --num-ideas 8 \
  --temperature 0.8 \
  --system "Target: Individual developers and small teams. Must integrate with VS Code and GitHub."
```

**Expected output:**
- **Idea 1:** AI-powered code completion
- **Idea 2:** Automated test generation
- **Idea 3:** Smart code review suggestions
- **Idea 4:** Project dependency analyzer
- **Idea 5:** Focus mode with distraction blocking
- **Idea 6:** Collaborative debugging sessions
- **Idea 7:** Performance profiling integration
- **Idea 8:** Automated documentation generation
- **Summary:** Recommended top 3 ideas with implementation priorities

### Example 2: Marketing Campaign Ideas

```bash
model-chorus ideate "Creative marketing campaigns for sustainable fashion brand" \
  --num-ideas 6 \
  --temperature 1.0
```

**Expected output:**
- **Idea 1:** "Wear Your Values" influencer partnership
- **Idea 2:** Virtual fashion shows in the metaverse
- **Idea 3:** Clothing lifecycle transparency app
- **Idea 4:** Trade-in program with gamification
- **Idea 5:** Sustainable fashion education series
- **Idea 6:** Community-driven design contests
- **Summary:** Best campaigns for different channels and audiences

### Example 3: Technical Problem-Solving

```bash
model-chorus ideate "Ways to reduce API latency by 50%" \
  --file docs/current_architecture.md \
  --file logs/performance_analysis.log \
  --num-ideas 7 \
  --temperature 0.7
```

**Expected output:**
- **Idea 1:** Multi-tier caching strategy
- **Idea 2:** Database query optimization
- **Idea 3:** CDN integration for static assets
- **Idea 4:** GraphQL for precise data fetching
- **Idea 5:** Connection pooling improvements
- **Idea 6:** Async processing for heavy operations
- **Idea 7:** Geographic load balancing
- **Summary:** Quick wins vs long-term improvements

---

## See Also

**Related Documentation:**
- [WORKFLOWS.md](../WORKFLOWS.md) - Complete workflow comparison guide
- [ARGUMENT Workflow](ARGUMENT.md) - Dialectical reasoning for argument analysis
- [CHAT Workflow](CHAT.md) - Simple conversation (coming soon)
- [CONSENSUS Workflow](CONSENSUS.md) - Multi-model perspectives (coming soon)
- [CONVERSATION_INFRASTRUCTURE.md](../CONVERSATION_INFRASTRUCTURE.md) - Threading details

**Code Examples:**
- [examples/workflow_examples.py](../../examples/workflow_examples.py) - Complete Python examples
- [examples/ideate_basic.py](../../examples/) - Basic IDEATE usage

**API Reference:**
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Complete API documentation
- [README.md](../../README.md) - Getting started guide

**Related Workflows:**
- Use **CHAT** for simple questions without structured ideation
- Use **ARGUMENT** to evaluate generated ideas dialectically
- Use **CONSENSUS** to get multi-model perspectives on ideas
- Combine **IDEATE** + **ARGUMENT** to generate then evaluate ideas
