# ARGUMENT Workflow

Structured dialectical reasoning for analyzing arguments, claims, and proposals through multi-perspective debate.

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Type** | Single-model dialectical reasoning |
| **Models** | 1 provider (Creator + Skeptic + Moderator roles) |
| **Best For** | Argument analysis, policy debates, technology decisions, balanced perspectives |
| **Conversation** | ✅ Yes (supports continuation) |
| **State** | ✅ Stateful (conversation threading) |
| **Output** | Creator argument, Skeptic critique, Moderator synthesis |

---

## What It Does

The ARGUMENT workflow analyzes arguments, claims, and proposals through structured dialectical reasoning. Unlike simple Q&A, it examines the topic from multiple angles through three distinct roles:

**1. Creator (Pro)** - Presents arguments **in favor** of the claim
- Constructs strongest possible case supporting the position
- Identifies benefits, evidence, and reasoning
- Advocates for the proposition

**2. Skeptic (Con)** - Challenges and critiques the argument
- Identifies weaknesses, counterarguments, and limitations
- Questions assumptions and evidence
- Presents opposing viewpoints

**3. Moderator (Synthesis)** - Provides balanced analysis
- Weighs both perspectives fairly
- Identifies common ground and key trade-offs
- Delivers nuanced conclusion incorporating both sides

**Result:** A comprehensive, balanced analysis that avoids single-perspective bias and reveals the complexity of the topic.

---

## When to Use

Use ARGUMENT when you need:

### Policy & Governance Analysis
```bash
# Analyze policy proposals from multiple angles
model-chorus argument "Universal healthcare should be implemented in the US"

# Evaluate regulatory changes
model-chorus argument "Cryptocurrency should be regulated like traditional securities"
```

### Technology & Architecture Decisions
```bash
# Debate technical approaches
model-chorus argument "GraphQL should replace all REST APIs in our application"

# Evaluate framework choices
model-chorus argument "Our team should migrate from React to Svelte"
```

### Feature & Product Proposals
```bash
# Analyze feature proposals
model-chorus argument "We should add AI-powered code review to our platform"

# Evaluate business strategies
model-chorus argument "Freemium model would be better than enterprise-only licensing"
```

### Research & Critical Analysis
```bash
# Academic arguments
model-chorus argument "Remote work increases overall productivity for knowledge workers"

# Scientific hypotheses
model-chorus argument "Quantum computing will make classical cryptography obsolete within 10 years"
```

### Code Review Alternatives
```bash
# Evaluate implementation approaches
model-chorus argument "This refactoring will improve maintainability without introducing bugs" \
  -f src/auth.py -f src/auth_refactored.py
```

---

## When NOT to Use

**Don't use ARGUMENT when:**

| Situation | Use Instead | Reason |
|-----------|-------------|--------|
| Simple questions or facts | **CHAT** | No need for dialectical debate |
| Need multiple model perspectives | **CONSENSUS** | ARGUMENT uses single model for all roles |
| Systematic investigation required | **THINKDEEP** | Need hypothesis tracking, not debate |
| Quick consultation | **CHAT** | ARGUMENT adds overhead with 3-role structure |
| Cost is primary concern | **CHAT** | ARGUMENT generates more tokens (3 responses) |

---

## CLI Usage

### Basic Invocation

**Simple argument analysis:**
```bash
model-chorus argument "Universal basic income would reduce poverty"
```

**Expected output:**
```
Analyzing argument: Universal basic income would reduce poverty

--- Creator (Pro) ---
[Arguments in favor of UBI reducing poverty]

--- Skeptic (Con) ---
[Counterarguments and limitations]

--- Moderator (Synthesis) ---
[Balanced analysis incorporating both perspectives]

Thread ID: argument-thread-abc123
```

### With Provider Selection

**Specify which AI provider to use:**
```bash
# Use Claude
model-chorus argument "Remote work increases productivity" --provider claude

# Use Gemini
model-chorus argument "Remote work increases productivity" --provider gemini

# Use Codex
model-chorus argument "Remote work increases productivity" --provider codex
```

### With File Context

**Provide supporting documents:**
```bash
# Single file
model-chorus argument "This refactoring improves code quality" \
  --file src/auth.py

# Multiple files
model-chorus argument "GraphQL would improve our API" \
  --file docs/api_requirements.md \
  --file docs/current_rest_api.md \
  --file docs/performance_analysis.md
```

### With Continuation (Threading)

**Continue a previous argument analysis:**
```bash
# Initial analysis
model-chorus argument "TypeScript should replace JavaScript in our codebase"
# Returns: Thread ID: argument-thread-xyz789

# Follow-up analysis
model-chorus argument "But what about the migration cost and learning curve?" \
  --continue argument-thread-xyz789

# Further exploration
model-chorus argument "How would this affect our existing libraries?" \
  --continue argument-thread-xyz789
```

**Why use continuation:**
- Builds on previous analysis context
- Explores specific angles in depth
- Maintains conversation coherence
- Saves tokens by reusing established context

### With Custom Configuration

**Temperature control (creativity vs focus):**
```bash
# Lower temperature for focused, analytical debate
model-chorus argument "Should we use microservices?" --temperature 0.5

# Higher temperature for more creative perspectives
model-chorus argument "How can we improve team productivity?" --temperature 0.9
```

**Token limit:**
```bash
# Limit response length
model-chorus argument "Evaluate our caching strategy" --max-tokens 1500
```

**System prompt for context:**
```bash
# Add specific context or constraints
model-chorus argument "Should we adopt Kubernetes?" \
  --system "Our team has 5 developers, 3 microservices, budget under $10k/month"
```

---

## Python API Usage

### Basic Usage

```python
import asyncio
from model_chorus.workflows import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

async def analyze_argument():
    # Initialize provider and memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create workflow
    workflow = ArgumentWorkflow(
        provider=provider,
        conversation_memory=memory
    )

    # Run analysis
    result = await workflow.run(
        prompt="Universal basic income would significantly reduce poverty",
        temperature=0.7
    )

    if result.success:
        # Access each role's perspective
        for step in result.steps:
            role_name = step.metadata.get('name', 'Unknown')
            print(f"\n--- {role_name} ---")
            print(step.content)

        # Access final synthesis
        print("\n--- Synthesis ---")
        print(result.synthesis)

        # Get thread ID for continuation
        thread_id = result.metadata.get('thread_id')
        print(f"\nThread ID: {thread_id}")
    else:
        print(f"Analysis failed: {result.error}")

# Run
asyncio.run(analyze_argument())
```

### With File Context

```python
async def analyze_with_context():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    result = await workflow.run(
        prompt="This API design follows best practices",
        files=["docs/api_spec.yaml", "docs/rest_guidelines.md"],
        temperature=0.7
    )

    if result.success:
        print(result.synthesis)
```

### With Continuation

```python
async def continuing_analysis():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # Initial analysis
    result1 = await workflow.run(
        prompt="GraphQL should replace our REST APIs"
    )

    thread_id = result1.metadata['thread_id']

    # Continue with follow-up
    result2 = await workflow.run(
        prompt="But what about caching and complexity?",
        continuation_id=thread_id
    )

    print(result2.synthesis)
```

### Accessing Result Components

```python
result = await workflow.run(prompt="Some argument")

if result.success:
    # Individual role responses
    creator_response = result.steps[0].content
    skeptic_response = result.steps[1].content

    # Final synthesis
    moderator_synthesis = result.synthesis

    # Metadata
    thread_id = result.metadata['thread_id']
    model_used = result.metadata.get('model', 'unknown')
    conversation_length = result.metadata.get('conversation_length', 0)

    # Step metadata
    for step in result.steps:
        role_name = step.metadata.get('name')
        timestamp = step.metadata.get('timestamp')
        print(f"{role_name} at {timestamp}")
```

### Error Handling

```python
async def robust_analysis():
    try:
        provider = ClaudeProvider()
        memory = ConversationMemory()
        workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

        result = await workflow.run(
            prompt="Should we adopt Kubernetes?",
            temperature=0.7
        )

        if result.success:
            print(result.synthesis)
        else:
            # Handle workflow failure
            print(f"Workflow failed: {result.error}")
            # Retry with different parameters, log error, etc.

    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        # Check API keys, provider availability, network, etc.
```

---

## Advanced Features

### Custom System Prompts

Add context or constraints to guide the analysis:

```python
result = await workflow.run(
    prompt="Should we implement feature X?",
    system_prompt="""
    Context:
    - Small team (5 developers)
    - Limited budget ($50k)
    - 6-month timeline
    - Must integrate with existing legacy systems

    Focus on practical feasibility and resource constraints.
    """
)
```

### Temperature Tuning

Balance analytical rigor with creative perspectives:

```python
# Low temperature (0.3-0.5): More focused, analytical debate
result = await workflow.run(
    prompt="Evaluate this security architecture",
    temperature=0.4
)

# Medium temperature (0.6-0.8): Balanced analysis (recommended)
result = await workflow.run(
    prompt="Should we migrate to microservices?",
    temperature=0.7
)

# High temperature (0.9-1.0): More creative, exploratory perspectives
result = await workflow.run(
    prompt="How can we innovate our product strategy?",
    temperature=0.9
)
```

### Multi-Turn Deep Dives

Use continuation to explore specific aspects:

```python
# Initial broad analysis
result1 = await workflow.run(
    prompt="Cloud-native architecture vs traditional deployment"
)
thread = result1.metadata['thread_id']

# Deep dive into specific concerns
result2 = await workflow.run(
    prompt="Focus specifically on cost implications",
    continuation_id=thread
)

result3 = await workflow.run(
    prompt="What about team skills and training?",
    continuation_id=thread
)

result4 = await workflow.run(
    prompt="Security and compliance considerations?",
    continuation_id=thread
)
```

### Comparing Multiple Providers

Get diverse perspectives by running with different providers:

```python
from model_chorus.providers import ClaudeProvider, GeminiProvider

providers = [
    ("Claude", ClaudeProvider()),
    ("Gemini", GeminiProvider()),
]

prompt = "Is serverless architecture suitable for our use case?"

for name, provider in providers:
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)
    result = await workflow.run(prompt=prompt)

    print(f"\n=== {name} Analysis ===")
    print(result.synthesis)
```

---

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | The argument/claim to analyze |
| `temperature` | `float` | `0.7` | Creativity level (0.0-1.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `system_prompt` | `str` | `None` | Additional context or constraints |
| `continuation_id` | `str` | `None` | Thread ID to continue previous analysis |
| `files` | `List[str]` | `[]` | File paths for context |

### Temperature Guidelines

| Range | Use Case | Characteristics |
|-------|----------|----------------|
| **0.0-0.4** | Technical analysis, security reviews | Focused, conservative, analytical |
| **0.5-0.7** | General arguments, policy analysis | Balanced, recommended for most uses |
| **0.8-1.0** | Creative proposals, brainstorming | Exploratory, unconventional perspectives |

### Provider-Specific Notes

**Claude:**
- Best for nuanced reasoning and balanced synthesis
- Recommended temperature: 0.6-0.8

**Gemini:**
- Strong analytical capabilities
- Recommended temperature: 0.5-0.7

**Codex:**
- Excellent for technical/code-related arguments
- Recommended temperature: 0.4-0.7

---

## Best Practices

### 1. Frame Arguments Clearly

**Good:**
```bash
# Clear, specific claim
model-chorus argument "Adopting TypeScript would improve code quality in our project"
```

**Less Effective:**
```bash
# Vague, open-ended question
model-chorus argument "What do you think about TypeScript?"
```

### 2. Provide Context When Needed

**Without context:**
```python
result = await workflow.run(
    prompt="Should we use Redis?"
)
```

**With helpful context:**
```python
result = await workflow.run(
    prompt="Should we use Redis for session storage?",
    system_prompt="We have 10k daily active users, need < 50ms latency, budget $200/month",
    files=["docs/current_architecture.md"]
)
```

### 3. Use Continuation for Deep Dives

Don't cram everything into one prompt:

```python
# Better: Start broad, then drill down
result1 = await workflow.run("Microservices vs monolith for our scale")
thread = result1.metadata['thread_id']

result2 = await workflow.run("Focus on operational complexity", continuation_id=thread)
result3 = await workflow.run("What about team organization?", continuation_id=thread)
```

### 4. Match Temperature to Use Case

```python
# Security analysis: be conservative
await workflow.run("Is this auth approach secure?", temperature=0.4)

# Product strategy: be creative
await workflow.run("How should we position our product?", temperature=0.9)
```

### 5. Clean Up Threads

Archive or delete completed conversation threads:

```python
# When analysis is complete
memory.archive_thread(thread_id)

# Or delete if no longer needed
memory.delete_thread(thread_id)
```

### 6. Handle Both Success and Failure

```python
result = await workflow.run(prompt="...")

if result.success:
    # Use the analysis
    print(result.synthesis)
else:
    # Handle failure gracefully
    logging.error(f"Analysis failed: {result.error}")
    # Retry, use fallback, notify user, etc.
```

---

## Common Use Patterns

### Pattern 1: Decision Support

```python
async def evaluate_decision(decision: str, context: dict):
    """Get structured analysis for a decision."""
    workflow = ArgumentWorkflow(provider, memory)

    system_prompt = f"""
    Context: {context['background']}
    Constraints: {context['constraints']}
    Timeline: {context['timeline']}
    """

    result = await workflow.run(
        prompt=decision,
        system_prompt=system_prompt,
        temperature=0.6
    )

    return {
        'creator_case': result.steps[0].content,
        'skeptic_critique': result.steps[1].content,
        'recommendation': result.synthesis
    }
```

### Pattern 2: Policy Analysis

```python
async def analyze_policy(policy_statement: str, supporting_docs: List[str]):
    """Analyze policy with evidence from documents."""
    workflow = ArgumentWorkflow(provider, memory)

    result = await workflow.run(
        prompt=policy_statement,
        files=supporting_docs,
        temperature=0.6
    )

    # Extract structured insights
    return {
        'benefits': extract_benefits(result.steps[0].content),
        'risks': extract_risks(result.steps[1].content),
        'conclusion': result.synthesis
    }
```

### Pattern 3: Iterative Refinement

```python
async def iterative_analysis(initial_prompt: str, follow_ups: List[str]):
    """Conduct multi-turn deep analysis."""
    workflow = ArgumentWorkflow(provider, memory)

    # Initial analysis
    result = await workflow.run(prompt=initial_prompt)
    thread_id = result.metadata['thread_id']

    results = [result]

    # Follow-up analyses
    for follow_up in follow_ups:
        result = await workflow.run(
            prompt=follow_up,
            continuation_id=thread_id
        )
        results.append(result)

    return results
```

---

## Troubleshooting

### Issue: Biased or One-Sided Output

**Symptoms:** Skeptic doesn't challenge strongly, or Moderator favors one side

**Solutions:**
```python
# Adjust temperature slightly higher
result = await workflow.run(prompt="...", temperature=0.8)

# Add explicit instruction
result = await workflow.run(
    prompt="...",
    system_prompt="Ensure Skeptic provides strong counterarguments and Moderator stays neutral"
)

# Try different provider (Claude often best for balanced analysis)
provider = ClaudeProvider()
```

### Issue: Responses Too Short or Shallow

**Symptoms:** Brief responses without depth

**Solutions:**
```python
# Increase max_tokens
result = await workflow.run(prompt="...", max_tokens=3000)

# Provide more context
result = await workflow.run(
    prompt="...",
    system_prompt="Provide detailed analysis with specific examples and evidence",
    files=["relevant_docs.md"]
)
```

### Issue: Lost Thread Context

**Symptoms:** Continuation doesn't reference previous analysis

**Solutions:**
```python
# Check thread ID is correct
print(f"Thread ID: {result.metadata['thread_id']}")

# Verify thread hasn't expired (default 72hr TTL)
threads = memory.list_threads()
print(threads)

# Check conversation history
history = memory.get_thread_history(thread_id)
print(f"Messages in thread: {len(history)}")
```

### Issue: Provider Errors

**Symptoms:** "Provider failed" or "API error" messages

**Solutions:**
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY

# Verify provider CLI is working
claude --version
gemini --version

# Test provider directly
claude "test message"

# Try different provider
model-chorus argument "test" --provider gemini
```

### Issue: High Costs

**Symptoms:** Expensive for frequent use

**Solutions:**
```python
# Use cheaper providers (Gemini Flash, Claude Haiku)
provider = GeminiProvider()  # Often cheaper than Claude

# Lower temperature (reduces token variance)
result = await workflow.run(prompt="...", temperature=0.5)

# Limit max_tokens
result = await workflow.run(prompt="...", max_tokens=1500)

# Use CHAT for simpler questions
# Only use ARGUMENT when dialectical analysis truly needed
```

---

## Comparison with Other Workflows

### ARGUMENT vs CHAT

| Aspect | ARGUMENT | CHAT |
|--------|----------|------|
| **Structure** | Three roles (Creator/Skeptic/Moderator) | Simple back-and-forth |
| **Best for** | Analyzing arguments, balanced perspectives | Quick questions, iteration |
| **Cost** | Higher (3 responses per query) | Lower (1 response per query) |
| **Output** | Pro/Con/Synthesis | Direct answer |
| **Use when** | Need balanced analysis of a claim | Need quick consultation |

**Example:**
```bash
# Use ARGUMENT for debate
model-chorus argument "Should we adopt GraphQL?"

# Use CHAT for quick question
model-chorus chat "What is GraphQL?" -p claude
```

### ARGUMENT vs CONSENSUS

| Aspect | ARGUMENT | CONSENSUS |
|--------|----------|-----------|
| **Models** | Single model (3 roles) | Multiple models |
| **Perspectives** | Pro/Con from same model | Different model perspectives |
| **Best for** | Dialectical reasoning | Multi-model comparison |
| **Threading** | Yes | No |
| **Use when** | Analyze arguments structurally | Compare model approaches |

**Example:**
```bash
# Use ARGUMENT for structured debate
model-chorus argument "Microservices vs monolith for our scale"

# Use CONSENSUS for multi-model perspectives
model-chorus consensus "Microservices vs monolith for our scale" \
  -p claude -p gemini -p codex -s synthesize
```

### ARGUMENT vs THINKDEEP

| Aspect | ARGUMENT | THINKDEEP |
|--------|----------|-----------|
| **Process** | Dialectical debate | Systematic investigation |
| **Structure** | Pro/Con/Synthesis | Hypothesis testing |
| **Best for** | Evaluating claims | Debugging, research |
| **Confidence** | Not tracked | Tracked (exploring → certain) |
| **Use when** | Need balanced argument analysis | Need evidence-based investigation |

**Example:**
```bash
# Use ARGUMENT for claim analysis
model-chorus argument "Remote work increases productivity"

# Use THINKDEEP for investigation
model-chorus thinkdeep "Why is our API slow?" -f src/api.py
```

---

## Real-World Examples

### Example 1: Technology Decision

```bash
model-chorus argument "Our team should migrate from REST to GraphQL" \
  --file docs/current_api.md \
  --file docs/client_requirements.md \
  --temperature 0.7
```

**Expected output:**
- **Creator:** Benefits of GraphQL (precise queries, type safety, single endpoint)
- **Skeptic:** Challenges (caching complexity, migration cost, learning curve)
- **Moderator:** Balanced recommendation based on team size, timeline, constraints

### Example 2: Policy Analysis

```bash
model-chorus argument "Universal healthcare would improve public health outcomes" \
  --file research/healthcare_studies.pdf \
  --file research/cost_analysis.xlsx \
  --temperature 0.6
```

**Expected output:**
- **Creator:** Evidence for improved health outcomes, reduced inequality
- **Skeptic:** Cost concerns, implementation challenges, funding mechanisms
- **Moderator:** Nuanced analysis of trade-offs and conditions for success

### Example 3: Code Review Alternative

```bash
model-chorus argument "This refactoring improves maintainability without bugs" \
  --file src/old_implementation.py \
  --file src/new_implementation.py \
  --file tests/test_suite.py \
  --temperature 0.5
```

**Expected output:**
- **Creator:** Code quality improvements, better structure
- **Skeptic:** Potential edge cases, performance concerns, test coverage gaps
- **Moderator:** Assessment of refactoring quality and recommended next steps

---

## See Also

**Related Documentation:**
- [WORKFLOWS.md](../WORKFLOWS.md) - Complete workflow comparison guide
- [CHAT Workflow](CHAT.md) - Simple single-model conversation (coming soon)
- [CONSENSUS Workflow](CONSENSUS.md) - Multi-model perspectives (coming soon)
- [THINKDEEP Workflow](THINKDEEP.md) - Systematic investigation (coming soon)
- [CONVERSATION_INFRASTRUCTURE.md](../CONVERSATION_INFRASTRUCTURE.md) - Threading and state management details

**Code Examples:**
- [examples/workflow_examples.py](../../examples/workflow_examples.py) - Complete Python code examples
- [examples/argument_basic.py](../../examples/) - Basic ARGUMENT usage patterns

**API Reference:**
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Complete API documentation
- [README.md](../../README.md) - Getting started guide

**Related Workflows:**
- Use **CHAT** for simple questions without dialectical analysis
- Use **CONSENSUS** when you need multiple model perspectives
- Use **THINKDEEP** for systematic investigation with hypothesis tracking
- Combine **ARGUMENT** + **CONSENSUS** for multi-model dialectical analysis
