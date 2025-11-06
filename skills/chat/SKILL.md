---
name: model-chorus:chat
description: Single-model conversational peer consultation with threading and context continuity for rapid iteration and second opinions
---

# ModelChorus: Chat

Engage in contextual, threaded conversations with a peer AI model for rapid iteration, second opinions, and focused sub-task delegation. Think of this as tapping a colleague on the shoulder for a quick question or brainstorming session.

## When to Use

Use this skill when you need:

### Seeking a Second Opinion
- Validating a solution before implementing
- Checking for flaws, alternatives, or improvements
- Quick sanity check between larger workflow steps
- **Trigger phrases:** "Is this the best way?", "Can you double-check this?", "What's another perspective on this?"

### Iterative Refinement
- Back-and-forth adjustments on code, UI descriptions, or creative writing
- Maintaining discussion thread where context continuity matters
- Building on previous exchanges without re-explaining context
- **Trigger phrases:** "That's close, but can you make it more...", "Let's try another approach", "Continue from where we left off"

### Clarifying Ambiguity
- User's request is unclear and needs exploration
- Multiple possible interpretations exist
- Need to understand requirements before diving deep
- **Trigger phrases:** Broad requests like "build me an app", "help me with this problem"

### Specialist Consultation
- Need expertise outside your core domain
- Specialized model would provide superior response
- Want consistent tone/voice from a specific model
- **Trigger phrases:** Niche technologies, specialized domains, specific model preferences

## How It Works

The `chat` workflow initiates a threaded session with a target model:

1. **Send prompt** - Your question or task to the peer model
2. **Receive response** - Model's answer with context awareness
3. **Continue thread** - Use `--continue` with thread ID to maintain conversation
4. **Build context** - Each turn remembers previous exchanges

**Key Features:**
- Session-based threading with unique thread IDs
- Rolling context window maintains conversation history
- Lightweight branching for testing alternatives
- Auto-summarization after extended exchanges

## Usage

```bash
modelchorus chat [prompt] \
  --provider [model] \
  --continue [thread-id] \
  [options]
```

## Parameters

### Required
- **prompt** (positional argument): The message or question to send to the peer model

### Optional
- `--provider, -p`: Model to chat with
  - Default: `gemini`
  - Options: `gemini`, `claude`, `codex`, `cursor-agent`
  - Choose based on model strengths (reasoning, code, creativity)

- `--continue`: Thread ID to continue existing conversation
  - Omit to start new conversation
  - Use same ID to maintain context across turns
  - Format: UUID string returned from previous interaction

- `--output, -o`: Save conversation to JSON file
  - Useful for documentation or later reference
  - Contains full thread history and metadata

- `--verbose, -v`: Show detailed execution information
  - Displays provider status, timing, token usage
  - Helpful for debugging or understanding model behavior

- `--temperature, -t`: Control response creativity (0.0-1.0)
  - Default: `0.7`
  - Lower for factual/deterministic responses
  - Higher for creative/exploratory discussions

- `--max-tokens`: Maximum tokens in response
  - Useful for controlling response length
  - Prevents overly verbose answers

- `--timeout`: Request timeout in seconds
  - Default: `120.0`
  - Increase for complex queries

## Strategic Examples

### Example 1: Iterative Code Review

**Goal:** Review and refine a Python function with context continuity

```bash
# Turn 1: Initial Review
modelchorus chat "You are a Python expert. Please review the following function for correctness, efficiency, and PEP8 adherence:

\`\`\`python
def factorial(n):
  if n == 0:
    return 1
  else:
    return n * factorial(n-1)
\`\`\`" \
  --provider gemini \
  --output factorial-review.json

# Returns: thread-id: abc-123-def-456
# Response: Provides feedback on recursion depth, suggests iterative alternative

# Turn 2: Follow-up Question
modelchorus chat "That's a great point about recursion limits. Could you rewrite using an iterative approach?" \
  --provider gemini \
  --continue abc-123-def-456

# Response: Provides refactored iterative version
```

### Example 2: Creative Brainstorming

**Goal:** Generate and refine project names

```bash
# Initial brainstorm
modelchorus chat "Let's brainstorm names for a CLI tool that orchestrates AI models. Core concepts: collaboration, harmony, intelligence. Give me 5 initial ideas." \
  --provider codex \
  --temperature 0.9

# Refine favorites
modelchorus chat "I like 'ModelChorus' and 'SynapseHub'. Can you create variations on these themes?" \
  --provider codex \
  --continue [thread-id] \
  --temperature 0.8
```

### Example 3: Debug Session

**Goal:** Diagnose intermittent bug with peer assistance

```bash
# Start investigation
modelchorus chat "I'm seeing intermittent 503 errors. Here are the symptoms: [symptoms]. What should I investigate first?" \
  --provider claude \
  --output debug-session.json

# Follow up with findings
modelchorus chat "I checked the database connections - pool is at 95% capacity. What's the likely root cause?" \
  --provider claude \
  --continue [thread-id]

# Get specific recommendation
modelchorus chat "Confirmed: missing index on users table. What's the best way to add this index with minimal downtime?" \
  --provider claude \
  --continue [thread-id]
```

### Example 4: Requirements Clarification

**Goal:** Understand ambiguous user request before implementing

```bash
# Explore interpretations
modelchorus chat "User asked to 'add authentication'. What are the most common interpretations and implementation approaches?" \
  --provider gemini

# Narrow down based on context
modelchorus chat "It's a microservices architecture with existing OAuth setup. What's the recommended approach?" \
  --provider gemini \
  --continue [thread-id]
```

## Best Practices

### DO:
- ✅ **Be specific in prompting** - Clearly define role and expectations ("You are a security expert analyzing code for vulnerabilities")
- ✅ **Manage session context** - Use `--continue` for related exchanges, start fresh for unrelated topics
- ✅ **Keep turns focused** - One clear question or instruction per turn
- ✅ **Choose the right peer** - Select models based on known strengths (gemini for reasoning, codex for code)
- ✅ **Seed first turn well** - Include both the question and desired end state to minimize back-and-forth
- ✅ **Save important sessions** - Use `--output` to preserve valuable conversations
- ✅ **Adjust temperature** - Lower for facts (0.3-0.5), higher for creativity (0.8-0.9)

### DON'T:
- ❌ **Mix unrelated topics** - Start new conversation rather than contaminating context
- ❌ **Forget thread IDs** - You'll lose conversation history without them
- ❌ **Bundle multiple queries** - Keep each turn focused on one topic
- ❌ **Skip role definition** - Models perform better with clear expertise framing
- ❌ **Let threads drift** - Periodically restate the objective if conversation wanders
- ❌ **Ignore temperature** - Default may not be optimal for your use case

## Workflow Integration

### Handoff to THINKDEEP
When a chat reveals complexity requiring structured investigation:

```bash
# Chat uncovers need for deep analysis
modelchorus chat "This caching strategy has multiple trade-offs..." \
  --provider gemini \
  --output chat-cache-discussion.json

# Escalate to THINKDEEP for systematic analysis
modelchorus thinkdeep "Analyze caching strategy trade-offs discovered in chat-cache-discussion.json" \
  --provider gemini
```

### Handoff from CONSENSUS
Use chat to explore consensus findings in depth:

```bash
# Run consensus on question
modelchorus consensus "Best database for real-time analytics?" \
  -p gemini -p codex \
  --output db-consensus.json

# Deep dive with one model
modelchorus chat "Based on db-consensus.json, let's explore the PostgreSQL + TimescaleDB approach in detail" \
  --provider gemini
```

## Error Handling

### Model Not Responding
- **Symptom:** Request times out or fails
- **Action:** Retry once, then try different provider
- **Command:** `--timeout 300` to increase wait time

### Invalid Thread ID
- **Symptom:** Error about expired or invalid session
- **Action:** Context is lost; start fresh conversation
- **Prevention:** Save thread IDs; don't wait too long between turns

### Unexpected Output
- **Symptom:** Response is irrelevant or nonsensical
- **Action:** Terminate session and retry with more precise prompt
- **Tip:** Add specific instructions like "Format response as bullet points"

### Context Drift
- **Symptom:** Model loses focus or misunderstands objective
- **Action:** Restate the main goal explicitly
- **Command:** Start new message with "To clarify, our objective is..."

### Hallucinated Information
- **Symptom:** Model invents facts, APIs, or code that don't exist
- **Action:** Request citations and verify manually
- **Prevention:** Lower temperature, ask for sources

## Output Format

**Terminal Output:**
```
Chat with gemini...
Prompt: Review this Python function for...

[Response from model]

Thread ID: abc-123-def-456
✓ Success (2.3s, 450 tokens)
```

**JSON Output (with --output):**
```json
{
  "thread_id": "abc-123-def-456",
  "provider": "gemini",
  "turns": [
    {
      "role": "user",
      "content": "Review this Python function...",
      "timestamp": "2025-11-06T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "...",
      "model": "gemini-2.5-pro",
      "usage": {"input_tokens": 120, "output_tokens": 450},
      "timestamp": "2025-11-06T10:30:02Z"
    }
  ],
  "metadata": {
    "temperature": 0.7,
    "total_tokens": 570
  }
}
```

## Performance

- **Typical latency:** 1-5 seconds depending on model and response length
- **Context window:** Models maintain ~10-20 previous turns
- **Auto-summary:** Triggered after ~15 exchanges to compress context
- **Concurrency:** One turn at a time per thread (sequential)

## Limitations

- Thread persistence limited to session duration (implementation-dependent)
- Context window may truncate very long conversations
- Model-specific rate limits apply
- Threading requires unique ID management
- No built-in conversation branching (use separate threads)

## Provider Selection Guide

| Provider | Best For | Strengths |
|----------|----------|-----------|
| `gemini` | General reasoning, fast responses | Balanced performance, good context retention |
| `claude` | Code review, analysis, writing | Deep reasoning, nuanced understanding |
| `codex` | Code generation, debugging | Technical accuracy, API knowledge |
| `cursor-agent` | IDE integration, refactoring | Context from development environment |

## See Also

- **THINKDEEP skill** - For complex multi-step investigations
- **CONSENSUS skill** - For multi-model parallel consultation
- **ModelChorus CLI** - Run `modelchorus --help` for all options
