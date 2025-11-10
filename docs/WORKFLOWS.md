# ModelChorus Workflows Guide

Complete guide to choosing and using ModelChorus workflows for different use cases.

---

## Quick Reference

| Workflow | Type | Models | Best For | Conversation | State |
|----------|------|--------|----------|--------------|-------|
| **CONSENSUS** | Multi-model | 2-4 providers | Multiple perspectives, robust answers | ❌ No | ❌ Stateless |
| **CHAT** | Single-model | 1 provider | Quick consultations, iterative refinement | ✅ Yes | ✅ Stateful |
| **THINKDEEP** | Single-model + Expert | 1-2 providers | Complex investigation, hypothesis testing | ✅ Yes | ✅ Stateful |
| **ARGUMENT** | Single-model | 1 provider | Structured debate, balanced critique | ✅ Yes | ✅ Stateful |
| **IDEATE** | Multi-model | 1-4 providers | Creative brainstorming, idea generation | ✅ Yes | ✅ Stateful |

---

## Workflow Comparison

### CONSENSUS

**Multi-model consultation with synthesis**

**What it does:**
- Queries multiple AI models in parallel
- Applies consensus strategy (synthesize, majority, all_responses, etc.)
- Returns combined or selected results

**When to use:**
- Need multiple expert perspectives
- Want to reduce single-model bias
- Comparing how different models approach a problem
- Building robust answers from diverse viewpoints
- Making important decisions (architecture, technology choices)

**When NOT to use:**
- Need conversation continuity (use CHAT or THINKDEEP)
- Single model is sufficient
- Cost is a primary concern (calls multiple APIs)
- Quick iteration is needed (use CHAT)

**Example scenarios:**
```bash
# Architecture decision
model-chorus consensus "REST vs GraphQL for our API?" \
  -p claude -p gemini -p codex -s synthesize

# Code review from multiple perspectives
model-chorus consensus "Review this authentication implementation" \
  -p claude -p gemini -s all_responses

# Technology evaluation
model-chorus consensus "Should we use TypeScript or JavaScript?" \
  -p claude -p codex -s majority
```

**Provider defaults:** The consensus workflow honors provider-level model preferences defined in `.model-chorusrc`. For example:

```yaml
providers:
  gemini:
    model: gemini-2.5-pro
```

Running `model-chorus consensus "Explain quantum computing" -p claude -p gemini -v` now prints `Applied model override for gemini: gemini-2.5-pro`, and the underlying Gemini CLI call receives the `-m gemini-2.5-pro` flag automatically. Adjust models permanently in the config file or override them ad hoc with the CLI's `--provider/-p` options when you need a different combination for a single run.

**Strengths:**
- ✅ Multiple expert perspectives
- ✅ Reduces model bias
- ✅ Parallel execution (fast)
- ✅ Flexible consensus strategies

**Limitations:**
- ❌ No conversation continuity
- ❌ Higher API costs (multiple models)
- ❌ No state preservation
- ❌ Cannot build on previous context

---

### CHAT

**Simple single-model conversation**

**What it does:**
- Creates conversational thread with one AI model
- Maintains conversation history across turns
- Supports file context

**When to use:**
- Quick consultations and Q&A
- Iterative refinement of ideas
- Code reviews with follow-up questions
- Learning and exploration
- Building on previous responses

**When NOT to use:**
- Need systematic investigation (use THINKDEEP)
- Want multiple perspectives (use CONSENSUS)
- Solving complex multi-step problems (use THINKDEEP)

**Example scenarios:**
```bash
# Iterative API design
model-chorus chat "Design a user API" -p claude
# Thread: thread-001

model-chorus chat "Add rate limiting" --continue thread-001
model-chorus chat "Add pagination" --continue thread-001

# Code review with follow-ups
model-chorus chat "Review auth.py" -f src/auth.py -p claude
# Thread: thread-002

model-chorus chat "How would you refactor the token validation?" --continue thread-002

# Learning conversation
model-chorus chat "Explain async/await in Python" -p claude
# Thread: thread-003

model-chorus chat "Show me an example with error handling" --continue thread-003
```

**Strengths:**
- ✅ Conversation continuity
- ✅ Simple and fast
- ✅ Low cost (single model)
- ✅ Good for iteration
- ✅ File context support

**Limitations:**
- ❌ Single model perspective
- ❌ No systematic investigation structure
- ❌ No hypothesis tracking
- ❌ No confidence progression

---

### THINKDEEP

**Extended reasoning with systematic investigation**

**What it does:**
- Multi-step investigation with hypothesis tracking
- Maintains investigation state across turns
- Tracks confidence progression
- Optional expert validation from second model

**When to use:**
- Debugging complex issues
- Security analysis and threat modeling
- Architecture decisions requiring deep analysis
- Performance troubleshooting
- Any problem requiring systematic, evidence-based investigation

**When NOT to use:**
- Simple questions (use CHAT)
- Need multiple perspectives on same topic (use CONSENSUS)
- Quick answers without investigation
- Cost is primary concern and simple chat suffices

**Example scenarios:**
```bash
# Debug intermittent bug
model-chorus thinkdeep "Users report 500 errors intermittently" \
  -f src/api/users.py -f logs/errors.log \
  -p claude -e gemini
# Thread: thread-004

model-chorus thinkdeep "Found race condition pattern" --continue thread-004

# Security analysis
model-chorus thinkdeep "Analyze SQL injection vectors" \
  -f src/db/queries.py -f src/api/handlers.py \
  -p claude -e gemini
# Thread: thread-005

# Performance investigation
model-chorus thinkdeep "Response time degrades under load" \
  -f src/cache/redis.py -f monitoring/metrics.json \
  -p claude
# Thread: thread-006

# Check investigation progress
model-chorus thinkdeep-status thread-006 --steps --files
```

**Strengths:**
- ✅ Systematic investigation structure
- ✅ Hypothesis tracking and evolution
- ✅ Confidence progression (exploring → certain)
- ✅ Optional expert validation
- ✅ File examination tracking
- ✅ State persistence across sessions
- ✅ Prevents jumping to conclusions

**Limitations:**
- ❌ Higher overhead than simple chat
- ❌ May be overkill for simple problems
- ❌ Requires more tokens (investigation context)
- ❌ Expert validation adds cost (but optional)

---

## Decision Tree

### Which workflow should I use?

```
Start here:
├─ Do I need multiple model perspectives?
│  └─ YES → Use CONSENSUS
│     - Architecture decisions
│     - Technology evaluations
│     - Comparing approaches
│
└─ NO → Need conversation continuity?
   ├─ NO → Use CONSENSUS (one-off question)
   │
   └─ YES → Is this a complex investigation?
      ├─ YES → Use THINKDEEP
      │  - Debugging complex bugs
      │  - Security analysis
      │  - Performance troubleshooting
      │  - Systematic problem-solving
      │
      └─ NO → Use CHAT
         - Quick consultations
         - Iterative refinement
         - Learning conversations
         - Simple code reviews
```

---

## Advanced Patterns

### Pattern 1: Consensus → THINKDEEP → CHAT

**Use case:** Major decision requiring deep analysis and refinement

**Step 1: Get multiple perspectives (CONSENSUS)**
```bash
model-chorus consensus "Microservices vs Monolith for our scale?" \
  -p claude -p gemini -p codex -s synthesize
```

**Step 2: Deep investigation of chosen approach (THINKDEEP)**
```bash
model-chorus thinkdeep "Investigate microservices scalability patterns" \
  -p claude -e gemini
# Thread: thread-007
```

**Step 3: Refine implementation details (CHAT)**
```bash
model-chorus chat "Design service discovery mechanism" \
  -p claude
# Thread: thread-008

model-chorus chat "Add health checks" --continue thread-008
```

### Pattern 2: THINKDEEP with Expert Escalation

**Use case:** Start investigation, escalate to expert if stuck

**Step 1: Initial investigation**
```bash
model-chorus thinkdeep "Memory leak in production" \
  -f src/cache/manager.py \
  -p claude --disable-expert
# Thread: thread-009
```

**Step 2: Check confidence**
```bash
model-chorus thinkdeep-status thread-009
# Confidence: medium (not certain)
```

**Step 3: Continue with expert validation**
```bash
model-chorus thinkdeep "Found weak reference issues" \
  --continue thread-009 \
  -e gemini  # Now enable expert
```

### Pattern 3: CHAT → CONSENSUS for Validation

**Use case:** Iterative design, then validate with multiple models

**Step 1: Iterate design with CHAT**
```bash
model-chorus chat "Design caching strategy" -p claude
# Thread: thread-010

model-chorus chat "Add Redis tier" --continue thread-010
model-chorus chat "Add invalidation logic" --continue thread-010
```

**Step 2: Validate final design with CONSENSUS**
```bash
model-chorus consensus "Review this caching design: [paste from thread-010]" \
  -p claude -p gemini -p codex -s synthesize
```

### Pattern 4: Conversation Branching

**Use case:** Explore alternatives without losing main thread

```python
from model_chorus.core.conversation import ConversationMemory
from model_chorus.workflows import ChatWorkflow

memory = ConversationMemory()
chat = ChatWorkflow(provider, conversation_memory=memory)

# Main conversation
result1 = await chat.run("Design user authentication")
main_thread = result1.metadata['thread_id']

result2 = await chat.run("Add JWT tokens", continuation_id=main_thread)

# Branch to explore alternative
alt_thread = memory.create_thread(
    workflow_name="chat",
    parent_thread_id=main_thread,
    initial_context={"exploring": "OAuth alternative"}
)

result3 = await chat.run(
    "What if we use OAuth instead?",
    continuation_id=alt_thread
)

# Return to main thread
result4 = await chat.run(
    "Add refresh tokens",
    continuation_id=main_thread
)
```

### Pattern 5: THINKDEEP Hypothesis Management

**Use case:** Complex investigation with multiple competing hypotheses

```python
from model_chorus.workflows import ThinkDeepWorkflow

workflow = ThinkDeepWorkflow(provider, conversation_memory=memory)

# Start investigation
result1 = await workflow.run(
    "API latency spikes every hour",
    files=["src/api/server.py", "logs/performance.log"]
)
thread_id = result1.metadata['thread_id']

# Add multiple hypotheses
workflow.add_hypothesis(thread_id, "Hourly cron job causes contention")
workflow.add_hypothesis(thread_id, "Cache expiration causes thundering herd")
workflow.add_hypothesis(thread_id, "Connection pool exhaustion")

# Investigate and update
result2 = await workflow.run(
    "Found scheduled backup at :00 past each hour",
    continuation_id=thread_id
)

workflow.update_hypothesis(
    thread_id,
    "Hourly cron job causes contention",
    evidence="Backup job locks users table for 10-15 seconds"
)

# Validate hypothesis
workflow.validate_hypothesis(thread_id, "Hourly cron job causes contention")

# Disprove others
workflow.disprove_hypothesis(thread_id, "Cache expiration causes thundering herd")
workflow.disprove_hypothesis(thread_id, "Connection pool exhaustion")

# Check final state
state = workflow.get_investigation_state(thread_id)
print(f"Confidence: {state.current_confidence}")  # Should be "very_high" or "certain"
print(f"Validated hypotheses: {len([h for h in state.hypotheses if h.status == 'validated'])}")
```

---

## Cost Optimization

### By Workflow

**CONSENSUS:**
- Cost: **High** (N × single model call)
- Optimization: Use fewer providers, choose first_valid strategy for simple questions

**CHAT:**
- Cost: **Low** (single model, growing context per turn)
- Optimization: Clear threads when done, use context window limits

**THINKDEEP:**
- Cost: **Medium-High** (single model + optional expert, larger context)
- Optimization: Disable expert validation when not needed, use lower-cost models

### Cost-Saving Strategies

1. **Start simple, escalate as needed**
   - Start with CHAT for exploration
   - Escalate to THINKDEEP only if systematic investigation needed
   - Use CONSENSUS only for important decisions

2. **Choose providers wisely**
   - Use cost-effective providers (Gemini Flash, Claude Haiku)
   - Reserve expensive models (GPT-4, Claude Opus) for critical tasks

3. **Manage conversation context**
   - Complete and archive threads when done
   - Don't continue conversations indefinitely
   - Use focused prompts to reduce token usage

4. **Optimize CONSENSUS**
   - Use `first_valid` strategy for simple questions
   - Use 2 providers instead of 3-4 when sufficient
   - Reserve `synthesize` for complex decisions

---

## Best Practices

### General

1. **Choose the right workflow**
   - Don't use CONSENSUS when CHAT suffices
   - Don't use THINKDEEP for simple questions
   - Match workflow to problem complexity

2. **Use continuation_id consistently**
   - Save thread IDs for later resumption
   - Document what each thread is for
   - Clean up completed threads

3. **Leverage file context**
   - Include relevant files with `-f` flag
   - Keep file count reasonable (1-5 files)
   - Use focused file excerpts when possible

4. **Monitor costs**
   - Track API usage across workflows
   - Set budget alerts
   - Review usage patterns regularly

### CONSENSUS-specific

1. **Choose appropriate strategy**
   - `synthesize` for comprehensive answers
   - `majority` for clear yes/no questions
   - `all_responses` to see all perspectives
   - `first_valid` for simple questions

2. **Select complementary providers**
   - Mix models with different strengths
   - Claude for reasoning, Codex for code, Gemini for analysis

### CHAT-specific

1. **Keep conversations focused**
   - One topic per thread
   - Start new thread for new topics
   - Archive completed conversations

2. **Build context progressively**
   - Reference previous responses
   - Add files as conversation evolves
   - Use clear, specific follow-up questions

### THINKDEEP-specific

1. **Use hypothesis tracking**
   - Add hypotheses early
   - Update with evidence
   - Validate or disprove explicitly

2. **Track confidence progression**
   - Check confidence level regularly
   - Continue investigation until "high" or "very_high"
   - Use expert validation when stuck at "medium"

3. **Examine files systematically**
   - Start with likely suspects
   - Add files as investigation progresses
   - Review files_checked to avoid redundancy

4. **Enable expert validation strategically**
   - Use for critical investigations
   - Enable when confidence is stuck
   - Disable for cost-sensitive work

---

## Examples by Use Case

### Debugging

**Simple bug** → CHAT
```bash
model-chorus chat "Why is this function returning None?" \
  -f src/utils.py -p claude
```

**Complex bug** → THINKDEEP
```bash
model-chorus thinkdeep "Intermittent race condition in order processing" \
  -f src/orders/processor.py -f src/db/transactions.py \
  -p claude -e gemini
```

### Architecture Decisions

**Quick consultation** → CHAT
```bash
model-chorus chat "Should I use Redis or Memcached?" -p claude
```

**Important decision** → CONSENSUS
```bash
model-chorus consensus "Evaluate Redis vs Memcached for session storage" \
  -p claude -p gemini -p codex -s synthesize
```

**Deep analysis** → THINKDEEP
```bash
model-chorus thinkdeep "Analyze scalability of current caching architecture" \
  -f src/cache/ -p claude -e gemini
```

### Code Review

**Quick review** → CHAT
```bash
model-chorus chat "Review this function" -f src/auth.py -p claude
```

**Multiple perspectives** → CONSENSUS
```bash
model-chorus consensus "Review authentication implementation" \
  -f src/auth.py -p claude -p codex -s all_responses
```

**Security review** → THINKDEEP
```bash
model-chorus thinkdeep "Security audit of authentication system" \
  -f src/auth.py -f src/middleware/jwt.py \
  -p claude -e gemini
```

### Learning & Exploration

**Q&A** → CHAT
```bash
model-chorus chat "Explain async/await in Python" -p claude
# Thread: thread-020

model-chorus chat "Show example with error handling" --continue thread-020
```

**Compare approaches** → CONSENSUS
```bash
model-chorus consensus "Explain async/await patterns" \
  -p claude -p gemini -s all_responses
```

---

## Migration from Other Tools

### From Direct CLI Usage

**Before:**
```bash
claude "Explain quantum computing"
```

**After (CHAT):**
```bash
model-chorus chat "Explain quantum computing" -p claude
# Gets conversation tracking + state management
```

### From Sequential Model Calls

**Before:**
```bash
claude "Design API" > claude.txt
gemini "Design API" > gemini.txt
codex "Design API" > codex.txt
# Manually compare outputs
```

**After (CONSENSUS):**
```bash
model-chorus consensus "Design API" \
  -p claude -p gemini -p codex -s synthesize
# Automatic synthesis
```

### From Manual Investigation

**Before:**
```bash
# Multiple separate Claude calls
claude "Analyze bug in auth.py"
claude "Check database queries"
claude "Review error logs"
# Manually track findings
```

**After (THINKDEEP):**
```bash
model-chorus thinkdeep "Authentication bug investigation" \
  -f src/auth.py -f logs/errors.log \
  -p claude
# Automatic hypothesis tracking + confidence progression
```

---

## Troubleshooting

### CONSENSUS Issues

**All providers fail**
- Check API keys (`echo $ANTHROPIC_API_KEY`)
- Verify CLI tools installed
- Test each provider individually

**Inconsistent results**
- Increase temperature for more variation
- Try different consensus strategies
- Add system prompt for context

### CHAT Issues

**Lost thread context**
- Check thread ID is correct
- Verify conversation not expired (72hr default TTL)
- Check `~/.model-chorus/conversations/` directory

**Context too long**
- Clear conversation and start fresh
- Reduce message history limit
- Use more focused prompts

### THINKDEEP Issues

**Confidence not progressing**
- Add more evidence and files
- Enable expert validation
- Update hypotheses with findings

**Investigation too long**
- Focus on specific hypothesis
- Narrow file examination scope
- Use more specific prompts

---

## See Also

- [DOCUMENTATION.md](DOCUMENTATION.md) - Complete API reference
- [CONVERSATION_INFRASTRUCTURE.md](CONVERSATION_INFRASTRUCTURE.md) - Conversation system details
- [../examples/](../examples/) - Code examples
- [../README.md](../README.md) - Getting started guide
