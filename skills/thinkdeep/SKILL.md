---
name: thinkdeep
description: Extended reasoning with systematic investigation, hypothesis tracking, and confidence progression
---

# THINKDEEP

## Overview

The THINKDEEP workflow provides multi-step investigation capabilities with hypothesis tracking, evidence collection, and confidence progression. Unlike simple conversations, THINKDEEP maintains investigation state across turns, allowing systematic analysis of complex problems through hypothesis formation, testing, and refinement.

**Key Capabilities:**
- Multi-step investigation with state persistence across conversation turns
- Hypothesis tracking and evolution as evidence accumulates
- Confidence level progression (exploring � low � medium � high � very_high � almost_certain � certain)
- Investigation step history with findings recorded
- File examination tracking across the investigation
- Optional expert validation for hypothesis verification

**Use Cases:**
- Complex debugging scenarios requiring systematic hypothesis testing
- Architecture decisions needing evidence-based reasoning
- Security analysis with methodical investigation
- Root cause analysis with confidence tracking
- Performance optimization requiring step-by-step exploration

## When to Use

Use the THINKDEEP workflow when you need to:

- **Systematic investigation** - Complex problems requiring methodical, step-by-step analysis with hypothesis testing
- **Evidence accumulation** - Building confidence through progressive evidence gathering across multiple investigation steps
- **Hypothesis evolution** - Starting with initial theories and refining them based on findings
- **State persistence** - Multi-turn investigations where context and progress must be maintained
- **Confidence tracking** - Knowing how certain you are about conclusions as investigation progresses

## When NOT to Use

Avoid the THINKDEEP workflow when:

| Situation | Use Instead |
|-----------|-------------|
| Simple conversational queries | **CHAT** - Single-model conversation for straightforward questions |
| Need multiple model perspectives | **CONSENSUS** - Parallel multi-model consultation |
| Need structured debate | **ARGUMENT** - Dialectical analysis with creator/skeptic/moderator |
| Creative brainstorming | **IDEATE** - Structured idea generation |
| Comprehensive research with citations | **RESEARCH** - Systematic information gathering |

## Hypothesis Tracking

THINKDEEP maintains hypothesis state across the investigation, allowing theories to evolve as evidence is gathered.

### How Hypothesis Tracking Works

**Initial Step:**
- First prompt starts with "exploring" confidence
- Model forms initial hypothesis based on available information
- Investigation begins with this working theory

**Subsequent Steps:**
- Each turn adds evidence supporting or contradicting the hypothesis
- Hypothesis may be refined, replaced, or strengthened
- Multiple hypotheses can exist simultaneously
- Investigation state tracks all hypotheses and their evolution

**Hypothesis Evolution Example:**

```
Step 1: "Why is authentication failing?"
� Hypothesis: "Session token expiration causing failures"
� Confidence: LOW (initial theory)

Step 2: "Check token validation logic"
� Evidence: Token expiration logic correct, but timing issue found
� Updated Hypothesis: "Race condition in async token validation"
� Confidence: MEDIUM (supporting evidence found)

Step 3: "Examine async/await patterns in auth flow"
� Evidence: Missing await in validation, non-blocking check
� Confirmed Hypothesis: "Race condition - token validated before expiry check completes"
� Confidence: HIGH (root cause identified with evidence)
```

### State Persistence

- Hypotheses are stored in conversation memory
- Thread ID links all investigation steps
- Each turn builds on previous findings
- Investigation can be paused and resumed using continuation

## Confidence Progression

THINKDEEP tracks confidence levels that progress as evidence accumulates, helping you understand how certain the conclusions are.

### Confidence Levels

| Level | Code | Description | When to Use |
|-------|------|-------------|-------------|
| **Exploring** | `exploring` | Just starting, no clear hypothesis | Initial investigation phase |
| **Low** | `low` | Early investigation, hypothesis forming | First theories with minimal evidence |
| **Medium** | `medium` | Some supporting evidence found | Partial evidence supporting hypothesis |
| **High** | `high` | Strong evidence supporting hypothesis | Substantial evidence, likely correct |
| **Very High** | `very_high` | Very strong evidence, high confidence | Comprehensive evidence, minimal doubt |
| **Almost Certain** | `almost_certain` | Near complete confidence | Overwhelming evidence, virtually certain |
| **Certain** | `certain` | 100% confidence, hypothesis validated | Hypothesis proven beyond reasonable doubt |

### Confidence Progression Patterns

**Typical Investigation Flow:**

```
exploring � low � medium � high � very_high � almost_certain
```

**Quick Resolution:**
```
exploring � low � high (direct evidence found immediately)
```

**Complex Investigation:**
```
exploring � low � medium (blocked) � low (new hypothesis) � medium � high � very_high
```

**Confidence can decrease** if:
- New evidence contradicts current hypothesis
- Investigation reveals complexity not initially apparent
- Hypothesis needs significant revision

### Using Confidence Levels

**For AI Agents:**
- **exploring/low**: Continue investigating, gather more evidence
- **medium**: Validate findings, look for confirming/contradicting evidence
- **high**: Consider hypothesis likely correct, verify edge cases
- **very_high/almost_certain**: Hypothesis well-supported, ready for conclusion
- **certain**: Hypothesis validated, investigation complete

**When to Stop Investigating:**
- Confidence reaches `high` or above AND sufficient evidence collected
- All relevant files examined and findings documented
- Hypothesis explains all observed behavior
- No contradicting evidence remains

### Expert Validation (Optional)

THINKDEEP supports optional expert validation using a secondary AI model:

**How it works:**
- Primary provider conducts investigation
- Expert provider (optional) reviews hypothesis and evidence
- Expert provides independent assessment of confidence level
- Helps verify conclusions and identify blind spots

**When to use expert validation:**
- High-stakes decisions requiring verification
- Complex investigations where confirmation valuable
- Security analysis requiring independent review
- Architecture decisions benefiting from second opinion

## Investigation Continuation

THINKDEEP supports multi-turn investigations where you can pause, resume, and build upon previous work using the continuation_id parameter.

### How Continuation Works

**continuation_id Parameter:**
- Unique thread identifier linking investigation steps
- Preserves full conversation history and context
- Maintains hypothesis state, findings, and confidence levels
- Enables seamless resumption across sessions

**State Preservation:**
- All previous findings and evidence
- Hypothesis evolution history
- Files checked and examined
- Confidence progression
- Investigation step history

### Continuation Patterns

**Pattern 1: Multi-Step Investigation**

```bash
# Step 1: Initial investigation
modelchorus thinkdeep \
  --model gpt5 \
  --step "Investigate authentication failures in production" \
  --step-number 1 \
  --total-steps 3 \
  --next-step-required true \
  --findings "Examining auth service logs..." \
  --confidence exploring

# Returns: continuation_id = "auth-inv-abc123"

# Step 2: Continue investigation with same thread
modelchorus thinkdeep \
  --continuation-id "auth-inv-abc123" \
  --model gpt5 \
  --step "Check token validation logic based on log findings" \
  --step-number 2 \
  --total-steps 3 \
  --next-step-required true \
  --findings "Found race condition in async token validation" \
  --confidence medium

# Step 3: Final analysis
modelchorus thinkdeep \
  --continuation-id "auth-inv-abc123" \
  --model gpt5 \
  --step "Verify race condition hypothesis with code analysis" \
  --step-number 3 \
  --total-steps 3 \
  --next-step-required false \
  --findings "Confirmed: missing await in auth middleware" \
  --confidence high
```

**Pattern 2: Investigation Branching**

Start a new investigation branch while preserving original:

```bash
# Original investigation continues...
modelchorus thinkdeep \
  --continuation-id "original-thread-xyz" \
  ...

# Branch to explore alternative hypothesis
# (Omit continuation-id to start fresh branch)
modelchorus thinkdeep \
  --model gpt5 \
  --step "Explore alternative: network latency causing timeouts" \
  --findings "Investigating network layer..." \
  --confidence low
```

**Pattern 3: Cross-Session Resume**

Resume investigation after break or context reset:

```bash
# Original session (day 1)
modelchorus thinkdeep \
  --step "Analyze memory leak in service" \
  --findings "Found increasing heap usage over 24h" \
  --confidence medium
# Returns continuation_id: "mem-leak-xyz789"

# Resume next session (day 2)
# continuation_id preserves all previous context
modelchorus thinkdeep \
  --continuation-id "mem-leak-xyz789" \
  --step "Trace heap allocations to identify source" \
  --findings "Identified unclosed database connections" \
  --confidence high
```

### State Inspection

**What Gets Preserved:**

| State Element | Preserved? | Notes |
|---------------|------------|-------|
| Hypothesis history | ✅ Yes | All hypotheses and revisions |
| Findings | ✅ Yes | Complete findings log |
| Files checked | ✅ Yes | Full file examination history |
| Confidence levels | ✅ Yes | Progression tracking |
| Step history | ✅ Yes | All investigation steps |
| Model context | ✅ Yes | Full conversation thread |

**What Gets Reset:**

When starting a new investigation (no continuation_id):
- Hypothesis state (starts fresh)
- Findings log (empty)
- Confidence level (begins at exploring)
- Files checked (new list)
- Step counter (resets to 1)

### Best Practices

**When to Use Continuation:**
- ✅ Multi-step investigation requiring state persistence
- ✅ Building on previous findings
- ✅ Resuming after pause or break
- ✅ Complex analysis spanning multiple sessions
- ✅ Hypothesis refinement based on new evidence

**When to Start Fresh:**
- ✅ Completely different investigation
- ✅ Previous hypothesis proven wrong (branch instead)
- ✅ Investigation scope changed significantly
- ✅ Context window becoming saturated
- ✅ Need clean slate for new approach

### Continuation Examples

**Example 1: Debugging Race Condition**

```python
# Step 1: Initial symptoms
{
  "step": "Users report intermittent 401 errors",
  "step_number": 1,
  "total_steps": 4,
  "next_step_required": True,
  "findings": "Error rate: 5% of requests, no pattern found in logs",
  "confidence": "exploring",
  "hypothesis": "Unknown cause - investigate auth flow"
}
# Returns: continuation_id = "auth-race-001"

# Step 2: Investigate auth flow
{
  "continuation_id": "auth-race-001",  # Links to step 1
  "step": "Examine token validation sequence",
  "step_number": 2,
  "total_steps": 4,
  "next_step_required": True,
  "findings": "Token checked before async validation completes",
  "confidence": "medium",
  "hypothesis": "Race condition in token validation"
}

# Step 3: Verify hypothesis
{
  "continuation_id": "auth-race-001",  # Links to steps 1-2
  "step": "Trace async execution order",
  "step_number": 3,
  "total_steps": 4,
  "next_step_required": True,
  "findings": "Missing await causes request to proceed before validation",
  "confidence": "high",
  "hypothesis": "Confirmed: race condition due to missing await"
}

# Step 4: Verify fix
{
  "continuation_id": "auth-race-001",  # Links to steps 1-3
  "step": "Verify adding await resolves issue",
  "step_number": 4,
  "total_steps": 4,
  "next_step_required": False,
  "findings": "With await added, validation completes before auth check",
  "confidence": "very_high",
  "hypothesis": "Root cause: missing await in middleware"
}
```

**Example 2: Architecture Decision**

```python
# Step 1: Analyze requirements
{
  "step": "Should we use microservices or monolith?",
  "findings": "Team size: 5 devs, expected scale: 10k users",
  "confidence": "exploring",
  "hypothesis": "Need to evaluate tradeoffs"
}
# Returns: continuation_id = "arch-decision-002"

# Step 2: Continue analysis
{
  "continuation_id": "arch-decision-002",
  "step": "Evaluate team experience and deployment complexity",
  "findings": "Team has limited k8s experience, deployment simplicity important",
  "confidence": "medium",
  "hypothesis": "Monolith may be better fit for team/scale"
}

# Step 3: Final recommendation
{
  "continuation_id": "arch-decision-002",
  "step": "Consider future scaling and migration path",
  "findings": "Can start monolith, extract services later if needed",
  "confidence": "high",
  "hypothesis": "Monolith is optimal: simpler ops, team fit, migration path exists"
}
```
