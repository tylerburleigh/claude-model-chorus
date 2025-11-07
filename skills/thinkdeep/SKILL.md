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

## Detailed Use Cases

### Debugging Scenarios

**Race Conditions and Timing Issues:**
- Intermittent bugs that don't reproduce consistently
- Authentication failures with no clear pattern
- State management issues in async code
- Example: "Users report random 401 errors, 5% of requests fail but logs show valid tokens"

**Memory Leaks and Resource Issues:**
- Gradually increasing memory usage
- Connection pool exhaustion
- File descriptor leaks
- Example: "Service memory grows from 200MB to 2GB over 24 hours, then OOM crashes"

**Integration Failures:**
- API calls failing with unclear errors
- Database connection issues
- Third-party service timeouts
- Example: "Payment processing fails 10% of the time with 'transaction timeout' but payment provider shows success"

**Configuration Problems:**
- Environment-specific bugs
- Deployment issues
- Feature flag interactions
- Example: "Feature works in staging but fails in production, configuration appears identical"

### Investigation Scenarios

**Performance Analysis:**
- Response time degradation
- Query performance issues
- CPU/memory spikes under load
- Example: "API latency increased from 100ms to 2s after deployment, need to identify bottleneck"

**Security Analysis:**
- Vulnerability assessment
- Access control verification
- Input validation checking
- Example: "Audit authentication flow for potential bypass vulnerabilities before security review"

**Architecture Decisions:**
- Technology selection (microservices vs monolith)
- Database choice (SQL vs NoSQL)
- Caching strategy evaluation
- Example: "Team of 5 developers, 10k users expected, evaluate architecture tradeoffs"

**Root Cause Analysis:**
- Production incident investigation
- Cascading failure analysis
- Error spike investigation
- Example: "500 error rate spiked from 0.1% to 15% at 2pm, trace back to root cause"

### When to Use THINKDEEP for These Scenarios

**Characteristics that indicate THINKDEEP is the right choice:**
- ✅ Problem requires multiple investigation steps
- ✅ Initial hypothesis may be wrong and need refinement
- ✅ Evidence needs to accumulate across steps
- ✅ Confidence level matters for decision-making
- ✅ Investigation may span multiple sessions
- ✅ Need to track what's been examined

**Example decision:**
```
Problem: "Users report intermittent 401 errors"

Simple fix? → NO (intermittent, unclear cause)
Need hypothesis testing? → YES (multiple possible causes)
Multi-step investigation? → YES (need to examine logs, code, flow)
Confidence tracking valuable? → YES (want to know how certain about root cause)

Decision: Use THINKDEEP ✓
```

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

THINKDEEP supports multi-turn investigations where you can pause, resume, and build upon previous work using the session_id parameter.

### How Continuation Works

**session_id Parameter:**
- Unique session identifier linking investigation steps
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

Step 1 - Initial investigation:
```bash
modelchorus thinkdeep --provider claude --step "Investigate authentication failures in production" --step-number 1 --total-steps 3 --next-step-required true --findings "Examining auth service logs..." --confidence exploring
```
Returns: `session_id = "auth-inv-abc123"`

Step 2 - Continue with same thread:
```bash
modelchorus thinkdeep --session-id "auth-inv-abc123" --provider claude --step "Check token validation logic based on log findings" --step-number 2 --total-steps 3 --next-step-required true --findings "Found race condition in async token validation" --confidence medium
```

Step 3 - Final analysis:
```bash
modelchorus thinkdeep --session-id "auth-inv-abc123" --provider claude --step "Verify race condition hypothesis with code analysis" --step-number 3 --total-steps 3 --next-step-required false --findings "Confirmed: missing await in auth middleware" --confidence high
```

**Pattern 2: Investigation Branching**

Start a new investigation branch while preserving original:

Original investigation continues:
```bash
modelchorus thinkdeep --session-id "original-thread-xyz" --step "Continue original investigation..." --step-number 4 --total-steps 5 --next-step-required true --findings "..." --confidence medium
```

New branch (omit continuation-id to start fresh):
```bash
modelchorus thinkdeep --provider claude --step "Explore alternative: network latency causing timeouts" --step-number 1 --total-steps 2 --next-step-required true --findings "Investigating network layer..." --confidence low
```

**Pattern 3: Cross-Session Resume**

Day 1 - Original session:
```bash
modelchorus thinkdeep --step "Analyze memory leak in service" --step-number 1 --total-steps 3 --next-step-required true --findings "Found increasing heap usage over 24h" --confidence medium
```
Returns: `session_id = "mem-leak-xyz789"`

Day 2 - Resume with preserved context:
```bash
modelchorus thinkdeep --session-id "mem-leak-xyz789" --step "Trace heap allocations to identify source" --step-number 2 --total-steps 3 --next-step-required true --findings "Identified unclosed database connections" --confidence high
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

When starting a new investigation (no session_id):
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

Step 1 - Initial symptoms:
```bash
modelchorus thinkdeep --step "Users report intermittent 401 errors" --step-number 1 --total-steps 4 --next-step-required true --findings "Error rate: 5% of requests, no pattern found in logs" --confidence exploring --hypothesis "Unknown cause - investigate auth flow"
```
Returns: `session_id = "auth-race-001"`

Step 2 - Investigate auth flow:
```bash
modelchorus thinkdeep --session-id "auth-race-001" --step "Examine token validation sequence" --step-number 2 --total-steps 4 --next-step-required true --findings "Token checked before async validation completes" --confidence medium --hypothesis "Race condition in token validation"
```

Step 3 - Verify hypothesis:
```bash
modelchorus thinkdeep --session-id "auth-race-001" --step "Trace async execution order" --step-number 3 --total-steps 4 --next-step-required true --findings "Missing await causes request to proceed before validation" --confidence high --hypothesis "Confirmed: race condition due to missing await"
```

Step 4 - Verify fix:
```bash
modelchorus thinkdeep --session-id "auth-race-001" --step "Verify adding await resolves issue" --step-number 4 --total-steps 4 --next-step-required false --findings "With await added, validation completes before auth check" --confidence very_high --hypothesis "Root cause: missing await in middleware"
```

**Example 2: Architecture Decision**

Step 1 - Analyze requirements:
```bash
modelchorus thinkdeep --step "Should we use microservices or monolith?" --step-number 1 --total-steps 3 --next-step-required true --findings "Team size: 5 devs, expected scale: 10k users" --confidence exploring --hypothesis "Need to evaluate tradeoffs"
```
Returns: `session_id = "arch-decision-002"`

Step 2 - Continue analysis:
```bash
modelchorus thinkdeep --session-id "arch-decision-002" --step "Evaluate team experience and deployment complexity" --step-number 2 --total-steps 3 --next-step-required true --findings "Team has limited k8s experience, deployment simplicity important" --confidence medium --hypothesis "Monolith may be better fit for team/scale"
```

Step 3 - Final recommendation:
```bash
modelchorus thinkdeep --session-id "arch-decision-002" --step "Consider future scaling and migration path" --step-number 3 --total-steps 3 --next-step-required false --findings "Can start monolith, extract services later if needed" --confidence high --hypothesis "Monolith is optimal: simpler ops, team fit, migration path exists"
```

## Basic Usage

### Simple Example

Basic investigation with single step:

```bash
modelchorus thinkdeep --step "Investigate why API latency increased from 100ms to 2s" --step-number 1 --total-steps 1 --next-step-required false --findings "Need to analyze recent deployment changes" --confidence exploring
```

### Common Options

**Required Parameters:**
- `--step`: Investigation step description
- `--step-number`: Current step index (starts at 1)
- `--total-steps`: Estimated total investigation steps
- `--next-step-required`: Whether more steps are needed (true/false)
- `--findings`: What was discovered in this step

**Optional Parameters:**
- `--provider`: AI provider to use (claude, gemini, codex, cursor-agent; default: claude)
- `--session-id`: Resume previous investigation
- `--hypothesis`: Current working theory
- `--confidence`: Confidence level (exploring, low, medium, high, very_high, almost_certain, certain)
- `--files-checked`: List of files examined
- `--temperature`: Creativity level (0.0-1.0, default: 0.7)
- `--thinking-mode`: Reasoning depth (minimal, low, medium, high, max)

## Advanced Usage

### With Model Selection

Specify which AI model to use for the investigation:

```bash
modelchorus thinkdeep --provider claude --step "Analyze security vulnerability in auth flow" --step-number 1 --total-steps 3 --next-step-required true --findings "Reviewing authentication middleware" --confidence exploring
```

### With File Context

Include specific files relevant to the investigation:

```bash
modelchorus thinkdeep --step "Debug race condition in token validation" --step-number 2 --total-steps 3 --next-step-required true --findings "Found async timing issue" --confidence medium --files-checked "src/auth/middleware.ts,src/services/token.ts"
```

### Multi-Step Investigation

Conduct systematic multi-step investigation with continuation:

```bash
modelchorus thinkdeep --step "Initial analysis of memory leak" --step-number 1 --total-steps 4 --next-step-required true --findings "Heap growing 50MB/hour" --confidence exploring
```

Then continue:

```bash
modelchorus thinkdeep --session-id "RETURNED_ID" --step "Trace allocation sources" --step-number 2 --total-steps 4 --next-step-required true --findings "Database connection pool not releasing" --confidence medium
```

### Adjusting Reasoning Depth

Control how deeply the model thinks about the problem:

```bash
modelchorus thinkdeep --step "Complex architectural decision" --step-number 1 --total-steps 1 --next-step-required false --findings "Need thorough analysis" --confidence exploring --thinking-mode max
```

Options: `minimal`, `low`, `medium` (default), `high`, `max`

### Expert Validation

Get independent expert review of investigation findings:

```bash
modelchorus thinkdeep --step "Final hypothesis verification" --step-number 3 --total-steps 3 --next-step-required false --findings "Root cause identified" --confidence high --use-assistant-model true
```

## Best Practices

### Investigation Planning

**Start with clear problem statement:**
- ✅ "Users report intermittent 401 errors, 5% failure rate, no pattern in logs"
- ❌ "Auth is broken"

**Estimate steps realistically:**
- Simple bugs: 1-2 steps
- Medium complexity: 3-5 steps
- Complex investigations: 6-10 steps
- Adjust `total-steps` as investigation progresses

**Use appropriate confidence levels:**
- Start at `exploring` or `low`
- Progress through evidence accumulation
- Reach `high` or `very_high` before concluding
- Use `certain` only when hypothesis is proven

### Hypothesis Management

**Form specific hypotheses:**
- ✅ "Race condition in async token validation due to missing await"
- ❌ "Something wrong with auth"

**Revise based on evidence:**
- Update hypothesis when new evidence contradicts it
- Track hypothesis evolution through investigation
- Document why hypothesis changed

**Test hypotheses systematically:**
- Identify what evidence would support/refute
- Gather that specific evidence
- Evaluate fairly (avoid confirmation bias)

### State Management

**Use continuation for multi-step work:**
- Always save session_id from first step
- Pass to subsequent steps to maintain state
- Enables cross-session resume

**Track files examined:**
- List all files checked in `--files-checked`
- Prevents re-examining same files
- Shows investigation coverage

**Document findings clearly:**
- Be specific about what was found
- Include relevant details (error messages, patterns, metrics)
- Note what was ruled out

### When to Stop Investigating

**Stop when:**
- Confidence reaches `high` or above AND hypothesis explains all evidence
- All relevant areas examined
- Cost of further investigation exceeds value
- Need input from domain expert or stakeholder

**Don't stop when:**
- Confidence still at `low` or `medium` with unanswered questions
- Hypothesis doesn't explain all observed behavior
- Contradicting evidence exists
- Investigation feels incomplete

## Examples

### Example 1: Quick Bug Investigation

Problem: Users seeing 500 errors on checkout

```bash
modelchorus thinkdeep --step "Investigate 500 errors in checkout flow" --step-number 1 --total-steps 1 --next-step-required false --findings "Error: 'payment_processor timeout'. Third-party API latency spike to 30s." --confidence high --hypothesis "Payment provider experiencing outage, not our bug"
```

Single-step investigation, clear finding, high confidence → done.

### Example 2: Multi-Step Performance Investigation

Step 1 - Initial analysis:
```bash
modelchorus thinkdeep --step "API latency increased from 100ms to 2s after deployment" --step-number 1 --total-steps 3 --next-step-required true --findings "Latency affects all endpoints equally, started at 3pm deployment" --confidence low --hypothesis "Deployment introduced performance regression"
```

Step 2 - Narrow down cause:
```bash
modelchorus thinkdeep --session-id "perf-inv-001" --step "Examine deployment changes" --step-number 2 --total-steps 3 --next-step-required true --findings "New logging middleware added, logs every request body. Bodies average 50KB." --confidence medium --hypothesis "Excessive logging causing I/O bottleneck"
```

Step 3 - Verify:
```bash
modelchorus thinkdeep --session-id "perf-inv-001" --step "Test hypothesis by disabling verbose logging" --step-number 3 --total-steps 3 --next-step-required false --findings "Latency drops to 120ms with logging disabled" --confidence very_high --hypothesis "Confirmed: verbose body logging causing 20x slowdown"
```

### Example 3: Security Audit

Investigation with expert validation:

```bash
modelchorus thinkdeep --provider claude --step "Audit authentication flow for bypass vulnerabilities" --step-number 1 --total-steps 2 --next-step-required true --findings "Token validation occurs before permission check. JWT expiry not verified in middleware." --confidence medium --hypothesis "Potential bypass: expired tokens may pass through"
```

Then verify with expert:
```bash
modelchorus thinkdeep --session-id "sec-audit-001" --provider claude --step "Verify vulnerability hypothesis" --step-number 2 --total-steps 2 --next-step-required false --findings "Confirmed: expired tokens accepted if permission check passes. Critical vulnerability." --confidence very_high --use-assistant-model true
```

## Troubleshooting

### Issue: Investigation feels stuck

**Symptoms:**
- Can't increase confidence past `low` or `medium`
- Hypothesis keeps changing without progress
- Findings don't lead anywhere

**Solutions:**
- Branch investigation: start new thread exploring alternative hypothesis
- Consult expert or stakeholder for domain knowledge
- Widen search: examine adjacent systems/layers
- Narrow focus: concentrate on one specific aspect
- Take break: resume with fresh perspective using session_id

### Issue: Confidence too low despite strong evidence

**Symptoms:**
- Strong evidence supporting hypothesis
- All tests pass, behavior explained
- But still feel uncertain

**Solutions:**
- Review all evidence systematically
- Check for contradicting evidence
- Verify hypothesis explains ALL observations
- Consider if seeking perfect certainty (which is rare)
- Use expert validation for independent assessment

### Issue: Multi-step investigation losing context

**Symptoms:**
- Later steps don't reference earlier findings
- Repeating work from previous steps
- Losing track of what's been examined

**Solutions:**
- Always use `--session-id` for multi-step investigations
- Include comprehensive findings in each step
- Reference earlier findings explicitly: "Building on Step 2's discovery of X..."
- Use `--files-checked` to track examination history
- Review session_id state before continuing

### Issue: Investigation taking too long

**Symptoms:**
- 10+ steps without resolution
- Total_steps keeps increasing
- Confidence not progressing

**Solutions:**
- Reassess hypothesis: may be fundamentally wrong
- Narrow scope: focus on specific sub-problem first
- Check if problem actually solvable through investigation
- Consider if need different approach (experiments, monitoring, etc.)
- Set confidence threshold for "good enough" conclusion

## Related Workflows

**For single-turn analysis:** Use **CHAT** - Simple conversational queries don't need multi-step investigation state.

**For multiple perspectives:** Use **CONSENSUS** - When you need diverse viewpoints on a problem rather than systematic investigation.

**For structured debate:** Use **ARGUMENT** - When pros/cons analysis more appropriate than hypothesis testing.

**For research:** Use **RESEARCH** - When gathering information is primary goal rather than solving specific problem.

**For brainstorming:** Use **IDEATE** - When generating ideas rather than investigating existing problem.

**Combining workflows:**
- Start with **THINKDEEP** to identify root cause
- Then use **CONSENSUS** to decide on solution approach
- Then use **CHAT** for implementation questions
- Use **RESEARCH** to gather context before investigation
