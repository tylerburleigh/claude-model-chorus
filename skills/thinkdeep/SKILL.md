---
name: model-chorus:thinkdeep
description: Structured multi-step investigation with hypothesis tracking for complex problem analysis and systematic reasoning
---

# ModelChorus: ThinkDeep

Conduct structured, multi-step investigations into complex problems with systematic analysis, hypothesis formulation and testing, and evidence tracking. This is your tool for deep, methodical thinking when simple answers are insufficient.

## When to Use

Use this skill for complex, analytical tasks that require systematic investigation:

### Root Cause Analysis
- Diagnosing problems with unknown causes
- Debugging elusive bugs or performance issues
- Investigating production incidents or system failures
- **Trigger phrases:** "Why is my app crashing?", "What's causing this error?", "Debug this issue for me"

### Complex System Design
- Designing non-trivial architectures
- Breaking down requirements and exploring trade-offs
- Evaluating multiple architectural approaches
- **Trigger phrases:** "Design an architecture for...", "How would I build a system that...", "Plan the implementation of..."

### Strategic Planning
- Creating detailed implementation plans
- Exploring multiple scenarios or approaches
- Producing comprehensive analysis reports
- **Trigger phrases:** "Create a test plan for...", "Outline the steps to migrate...", "Compare technologies X and Y"

### Decision Making
- Architecture decisions requiring structured analysis
- Technology evaluation with explicit trade-offs
- Risk assessment and mitigation planning
- **Trigger phrases:** "Should we use X or Y?", "What are the pros and cons?", "Help me decide between..."

## How It Works

The `thinkdeep` workflow orchestrates a model through structured reasoning phases:

1. **Problem Framing** - Define the question and success criteria
2. **Hypothesis Generation** - Enumerate possible explanations or approaches
3. **Investigation Loop** - Systematically test each hypothesis
4. **Evidence Gathering** - Collect data to validate or invalidate hypotheses
5. **Synthesis** - Draw conclusions based on accumulated evidence
6. **Report Generation** - Produce structured decision document

**Key Features:**
- Hypothesis lifecycle tracking (`proposed → examining → confirmed/rejected/pending`)
- Evidence logging for audit trail
- Structured reasoning trace across sessions
- Automatic sanity checks and counterexample generation
- Digestible decision reports

## Usage

```bash
modelchorus thinkdeep [problem] \
  --provider [model] \
  --continue [thread-id] \
  [options]
```

## Parameters

### Required
- **problem** (positional argument): The central question or issue to investigate

### Optional
- `--provider, -p`: Model for deep reasoning
  - Default: `gemini`
  - Recommended: Use powerful reasoning models
  - Options: `gemini`, `claude`, `codex`

- `--continue`: Thread ID to continue existing investigation
  - Maintains hypothesis tracking across sessions
  - Preserves evidence and reasoning history
  - Omit to start fresh investigation

- `--expert, -e`: Optional expert model for validation
  - Provides second opinion on conclusions
  - Can challenge assumptions
  - Useful for critical decisions

- `--output, -o`: Save investigation ledger to file
  - JSON format with full reasoning trace
  - Includes hypotheses, evidence, conclusions
  - Useful for documentation and audit

- `--verbose, -v`: Show detailed reasoning steps
  - Displays hypothesis states
  - Shows evidence collection
  - Helpful for understanding process

- `--max-depth`: Maximum reasoning iterations
  - Default: `5`
  - Prevents runaway loops
  - Balance depth vs. time

- `--confidence-target`: Required confidence level (0.0-1.0)
  - Default: `0.8`
  - Higher values demand more rigor
  - Lower for exploratory analysis

- `--timeout`: Request timeout in seconds
  - Default: `300.0` (5 minutes)
  - Increase for complex investigations

## Strategic Examples

### Example 1: Root Cause Analysis

**Goal:** Diagnose intermittent 503 Service Unavailable errors

```bash
# Turn 1: Initialize investigation
modelchorus thinkdeep "PROBLEM: Intermittent 503 errors on web server.
CONTEXT: Errors occur during peak traffic, no pattern in logs.
OBJECTIVE: Identify root cause and recommend fix.

Please formulate 3 primary hypotheses for investigation." \
  --provider gemini \
  --output rca-503.json

# Response generates hypotheses:
# 1. Database connection pool exhaustion
# 2. Upstream API timeout
# 3. Web server resource overload

# Turn 2: Investigate first hypothesis
# (After checking DB connections show 95% capacity)
modelchorus thinkdeep "EVIDENCE COLLECTED: Database connection pool at 95% capacity during incidents.

Update Hypothesis #1 status and suggest next investigation step." \
  --provider gemini \
  --continue [thread-id] \
  --output rca-503.json

# Turn 3: Conclude investigation
# (After confirming missing index on users table)
modelchorus thinkdeep "CONFIRMED: Missing index on users.email causing slow queries and pool exhaustion.

Synthesize findings into final report with:
- Root cause explanation
- Supporting evidence
- Recommended solution
- Implementation steps" \
  --provider gemini \
  --continue [thread-id] \
  --output rca-503.json \
  --expert claude
```

### Example 2: Architecture Decision

**Goal:** Choose caching strategy for high-traffic API

```bash
# Initialize decision analysis
modelchorus thinkdeep "DECISION: Select caching strategy for REST API serving 10k req/sec.
CONSTRAINTS:
- Budget: $5k/month max
- Latency: p99 <50ms
- Data freshness: 5-minute staleness acceptable

Enumerate caching approaches and evaluation criteria." \
  --provider gemini \
  --max-depth 6 \
  --output cache-decision.json

# Evaluate each approach
modelchorus thinkdeep "Analyze Redis cluster approach:
- Cost estimation
- Performance characteristics
- Operational complexity
- Failure scenarios

Compare against constraints." \
  --provider gemini \
  --continue [thread-id]

# Final decision with expert validation
modelchorus thinkdeep "Based on analysis, recommend best approach with:
- Decision matrix
- Risk assessment
- Implementation checklist
- Monitoring strategy" \
  --provider gemini \
  --continue [thread-id] \
  --expert claude \
  --output cache-decision.json
```

### Example 3: Migration Planning

**Goal:** Plan database migration from MySQL to PostgreSQL

```bash
modelchorus thinkdeep "OBJECTIVE: Plan migration from MySQL 8.0 to PostgreSQL 15.
SCOPE: 500GB database, 24/7 uptime required.
CONSTRAINTS:
- Zero data loss
- Max downtime: 15 minutes
- Rollback plan required

Create migration strategy with phases, risks, and validation steps." \
  --provider gemini \
  --max-depth 8 \
  --confidence-target 0.9 \
  --output mysql-pg-migration.json
```

### Example 4: Incident Postmortem

**Goal:** Comprehensive postmortem analysis

```bash
modelchorus thinkdeep "INCIDENT ANALYSIS: Complete outage on 2025-11-05 14:30-16:45 UTC.
SYMPTOMS: All services unresponsive, database deadlocks in logs.
TIMELINE: [attach detailed timeline]

Conduct systematic postmortem:
1. Identify all contributing factors
2. Establish causal chain
3. Validate hypotheses against logs
4. Recommend preventive measures
5. Define follow-up actions

Use evidence-based reasoning throughout." \
  --provider gemini \
  --max-depth 10 \
  --expert claude \
  --output postmortem-2025-11-05.json
```

## Best Practices

### DO:
- ✅ **Frame problem precisely** - Clear objective, constraints, success criteria
- ✅ **Start with hypotheses** - Enumerate possibilities before investigating
- ✅ **One step at a time** - Each prompt should focus on discrete analytical step
- ✅ **Log evidence explicitly** - "Add finding:", "Evidence supports:", "Contradicts:"
- ✅ **Use external tools** - Gather data with other commands, feed into thinkdeep
- ✅ **Request structured output** - Specify format (matrix, timeline, decision tree)
- ✅ **Challenge assumptions** - Ask model to question its own reasoning
- ✅ **Involve expert** - Use `--expert` for critical decisions
- ✅ **Archive findings** - Save ledgers for institutional knowledge
- ✅ **Start broad, narrow** - Generate many hypotheses, systematically eliminate

### DON'T:
- ❌ **Skip problem definition** - Vague goals lead to unfocused analysis
- ❌ **Rush to conclusions** - Ensure adequate evidence before deciding
- ❌ **Ignore contradictions** - Address conflicting evidence explicitly
- ❌ **Let model drift** - Keep focused on stated objective
- ❌ **Forget state management** - Update hypothesis status deliberately
- ❌ **Mix investigations** - One thread per problem
- ❌ **Exceed depth limit** - If stuck, reframe or break into subtasks
- ❌ **Skip validation** - Critical decisions need expert review

## Hypothesis Management

### Lifecycle States
```
PROPOSED     → Initial hypothesis generated
EXAMINING    → Currently under investigation
CONFIRMED    → Evidence supports hypothesis
REJECTED     → Evidence contradicts hypothesis
PENDING      → Needs more evidence
```

### Explicit State Updates
```bash
# Update hypothesis status clearly
"Hypothesis #1: CONFIRMED - Database pool exhaustion verified"
"Hypothesis #2: REJECTED - No evidence of API timeouts"
"Hypothesis #3: PENDING - Need to check memory metrics"
"New Hypothesis #4: PROPOSED - SSL handshake overhead"
```

## Workflow Integration

### From CHAT
Escalate when conversation reveals complexity:

```bash
# Chat uncovers multi-faceted problem
modelchorus chat "This caching issue has performance and consistency trade-offs..." \
  --provider gemini \
  --output cache-chat.json

# Escalate to THINKDEEP for systematic analysis
modelchorus thinkdeep "Systematically analyze caching trade-offs from cache-chat.json" \
  --provider gemini
```

### To CONSENSUS
Validate conclusions with multiple models:

```bash
# Complete THINKDEEP investigation
modelchorus thinkdeep "..." \
  --output architecture-analysis.json

# Get consensus on recommendation
modelchorus consensus "Review architecture-analysis.json and validate recommendation" \
  -p gemini -p claude -p codex \
  -s synthesize
```

### With External Tools
Integrate data gathering:

```bash
# Formulate hypothesis
modelchorus thinkdeep "Generate hypotheses for high memory usage"

# Gather evidence (external tool)
ps aux --sort=-%mem | head -20 > memory-report.txt

# Feed back to investigation
modelchorus thinkdeep "Evidence from memory-report.txt shows Python process at 85% memory. Update Hypothesis #1 status." \
  --continue [thread-id]
```

## Error Handling

### Reasoning Loops
- **Symptom:** Model reiterates without progress
- **Action:** Increase `--confidence-target` or inject fresh evidence
- **Command:** Add "Summarize current state and propose 3 alternative next steps"

### Corrupted State
- **Symptom:** Investigation ledger becomes invalid
- **Action:** No recovery - must restart investigation
- **Prevention:** Save ledger frequently with `--output`

### Contradictory Evidence
- **Symptom:** Evidence supports multiple conflicting hypotheses
- **Action:** Mark hypotheses as `PENDING` and request more data
- **Escalation:** Use `--expert` for second opinion

### Depth Exceeded
- **Symptom:** Max iterations reached without conclusion
- **Action:** Summarize findings and identify blocking questions
- **Option:** Break into sub-investigations or escalate to human

### Model Refusal
- **Symptom:** Model cannot fulfill reasoning step
- **Action:** Rephrase with more context or break into smaller steps
- **Fallback:** Try different provider with `--provider`

## Output Format

**Terminal Output:**
```
ThinkDeep Investigation...
Problem: Diagnose 503 errors
Provider: gemini

[Structured reasoning output]

Hypotheses:
  #1: Database pool exhaustion [EXAMINING]
  #2: Upstream timeout [PROPOSED]
  #3: Resource overload [PROPOSED]

Thread ID: xyz-789-abc-123
✓ Investigation step complete (5.2s)
```

**JSON Ledger (with --output):**
```json
{
  "investigation_id": "xyz-789-abc-123",
  "problem": "Diagnose 503 errors",
  "provider": "gemini",
  "hypotheses": [
    {
      "id": 1,
      "description": "Database connection pool exhaustion",
      "status": "CONFIRMED",
      "evidence": [
        "Pool at 95% capacity during incidents",
        "Missing index on users.email causing slow queries"
      ],
      "confidence": 0.95
    },
    {
      "id": 2,
      "description": "Upstream API timeout",
      "status": "REJECTED",
      "evidence": ["API logs show normal response times"],
      "confidence": 0.05
    }
  ],
  "conclusions": {
    "root_cause": "Missing database index causing pool exhaustion",
    "recommendation": "Add index on users.email column",
    "confidence": 0.95
  },
  "reasoning_trace": [
    {
      "step": 1,
      "action": "Formulate hypotheses",
      "output": "...",
      "timestamp": "2025-11-06T10:30:00Z"
    }
  ],
  "metadata": {
    "total_steps": 3,
    "depth": 3,
    "expert_validation": true
  }
}
```

## Reasoning Patterns

### Pattern 1: Hypothesis-Driven Investigation
```
1. Formulate 3-5 testable hypotheses
2. Prioritize by likelihood/impact
3. Test highest-priority first
4. Update hypothesis states based on evidence
5. Repeat until one confirmed or all rejected
```

### Pattern 2: Comparative Analysis
```
1. Enumerate options (technologies, approaches)
2. Define evaluation criteria
3. Score each option against criteria
4. Create decision matrix
5. Recommend based on weighted scores
```

### Pattern 3: Root Cause Drill-Down
```
1. Identify symptom
2. Ask "Why does this occur?" → hypothesis
3. Gather evidence
4. If confirmed, ask "Why does this cause occur?"
5. Repeat until root cause found
```

### Pattern 4: Risk Assessment
```
1. Identify potential failure scenarios
2. Estimate probability and impact
3. Classify by severity (critical/high/medium/low)
4. Propose mitigations
5. Prioritize by risk reduction ROI
```

## Performance

- **Typical duration:** 5-30 minutes depending on complexity
- **Iteration depth:** Usually 3-7 steps for most investigations
- **Context retention:** Full reasoning trace preserved
- **Concurrency:** Sequential (one step at a time)
- **Token usage:** Higher than chat due to reasoning depth

## Limitations

- Cannot execute code or gather data directly (use external tools)
- Requires explicit state management between steps
- Investigation quality depends on evidence quality
- Time-intensive for simple problems (use CHAT instead)
- Hypothesis tracking requires structured prompting
- No automatic rollback if investigation goes off-track

## Provider Selection Guide

| Provider | Best For | Reasoning Style |
|----------|----------|-----------------|
| `gemini` | General investigation, system design | Balanced, thorough |
| `claude` | Code analysis, nuanced decisions | Deep, contextual |
| `codex` | Technical debugging, API evaluation | Technical, precise |

**Expert Validation:**
Use different provider as expert to get diverse perspective (e.g., `--provider gemini --expert claude`)

## See Also

- **CHAT skill** - For quick iterations before deep investigation
- **CONSENSUS skill** - For validating THINKDEEP conclusions
- **ModelChorus CLI** - Run `modelchorus --help` for all options
