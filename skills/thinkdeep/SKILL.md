---
name: model-chorus:thinkdeep
description: Performs multi-stage investigation and reasoning for complex problem analysis, architecture decisions, performance challenges, and security analysis
---

# ModelChorus: ThinkDeep

Systematic, multi-step investigation and deep reasoning workflow for complex technical problems that require careful analysis, hypothesis testing, and expert validation.

## When to Use

Use this skill when you need:
- **Architecture decisions** - Evaluating design patterns and trade-offs
- **Complex bug investigation** - Root cause analysis requiring multiple steps
- **Performance challenges** - Systematic performance analysis and optimization
- **Security analysis** - Comprehensive security review and threat modeling
- **Technology evaluation** - Comparing technologies with deep analysis
- **Code archaeology** - Understanding unfamiliar or complex codebases

## How It Works

ThinkDeep uses a structured investigation workflow:

1. **Hypothesis Formation** - Generate initial theories about the problem
2. **Evidence Gathering** - Systematically collect data and test hypotheses
3. **Iterative Refinement** - Adjust theories based on findings
4. **Expert Validation** - Optional external model validates conclusions
5. **Comprehensive Report** - Detailed findings with recommendations

The workflow tracks confidence levels and adjusts investigation depth automatically.

## Usage

Invoke ThinkDeep with your investigation goal:

```
Use ThinkDeep to investigate:
[Problem description or question]

Context:
- Relevant files: [file paths]
- Focus areas: [architecture/performance/security/etc.]
- Confidence target: [high/very_high/certain]
```

## Parameters

- **step** (required): Current investigation step description
- **step_number** (required): Current step number (starts at 1)
- **total_steps** (required): Estimated total steps needed
- **next_step_required** (required): True if more investigation needed
- **findings** (required): Discoveries and evidence from this step
- **model** (required): AI model to use for investigation

**Optional:**
- **hypothesis**: Current theory based on evidence
- **confidence**: exploring, low, medium, high, very_high, almost_certain, certain
- **files_checked**: Absolute paths of examined files
- **relevant_files**: Files directly relevant to the issue
- **relevant_context**: Methods/functions involved
- **issues_found**: List of issues with severity levels
- **focus_areas**: Specific aspects to emphasize
- **problem_context**: Additional background information
- **continuation_id**: Resume previous investigation
- **temperature**: 0=deterministic, 1=creative
- **thinking_mode**: minimal, low, medium, high, max

## Examples

### Example 1: Architecture Decision

```
ThinkDeep Investigation:

Step 1: "Analyzing caching strategies for high-traffic API"
Total steps: 4
Next step required: true

Problem context: "API serves 50k req/sec, current in-memory cache
causing memory issues. Need to evaluate Redis vs Memcached vs CDN."

Focus areas: ["performance", "scalability", "cost"]

Model: gemini-2.5-pro
Thinking mode: high
```

### Example 2: Performance Investigation

```
ThinkDeep Investigation:

Step 1: "Identifying bottleneck in slow database queries"
Total steps: 5
Next step required: true

Findings: "Initial profiling shows queries taking 2+ seconds.
Suspect missing indexes on user_events table."

Hypothesis: "Missing composite index on (user_id, created_at)"

Relevant files:
- /home/user/project/db/models/user_event.py
- /home/user/project/api/events.py

Confidence: medium
Model: gpt-5-codex
```

### Example 3: Security Analysis

```
ThinkDeep Investigation:

Step 1: "Reviewing authentication implementation for vulnerabilities"
Total steps: 6
Next step required: true

Problem context: "OAuth2 implementation with JWT tokens.
Need comprehensive security review before production deployment."

Focus areas: ["security", "authentication"]

Files checked:
- /home/user/project/auth/oauth.py
- /home/user/project/auth/tokens.py
- /home/user/project/middleware/auth.py

Model: gpt-5-pro
Thinking mode: max
```

### Example 4: Continuing Investigation

```
ThinkDeep Investigation (Continued):

Continuation ID: abc123

Step 3: "Testing hypothesis about missing index"
Total steps: 5
Next step required: true

Findings: "Added EXPLAIN ANALYZE - confirmed sequential scan.
Index would reduce query time from 2.1s to 0.03s."

Hypothesis: "Composite index (user_id, created_at) will solve issue"

Confidence: high
```

## Confidence Levels

The workflow tracks confidence as investigation progresses:

- **exploring** - Just starting, initial observations
- **low** - Early investigation, few data points
- **medium** - Some evidence gathered, pattern emerging
- **high** - Strong evidence supporting hypothesis
- **very_high** - Comprehensive understanding, validated findings
- **almost_certain** - Near complete confidence in conclusion
- **certain** - 100% confidence, no external validation needed

## Investigation Workflow

**Step 1: Problem Framing**
- State the problem clearly
- Define success criteria
- Identify initial hypotheses

**Steps 2-N: Investigation**
- Gather evidence systematically
- Test hypotheses
- Update confidence levels
- Adjust course based on findings

**Final Step: Conclusions**
- Summarize findings
- Provide recommendations
- Document confidence level
- Note any limitations

## Best Practices

**DO:**
- ✅ Start with clear problem statement
- ✅ Track all examined files
- ✅ Update hypothesis based on evidence
- ✅ Use specific focus areas
- ✅ Document findings thoroughly
- ✅ Adjust total_steps as you learn more

**DON'T:**
- ❌ Skip evidence gathering steps
- ❌ Jump to conclusions without data
- ❌ Ignore contradicting evidence
- ❌ Use "certain" confidence prematurely
- ❌ Forget to track relevant files

## Model Selection

**Recommended models:**
- `gemini-2.5-pro` - Best for deep analysis (1M context)
- `gpt-5-pro` - Best for complex reasoning (400K context)
- `gpt-5-codex` - Best for code analysis (400K context)

## Expert Validation

By default, ThinkDeep uses external model validation. Set `use_assistant_model: false` to skip external validation and rely solely on your investigation.

## Integration with ModelChorus

This skill uses the `mcp__zen__thinkdeep` tool from the Zen MCP server, providing structured workflows for complex problem analysis.

## See Also

- **debug** - For specific bug debugging workflows
- **codereview** - For comprehensive code reviews
- **chat** - For quick questions without full investigation
- **planner** - For breaking down implementation plans
