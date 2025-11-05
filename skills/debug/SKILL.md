---
name: model-chorus:debug
description: Systematic debugging and root cause analysis for any type of issue including complex bugs, mysterious errors, performance problems, race conditions, and memory leaks
---

# ModelChorus: Debug

Systematic debugging workflow with hypothesis testing and expert analysis for identifying and resolving bugs, errors, and unexpected behavior.

## When to Use

Use this skill when you need to:
- **Debug complex bugs** - Multi-layered issues requiring investigation
- **Investigate mysterious errors** - Intermittent or hard-to-reproduce problems
- **Performance debugging** - Memory leaks, slow operations, bottlenecks
- **Race conditions** - Concurrency issues and timing problems
- **Integration issues** - Problems across multiple systems or components
- **Root cause analysis** - Finding the real cause, not just symptoms

## How It Works

The debug workflow follows a structured approach:

1. **Problem Documentation** - Clearly state the issue and symptoms
2. **Hypothesis Formation** - Develop theories about root cause
3. **Evidence Collection** - Gather data to test hypotheses
4. **Systematic Testing** - Test each hypothesis methodically
5. **Root Cause Identification** - Pinpoint the actual cause
6. **Expert Validation** - Optional external model validates findings

The workflow tracks confidence and can conclude "no bug found" if the issue is a misunderstanding.

## Usage

Invoke Debug with your issue description:

```
Use Debug to investigate:
[Describe the bug, error, or unexpected behavior]

Steps to reproduce:
1. [Step 1]
2. [Step 2]
3. [Expected vs Actual]

Context:
- Relevant files: [file paths]
- Error messages: [any errors]
- Environment: [OS, versions, etc.]
```

## Parameters

- **step** (required): Investigation step description
- **step_number** (required): Current step number (starts at 1)
- **total_steps** (required): Estimated total investigation steps
- **next_step_required** (required): True if more investigation needed
- **findings** (required): Discoveries and evidence from this step
- **model** (required): AI model to use

**Optional:**
- **hypothesis**: Current theory about root cause
- **confidence**: exploring, low, medium, high, very_high, almost_certain, certain
- **files_checked**: All examined files (including ruled-out ones)
- **relevant_files**: Files directly related to the issue
- **relevant_context**: Methods/functions involved
- **issues_found**: Identified issues with severity levels
- **images**: Screenshots or error dialogs
- **continuation_id**: Resume previous debugging session
- **temperature**: 0=deterministic, 1=creative
- **thinking_mode**: minimal, low, medium, high, max

## Examples

### Example 1: Intermittent Error

```
Debug Investigation:

Step 1: "API endpoint randomly returns 500 errors (5% of requests)"

Hypothesis: "Race condition in database connection pool"

Files checked:
- /home/user/project/api/endpoints.py
- /home/user/project/db/connection.py

Findings: "Error logs show 'connection pool exhausted'. Occurs
under high load (>100 concurrent requests)."

Confidence: medium
Step: 1/5
Next: true

Model: gpt-5-codex
Thinking mode: high
```

### Example 2: Memory Leak

```
Debug Investigation:

Step 1: "Application memory grows from 200MB to 4GB over 24 hours"

Problem context: "Python web service processing file uploads.
Memory never releases even after files are processed."

Hypothesis: "File handles or buffers not being properly closed"

Files checked:
- /home/user/project/upload/handler.py
- /home/user/project/storage/files.py

Findings: "upload_handler() keeps references to uploaded files
in global cache dict. No cleanup mechanism implemented."

Confidence: high
Step: 2/4
Next: true

Model: gemini-2.5-pro
```

### Example 3: "No Bug Found" Scenario

```
Debug Investigation:

Step 1: "User reports login button doesn't work"

Files checked:
- /home/user/project/frontend/login.tsx
- /home/user/project/api/auth.py

Findings: "Button works correctly. User was testing with invalid
credentials. Error message wasn't clear enough."

Hypothesis: "No bug - UX issue with error messaging"

Issues found:
- severity: low
  description: "Error message 'Invalid input' too generic"

Confidence: very_high
Step: 2/2
Next: false

Model: gpt-5
```

### Example 4: Race Condition

```
Debug Investigation:

Step 1: "Order processing occasionally creates duplicate charges"

Hypothesis: "Multiple workers processing same order due to missing lock"

Files checked:
- /home/user/project/orders/processor.py
- /home/user/project/db/models.py
- /home/user/project/workers/celery_tasks.py

Findings: "order_processor() checks order.status but doesn't use
database-level lock. Window between check and update allows duplicate
processing."

Issues found:
- severity: critical
  description: "Missing SELECT FOR UPDATE in order processing"

Confidence: very_high
Step: 3/4
Next: true

Model: gpt-5-pro
Thinking mode: max
```

## Confidence Levels

**exploring** - Just starting to investigate
**low** - Early findings, multiple possibilities
**medium** - Narrowing down, some evidence
**high** - Strong evidence for root cause
**very_high** - Very strong evidence, validated theory
**almost_certain** - Nearly confirmed root cause
**certain** - 100% confidence - prevents external validation

**Important**: Only use "certain" when you have absolute confidence locally. This skips external expert validation.

## Debug Workflow Phases

**Phase 1: Problem Documentation (Step 1)**
- Clearly state the issue
- Document symptoms
- List reproduction steps
- Gather initial observations

**Phase 2: Investigation (Steps 2-N)**
- Form hypotheses based on evidence
- Check relevant code paths
- Test theories systematically
- Update confidence as you learn
- Note all files examined (even dead ends)

**Phase 3: Root Cause & Validation**
- Identify specific cause
- Confirm with evidence
- Get expert validation (if not certain)
- Document fix recommendations

## Valid "No Bug" Conclusions

Sometimes debugging reveals the issue isn't a bug:

- **User misunderstanding** - Feature working as designed
- **Documentation issue** - Behavior unclear to users
- **Environmental issue** - Problem in user's setup, not code
- **UX problem** - Not broken, just confusing

Document these findings clearly with supporting evidence.

## Best Practices

**DO:**
- ✅ Document all reproduction steps
- ✅ Track every file you examine
- ✅ Update hypothesis based on findings
- ✅ Include error messages and logs
- ✅ Note environmental factors
- ✅ Accept "no bug found" if evidence supports it

**DON'T:**
- ❌ Skip reproduction attempt
- ❌ Only check obvious suspects
- ❌ Ignore contradicting evidence
- ❌ Use "certain" prematurely
- ❌ Force a bug conclusion if there isn't one

## Model Selection

**Recommended models:**
- `gpt-5-codex` - Best for code-specific bugs
- `gemini-2.5-pro` - Best for complex analysis
- `gpt-5-pro` - Best for multi-system issues

## Expert Validation

Default: External model validates findings. Set `use_assistant_model: false` to skip and rely only on your investigation.

## Integration with ModelChorus

Uses `mcp__zen__debug` tool from Zen MCP server for structured debugging workflows with hypothesis testing and validation.

## See Also

- **thinkdeep** - For broader problem investigation
- **codereview** - For preventive bug finding
- **precommit** - For catching bugs before commit
- **chat** - For quick debugging discussions
