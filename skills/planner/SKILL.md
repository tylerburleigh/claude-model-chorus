---
name: model-chorus:planner
description: Breaks down complex tasks through interactive, sequential planning with revision and branching capabilities for complex project planning, system design, migration strategies, and architectural decisions
---

# ModelChorus: Planner

Interactive, incremental planning workflow with deep reflection for complex tasks. Build plans step-by-step with revision and branching capabilities.

## When to Use

Use this skill when you need:
- **Complex project planning** - Multi-phase implementations
- **System design** - Architecture planning with alternatives
- **Migration strategies** - Planning complex migrations
- **Architectural decisions** - Exploring design options
- **Feature breakdown** - Breaking features into tasks
- **Refactoring plans** - Systematic code refactoring

## How It Works

Planner uses incremental, interactive planning:

1. **Problem Framing** (Step 1) - Describe task, problem, scope
2. **Incremental Planning** (Steps 2-N) - Build plan piece by piece:
   - Add planning steps
   - Revise previous steps
   - Branch to explore alternatives
   - Capture open questions
3. **Expert Synthesis** - Optional external model validates plan

The workflow supports:
- **Revisions** - Update previous steps as you learn
- **Branching** - Explore alternative approaches
- **Progressive disclosure** - Add detail incrementally

## Usage

Invoke Planner with your planning task:

```
Use Planner to plan:
[Task or project to plan]

Scope:
- What needs to be accomplished
- Constraints and requirements
- Success criteria
```

## Parameters

- **step** (required): Planning content for this step
- **step_number** (required): Current step (starts at 1)
- **total_steps** (required): Estimated total steps
- **next_step_required** (required): True if more planning needed
- **model** (required): AI model to use

**Optional:**
- **is_step_revision**: True when replacing a previous step
- **revises_step_number**: Step number being replaced
- **is_branch_point**: True when creating alternative path
- **branch_id**: Name for this branch (e.g., "approach-A")
- **branch_from_step**: Step number where branch starts
- **more_steps_needed**: True if extending beyond prior estimate
- **continuation_id**: Resume previous planning session
- **use_assistant_model**: True for expert validation (default)

## Examples

### Example 1: Feature Planning

```
Planner:

Step 1: Plan user authentication system

Task: "Implement OAuth2 authentication with JWT tokens for
REST API. Support Google and GitHub providers."

Scope:
- OAuth2 flow implementation
- JWT token generation/validation
- Provider integration (Google, GitHub)
- User session management
- Must integrate with existing User model

Success criteria:
- Users can log in with Google/GitHub
- JWT tokens properly validated
- Sessions persist across requests
- All endpoints properly secured

Step: 1/5
Next: true
Model: gemini-2.5-pro
```

### Example 2: Migration Planning

```
Planner:

Step 1: Plan database migration from PostgreSQL to MongoDB

Problem: "Current PostgreSQL schema too rigid for evolving
product requirements. Need document flexibility."

Context:
- 500GB database, 50M users
- Zero downtime required
- Must maintain data integrity
- 6-month timeline acceptable

Step 1 content: "Analyzing current PostgreSQL schema and
identifying challenges. Key concerns: data migration strategy,
zero-downtime approach, rollback plan."

Step: 1/8
Next: true
Model: gpt-5-pro
Use assistant: true
```

### Example 3: Step Revision

```
Planner:

Step 3: Revise deployment approach

Is step revision: true
Revises step number: 2

Original Step 2 said: "Deploy all services simultaneously"

Revised Step 2: "Rolling deployment of services in 3 phases:
Phase 1 - Auth service (day 1)
Phase 2 - API services (day 3-5)
Phase 3 - Frontend (day 7)

Reason for revision: Simultaneous deployment too risky.
Rolling deployment allows rollback points."

Step: 3/7
Next: true
More steps needed: true (originally estimated 5, now need 7)
```

### Example 4: Branch Creation

```
Planner:

Step 4: Explore caching alternative

Is branch point: true
Branch ID: "redis-approach"
Branch from step: 3

Main branch (Step 3): Use in-memory caching with LRU eviction

Alternative (redis-approach): "Use Redis for distributed caching.

Pros:
- Shared cache across instances
- Persistence options
- Better scalability

Cons:
- Additional infrastructure
- Network latency
- Operational complexity

Will explore both paths and compare."

Step: 4/6
Next: true
```

## Planning Workflow

**Step 1: Problem Framing**
- Describe task/problem clearly
- Define scope and constraints
- List success criteria
- Identify initial approach

**Steps 2-N: Incremental Planning**
- Add planning details step by step
- Revise previous steps as understanding improves
- Branch to explore alternatives
- Document open questions
- Refine estimates

**Final Step: Plan Synthesis**
- External model (if enabled) provides expert review
- Identifies gaps or risks
- Suggests improvements
- Validates approach

## Revision vs. Branching

**Revision** - Fix or improve previous step
```
is_step_revision: true
revises_step_number: 2
```

**Branching** - Explore alternative approach
```
is_branch_point: true
branch_id: "alternative-a"
branch_from_step: 3
```

## Managing Complexity

**Start Simple:**
- Begin with high-level plan
- Add detail incrementally
- Don't plan everything upfront

**Revise As You Learn:**
- Update steps when understanding improves
- Don't be afraid to change direction
- Track why you revised

**Explore Alternatives:**
- Branch to compare approaches
- Document trade-offs
- Merge or choose best path

## Best Practices

**DO:**
- ✅ Start with clear problem statement
- ✅ Build plan incrementally
- ✅ Revise steps as you learn more
- ✅ Branch to explore alternatives
- ✅ Document constraints and requirements
- ✅ Track open questions
- ✅ Use more_steps_needed to adjust estimates

**DON'T:**
- ❌ Try to plan everything in step 1
- ❌ Stick to bad plan (revise instead!)
- ❌ Ignore alternative approaches
- ❌ Plan without understanding problem
- ❌ Skip success criteria
- ❌ Forget to document trade-offs

## Progressive Disclosure

Build plans in layers:

**Layer 1: High-Level** (Steps 1-3)
- Major phases
- Key decisions
- Critical path

**Layer 2: Detail** (Steps 4-6)
- Specific tasks
- Dependencies
- Resource needs

**Layer 3: Refinement** (Steps 7+)
- Edge cases
- Risk mitigation
- Alternative paths

## Model Selection

**Recommended models:**
- `gemini-2.5-pro` - Best for complex planning (1M context)
- `gpt-5-pro` - Excellent reasoning (400K context)
- `gpt-5` - Balanced general planning

## Expert Synthesis

Default: External model reviews plan. Set `use_assistant_model: false` to skip external review and rely on your own planning.

## Integration with ModelChorus

Uses `mcp__zen__planner` tool from Zen MCP server for structured, incremental planning with reflection.

## Planning Patterns

**Waterfall Planning:**
- Linear steps
- No branches
- Good for well-understood tasks

**Exploratory Planning:**
- Multiple branches
- Frequent revisions
- Good for novel/complex tasks

**Agile Planning:**
- Small increments
- Frequent revisions
- Good for evolving requirements

## See Also

- **consensus** - Get multiple opinions on plan
- **thinkdeep** - Deep investigation before planning
- **chat** - Quick planning discussions
- **codereview** - Review planned implementation
