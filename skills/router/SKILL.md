---
name: router
description: Intelligent workflow selection that analyzes user requests and recommends the optimal ModelChorus workflow with parameters
---

# ROUTER

## Overview

The ROUTER skill helps AI agents make intelligent decisions about which ModelChorus workflow to invoke for a given user request. Instead of guessing or defaulting to CHAT for everything, this skill provides a systematic analysis that considers the user's intent, problem complexity, constraints, and desired outcomes.

**Available Workflows:**
- **CHAT** - Single-model conversation with threading
- **CONSENSUS** - Multi-model parallel consultation with synthesis
- **THINKDEEP** - Extended reasoning with hypothesis tracking
- **ARGUMENT** - Three-role dialectical reasoning (Creator/Skeptic/Moderator)
- **IDEATE** - Creative brainstorming with quantity/creativity control

**When to Use This Skill:**

The main AI agent should invoke this skill when:
- User makes a request that could map to multiple workflows
- Uncertain which workflow would best serve the user's needs
- Need to extract appropriate parameters for the selected workflow
- Want to suggest workflow sequences for complex multi-step tasks

**This skill outputs a structured decision** including:
- Primary workflow recommendation
- Confidence level (high/medium/low)
- Workflow-specific parameters
- Alternative workflows (when applicable)
- Suggested sequences for multi-step tasks
- Clarifying questions (when confidence is low)

---

## Signal Extraction Guide

Before selecting a workflow, extract these signals from the user's request:

### 1. Primary Goal (The Core Intent)

| Goal Category | Keywords/Patterns | Maps To |
|---------------|-------------------|---------|
| **Answer/Opinion** | "what is", "explain", "how does", "why", "quick question" | CHAT |
| **Evaluation** | "evaluate", "critique", "review", "pros and cons", "pre-mortem", "assess" | ARGUMENT, CONSENSUS |
| **Generation** | "brainstorm", "ideas for", "suggest ways", "creative solutions", "come up with" | IDEATE |
| **Investigation** | "debug", "investigate", "root cause", "why is X happening", "analyze problem" | THINKDEEP, CHAT |
| **Decision** | "choose between", "compare A vs B", "decide on", "select", "which should" | CONSENSUS, ARGUMENT |

### 2. Complexity & Scope Indicators

| Complexity Level | Signals | Workflow Implications |
|------------------|---------|----------------------|
| **Simple** | "quick", "simple", "straightforward", factual questions | Favors CHAT |
| **Moderate** | Standard feature requests, comparisons, evaluations | ARGUMENT, IDEATE, RESEARCH |
| **Complex** | "intermittent", "ambiguous", "multi-step", "complex system", "race condition", "unclear cause" | Strongly favors THINKDEEP |

### 3. Perspective Requirements

| Requirement | Signals | Workflow |
|-------------|---------|----------|
| **Single perspective sufficient** | No explicit need for validation | CHAT, THINKDEEP, ARGUMENT, IDEATE |
| **Multiple perspectives needed** | "validate", "cross-check", "multiple viewpoints", "blind spots", "consensus", "compare approaches" | CONSENSUS |

### 4. Decision Impact & Risk

| Impact Level | Signals | Workflow |
|--------------|---------|----------|
| **High stakes** | "architecture", "production", "security review", "major decision", "high-impact", "critical" | CONSENSUS (for validation) |
| **Moderate stakes** | Standard feature decisions, design choices | ARGUMENT, CHAT |
| **Low stakes** | Exploratory questions, learning | CHAT, IDEATE |

### 5. Output Requirements

| Requirement | Signals | Workflow |
|-------------|---------|----------|
| **Multiple options** | "different ways", "explore possibilities", "alternatives", "various approaches" | IDEATE, CONSENSUS |
| **Structured debate** | "pro and con", "strengths and weaknesses", "balanced view", "critique" | ARGUMENT |
| **Systematic analysis** | "step by step", "methodical", "build evidence", "track progress" | THINKDEEP |

### 6. Constraints

| Constraint Type | Signals | Impact |
|-----------------|---------|--------|
| **Time limited** | "quick", "fast", "urgent", "in a hurry", "ASAP" | Favor CHAT, IDEATE, ARGUMENT; avoid THINKDEEP, CONSENSUS |
| **Budget limited** | "cheap", "low-cost", "economical", "minimal expense" | Favor CHAT, IDEATE; avoid CONSENSUS |
| **Quality priority** | "thorough", "comprehensive", "validated", "high quality", "certain" | Favor CONSENSUS, THINKDEEP |

### 7. Interaction Style

| Style | Signals | Workflow |
|-------|---------|----------|
| **Conversational/iterative** | "let's explore", "I'll have follow-ups", "iterative", ongoing discussion | CHAT (supports threading) |
| **One-shot synthesis** | "give me a complete analysis", "comprehensive answer" | CONSENSUS, ARGUMENT |
| **Multi-step investigation** | "investigate step by step", "build confidence", "test hypotheses" | THINKDEEP (supports state persistence) |

### 8. Artifact to Analyze

| Presence | Signals | Workflow |
|----------|---------|----------|
| **Specific proposal/design provided** | User includes code, design doc, architecture diagram, or explicit proposal to evaluate | Strong signal for ARGUMENT |
| **No artifact** | General question or request | Other workflows |

---

## Routing Procedure

Follow this 6-step procedure to select the optimal workflow:

### Step 0: Parse Constraints and Context

**Extract immediately:**
- Budget constraints (limited/flexible)
- Time constraints (urgent/flexible)
- Quality requirements (standard/high)
- Conversation continuation need (threading expected?)

**References:**
- WORKFLOW_SELECTION_GUIDE.md lines 445-468 ("By Constraints")
- CHAT supports threading (line 65)
- CONSENSUS does not support threading (line 110)
- THINKDEEP supports state persistence (lines 143-145)

### Step 1: Identify Primary Goal and Complexity

**Determine:**
- What is the user's core intent? (Answer/Ideas/Evaluation/Investigation/Decision/Information)
- What is the problem complexity? (Simple/Moderate/Complex)

**References:**
- WORKFLOW_SELECTION_GUIDE.md line 565 ("What's my goal?")
- Line 436-442 ("By Problem Type" table)

### Step 2: Apply Prioritized Rules (Highest to Lowest Precedence)

Apply these rules in order. First match wins unless overridden by constraints in Step 3:

1. **Specific proposal/decision to evaluate** (artifact provided) → **ARGUMENT**
   - Reference: Lines 193-199 ("Evaluating a specific proposal")

2. **Creative options/brainstorming needed** → **IDEATE**
   - Reference: Lines 232-238 ("Need creative options")

3. **High-impact decision needing multiple perspectives** → **CONSENSUS**
   - Reference: Lines 113-119 ("Decision has significant impact... Multiple perspectives")

4. **Complex/ambiguous investigation** (multi-step reasoning) → **THINKDEEP**
   - Reference: Lines 153-159 ("Problem is complex... Multi-step analysis beneficial")

5. **Otherwise** (default) → **CHAT**
   - Reference: Lines 74-80 ("Quick answer... One model's perspective")

###Step 3: Apply Constraint-Based Overrides

**If time/budget limited:**
- Demote CONSENSUS and THINKDEEP
- Promote CHAT, IDEATE, ARGUMENT
- Reference: Lines 445-459 ("When budget/time is limited")

**If quality is priority:**
- Promote CONSENSUS, THINKDEEP
- Reference: Lines 461-468 ("When quality is priority")

### Step 4: Conversation-Mode Routing

**If user expects multi-turn conversation:**
- Prefer CHAT or THINKDEEP (both support threading/persistence)
- Avoid CONSENSUS (no conversation continuation)
- Reference: Line 65 (CHAT threading), Line 110 (CONSENSUS no threading), Lines 143-145 (THINKDEEP persistence)

**If user asks to "validate" or "cross-check" after investigation:**
- Suggest CONSENSUS as next step
- Reference: Lines 491-497 ("Investigate → Validate Pattern")

### Step 5: Sequence Suggestion

**If multiple intents detected or staged work identified:**

Use recommended patterns from WORKFLOW_SELECTION_GUIDE.md lines 475-505:

- **Generate → Evaluate Pattern**: IDEATE → ARGUMENT → CONSENSUS
  - Use for: Feature planning, solution design

- **Investigate → Validate Pattern**: THINKDEEP → CONSENSUS → ARGUMENT
  - Use for: Complex debugging, root cause analysis

- **Quick → Deep Pattern**: CHAT → THINKDEEP → CONSENSUS
  - Use for: Exploratory analysis that becomes complex

### Step 6: Low-Confidence Routing

**If confidence is low or request is ambiguous:**
- Formulate ONE targeted clarifying question
- Provide a provisional default (usually CHAT for safety)
- Include rationale for why clarification is needed

**Examples of ambiguous requests:**
- "Help with my project" → Ask: "What kind of help? Generate ideas, debug an issue, or evaluate a plan?"
- "Look into this API" → Ask: "Should I investigate a specific problem, research best practices, or review the design?"

---

## Anti-Pattern Prevention

**DO NOT route to these workflows in these situations:**

### ❌ CONSENSUS for Simple Questions
- **Bad**: "What's the capital of France?" → CONSENSUS
- **Why wrong**: Overkill, wastes resources
- **Use instead**: CHAT
- **Reference**: Lines 511-515

### ❌ CHAT for Complex Investigations
- **Bad**: "Debug this intermittent race condition" → CHAT
- **Why wrong**: No systematic investigation, no hypothesis tracking
- **Use instead**: THINKDEEP
- **Reference**: Lines 516-520

### ❌ IDEATE for Evaluation
- **Bad**: "Is this database schema design good?" → IDEATE
- **Why wrong**: IDEATE generates, doesn't evaluate
- **Use instead**: ARGUMENT or CONSENSUS
- **Reference**: Lines 521-525

### ❌ THINKDEEP for Quick Facts
- **Bad**: "What HTTP status code means unauthorized?" → THINKDEEP
- **Why wrong**: Overly complex for simple lookup
- **Use instead**: CHAT
- **Reference**: Lines 531-535

---

## Workflow-Specific Parameter Suggestions

When you select a workflow, suggest appropriate parameters based on the user request:

### CHAT Parameters
- `continue`: thread ID if user is in ongoing conversation
- `provider`: claude (default), gemini, codex, cursor-agent

### CONSENSUS Parameters
- `strategy`: "synthesize" (default for structured view), "all_responses" (to see each model separately), "majority" (factual queries), "weighted" (most detailed), "first_valid" (speed priority)
- `provider`: List of 2-4 providers (default: ["claude", "gemini"])
- Reference: Lines 308, 420

### THINKDEEP Parameters
- `step`: Brief description of what to investigate in this step
- `step_number`: 1 (for new investigations)
- `total_steps`: Estimate based on complexity (1-3 for simple, 3-5 for moderate, 6-10 for complex)
- `next_step_required`: true (for multi-step), false (single step)
- `findings`: Initial observations or "Starting investigation"
- `confidence`: "exploring" (default for new), or carry forward from previous
- Reference: Lines 340-341

### ARGUMENT Parameters
- No specific parameters beyond the prompt

### IDEATE Parameters
- `num_ideas`: 3-4 (quick), 5-7 (standard), 8-10 (comprehensive), 10+ (exhaustive)
- Reference: Line 412

---

## Output Format Specification

Return a JSON-structured decision object:

```json
{
  "workflow": "CONSENSUS",
  "confidence": "high",
  "rationale": "User requests comparative evaluation for high-impact architecture decision. Multiple perspectives reduce blind spots and validate choice.",
  "parameters": {
    "strategy": "synthesize",
    "provider": ["claude", "gemini", "codex"]
  },
  "alternatives": [
    {
      "workflow": "ARGUMENT",
      "when_to_use": "If budget/time constraints tighten",
      "reason": "Faster structured critique of top option, single model"
    }
  ],
  "sequence": [
    {
      "step": 1,
      "workflow": "CONSENSUS",
      "trigger": "Initial evaluation"
    },
    {
      "step": 2,
      "workflow": "ARGUMENT",
      "trigger": "After short-listing top choice, critique it"
    }
  ],
  "clarifying_question": null,
  "used_signals": [
    "compare X vs Y",
    "high-impact decision",
    "architecture choice",
    "need validation"
  ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow` | string | Yes | One of: CHAT, CONSENSUS, THINKDEEP, ARGUMENT, IDEATE |
| `confidence` | string | Yes | One of: "high", "medium", "low" |
| `rationale` | string | Yes | 1-2 sentences explaining why this workflow was selected |
| `parameters` | object | No | Workflow-specific parameters extracted from request |
| `alternatives` | array | No | Other viable workflows with context for when to use them |
| `sequence` | array | No | Multi-step workflow plan for complex tasks |
| `clarifying_question` | string | No | Single targeted question when confidence is low |
| `used_signals` | array | Yes | List of extracted signals for transparency/debugging |

---

## Example Routing Decisions

### Example 1: Simple Question → CHAT

**User Request**: "What is JWT authentication?"

**Decision**:
```json
{
  "workflow": "CHAT",
  "confidence": "high",
  "rationale": "Simple factual question requiring quick explanation. Single perspective sufficient.",
  "parameters": {},
  "used_signals": ["what is", "simple question", "factual"]
}
```

### Example 2: High-Impact Decision → CONSENSUS

**User Request**: "Choose between PostgreSQL, MongoDB, and DynamoDB for our e-commerce platform (10k users, ACID required, team has SQL background)"

**Decision**:
```json
{
  "workflow": "CONSENSUS",
  "confidence": "high",
  "rationale": "High-impact technology selection with significant implications. Multiple model perspectives provide validation and reduce blind spots.",
  "parameters": {
    "strategy": "synthesize",
    "provider": ["claude", "gemini", "codex"]
  },
  "sequence": [
    {
      "step": 1,
      "workflow": "CONSENSUS",
      "trigger": "Make final decision with synthesis"
    },
    {
      "step": 2,
      "workflow": "ARGUMENT",
      "trigger": "Critique the selected choice"
    }
  ],
  "used_signals": [
    "choose between",
    "high-impact",
    "e-commerce platform",
    "need validation",
    "comparison"
  ]
}
```

### Example 3: Complex Debugging → THINKDEEP

**User Request**: "API latency increased from 100ms to 2s after deployment. Intermittent, affects all endpoints."

**Decision**:
```json
{
  "workflow": "THINKDEEP",
  "confidence": "high",
  "rationale": "Complex investigation requiring systematic hypothesis testing. Multi-step analysis needed to build confidence in root cause.",
  "parameters": {
    "step": "Investigate 20x latency increase after deployment",
    "step_number": 1,
    "total_steps": 4,
    "next_step_required": true,
    "findings": "Latency affects all endpoints equally, started immediately after 3pm deployment",
    "confidence": "exploring"
  },
  "sequence": [
    {
      "step": 1,
      "workflow": "THINKDEEP",
      "trigger": "Systematic investigation"
    },
    {
      "step": 2,
      "workflow": "CONSENSUS",
      "trigger": "After identifying likely cause, validate findings"
    }
  ],
  "used_signals": [
    "investigate",
    "complex problem",
    "intermittent",
    "root cause needed",
    "multi-step analysis"
  ]
}
```

### Example 4: Proposal Evaluation → ARGUMENT

**User Request**: "Evaluate this API design: [paste design]. Focus on scalability, security, and developer experience."

**Decision**:
```json
{
  "workflow": "ARGUMENT",
  "confidence": "high",
  "rationale": "Specific proposal provided for evaluation. Structured dialectical critique (Creator/Skeptic/Moderator) provides balanced analysis.",
  "parameters": {},
  "alternatives": [
    {
      "workflow": "CONSENSUS",
      "when_to_use": "If you need multiple real model perspectives",
      "reason": "More thorough but higher cost"
    }
  ],
  "used_signals": [
    "evaluate",
    "specific proposal provided",
    "critique needed",
    "focus on specific criteria"
  ]
}
```

### Example 5: Creative Brainstorming → IDEATE

**User Request**: "Brainstorm creative ways to improve user onboarding. Current drop-off is 40%."

**Decision**:
```json
{
  "workflow": "IDEATE",
  "confidence": "high",
  "rationale": "Creative idea generation needed. IDEATE provides structured brainstorming with configurable quantity.",
  "parameters": {
    "num_ideas": 6
  },
  "sequence": [
    {
      "step": 1,
      "workflow": "IDEATE",
      "trigger": "Generate creative ideas"
    },
    {
      "step": 2,
      "workflow": "ARGUMENT",
      "trigger": "After selecting top 2-3 ideas, evaluate each"
    }
  ],
  "used_signals": [
    "brainstorm",
    "creative ways",
    "generate ideas",
    "improve onboarding"
  ]
}
```

### Example 6: Ambiguous Request (Low Confidence) → Clarifying Question

**User Request**: "Help with my project"

**Decision**:
```json
{
  "workflow": "CHAT",
  "confidence": "low",
  "rationale": "Request is too vague to determine optimal workflow. Defaulting to CHAT as safe fallback but requesting clarification.",
  "clarifying_question": "What kind of help do you need? Are you trying to:\n- Generate creative ideas (brainstorming)\n- Debug or investigate a problem\n- Evaluate or critique a proposal\n- Get a quick answer or opinion",
  "used_signals": [
    "ambiguous intent",
    "no clear goal",
    "generic request"
  ]
}
```

### Example 7: Multiple Valid Workflows → Alternatives Provided

**User Request**: "Review this database schema design"

**Decision**:
```json
{
  "workflow": "ARGUMENT",
  "confidence": "medium",
  "rationale": "Evaluation of specific design. ARGUMENT provides structured critique (Creator/Skeptic/Moderator) at moderate cost.",
  "parameters": {},
  "alternatives": [
    {
      "workflow": "CONSENSUS",
      "when_to_use": "If you want multiple real model perspectives for higher confidence",
      "reason": "More thorough validation but higher cost"
    },
    {
      "workflow": "CHAT",
      "when_to_use": "If budget/time is very limited",
      "reason": "Quick single-perspective review"
    }
  ],
  "used_signals": [
    "review",
    "specific artifact (schema)",
    "evaluation needed",
    "multiple valid approaches"
  ]
}
```

### Example 8: Time Constraint Override → CHAT

**User Request**: "Quick: Should we use microservices or monolith for our 5-person team?"

**Decision**:
```json
{
  "workflow": "CHAT",
  "confidence": "medium",
  "rationale": "Time constraint ('quick') overrides typical CONSENSUS recommendation for architecture decisions. CHAT provides fast guidance with note about more thorough options.",
  "parameters": {},
  "alternatives": [
    {
      "workflow": "CONSENSUS",
      "when_to_use": "If time permits and decision impact justifies it",
      "reason": "Multi-model validation for high-impact architecture choice"
    }
  ],
  "used_signals": [
    "quick (time constraint)",
    "architecture decision",
    "constraint override applied"
  ]
}
```

### Example 9: Sequence Suggestion → Multi-Step Plan

**User Request**: "We need to add a new payment provider. Help us choose and implement."

**Decision**:
```json
{
  "workflow": "CONSENSUS",
  "confidence": "high",
  "rationale": "Multi-step task detected: evaluate options → critique → implement. Starting with CONSENSUS to gather multiple perspectives on payment providers.",
  "parameters": {
    "strategy": "synthesize",
    "provider": ["claude", "gemini", "codex"]
  },
  "sequence": [
    {
      "step": 1,
      "workflow": "CONSENSUS",
      "trigger": "Evaluate top 2-3 payment provider options with multiple perspectives"
    },
    {
      "step": 2,
      "workflow": "ARGUMENT",
      "trigger": "Critique selected provider before implementation"
    },
    {
      "step": 3,
      "workflow": "CHAT",
      "trigger": "Implementation guidance and iterative support"
    }
  ],
  "used_signals": [
    "multi-step task",
    "decision + implementation",
    "payment provider selection"
  ]
}
```

---

## Technical Contract

### Input

**Required:**
- `user_request` (string): The user's full request/question

**Optional:**
- `conversation_context` (object): Information about ongoing conversation
  - `has_thread`: boolean indicating if user is in threaded conversation
  - `previous_workflow`: string indicating last workflow used
- `explicit_constraints` (object): User-specified constraints
  - `time_limited`: boolean
  - `budget_limited`: boolean
  - `quality_priority`: boolean

### Output

Returns a JSON object with the structure defined in "Output Format Specification" section.

**Required fields:**
- `workflow`: string
- `confidence`: string (high/medium/low)
- `rationale`: string
- `used_signals`: array of strings

**Optional fields:**
- `parameters`: object (workflow-specific)
- `alternatives`: array of objects
- `sequence`: array of objects
- `clarifying_question`: string

---

## Clarifying Question Templates

Use these templates when confidence is low:

### Goal Ambiguity
**Template**: "What's your primary goal? Are you trying to:\n- [Option A related to request]\n- [Option B related to request]\n- [Option C related to request]"

**Example**: "What's your primary goal? Are you trying to:\n- Generate creative ideas (brainstorming)\n- Evaluate an existing proposal\n- Get a quick answer or opinion"

### Complexity Unclear
**Template**: "How deep should I go? Do you need:\n- Quick guidance (fast, single perspective)\n- Thorough analysis (systematic investigation)\n- Validated decision (multiple perspectives)"

### Multiple Options Unclear
**Template**: "Should I:\n- Explore multiple options/ideas\n- Evaluate a specific proposal\n- Give you one recommendation"

---

## Best Practices for Using This Skill

1. **Always extract signals first** - Don't jump to conclusions; systematically analyze the request
2. **Consider constraints early** - Time/budget/quality requirements can override default selections
3. **Provide alternatives for medium confidence** - Help users understand trade-offs
4. **Suggest sequences for complex tasks** - Guide users through multi-step workflows
5. **Default to CHAT when uncertain** - It's the safest, most flexible option
6. **Be explicit in rationale** - Reference specific signals and guide sections
7. **Pre-populate parameters** - Save users time by suggesting sensible defaults
8. **Ask ONE clarifying question** - Don't overwhelm with multiple questions; focus on the decision boundary

---

**This skill is the intelligent decision layer for ModelChorus. Use it to ensure users get routed to the optimal workflow for their specific needs.**
