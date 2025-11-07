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
- Confidence level progression (exploring ’ low ’ medium ’ high ’ very_high ’ almost_certain ’ certain)
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
’ Hypothesis: "Session token expiration causing failures"
’ Confidence: LOW (initial theory)

Step 2: "Check token validation logic"
’ Evidence: Token expiration logic correct, but timing issue found
’ Updated Hypothesis: "Race condition in async token validation"
’ Confidence: MEDIUM (supporting evidence found)

Step 3: "Examine async/await patterns in auth flow"
’ Evidence: Missing await in validation, non-blocking check
’ Confirmed Hypothesis: "Race condition - token validated before expiry check completes"
’ Confidence: HIGH (root cause identified with evidence)
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
exploring ’ low ’ medium ’ high ’ very_high ’ almost_certain
```

**Quick Resolution:**
```
exploring ’ low ’ high (direct evidence found immediately)
```

**Complex Investigation:**
```
exploring ’ low ’ medium (blocked) ’ low (new hypothesis) ’ medium ’ high ’ very_high
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
