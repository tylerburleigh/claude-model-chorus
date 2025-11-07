---
name: consensus
description: Multi-model consultation with parallel execution and configurable synthesis strategies
---

# CONSENSUS

## Overview

The CONSENSUS workflow enables querying multiple AI models simultaneously and synthesizing their responses for improved accuracy, diverse perspectives, and reliable decision-making. This workflow executes all providers in parallel and applies configurable strategies to combine or select from their outputs.

**Key Capabilities:**
- Parallel multi-model execution for fast responses
- Five consensus strategies for different use cases (all_responses, synthesize, majority, weighted, first_valid)
- Per-provider timeout and error handling
- Response weighting and prioritization
- Graceful degradation when providers fail

**Use Cases:**
- Cross-validation of facts or recommendations across multiple AI models
- Gathering diverse perspectives on complex decisions or architectural choices
- Improving answer reliability through multi-model consensus
- Comparing provider strengths and response quality
- Building robust systems that don't depend on a single model

## When to Use

Use the CONSENSUS workflow when you need to:

- **Multiple perspectives** - Get different viewpoints from various AI models to make better-informed decisions
- **Fact verification** - Cross-check facts, calculations, or recommendations across models for accuracy
- **Critical decisions** - Use multi-model consensus for high-stakes decisions where a single model's view isn't sufficient
- **Provider comparison** - Evaluate and compare how different models approach the same problem
- **Reliability improvement** - Increase confidence in answers through multi-model agreement or synthesis

## When NOT to Use

Avoid the CONSENSUS workflow when:

| Situation | Use Instead |
|-----------|-------------|
| You need a simple conversational interaction | **CHAT** - Single-model conversation with threading |
| You need deep investigation with hypothesis tracking | **THINKDEEP** - Systematic investigation with confidence progression |
| You need structured debate analysis | **ARGUMENT** - Three-role dialectical creator/skeptic/moderator |
| You need creative brainstorming | **IDEATE** - Structured idea generation |
| You need systematic research with citations | **RESEARCH** - Comprehensive information gathering |

## Consensus Strategies

CONSENSUS provides five strategies for combining or selecting model responses. Choose the strategy based on your specific needs:

### 1. all_responses (Default)

**What it does:** Returns all model responses separately with clear provider labels, separated by dividers.

**When to use:**
- You want to see each model's complete response independently
- You need to manually compare and evaluate different perspectives
- You want full visibility into how each model approached the question
- You're evaluating provider performance or quality

**Output format:**
```
## CLAUDE

[Claude's complete response]

---

## GEMINI

[Gemini's complete response]

---

## CODEX

[Codex's complete response]
```

**Example use case:** Comparing code review feedback from different models to understand diverse approaches.

---

### 2. synthesize

**What it does:** Combines all responses into a synthesized view with numbered perspectives from each model.

**When to use:**
- You want structured presentation of all perspectives
- You need to see how different models complement each other
- You want organized comparison of viewpoints
- You're building documentation from multiple sources

**Output format:**
```
# Synthesized Response from Multiple Models

## Perspective 1: claude

[Claude's response]

## Perspective 2: gemini

[Gemini's response]

## Perspective 3: codex

[Codex's response]
```

**Example use case:** Gathering architectural recommendations where each model provides complementary insights.

---

### 3. majority

**What it does:** Returns the most common response among all models (basic string matching).

**When to use:**
- You want the answer that most models agree on
- You need consensus-driven results for factual queries
- You're looking for the most widely agreed-upon perspective
- You want to filter out outlier responses

**Output format:** Single response string (the most common one)

**Example use case:** Verifying a factual answer or calculation where multiple models should converge on the same result.

**Note:** Current implementation uses simple string matching. For complex responses, consider using `all_responses` or `synthesize` instead.

---

### 4. weighted

**What it does:** Returns a weighted response, currently using response length as a heuristic (longer = more detailed).

**When to use:**
- You want the most comprehensive response
- You prefer detailed explanations over brief answers
- You're willing to accept a single "best" response based on heuristics
- You trust that length correlates with quality for your use case

**Output format:** Single response string (the longest/most detailed one)

**Example use case:** Technical explanations where detail and thoroughness are valued.

**Note:** Current implementation uses length as a proxy for quality. Future versions may incorporate provider-specific weights.

---

### 5. first_valid

**What it does:** Returns the first successful response from any provider, ignoring subsequent responses.

**When to use:**
- You want the fastest possible response time
- All providers are roughly equivalent for your task
- You need basic fallback behavior
- Response time matters more than consensus

**Output format:** Single response string (first successful one)

**Example use case:** Quick queries where any provider's answer is sufficient, but you want automatic failover if the primary provider fails.

---

## Strategy Selection Guide

**Decision tree:**

```
Need to see all perspectives independently? ’ all_responses
Need organized, numbered perspectives? ’ synthesize
Need consensus answer (factual query)? ’ majority
Need most detailed/comprehensive response? ’ weighted
Need fastest response with fallback? ’ first_valid
```

**By use case:**

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| Fact-checking | `majority` or `synthesize` | Verify agreement across models |
| Architecture decisions | `all_responses` or `synthesize` | Compare different approaches |
| Code review | `all_responses` | See diverse feedback independently |
| Quick queries | `first_valid` | Speed with failover |
| Technical explanations | `weighted` or `synthesize` | Get comprehensive details |
| Provider comparison | `all_responses` | Evaluate each model separately |
