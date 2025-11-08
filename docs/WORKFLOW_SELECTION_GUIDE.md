# ModelChorus Workflow Selection Guide

**Version:** 1.0
**Last Updated:** 2025-11-07

This guide helps you choose the right ModelChorus workflow for your specific needs. Each workflow excels at different types of tasks - selecting the optimal one ensures better results and efficient resource usage.

---

## Quick Decision Matrix

**Use this flowchart to quickly identify the best workflow:**

```
START: What do you need?

├─ Quick answer or opinion?
│  └─ Single perspective sufficient?
│     ├─ YES → **CHAT** (Simple conversation)
│     └─ NO → **CONSENSUS** (Multiple perspectives)
│
├─ Complex problem needing investigation?
│  └─ Multi-step analysis required?
│     └─ YES → **THINKDEEP** (Systematic investigation)
│
├─ Evaluating a specific idea or proposal?
│  └─ Need balanced critique?
│     └─ YES → **ARGUMENT** (Structured debate)
│
└─ Need creative ideas or brainstorming?
   └─ Exploring possibilities?
      └─ YES → **IDEATE** (Idea generation)
```

---

## Workflow Comparison Table

| Workflow | Best For | Speed | Cost | Complexity | Output Type |
|----------|----------|-------|------|------------|-------------|
| **CHAT** | Quick questions, opinions | ⚡ Fast | $ Low | ⭐ Simple | Conversational response |
| **CONSENSUS** | Multi-perspective decisions | 🐌 Slower | $$$ High | ⭐⭐ Moderate | Synthesized analysis |
| **THINKDEEP** | Complex investigations | 🐢 Slowest | $$ Medium | ⭐⭐⭐ Complex | Investigation summary |
| **ARGUMENT** | Idea evaluation, critique | ⚡ Fast | $$ Medium | ⭐⭐ Moderate | Balanced analysis |
| **IDEATE** | Creative brainstorming | ⚡ Fast | $ Low | ⭐ Simple | List of ideas |

---

## Detailed Workflow Profiles

### CHAT - Simple Conversations

**What it is:**
Single-model conversational interface for straightforward questions and discussions.

**Strengths:**
- ✅ Fast responses (seconds)
- ✅ Low cost (single model call)
- ✅ Simple to use
- ✅ Supports multi-turn conversations
- ✅ Good for iterative exploration

**Weaknesses:**
- ❌ Single perspective only
- ❌ No systematic investigation
- ❌ Limited structure in responses
- ❌ No built-in validation

**Use CHAT when:**
- You need a quick answer or opinion
- One model's perspective is sufficient
- Cost and speed are priorities
- You want to iterate conversationally
- The problem is straightforward

**Don't use CHAT when:**
- You need multiple perspectives (use CONSENSUS)
- Problem requires systematic analysis (use THINKDEEP)
- You need structured debate (use ARGUMENT)

**Example use cases:**
- Code review: "Review this function for potential bugs"
- Quick explanation: "Explain how JWT authentication works"
- Opinion: "Should I use REST or GraphQL for this API?"
- Iteration: "That makes sense. What about error handling?"

---

### CONSENSUS - Multi-Model Collaboration

**What it is:**
Parallel consultation of multiple AI models with strategy-based synthesis of their responses.

**Strengths:**
- ✅ Multiple perspectives reduce blind spots
- ✅ Built-in validation (models check each other)
- ✅ Comprehensive analysis
- ✅ Flexible synthesis strategies
- ✅ Identifies areas of agreement/disagreement

**Weaknesses:**
- ❌ Higher cost (multiple model calls)
- ❌ Slower (parallel but still multiple calls)
- ❌ No conversation continuation
- ❌ Can be overwhelming with conflicting views

**Use CONSENSUS when:**
- Decision has significant impact
- Multiple perspectives add value
- Need validation and cross-checking
- Want to identify blind spots
- Budget allows for thorough analysis

**Don't use CONSENSUS when:**
- Budget/time constrained (use CHAT)
- Need iterative investigation (use THINKDEEP)
- Problem is straightforward (use CHAT)
- Need structured debate format (use ARGUMENT)
- Need creative ideas (use IDEATE)

**Example use cases:**
- Architecture decision: "Microservices vs monolith for our scale?"
- Technology selection: "Choose database: PostgreSQL, MongoDB, or DynamoDB?"
- Security review: "Evaluate authentication flow for vulnerabilities"
- Design review: "Review this API design for potential issues"

---

### THINKDEEP - Systematic Investigation

**What it is:**
Multi-step investigation workflow with hypothesis tracking, evidence accumulation, and confidence progression.

**Strengths:**
- ✅ Systematic, methodical analysis
- ✅ Hypothesis evolution based on evidence
- ✅ Confidence tracking shows certainty
- ✅ State persistence across sessions
- ✅ Investigation history maintained
- ✅ Best for complex, ambiguous problems

**Weaknesses:**
- ❌ Slower (multiple investigation steps)
- ❌ More complex to use
- ❌ Single model perspective
- ❌ Requires more user engagement

**Use THINKDEEP when:**
- Problem is complex and ambiguous
- Need systematic investigation
- Building confidence through evidence
- Multi-step analysis beneficial
- Investigation may span sessions

**Don't use THINKDEEP when:**
- Need quick answer (use CHAT)
- Need multiple perspectives (use CONSENSUS)
- Problem is straightforward (use CHAT)
- Need creative ideas (use IDEATE)

**Example use cases:**
- Debugging: "Intermittent 401 errors, 5% of requests fail"
- Root cause analysis: "Memory leak growing 50MB/hour"
- Performance investigation: "API latency increased 20x after deployment"
- Security audit: "Find authentication bypass vulnerabilities"
- Architecture analysis: "Why is this system design causing scaling issues?"

---

### ARGUMENT - Structured Debate

**What it is:**
Dialectical analysis using three perspectives: Creator (proposes), Skeptic (challenges), Moderator (synthesizes).

**Strengths:**
- ✅ Balanced evaluation (pro and con)
- ✅ Identifies weaknesses and strengths
- ✅ Structured format
- ✅ Good for pre-mortem analysis
- ✅ Faster than full CONSENSUS

**Weaknesses:**
- ❌ Single model (simulated perspectives)
- ❌ Less thorough than CONSENSUS
- ❌ Limited to evaluation (not generation)
- ❌ Can be overly critical

**Use ARGUMENT when:**
- Evaluating a specific proposal
- Need balanced pro/con analysis
- Want structured critique
- Pre-mortem before decision
- Budget is moderate

**Don't use ARGUMENT when:**
- Need multiple real models (use CONSENSUS)
- Need idea generation (use IDEATE)
- Need systematic investigation (use THINKDEEP)
- Just need information (use CHAT)

**Example use cases:**
- Proposal evaluation: "Evaluate: Use serverless for our backend"
- Pre-mortem: "What could go wrong with this architecture?"
- Design review: "Critique this database schema design"
- Decision validation: "Challenge our decision to adopt microservices"

---

### IDEATE - Creative Brainstorming

**What it is:**
Structured idea generation with divergent thinking followed by synthesis.

**Strengths:**
- ✅ Generates multiple creative options
- ✅ Explores possibility space
- ✅ Fast execution
- ✅ Low cost
- ✅ Synthesis combines best elements

**Weaknesses:**
- ❌ Ideas may need validation
- ❌ No deep analysis of ideas
- ❌ Single model perspective
- ❌ Limited to generation (not evaluation)

**Use IDEATE when:**
- Need creative options
- Exploring possibilities
- Brainstorming solutions
- Want fresh perspectives
- Early in problem-solving

**Don't use IDEATE when:**
- Need evaluation (use ARGUMENT or CONSENSUS)
- Need investigation (use THINKDEEP)
- Need information (use CHAT)
- Already have ideas to evaluate

**Example use cases:**
- Feature ideas: "Generate ideas for improving user onboarding"
- Solution brainstorming: "Creative solutions for reducing API latency"
- Naming: "Suggest names for our new product"
- Approach exploration: "Different ways to implement caching"

---

## Cross-Workflow Examples

### Example 1: API Design Problem

**Problem:** Designing a new REST API for user management

**CHAT Approach:**
```bash
model-chorus chat "Design a REST API for user management with CRUD operations"
```
- **Result:** Single design proposal
- **Pros:** Fast, actionable
- **Cons:** One perspective, no validation

**CONSENSUS Approach:**
```bash
model-chorus consensus --strategy synthesize "Design a REST API for user management. Consider endpoints, authentication, error handling, and versioning"
```
- **Result:** Multiple design approaches synthesized
- **Pros:** Validated, comprehensive
- **Cons:** Slower, higher cost

**ARGUMENT Approach:**
```bash
model-chorus argument "Evaluate this API design proposal: [paste design]. Focus on scalability, security, and developer experience"
```
- **Result:** Balanced critique with strengths/weaknesses
- **Pros:** Identifies issues, structured feedback
- **Cons:** Evaluative only, doesn't create design

**Recommendation:** Use **CONSENSUS** for initial design (high-impact decision), then **ARGUMENT** to critique before implementation.

---

### Example 2: Performance Problem

**Problem:** API latency increased from 100ms to 2s after deployment

**CHAT Approach:**
```bash
model-chorus chat "API latency went from 100ms to 2s after deployment. What could be wrong?"
```
- **Result:** List of possible causes
- **Pros:** Fast suggestions
- **Cons:** No investigation, guessing

**THINKDEEP Approach:**
```bash
model-chorus thinkdeep --step "API latency increased 20x after deployment" --step-number 1 --total-steps 3 --next-step-required true --findings "Affects all endpoints equally, started at 3pm deployment" --confidence low
```
- **Result:** Systematic investigation with hypothesis testing
- **Pros:** Methodical, builds confidence, finds root cause
- **Cons:** Slower, multiple steps

**Recommendation:** Use **THINKDEEP** for this complex debugging scenario. CHAT is too shallow for systematic investigation.

---

### Example 3: Technology Selection

**Problem:** Choose between PostgreSQL, MongoDB, and DynamoDB

**CHAT Approach:**
```bash
model-chorus chat "Should I use PostgreSQL, MongoDB, or DynamoDB for my e-commerce app?"
```
- **Result:** Single recommendation
- **Pros:** Fast decision
- **Cons:** May miss important tradeoffs

**CONSENSUS Approach:**
```bash
model-chorus consensus --strategy synthesize "Evaluate PostgreSQL, MongoDB, and DynamoDB for e-commerce platform. Consider: 10k users, ACID requirements, query patterns, team expertise (SQL background)"
```
- **Result:** Multi-model evaluation with consensus
- **Pros:** Comprehensive, validated, considers tradeoffs
- **Cons:** Higher cost

**ARGUMENT Approach:**
```bash
model-chorus argument "Proposal: Use MongoDB for our e-commerce platform. Team has SQL background, need ACID transactions, 10k users expected"
```
- **Result:** Balanced critique of MongoDB choice
- **Pros:** Identifies specific concerns
- **Cons:** Only evaluates one option

**Recommendation:** Use **CONSENSUS** for high-impact technology selection to get multiple perspectives on the tradeoffs.

---

### Example 4: Feature Brainstorming

**Problem:** Improve user onboarding experience

**CHAT Approach:**
```bash
model-chorus chat "Suggest ways to improve our user onboarding"
```
- **Result:** List of suggestions
- **Pros:** Fast ideas
- **Cons:** Limited exploration

**IDEATE Approach:**
```bash
model-chorus ideate --num-ideas 5 "Generate creative ideas to improve user onboarding experience. Current issues: 40% drop-off, users confused about first steps"
```
- **Result:** 5 diverse ideas with synthesis
- **Pros:** Structured brainstorming, explores space
- **Cons:** Still needs evaluation

**CONSENSUS Approach:**
```bash
model-chorus consensus --strategy all_responses "What are the best ways to improve user onboarding? Current drop-off: 40%"
```
- **Result:** Multiple perspectives on improvements
- **Pros:** Validated ideas from multiple models
- **Cons:** Higher cost for brainstorming phase

**Recommendation:** Use **IDEATE** for initial idea generation, then **ARGUMENT** to evaluate top 2-3 ideas before implementation.

---

## Decision Framework

### By Problem Type

| Problem Type | First Choice | Alternative | Why |
|--------------|--------------|-------------|-----|
| Quick question | CHAT | - | Speed and simplicity |
| Complex investigation | THINKDEEP | CHAT → THINKDEEP | Systematic analysis |
| High-impact decision | CONSENSUS | THINKDEEP | Multiple perspectives |
| Idea generation | IDEATE | CHAT | Structured brainstorming |
| Idea evaluation | ARGUMENT | CONSENSUS | Balanced critique |

### By Constraints

**When budget is limited:**
1. CHAT (cheapest)
2. IDEATE (low cost)
3. ARGUMENT (moderate)
4. THINKDEEP (moderate, but multi-step)
5. CONSENSUS (most expensive)

**When time is limited:**
1. CHAT (fastest)
2. IDEATE (fast)
3. ARGUMENT (fast)
4. THINKDEEP (slow, multi-step)
5. CONSENSUS (slower, parallel calls)

**When quality is priority:**
1. CONSENSUS (highest quality, validated)
2. THINKDEEP (high quality, systematic)
3. ARGUMENT (good quality, balanced)
4. CHAT (depends on prompt quality)
5. IDEATE (generation focus, not validation)

---

## Workflow Combinations

### Effective Workflow Sequences

**Generate → Evaluate Pattern:**
```
1. IDEATE (generate options)
2. ARGUMENT (evaluate top 2-3)
3. CONSENSUS (validate final choice)
```
Use for: Feature planning, solution design

**Investigate → Validate Pattern:**
```
1. THINKDEEP (investigate problem)
2. CONSENSUS (validate findings)
3. ARGUMENT (critique proposed solution)
```
Use for: Complex debugging, root cause analysis

**Quick → Deep Pattern:**
```
1. CHAT (quick exploration)
2. THINKDEEP (deep investigation)
3. CONSENSUS (validate conclusions)
```
Use for: Exploratory analysis that becomes complex

---

## Anti-Patterns to Avoid

### ❌ Using CONSENSUS for Simple Questions
**Problem:** "What's the capital of France?"
**Why wrong:** Overkill, wastes resources
**Use instead:** CHAT

### ❌ Using CHAT for Complex Investigations
**Problem:** "Debug this intermittent race condition"
**Why wrong:** No systematic investigation
**Use instead:** THINKDEEP

### ❌ Using IDEATE for Evaluation
**Problem:** "Is this database schema design good?"
**Why wrong:** IDEATE generates, doesn't evaluate
**Use instead:** ARGUMENT or CONSENSUS

### ❌ Using THINKDEEP for Quick Facts
**Problem:** "What HTTP status code means unauthorized?"
**Why wrong:** Overly complex for simple lookup
**Use instead:** CHAT

---

## Getting Started

### First-Time Users

**Start with CHAT:**
- Easiest to understand
- Fast feedback
- Build familiarity with system

**Then try IDEATE:**
- Still simple, but structured
- See how ModelChorus adds value
- Low risk, creative output

**Graduate to ARGUMENT:**
- Learn evaluation patterns
- Understand balanced analysis
- See structured critique

**Finally explore advanced workflows:**
- THINKDEEP for complex problems
- CONSENSUS for important decisions

### Choosing Your First Workflow

Ask yourself:
1. **What's my goal?** (Answer, Ideas, Evaluation, Investigation, Information)
2. **How complex is the problem?** (Simple, Moderate, Complex)
3. **Do I need multiple perspectives?** (Yes → CONSENSUS, No → others)
4. **How much time/budget do I have?** (Limited → CHAT/IDEATE, Flexible → others)

**Most common starting points:**
- Developers: CHAT → THINKDEEP → CONSENSUS
- Product managers: IDEATE → ARGUMENT → CONSENSUS
- Architects: CONSENSUS → ARGUMENT
- Security engineers: THINKDEEP → CONSENSUS → ARGUMENT

---

## Related Documentation

- **Terminology Glossary:** `docs/GLOSSARY.md` - Standard terms across workflows
- **Individual Workflow Documentation:** `skills/*/SKILL.md` - Detailed instructions per workflow
- **Getting Started:** [Add link when available]
- **Examples:** [Add link when available]

---

## Feedback & Questions

This guide is based on multi-model consensus review of all ModelChorus workflows. If you have:
- Questions about which workflow to use
- Suggestions for improving this guide
- Examples of successful workflow combinations
- Challenges choosing the right workflow

Please open an issue in the ModelChorus repository or contact the team.

---

*Remember: There's no single "right" workflow for every problem. Experiment with different approaches, combine workflows, and find what works best for your use cases.*
