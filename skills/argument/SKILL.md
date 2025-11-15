---
name: argument
description: Structured dialectical reasoning through three-role analysis (Creator/Skeptic/Moderator)
---

# ARGUMENT

## Overview

The ARGUMENT workflow provides structured dialectical reasoning for analyzing arguments, claims, and proposals through three distinct roles. Unlike simple Q&A or multi-model consensus, ARGUMENT examines topics from multiple angles using a single model that embodies different perspectives in sequence.

**Key Capabilities:**
- Three-role dialectical structure (Creator/Skeptic/Moderator)
- Single-model reasoning with structured debate
- Conversation threading for multi-turn analysis
- Balanced synthesis incorporating all perspectives
- Context-aware analysis with file support

**Use Cases:**
- Policy and governance analysis requiring balanced perspectives
- Technology and architecture decisions with trade-off evaluation
- Feature proposals needing pro/con analysis
- Critical evaluation of arguments or claims
- Decision support requiring structured debate

## When to Use

Use the ARGUMENT workflow when you need to:

- **Balanced perspectives** - Analyze arguments from pro, con, and synthesis viewpoints in structured format
- **Critical evaluation** - Examine claims with both supportive and skeptical lenses before reaching conclusions
- **Decision analysis** - Evaluate proposals, policies, or technical choices through dialectical reasoning
- **Trade-off assessment** - Understand benefits, risks, and nuances of complex decisions
- **Structured debate** - Move beyond simple answers to see arguments from multiple angles

## When NOT to Use

Avoid the ARGUMENT workflow when:

| Situation | Use Instead |
|-----------|-------------|
| Simple questions or facts | **CHAT** - Single-model conversation for straightforward queries |
| Need multiple model perspectives | **CONSENSUS** - Parallel multi-model consultation |
| Systematic investigation required | **THINKDEEP** - Hypothesis tracking and evidence-based research |
| Creative brainstorming | **IDEATE** - Structured idea generation |

## Three-Role Structure

The ARGUMENT workflow operates through three sequential roles, each providing a distinct perspective:

### 1. Creator (Pro) - Arguments in Favor

**Purpose:** Constructs the strongest possible case **supporting** the proposition.

**Responsibilities:**
- Present compelling arguments in favor of the claim
- Identify benefits, advantages, and positive outcomes
- Provide evidence and reasoning that supports the position
- Advocate for the proposition with conviction
- Highlight opportunities and potential value

**Perspective:** Optimistic advocate for the position

**Example outputs:**
- "GraphQL would significantly improve our API by enabling clients to request exactly the data they need..."
- "Adopting microservices provides better scalability and allows teams to work independently..."
- "Universal basic income could reduce poverty by providing guaranteed income security..."

### 2. Skeptic (Con) - Critical Challenge

**Purpose:** Challenges and critiques the argument with rigorous skepticism.

**Responsibilities:**
- Identify weaknesses, limitations, and potential problems
- Present counterarguments and opposing viewpoints
- Question assumptions, evidence, and reasoning
- Highlight risks, costs, and negative consequences
- Challenge claims with critical analysis

**Perspective:** Critical examiner questioning the position

**Example outputs:**
- "However, GraphQL introduces caching complexity and requires significant migration effort..."
- "Microservices add operational overhead, increase debugging complexity, and may be premature for our team size..."
- "Universal basic income faces funding challenges, potential work disincentives, and implementation barriers..."

### 3. Moderator (Synthesis) - Balanced Analysis

**Purpose:** Provides nuanced, balanced synthesis incorporating both perspectives.

**Responsibilities:**
- Weigh arguments from both Creator and Skeptic fairly
- Identify common ground and key trade-offs
- Acknowledge complexity and context-dependent factors
- Deliver conclusions that incorporate both sides
- Provide actionable recommendations when appropriate

**Perspective:** Impartial analyst seeking truth and balance

**Example outputs:**
- "GraphQL offers clear benefits for complex data needs but requires careful evaluation of caching strategies and migration costs. Best suited for..."
- "Microservices make sense at scale but may be premature for teams under 10 developers. Consider starting with a modular monolith..."
- "Universal basic income shows promise in pilot studies but faces significant implementation challenges. Success depends on..."

## How It Works

The ARGUMENT workflow provides structured dialectical analysis through three sequential roles:

### Workflow Steps

1. **Input Phase**
   - User provides argument, claim, or proposal
   - Optional: Context files for grounding
   - Optional: System prompt for constraints

2. **Creator Role (Pro)**
   - Constructs strongest possible case supporting the proposition
   - Presents benefits, advantages, and positive outcomes
   - Advocates for the position with evidence

3. **Skeptic Role (Con)**
   - Challenges arguments with rigorous criticism
   - Identifies weaknesses, limitations, and risks
   - Presents counterarguments and opposing viewpoints

4. **Moderator Role (Synthesis)**
   - Weighs arguments from both Creator and Skeptic
   - Identifies trade-offs and context-dependent factors
   - Delivers balanced conclusions incorporating both sides

5. **Output Phase**
   - Comprehensive multi-perspective analysis
   - Creator's argument + Skeptic's critique + Moderator's synthesis
   - Thread ID for conversation continuation

**Result:**

**Result:** A comprehensive, multi-perspective analysis that reveals complexity and avoids single-viewpoint bias.

## Detailed Use Cases

### Policy & Governance Analysis

**Example: Healthcare Policy**
```bash
model-chorus argument "Universal healthcare should be implemented in the US"
```

**What you get:**
- **Creator:** Evidence for improved health outcomes, reduced inequality, cost efficiency examples from other nations
- **Skeptic:** Implementation costs, funding mechanisms, potential wait times, system transition challenges
- **Moderator:** Balanced assessment of conditions for success, trade-offs, and context-dependent factors

**Example: Regulatory Decisions**
```bash
model-chorus argument "Cryptocurrency should be regulated like traditional securities"
```

### Technology & Architecture Decisions

**Example: API Architecture**
```bash
model-chorus argument "GraphQL should replace all REST APIs in our application" --file docs/current_api.md --file docs/client_requirements.md
```

**What you get:**
- **Creator:** Benefits of GraphQL (precise queries, type safety, single endpoint, reduced over-fetching)
- **Skeptic:** Challenges (caching complexity, migration cost, learning curve, tooling ecosystem)
- **Moderator:** Recommendation based on team size, timeline, and specific use cases

**Example: Framework Selection**
```bash
model-chorus argument "Our team should migrate from React to Svelte"
```

### Feature & Product Proposals

**Example: Feature Analysis**
```bash
model-chorus argument "We should add AI-powered code review to our platform"
```

**What you get:**
- **Creator:** User value, competitive advantage, automation benefits
- **Skeptic:** Development cost, accuracy concerns, support burden
- **Moderator:** Phased rollout recommendations with success metrics

**Example: Business Strategy**
```bash
model-chorus argument "Freemium model would be better than enterprise-only licensing"
```

### Research & Critical Analysis

**Example: Workplace Research**
```bash
model-chorus argument "Remote work increases overall productivity for knowledge workers"
```

**What you get:**
- **Creator:** Research showing productivity gains, reduced commute time, work-life balance improvements
- **Skeptic:** Collaboration challenges, communication overhead, potential isolation issues
- **Moderator:** Context-dependent analysis (depends on role type, team structure, company culture)

**Example: Technology Predictions**
```bash
model-chorus argument "Quantum computing will make classical cryptography obsolete within 10 years"
```

### Code Review Alternatives

**Example: Refactoring Evaluation**
```bash
model-chorus argument "This refactoring will improve maintainability without introducing bugs" --file src/auth.py --file src/auth_refactored.py --file tests/test_auth.py
```

**What you get:**
- **Creator:** Code quality improvements, better structure, reduced complexity
- **Skeptic:** Potential edge cases, performance concerns, test coverage gaps
- **Moderator:** Assessment of refactoring quality and recommended next steps

## CLI Usage

### Basic Invocation

**Simple argument analysis:**
```bash
model-chorus argument "Universal basic income would reduce poverty"
```

**Expected output:**
```
Analyzing argument: Universal basic income would reduce poverty

--- Creator (Pro) ---
[Arguments in favor of UBI reducing poverty]

--- Skeptic (Con) ---
[Counterarguments and limitations]

--- Moderator (Synthesis) ---
[Balanced analysis incorporating both perspectives]

Session ID: argument-thread-abc123
```

### With Provider Selection

**Specify which AI provider to use:**
```bash
# Use Claude (recommended for balanced analysis)
model-chorus argument "Remote work increases productivity" --provider claude

# Use Gemini (strong analytical capabilities)
model-chorus argument "Remote work increases productivity" --provider gemini

# Use Codex (excellent for technical arguments)
model-chorus argument "Remote work increases productivity" --provider codex
```

### With File Context

**Provide supporting documents:**
```bash
# Single file
model-chorus argument "This refactoring improves code quality" --file src/auth.py

# Multiple files
model-chorus argument "GraphQL would improve our API" --file docs/api_requirements.md --file docs/current_rest_api.md --file docs/performance_analysis.md
```

### With Continuation (Threading)

**Continue a previous argument analysis:**
```bash
# Initial analysis
model-chorus argument "TypeScript should replace JavaScript in our codebase"
# Returns: Session ID: argument-thread-xyz789

# Follow-up analysis
model-chorus argument "But what about the migration cost and learning curve?" --continue argument-thread-xyz789

# Further exploration
model-chorus argument "How would this affect our existing libraries?" --continue argument-thread-xyz789
```

**Why use continuation:**
- Builds on previous analysis context
- Explores specific angles in depth
- Maintains conversation coherence
- Saves tokens by reusing established context

### With Custom Configuration

**Temperature control:**
```bash
# Lower temperature for focused, analytical debate
model-chorus argument "Should we use microservices?" --temperature 0.5

# Higher temperature for more creative perspectives
model-chorus argument "How can we improve team productivity?" --temperature 0.9
```

**Token limit:**
```bash
model-chorus argument "Evaluate our caching strategy" --max-tokens 1500
```

**System prompt for context:**
```bash
model-chorus argument "Should we adopt Kubernetes?" --system "Our team has 5 developers, 3 microservices, budget under $10k/month"
```

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | The argument/claim to analyze |
| `system_prompt` | `str` | `None` | Additional context or constraints |
| `continuation_id` (aliases: `--continue`, `-c`, `--session-id`) | `str` | `None` | Thread ID to continue previous analysis (all aliases work identically) |
| `files` | `List[str]` | `[]` | File paths for context |

### Temperature Guidelines

| Range | Use Case | Characteristics |
|-------|----------|----------------|
| **0.0-0.4** | Technical analysis, security reviews | Focused, conservative, analytical |
| **0.5-0.7** | General arguments, policy analysis | Balanced, recommended for most uses |
| **0.8-1.0** | Creative proposals, brainstorming | Exploratory, unconventional perspectives |

### Provider-Specific Notes

**Claude:**
- Best for nuanced reasoning and balanced synthesis
- Recommended temperature: 0.6-0.8

**Gemini:**
- Strong analytical capabilities
- Recommended temperature: 0.5-0.7

**Codex:**
- Excellent for technical/code-related arguments
- Recommended temperature: 0.4-0.7

## Technical Contract

### Parameters

**Required:**
- `prompt` (string): The argument, claim, or proposal to analyze through three-role dialectical reasoning

**Optional:**
- `--provider` (string): AI provider to use for all three roles - Valid values: `claude`, `gemini`, `codex`, `cursor-agent` - Default: `claude`
- `--continue` / `-c` / `--session-id` (string): Session ID to continue previous argument analysis - All aliases work identically - Format: `argument-thread-{uuid}` - Maintains full conversation history across all three roles
- `--file, -f` (string, repeatable): File paths to include as context for analysis - Can be specified multiple times - Files must exist before execution - Provided to all three roles (Creator, Skeptic, Moderator)
- `--system` (string): Additional system prompt to customize analysis context - Applied to all three roles - Useful for providing constraints, background, or specific focus areas
- `--timeout` (float): Timeout per provider in seconds - Default: 120.0 - Prevents hanging on slow providers
- `--output, -o` (string): Path to save JSON output file - Creates or overwrites file at specified path
- `--verbose, -v` (boolean): Enable detailed execution information - Default: false - Shows per-role timing and execution details
- `--skip-provider-check` (boolean): Skip provider availability check for faster startup - Default: false

### Return Format

The ARGUMENT workflow returns a JSON object with the following structure:

```json
{
  "result": "=== Creator (Pro) ===\n[Arguments in favor...]\n\n=== Skeptic (Con) ===\n[Counterarguments...]\n\n=== Moderator (Synthesis) ===\n[Balanced analysis...]",
  "session_id": "argument-thread-abc-123-def-456",
  "metadata": {
    "provider": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "roles_executed": ["creator", "skeptic", "moderator"],
    "total_prompt_tokens": 450,
    "total_completion_tokens": 900,
    "total_tokens": 1350,
    "temperature": 0.7,
    "timestamp": "2025-11-07T10:30:00Z",
    "role_details": {
      "creator": {
        "prompt_tokens": 150,
        "completion_tokens": 300,
        "response_time_seconds": 2.1
      },
      "skeptic": {
        "prompt_tokens": 150,
        "completion_tokens": 300,
        "response_time_seconds": 2.3
      },
      "moderator": {
        "prompt_tokens": 150,
        "completion_tokens": 300,
        "response_time_seconds": 2.5
      }
    }
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | Complete three-role analysis with Creator's pro arguments, Skeptic's counterarguments, and Moderator's balanced synthesis - Formatted with clear role headers |
| `session_id` | string | Session ID for continuing this argument analysis (format: `argument-thread-{uuid}`) |
| `metadata.provider` | string | The AI provider used for all three roles |
| `metadata.model` | string | Specific model version used by the provider |
| `metadata.roles_executed` | array[string] | List of roles executed in sequence: `["creator", "skeptic", "moderator"]` |
| `metadata.total_prompt_tokens` | integer | Total tokens across all three role prompts |
| `metadata.total_completion_tokens` | integer | Total tokens across all three role responses |
| `metadata.total_tokens` | integer | Total tokens consumed (sum of prompt and completion) |
| `metadata.temperature` | float | Temperature setting used for all three roles (0.0-1.0) |
| `metadata.timestamp` | string | ISO 8601 timestamp of when the analysis was completed |
| `metadata.role_details` | object | Per-role execution details including tokens and timing for Creator, Skeptic, and Moderator |

**Usage Notes:**
- Save the `session_id` value to continue dialectical analysis with follow-up questions or deeper exploration
- The `result` contains all three perspectives in sequence with clear headers for each role
- Token counts reflect the cost of running all three roles (Creator + Skeptic + Moderator)
- Each role builds on context from previous roles within the same invocation
- When using `--continue`, all three roles have access to the full conversation history
- The Moderator's synthesis always incorporates both Creator and Skeptic perspectives

## Best Practices

### 1. Frame Arguments Clearly

**Good:**
```bash
# Clear, specific claim
model-chorus argument "Adopting TypeScript would improve code quality in our project"
```

**Less Effective:**
```bash
# Vague, open-ended question
model-chorus argument "What do you think about TypeScript?"
```

### 2. Provide Context When Needed

**Without context:**
```bash
model-chorus argument "Should we use Redis?"
```

**With helpful context:**
```bash
model-chorus argument "Should we use Redis for session storage?" --system "We have 10k daily active users, need < 50ms latency, budget $200/month" --file docs/current_architecture.md
```

### 3. Use Continuation for Deep Dives

Don't cram everything into one prompt:

```bash
# Better: Start broad, then drill down
model-chorus argument "Microservices vs monolith for our scale"
# Session ID: argument-thread-abc123

model-chorus argument "Focus on operational complexity" --continue argument-thread-abc123

model-chorus argument "What about team organization?" --continue argument-thread-abc123
```

### 4. Match Temperature to Use Case

```bash
# Security analysis: be conservative
model-chorus argument "Is this auth approach secure?" --temperature 0.4

# Product strategy: be creative
model-chorus argument "How should we position our product?" --temperature 0.9
```

## Real-World Examples

### Example 1: Technology Decision

```bash
model-chorus argument "Our team should migrate from REST to GraphQL" --file docs/current_api.md --file docs/client_requirements.md --temperature 0.7
```

**Expected output:**
- **Creator:** Benefits of GraphQL (precise queries, type safety, single endpoint)
- **Skeptic:** Challenges (caching complexity, migration cost, learning curve)
- **Moderator:** Balanced recommendation based on team size, timeline, constraints

### Example 2: Policy Analysis

```bash
model-chorus argument "Universal healthcare would improve public health outcomes" --file research/healthcare_studies.pdf --file research/cost_analysis.xlsx --temperature 0.6
```

**Expected output:**
- **Creator:** Evidence for improved health outcomes, reduced inequality
- **Skeptic:** Cost concerns, implementation challenges, funding mechanisms
- **Moderator:** Nuanced analysis of trade-offs and conditions for success

### Example 3: Code Review Alternative

```bash
model-chorus argument "This refactoring improves maintainability without bugs" --file src/old_implementation.py --file src/new_implementation.py --file tests/test_suite.py --temperature 0.5
```

**Expected output:**
- **Creator:** Code quality improvements, better structure
- **Skeptic:** Potential edge cases, performance concerns, test coverage gaps
- **Moderator:** Assessment of refactoring quality and recommended next steps

## Progress Reporting

The ARGUMENT workflow automatically displays progress updates to stderr as it executes. You will see messages like:

```
Starting argument workflow (estimated: 15-30s)...
[Creator] Starting...
✓ argument workflow complete
```

**Important:** Progress updates are emitted automatically - do NOT use `BashOutput` to poll for progress. Simply invoke the command and wait for completion. All progress information streams automatically to stderr without interfering with stdout.

## Comparison with Other Workflows

### ARGUMENT vs CHAT

| Aspect | ARGUMENT | CHAT |
|--------|----------|------|
| **Structure** | Three roles (Creator/Skeptic/Moderator) | Simple back-and-forth |
| **Best for** | Analyzing arguments, balanced perspectives | Quick questions, iteration |
| **Cost** | Higher (3 responses per query) | Lower (1 response per query) |
| **Output** | Pro/Con/Synthesis | Direct answer |
| **Use when** | Need balanced analysis of a claim | Need quick consultation |

### ARGUMENT vs CONSENSUS

| Aspect | ARGUMENT | CONSENSUS |
|--------|----------|-----------|
| **Models** | Single model (3 roles) | Multiple models |
| **Perspectives** | Pro/Con from same model | Different model perspectives |
| **Best for** | Dialectical reasoning | Multi-model comparison |
| **Threading** | Yes | No |
| **Use when** | Analyze arguments structurally | Compare model approaches |

### ARGUMENT vs THINKDEEP

| Aspect | ARGUMENT | THINKDEEP |
|--------|----------|-----------|
| **Process** | Dialectical debate | Systematic investigation |
| **Structure** | Pro/Con/Synthesis | Hypothesis testing |
| **Best for** | Evaluating claims | Debugging, research |
| **Confidence** | Not tracked | Tracked (exploring → certain) |
| **Use when** | Need balanced argument analysis | Need evidence-based investigation |

## See Also

**Related Workflows:**
- **CHAT** - Simple single-model conversation for quick questions
- **CONSENSUS** - Multi-model perspectives for cross-validation
- **THINKDEEP** - Systematic investigation with hypothesis tracking
- **IDEATE** - Creative brainstorming and idea generation
