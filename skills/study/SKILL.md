---
name: study
description: Persona-based collaborative research tool with intelligent role-based orchestration and conversation threading
---

# STUDY Workflow

## Overview

The STUDY workflow enables **multi-persona collaborative investigation and analysis** by orchestrating multiple AI personas with distinct expertise to explore complex topics from different perspectives. Unlike running the same model multiple times, STUDY uses intelligent routing to select relevant personas based on investigation context, then synthesizes their diverse findings into comprehensive analysis.

**Key Capabilities:**
- Multi-persona investigation with intelligent routing
- Conversation threading for multi-turn research sessions
- Persona-specific expertise and systematic synthesis
- Investigation continuation across sessions
- Context-aware analysis with file support

**Use Cases:**
- Complex topic exploration from multiple expert perspectives
- Code and system analysis with specialized viewpoints
- Research investigation with iterative deepening
- Collaborative problem analysis (simulating team expertise)
- Learning complex topics systematically

---

## When to Use

Use STUDY when you need to:

- **Multi-perspective analysis** - Topic benefits from multiple expert viewpoints (security, architecture, performance)
- **Complex exploration** - Requires systematic investigation from different angles
- **Research investigation** - Need to deepen understanding iteratively across sessions
- **Collaborative analysis** - Simulate team expertise and perspective diversity
- **Code/system review** - Analyze from multiple specialized perspectives

---

## When NOT to Use

| Situation | Use Instead | Key Difference |
|-----------|-------------|-----------------|
| Structured debate of single proposition | ARGUMENT | ARGUMENT uses pro/con/synthesis debate structure; STUDY explores topics from expert perspectives |
| Want multiple AI models' perspectives | CONSENSUS | CONSENSUS uses different AI models; STUDY uses single model with multiple personas |
| Hypothesis testing with confidence tracking | THINKDEEP | THINKDEEP tracks hypothesis evolution; STUDY explores collaboratively |
| Simple Q&A or conversational help | CHAT | CHAT is simpler and faster; STUDY adds persona expertise layer |
| Creative brainstorming without expertise | IDEATE | IDEATE generates ideas; STUDY analyzes using expert perspectives |

---

## Basic Usage

### Simple Example

```bash
model-chorus study start --scenario "What are the main patterns in authentication systems?"
```

**Expected Output:**
- Thread ID for later continuation
- Investigation status (new or continuation)
- Persona contributions (each labeled with persona name)
- Synthesis combining all perspectives
- Hint for continuation with thread ID

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--scenario` | (none) | (required) | Investigation topic or research question |
| `--provider` | `-p` | `gemini` | AI provider (`claude`, `gemini`, `codex`, `cursor-agent`) |
| `--persona` | (none) | `Researcher, Critic` | Specific personas to use (repeatable) |
| `--file` | `-f` | None | Context files to include (repeatable) |
| `--continue` / `--session-id` | `-c` | None | Thread ID to continue existing investigation (both aliases work identically) |
| `--output` | `-o` | None | Save results to JSON file |
| `--verbose` | `-v` | False | Show detailed execution information |

---

## Advanced Usage

### With Provider Selection

```bash
# Use specific provider
model-chorus study start --scenario "Analyze authentication patterns" -p gemini
```

All personas use the selected provider. Choose based on capabilities, cost, and speed.

### With Persona Selection (CRITICAL - Primary Differentiator)

```bash
model-chorus study start --scenario "Design distributed consensus algorithm" --persona TheoreticalMathematician --persona PracticalEngineer --persona SecurityExpert
```

**Why personas matter:**
- Personas are NOT just "run with different temperature"
- Each has specific expertise area (security, architecture, performance, etc.)
- Each has distinct role (primary investigator, critic, specialist)
- Intelligent routing means not all personas always contribute
- Synthesis intelligently combines findings, not just concatenates

### With File Context

```bash
model-chorus study start --scenario "Analyze this authentication system" --file src/auth.py --file src/middleware.ts --persona SecurityExpert --persona Architect
```

Files provide context grounding for all personas. Each analyzes from their perspective.

### With Conversation Threading (Continuation)

```bash
# Day 1: Initial investigation
thread=$(model-chorus study start --scenario "Learn Kubernetes architecture" --output step1.json | grep "Thread ID:" | awk '{print $NF}')

# Day 2: Continue investigation
model-chorus study start --scenario "Deep dive into service discovery" --continue "$thread" --output step2.json

# Day 3: Another angle
model-chorus study start --scenario "Networking and ingress patterns" --continue "$thread" --output step3.json
```

Full conversation history is preserved. Personas aware of all previous findings and context.

### Adjusting Creativity

```bash
# Lower temperature for analytical, deterministic responses
model-chorus study start --scenario "Security analysis of authentication flow" --temperature 0.3

# Higher temperature for more creative exploration
model-chorus study start --scenario "Generate architectural patterns" --temperature 0.9

# Default balanced setting
model-chorus study start --scenario "General investigation" --temperature 0.7
```

### Saving Results

```bash
model-chorus study start --scenario "Research topic" --output results.json
```

JSON output preserves all metadata: routing decisions, persona contributions, synthesis, token usage.

### Verbose Output

```bash
model-chorus study start --scenario "Investigation topic" --verbose
```

Shows provider, model version, execution timing, token counts, and detailed routing information.

---

## Workflow-Specific Features

### Multi-Persona Investigation

#### What are personas?

Personas are distinct AI entities with specific expertise and perspectives:

- **Expertise Area** (e.g., security, performance, architecture, code quality)
- **Role** (e.g., primary investigator, critical reviewer, specialist)
- **Perspective** - Brings unique viewpoint to investigation
- **Context Retention** - Remembers previous findings across phases

#### Default personas

- **Researcher**: Systematic investigation, factual analysis, synthesis-focused
- **Critic**: Challenges assumptions, identifies edge cases, questions conclusions

#### Why personas matter

```
Standard LLM approach:
  "Run same model twice with different temperatures"
  → Similar responses, minor variations

STUDY approach:
  "Invoke distinct personas with specific expertise"
  → Each brings unique perspective
  → Routing selects relevant personas
  → Synthesis identifies genuine insights and tradeoffs
```

#### Custom personas

```bash
# Security review team
model-chorus study start --scenario "Review authentication code" --persona SecurityExpert --persona CodeReviewer --persona Architect
```

**Persona selection framework:**
- Match personas to investigation question
- Avoid redundant persona combinations
- Use 2-3 personas for quick analysis, 4+ for comprehensive
- Consider which expertise areas are needed

#### Common persona combinations

| Investigation Type | Recommended Personas | Rationale |
|-------------------|----------------------|-----------|
| Security analysis | SecurityExpert, Architect | Security + design perspectives |
| Performance optimization | PerformanceEngineer, Architect | Technical + design angles |
| Code review | CodeReviewer, SecurityExpert, Architect | Maintainability, security, design |
| System learning | TeachingExpert, PracticalEngineer, Theorist | Conceptual + practical + foundational |
| Design decision | Architect, SecurityExpert, PerformanceEngineer | All major considerations |
| Quick analysis | Researcher, Critic | Fast, balanced baseline |

---

### Intelligent Routing

#### What is routing?

The system automatically selects which personas contribute to each investigation step:

- **Phase-aware**: Different personas relevant in different phases
- **Context-sensitive**: Routes based on investigation direction
- **Efficient**: Avoids unnecessary invocations
- **Transparent**: Explains why personas were selected/skipped

#### Routing benefits

- **Efficiency**: Only invokes relevant personas, reducing token usage
- **Relevance**: Routes to experts suited for current question
- **Flexibility**: Adapts based on investigation direction
- **Transparency**: Explains routing decisions

#### Example routing scenario

```
User: "Design a payment system architecture"

Personas selected:
✓ Architect (relevant: design patterns, system architecture)
✓ SecurityExpert (relevant: payment processing security)
✓ PerformanceEngineer (relevant: transaction throughput)

NOT selected:
✗ DatabaseSpecialist (less relevant for initial architecture)
✗ DevOpsEngineer (not needed for design phase)

Rationale: Architecture and security are critical; performance is important;
database and ops decisions come later.
```

#### Interpreting routing decisions

Check JSON output (`--output` flag) for `routing_metadata`:
- Each step shows which persona contributed
- Understand why certain personas were selected
- Learn what personas were considered and why skipped

---

### Investigation Phases

Investigation organizes into systematic phases with different personas becoming relevant:

**Phase 1: Exploration**
- Understanding context, gathering information
- Usually: All or most personas contribute
- Goal: Comprehensive overview

**Phase 2: Deep Dive**
- Detailed analysis of specific aspects
- Usually: Expert-specific personas
- Goal: Technical understanding

**Phase 3: Synthesis**
- Combining findings, identifying patterns, tradeoffs
- Usually: Synthesis-focused personas
- Goal: Actionable conclusions

**Phase 4: Validation** (if multi-turn)
- Checking edge cases, completeness
- Usually: Critical/questioning personas
- Goal: Robustness verification

#### Progression example

```
Turn 1: "Explain authentication patterns"
  → Exploration phase: Researcher + Critic explore basics

Turn 2: "Deep dive into OAuth 2.0 security"
  → Deep Dive phase: SecurityExpert + Architect focus

Turn 3: "Compare OAuth vs SAML in enterprise context"
  → Synthesis phase: All personas integrate perspectives

Turn 4: "What about SAML edge cases in federated scenarios?"
  → Validation phase: SecurityExpert challenges assumptions
```

---

### Conversation Memory and Persistence

#### Memory system

All investigations persist to enable resumption:

- **Thread IDs**: Unique identifier for investigation (UUID)
- **Full History**: All messages, timestamps, metadata
- **Persona Attribution**: Each message tracked with persona
- **Investigation Metadata**: Context, phase, routing decisions
- **Cross-Session**: Resume any time without context loss

#### Practical multi-turn investigation

```bash
# Day 1: Initial research
model-chorus study start --scenario "Learn Kubernetes"
# Returns: Thread ID: abc-123-def

# Day 2: Continue research - full context preserved
model-chorus study start --scenario "Deep dive into service discovery" --continue abc-123-def

# Day 3: Another angle - all previous findings available
model-chorus study start --scenario "Networking and ingress in detail" --continue abc-123-def

# Day 4: Review all findings
model-chorus study view --investigation abc-123-def --show-all
```

---

### Three-Command CLI Structure

STUDY has three commands (vs single command for other workflows):

#### `study start` - Primary command

Starts new investigation OR continues existing:
- Full investigation orchestration
- Persona routing and execution
- Returns thread ID
- Supports all parameters

#### `study-next` - Convenience wrapper

Streamlined syntax for continuation:
```bash
model-chorus study-next --investigation abc-123 --scenario "Next question"
```

Automatically uses previous thread context, less boilerplate.

#### `study view` - Investigation inspection

View investigation history and findings:
```bash
# View summary
model-chorus study view --investigation abc-123

# View specific persona
model-chorus study view --investigation abc-123 --persona SecurityExpert

# Full history
model-chorus study view --investigation abc-123 --show-all

# JSON for processing
model-chorus study view --investigation abc-123 --json
```

---

## Decision Guide

### Choosing Personas (PRIMARY DECISION POINT)

Persona selection is STUDY's most important decision point:

**Decision Framework:**
1. What expertise does the question need?
2. Which perspectives are missing?
3. What angles need exploration?
4. How comprehensive? (2-3 for quick, 4+ for thorough)

**Persona Selection Table:**

| Investigation Type | Recommended Personas | Rationale |
|-------------------|----------------------|-----------|
| Security analysis | SecurityExpert, Architect | Security + design perspectives |
| Performance optimization | PerformanceEngineer, Architect | Technical + design angles |
| Code review | CodeReviewer, SecurityExpert, Architect | Maintainability, security, design |
| System learning | TeachingExpert, PracticalEngineer, Theorist | Conceptual + practical + foundational |
| Design decision | Architect, SecurityExpert, PerformanceEngineer | All major considerations |
| Quick analysis | Researcher, Critic | Fast, balanced baseline |

### Provider Selection

Provider choice affects all personas (single provider for all):

| Provider | When to Use |
|----------|------------|
| claude | Default, balanced capabilities, most reliable |
| gemini | Try if Claude unavailable, different perspective |
| codex | Fast responses, good for code analysis |
| cursor-agent | Integration with Cursor development |

### Continuation Strategy

**Progressive Deepening:**
```
Turn 1: Overview/foundation
Turn 2: Specific aspects
Turn 3: Deeper analysis
Turn 4: Edge cases/validation
```

**Multi-Angle Analysis:**
```
Turn 1: Persona set A
Turn 2: Persona set B (different expertise)
Turn 3: All personas integrate
```

### File Context Strategy

| Scenario | Approach | Example |
|----------|----------|---------|
| Single file analysis | Include file, default personas | `--file auth.py` |
| Multi-file review | Multiple files, expert personas | `--file *.py --persona Architect` |
| Security review | Code files, SecurityExpert | `--file app.ts --persona SecurityExpert` |
| Learning from code | Include codebase, Teacher personas | `--file project/** --persona TeachingExpert` |

---

## Common Patterns

### Pattern 1: Quick Analysis with Default Personas

```bash
model-chorus study start --scenario "Microservices vs monolith for our team?"
```

**When:** Need fast, balanced perspective without deep customization

**Expected:** Researcher explores pros/cons, Critic questions assumptions

---

### Pattern 2: Specialized Expert Review

```bash
model-chorus study start --scenario "Security analysis of authentication flow" --persona SecurityExpert --persona Architect --file src/auth/*.ts --temperature 0.4
```

**When:** Need expert-specific analysis with code context

**Expected:** SecurityExpert identifies vulnerabilities, Architect reviews design

---

### Pattern 3: Iterative Multi-Turn Investigation

```bash
# Turn 1: Foundation
thread=$(model-chorus study start --scenario "What is event-driven architecture?" --output step1.json | grep "Thread ID:" | awk '{print $NF}')

# Turn 2: Deep dive
model-chorus study start --scenario "Compare event-driven vs request-response architectures" --continue "$thread" --output step2.json

# Turn 3: Practical application
model-chorus study start --scenario "Design event-driven order processing system" --continue "$thread" --persona Architect --persona PerformanceEngineer --output step3.json
```

**When:** Building knowledge progressively, each turn deepens understanding

**Expected:** Context preserved, insights build on previous findings

---

### Pattern 4: Code Review from Multiple Angles

```bash
model-chorus study start --scenario "Review microservice implementation for quality" --file src/service/main.ts --file src/service/handlers.ts --file tests/service.spec.ts --persona CodeReviewer --persona SecurityExpert --persona PerformanceEngineer --output code_review.json
```

**When:** Comprehensive code review needed from multiple perspectives

**Expected:** CodeReviewer finds maintainability issues, SecurityExpert finds vulnerabilities, PerformanceEngineer identifies optimizations

---

### Pattern 5: Team Expertise Simulation

```bash
model-chorus study start --scenario "Design distributed caching layer for our platform" --persona Architect --persona PerformanceEngineer --persona DatabaseSpecialist --persona DevOpsEngineer --temperature 0.6
```

**When:** Simulating team discussion and diverse perspectives

**Expected:** Each persona contributes from their domain, synthesis identifies tradeoffs

---

### Pattern 6: Systematic Learning

```bash
model-chorus study start --scenario "Learn GraphQL fundamentals and best practices" --persona TeachingExpert --persona PracticalEngineer --persona ArchitectureExpert --temperature 0.4
```

**When:** Structured learning from multiple teaching approaches

**Expected:** TeachingExpert explains concepts, PracticalEngineer shows implementation, Architect discusses design

---

### Pattern 7: Decision Documentation

```bash
# Initial comparison
thread=$(model-chorus study start --scenario "PostgreSQL vs MongoDB for our e-commerce platform" --output decision_step1.json | grep "Thread ID:")

# Deep dive
model-chorus study start --scenario "PostgreSQL vs MongoDB: ACID guarantees for financial operations" --continue "$thread" --persona DatabaseSpecialist --persona PerformanceEngineer --output decision_step2.json

# Performance analysis
model-chorus study start --scenario "Query performance comparison at 10k QPS load" --continue "$thread" --persona PerformanceEngineer --output decision_step3.json
```

**When:** Documenting technical decision-making process with rationale

**Expected:** Complete analysis with tradeoffs documented

---

## Output Format

### Console Output

```
✓ STUDY investigation completed

Thread ID: 550e8400-e29b-41d4-a716-446655440000
Status: New investigation started
Personas: SecurityExpert, Architect, PerformanceEngineer
Investigation Round: 1

[SecurityExpert]:
Security analysis of the proposed architecture...

[Architect]:
Design and scalability considerations...

[PerformanceEngineer]:
Performance implications and optimization opportunities...

Research Synthesis:
Integrated analysis combining all perspectives, tradeoffs, and recommendations...

To continue this investigation, use: --continue 550e8400-e29b-41d4-a716-446655440000
```

### JSON Output (with --output)

```json
{
  "scenario": "Investigation topic",
  "provider": "claude",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_continuation": false,
  "investigation_round": 1,
  "personas_used": ["SecurityExpert", "Architect", "PerformanceEngineer"],
  "routing_metadata": {
    "phase": "exploration",
    "routing_decisions": [
      {"persona": "SecurityExpert", "reason": "Security-focused question"},
      {"persona": "Architect", "reason": "Design consideration"},
      {"persona": "PerformanceEngineer", "reason": "Performance impact"}
    ]
  },
  "steps": [
    {
      "step_number": 1,
      "persona": "SecurityExpert",
      "content": "Security analysis...",
      "metadata": {"phase": "exploration"}
    }
  ],
  "synthesis": "Integrated findings...",
  "model": "claude-3-opus",
  "usage": {"total_tokens": 4500}
}
```

**Key JSON fields:**
- `thread_id`: Use for continuation with `--continue`
- `personas_used`: Actual personas that contributed
- `routing_metadata`: Why personas were selected
- `steps`: Individual persona contributions
- `synthesis`: Integrated findings

### View Command Output

```
STUDY Investigation Memory

Investigation ID: 550e8400-e29b-41d4-a716-446655440000
Total Messages: 12
Personas: SecurityExpert, Architect, PerformanceEngineer

Conversation History:

[Message 1]
  Role: user
  Content: Investigation topic...
  Timestamp: 2025-11-08T14:30:00Z

[Message 2]
  Role: assistant (SecurityExpert)
  Content: Security analysis...
  Timestamp: 2025-11-08T14:30:45Z

...
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No providers available` | API key not configured | Set `ANTHROPIC_API_KEY` environment variable |
| `Investigation not found` | Wrong thread ID | Verify thread ID with `ls ~/.model-chorus/conversations/` |
| `File not found` | Invalid file path in `-f` | Verify file paths exist with `ls` |
| `Invalid persona` | Persona name not recognized | Use default personas or check available ones |
| `Interrupted by user` | Ctrl+C during investigation | Resume with `--continue <thread-id>` |

### Troubleshooting Commands

```bash
# Check provider availability
model-chorus list-providers --check

# List all investigations
ls -la ~/.model-chorus/conversations/

# Inspect investigation
model-chorus study view --investigation <thread-id> --json | jq '.messages | length'

# Resume interrupted investigation
model-chorus study start --scenario "Continue investigation" --continue <thread-id>
```

---

## Best Practices

1. **Choose personas intentionally** - Select personas whose expertise matches investigation question; avoid redundant combinations

2. **Use continuation for complex investigations** - Multi-turn preserves context better than single long prompt

3. **Start with default personas for exploration** - Add specialists when specific expertise needed

4. **Adjust temperature based on investigation type** - 0.3-0.5 for analysis, 0.7+ for exploration

5. **Leverage file context for code analysis** - Ground investigation in real code, not abstractions

6. **Save investigation results** - JSON preserves metadata, routing decisions, token usage

7. **Review findings before continuing** - Use `study view` to refresh context before next turn

---

## Examples

### Example 1: Security Code Review

**Scenario:** Security team reviewing authentication code

**Command:**
```bash
model-chorus study start --scenario "Review authentication implementation for vulnerabilities, design issues, and code quality concerns" --file src/auth/oauth.ts --file src/auth/jwt.ts --file src/auth/middleware.ts --persona SecurityExpert --persona Architect --persona CodeReviewer --temperature 0.4 --output security_review.json
```

**Expected outcome:**
- SecurityExpert identifies potential vulnerabilities
- Architect evaluates design patterns and scalability
- CodeReviewer notes maintainability and code quality issues
- Synthesis provides comprehensive review with prioritized recommendations

---

### Example 2: Technology Selection Decision

**Scenario:** Database technology selection for e-commerce platform

**Initial comparison:**
```bash
thread=$(model-chorus study start --scenario "Compare PostgreSQL vs MongoDB for e-commerce transactions" --persona DatabaseSpecialist --persona PerformanceEngineer --output decision_comparison.json | grep "Thread ID:")
```

**Deep dive on requirements:**
```bash
model-chorus study start --scenario "Focus on ACID guarantees and transaction consistency for financial operations" --continue "$thread" --persona DatabaseSpecialist --persona SecurityExpert --output decision_acid.json
```

**Performance analysis:**
```bash
model-chorus study start --scenario "Query performance and throughput under 10k QPS load" --continue "$thread" --persona PerformanceEngineer --output decision_performance.json
```

**Expected outcome:**
- Complete analysis with all perspectives documented
- Tradeoffs clearly identified
- Recommendation with rationale from multiple angles

---

### Example 3: Codebase Learning for New Developer

**Scenario:** New team member learning authentication system

**Day 1 - Architecture overview:**
```bash
model-chorus study start --scenario "Explain the authentication system architecture and data flow" --file src/auth/** --persona TeachingExpert --persona PracticalEngineer --temperature 0.5 --output auth_day1.json
```

**Day 2 - OAuth implementation deep dive:**
```bash
thread=$(ls ~/.model-chorus/conversations/ | tail -1 | sed 's/.json//')
model-chorus study start --scenario "Deep dive into OAuth 2.0 implementation details and flow" --file src/auth/oauth.ts --continue "$thread" --persona DetailedEngineer --persona TeachingExpert --output auth_day2.json
```

**Day 3 - Security and edge cases:**
```bash
model-chorus study start --scenario "Security considerations and edge cases in the implementation" --continue "$thread" --persona SecurityExpert --persona Critic --output auth_day3.json
```

**Expected outcome:**
- Systematic learning progression
- Conceptual understanding + practical implementation + security awareness
- Full context preserved across days

---

## Troubleshooting

### Issue: Investigation feels incomplete

**Symptoms:** Synthesis doesn't address all aspects; obvious angles missed

**Solutions:**
- Re-run with different/additional personas
- Ask more specific follow-up questions
- Increase temperature for more creative exploration
- Add more context files
- Use continuation for deeper investigation

---

### Issue: Persona perspectives are redundant

**Symptoms:** Multiple personas saying similar things; lack of diversity

**Solutions:**
- Choose more distinct persona combinations
- Ask more specific, differentiated questions
- Highlight what unique perspective each persona should bring
- Use continuation to explore different angles separately

---

### Issue: Continuation losing context

**Symptoms:** Second turn ignores or contradicts earlier findings

**Solutions:**
- Verify correct thread ID with `model-chorus study view --investigation <id>`
- Review previous findings before continuing
- Add explicit context reminder in prompt
- Check JSON output for conversation history

---

## Related Workflows

### STUDY vs ARGUMENT

**Difference:**
- **ARGUMENT**: Structured debate with Creator/Skeptic/Moderator roles
- **STUDY**: Collaborative investigation with custom personas

**When to use each:**
- Use ARGUMENT: "Should we adopt X?" (binary decision with debate structure)
- Use STUDY: "What are patterns in X?" (exploratory investigation with expertise)

**Combining:** Use STUDY for investigation, then ARGUMENT to debate conclusions if decision needed

---

### STUDY vs CONSENSUS

**Difference:**
- **CONSENSUS**: Multiple different AI models (different models, same question)
- **STUDY**: Single model with multiple personas (same model, different roles)

**Cost/speed:** STUDY faster (single provider); CONSENSUS more diverse (multiple models)

**When:**
- Use CONSENSUS: Need different AI model perspectives (Claude vs Gemini vs Codex)
- Use STUDY: Need multi-expert analysis within single model

---

### STUDY vs THINKDEEP

**Difference:**
- **THINKDEEP**: Hypothesis testing with confidence tracking and evolution
- **STUDY**: Collaborative exploration with personas

**When:**
- Use THINKDEEP: Debugging, root cause analysis, hypothesis testing, confidence tracking
- Use STUDY: Complex topic exploration, learning, multi-expert analysis

---

### STUDY vs CHAT

**Difference:**
- **CHAT**: Simple conversation
- **STUDY**: Multi-persona investigation with expertise

**When:**
- Use CHAT: Quick Q&A, simple questions, general conversation
- Use STUDY: Need multiple expert perspectives, systematic investigation

---

### STUDY vs IDEATE

**Difference:**
- **IDEATE**: Generate creative ideas and brainstorm
- **STUDY**: Analyze topics from expert perspectives

**When:**
- Use IDEATE: Brainstorming, generating new ideas, creative exploration
- Use STUDY: Analysis, understanding, review, expert assessment

---

## See Also

- STUDY Workflow README: `model_chorus/src/model_chorus/workflows/study/README.md`
- CLI Reference: `model_chorus/src/model_chorus/workflows/study/CLI_REFERENCE.md`
- Memory System: `model_chorus/src/model_chorus/workflows/study/MEMORY_AND_LIFECYCLE.md`
- Provider Information: `model-chorus list-providers`
- General CLI Help: `model-chorus --help`
