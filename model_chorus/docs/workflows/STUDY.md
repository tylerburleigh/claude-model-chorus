# STUDY Workflow - Persona-Based Collaborative Research

## Overview

The STUDY workflow enables **multi-persona collaborative research and investigation** through intelligent role-based orchestration. Multiple AI personas with distinct expertise and perspectives investigate complex topics, with intelligent routing selecting relevant personas for each phase of investigation.

## Key Concepts

### What is the STUDY Workflow?

The STUDY workflow is designed for situations where a complex question or topic benefits from **multiple perspectives**. Instead of asking a single AI model to think through all angles, the STUDY workflow:

1. **Organizes multiple personas** with different expertise areas and perspectives
2. **Routes investigation questions** intelligently to the most relevant personas
3. **Maintains conversation memory** across multiple turns, preserving context and findings
4. **Synthesizes findings** from all personas into comprehensive conclusions
5. **Supports continuation** of investigations across multiple sessions

### Core Architecture

```
User Question/Topic
        ↓
   PersonaRouter (intelligent selection)
        ↓
  [Persona 1] [Persona 2] [Persona 3] ...
        ↓         ↓         ↓
     Findings  Findings  Findings
        ↓
    Synthesis Engine
        ↓
  Comprehensive Analysis
        ↓
ConversationMemory (persistence)
```

## Use Cases

### 1. Complex Topic Exploration
**Scenario**: You need to understand authentication system patterns from multiple angles.

**Example**:
```bash
model-chorus study start \
  --scenario "Explore the implementation patterns of authentication systems" \
  --personas SecurityExpert Architect Developer
```

**What happens**:
- **Security Expert** analyzes threat models and security best practices
- **Architect** examines scalability and design patterns
- **Developer** considers implementation practicality and maintenance

**Output**: A comprehensive analysis combining all three perspectives

### 2. Research Investigation with Continuation
**Scenario**: You're conducting ongoing research that benefits from iterative exploration.

**Initial investigation**:
```bash
model-chorus study start \
  --scenario "Analyze OAuth 2.0 security model" \
  --scenario-context authentication_system.md
```

**Later, continue deeper investigation**:
```bash
model-chorus study start \
  --scenario "Dive deeper into OAuth 2.0 refresh token rotation" \
  --continue thread-id-from-previous
```

**What happens**:
- The workflow retrieves the previous investigation thread
- New personas build on previous findings
- Conversation history provides full context
- Investigation deepens with new insights

### 3. Collaborative Problem Analysis
**Scenario**: Your team is solving a distributed systems problem that needs multiple expert viewpoints.

**Example**:
```bash
model-chorus study start \
  --scenario "Design a consensus algorithm for blockchain" \
  --personas TheoreticianWhoFocusesOnProofsAndMath \
               PracticalEngineerWhoFocusesOnImplementation \
               SecurityExpertWhoFocusesOnAttacks
```

**What happens**:
- Each persona approaches the design from their specialization
- Different personas question each other's assumptions
- The synthesis identifies tradeoffs and optimal solutions
- Results capture both theoretical correctness and practical viability

### 4. Document and Code Analysis
**Scenario**: You need to understand complex code or documentation from multiple viewpoints.

**Example**:
```bash
model-chorus study start \
  --scenario "Analyze this authentication codebase for patterns and vulnerabilities" \
  --file src/auth.py \
  --file src/middleware/auth_middleware.ts \
  --personas SecurityExpert Architect Maintainer
```

**What happens**:
- Security Expert identifies potential vulnerabilities
- Architect evaluates overall design quality
- Maintainer perspective highlights sustainability issues
- Synthesis shows the complete picture

### 5. Learning and Knowledge Building
**Scenario**: You're learning a new technology and want systematic exploration.

**Example**:
```bash
model-chorus study start \
  --scenario "Learn about Kubernetes architecture" \
  --personas ConceptualTeacher DetailOrientedEngineer PracticalApplicationExpert
```

**What happens**:
- Conceptual Teacher explains foundational ideas
- Detailed Engineer explores implementation specifics
- Practical Expert shows real-world patterns
- You get a complete learning path with multiple reinforcement angles

## Workflow Features

### Multi-Persona Investigation

The STUDY workflow organizes investigation through distinct personas:

**Persona Characteristics**:
- **Name**: Unique identifier (e.g., "SecurityExpert", "Researcher")
- **Expertise**: Area of specialization (e.g., "security analysis and threat modeling")
- **Role**: Function in investigation (e.g., "primary investigator", "critical reviewer")
- **Perspective**: Unique viewpoint they bring
- **Memory**: Retained context from previous turns

**Default Personas**:
- **Researcher**: Systematic investigation and analysis
- **Critic**: Identifying assumptions, edge cases, and weaknesses

You can customize personas via CLI:
```bash
model-chorus study start \
  --scenario "..." \
  --persona SecurityExpert \
  --persona Architect \
  --persona Developer
```

### Intelligent Persona Routing

The PersonaRouter intelligently selects which personas should contribute to each investigation phase:

**Routing Decisions**:
- Identifies the current investigation phase
- Analyzes which personas have relevant expertise
- Routes the question to the most relevant personas
- Tracks why each persona was selected
- Avoids redundant persona contributions

**Benefits**:
- **Efficiency**: Doesn't query all personas for every question
- **Relevance**: Routes to experts most suited for current question
- **Flexibility**: Can adapt routing based on findings
- **Transparency**: Explains why personas were selected

### Conversation Memory and Continuation

The STUDY workflow maintains full conversation history:

**Memory Features**:
- All messages stored with timestamps
- Per-message metadata (persona, phase, context)
- Thread IDs for resuming investigations
- Investigation metadata preserved across sessions

**Continuation Pattern**:
```bash
# Start investigation
result=$(model-chorus study start --scenario "..." )
# Extract thread-id from result.metadata.thread_id

# Later, continue with same context
model-chorus study start --scenario "..." --continue <thread-id>
```

### Conversation Viewing

View previous investigation discussions:

```bash
# View entire investigation
model-chorus study view --investigation thread-id-123

# Filter by specific persona
model-chorus study view --investigation thread-id-123 --persona SecurityExpert

# Show all messages (don't truncate)
model-chorus study view --investigation thread-id-123 --show-all

# Output as JSON for processing
model-chorus study view --investigation thread-id-123 --json
```

## API Usage

### Basic Investigation

```python
from model_chorus.providers import ClaudeProvider
from model_chorus.workflows.study import StudyWorkflow
from model_chorus.core.conversation import ConversationMemory

# Create provider and memory
provider = ClaudeProvider()
memory = ConversationMemory()

# Create workflow
workflow = StudyWorkflow(provider, conversation_memory=memory)

# Conduct research
result = await workflow.run(
    "Explore the implementation patterns of authentication systems"
)

# Access results
print(f"Success: {result.success}")
print(f"Synthesis: {result.synthesis}")
print(f"Thread ID: {result.metadata['thread_id']}")
```

### Continuing an Investigation

```python
# Continue previous investigation
result2 = await workflow.run(
    "Dive deeper into OAuth 2.0 flow patterns",
    continuation_id=result.metadata['thread_id']
)
```

### Customizing Personas

```python
result = await workflow.run(
    "Design a distributed consensus algorithm",
    personas=[
        {
            "name": "TheoreticalMathematician",
            "expertise": "formal proofs and mathematical correctness",
            "role": "primary investigator"
        },
        {
            "name": "PracticalEngineer",
            "expertise": "implementation efficiency and real-world constraints",
            "role": "critical reviewer"
        }
    ]
)
```

### Accessing Investigation Steps

```python
# Each step represents a persona contribution
for step in result.steps:
    persona = step.metadata.get('persona', 'Unknown')
    print(f"\n{persona}:")
    print(step.content)
```

## Command-Line Interface

### study start - Start New Investigation

```bash
model-chorus study start \
  --scenario "Investigation topic or question" \
  [--provider claude|gemini|codex|cursor-agent] \
  [--persona persona-name] [--persona another-persona] \
  [--file context-file.py] [--file another-file.md] \
  [--temperature 0.0-1.0] \
  [--max-tokens number] \
  [--output result.json] \
  [--verbose]
```

**Options**:
- `--scenario` (required): The investigation topic or research question
- `--provider`: Which model to use (defaults to config)
- `--persona`: Specific personas to use (can specify multiple times)
- `--file`: Context files to include (can specify multiple times)
- `--temperature`: Generation temperature for creativity
- `--max-tokens`: Maximum response length
- `--output`: Save results to JSON file
- `--verbose`: Show detailed execution information
- `--skip-provider-check`: Skip provider availability check

**Examples**:
```bash
# Simple investigation
model-chorus study start --scenario "What are the main authentication patterns?"

# With multiple personas
model-chorus study start \
  --scenario "Analyze this code" \
  --persona SecurityExpert \
  --persona Architect

# With context files
model-chorus study start \
  --scenario "Analyze this authentication system" \
  --file src/auth.py \
  --file src/middleware/auth.ts

# Continue previous investigation
model-chorus study start \
  --scenario "Deeper analysis of OAuth 2.0" \
  --continue thread-id-123

# Save results
model-chorus study start \
  --scenario "..." \
  --output results.json
```

### study next - Continue Investigation

```bash
model-chorus study start \
  --scenario "Continue the investigation by exploring the next aspect" \
  --continue <thread-id>
```

**Convenience alternative**:
```bash
model-chorus study-next \
  --investigation <thread-id> \
  [--provider provider] \
  [--file additional-context.py] \
  [--output results.json]
```

### study view - View Investigation History

```bash
model-chorus study view \
  --investigation <thread-id> \
  [--persona persona-name] \
  [--show-all] \
  [--json] \
  [--verbose]
```

**Options**:
- `--investigation` (required): Thread ID to view
- `--persona`: Filter by specific persona
- `--show-all`: Don't truncate long messages
- `--json`: Output in JSON format
- `--verbose`: Show timestamps and metadata

**Examples**:
```bash
# View investigation summary
model-chorus study view --investigation thread-id-123

# See messages from specific persona
model-chorus study view --investigation thread-id-123 --persona SecurityExpert

# Show full messages
model-chorus study view --investigation thread-id-123 --show-all

# Get machine-readable output
model-chorus study view --investigation thread-id-123 --json
```

## Output Format

### Command Output

**For start/next commands**:
```
✓ STUDY investigation completed

Thread ID: thread-id-123
Status: New investigation started
Personas: Researcher, Critic

Investigation Steps:

[Researcher]:
[persona response content...]

[Critic]:
[persona response content...]

Research Synthesis:
[synthesized findings combining all personas...]

To continue this investigation, use: --continue thread-id-123
```

### JSON Output (with --output flag)

```json
{
  "scenario": "Investigation topic",
  "provider": "claude",
  "thread_id": "thread-id-123",
  "is_continuation": false,
  "personas_used": ["Researcher", "Critic"],
  "steps": [
    {
      "persona": "Researcher",
      "content": "Investigation findings...",
      "metadata": { ... }
    },
    {
      "persona": "Critic",
      "content": "Critical analysis...",
      "metadata": { ... }
    }
  ],
  "synthesis": "Combined analysis from all personas...",
  "model": "claude-3-opus",
  "usage": {
    "total_tokens": 2500
  }
}
```

### Python API Output

```python
result = WorkflowResult(
    success=True,
    steps=[
        WorkflowStep(
            step_number=1,
            content="Researcher findings...",
            model="claude-3-opus",
            metadata={"persona": "Researcher", ...}
        ),
        WorkflowStep(
            step_number=2,
            content="Critic analysis...",
            model="claude-3-opus",
            metadata={"persona": "Critic", ...}
        )
    ],
    synthesis="Combined findings...",
    metadata={
        "thread_id": "thread-id-123",
        "workflow_type": "study",
        "personas_used": ["Researcher", "Critic"],
        "investigation_rounds": 2,
        "is_continuation": False
    }
)
```

## Performance Characteristics

### Token Usage

- **Simple Investigation**: 2,000-5,000 tokens
- **With Context Files**: 5,000-15,000 tokens
- **Multi-turn Continuation**: Varies based on history length

### Response Time

- **Single Persona**: 15-30 seconds
- **Multiple Personas**: 30-60 seconds
- **With Large Context**: 60-120 seconds

### Storage

- Investigation threads stored in: `~/.model-chorus/conversations/`
- File size per investigation: ~10-50 KB
- No cleanup required (threads persist for later continuation)

## Advanced Features

### Routing History

Access historical routing decisions:

```python
history = workflow.get_routing_history(
    investigation_id="thread-id-123",
    limit=10  # Get last 10 routing decisions
)

for entry in history:
    print(f"{entry.timestamp}: Selected {entry.selected_persona}")
    print(f"  Reason: {entry.routing_reason}")
```

### Custom Routing Strategies

(To be implemented in Phase 6-2)

- **Adaptive Routing**: System learns which personas contribute most
- **Round-Robin**: Systematically rotate through personas
- **Expertise-Based**: Route based on topic-persona expertise matching

## Integration with Other Workflows

The STUDY workflow works well with:

- **ARGUMENT**: Use STUDY for detailed investigation, ARGUMENT for structured debate
- **CONSENSUS**: Use STUDY to gather perspectives, CONSENSUS to synthesize
- **THINKDEEP**: Use STUDY for multi-perspective thinking alongside deep single-perspective analysis

## Error Handling

### Provider Unavailable

```
Error: No providers available for study workflow:
  - claude: API key not configured

To fix this:
  Check installations: model-chorus list-providers --check
  Install missing providers or update .model-chorusrc
```

### Missing Investigation Thread

```
Error: Investigation not found: invalid-thread-id
Make sure you're using the correct thread ID from a previous investigation.

To start a new investigation, use: model-chorus study --scenario "..."
```

### File Not Found

```
Error: File not found: src/missing-file.py
Specify file paths that exist relative to current directory.
```

## Configuration

STUDY workflow can be configured in `~/.model-chorusrc`:

```yaml
workflows:
  study:
    provider: claude          # Primary provider
    temperature: 0.7          # Generation temperature
    max_tokens: 4000         # Response limit
    fallback_providers:      # Fallback providers
      - gemini
      - codex
```

## Related Workflows

- **ARGUMENT**: Role-based debate with structured perspectives
- **CONSENSUS**: Multi-model consensus with synthesis
- **THINKDEEP**: Extended single-model investigation
- **CHAT**: Simple conversation with memory
- **IDEATE**: Creative brainstorming with personas

## See Also

- [Study Workflow Architecture](../../../ideas/persona_orchestration/misc/study_workflow_architecture_guide.md)
- [PersonaRouter Implementation](./persona_router.py)
- [Base Workflow Documentation](../../core/base_workflow.py)
- [Conversation Memory Guide](../../core/conversation.py)
