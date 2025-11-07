# ModelChorus Skill Instructions Research

**Research Date:** 2025-11-07
**Spec:** modelchorus-skill-instructions-2025-11-07-001
**Phase:** Phase 1 - Research & Analysis

## Executive Summary

This document consolidates research findings for creating SKILL.md files for ModelChorus workflows. The key discovery is that SKILL.md files are **AI agent instructions** (not user documentation), should be CLI-focused, and self-contained per workflow.

## 1. SKILL.md Format: AI Agent Instructions

### Critical Distinction

**SKILL.md files are NOT:**
- User documentation
- API reference guides
- Comprehensive feature explanations
- Cross-workflow comparisons

**SKILL.md files ARE:**
- Instructions for AI agents on HOW to use the workflow
- CLI command-focused guidance
- Step-by-step procedures and decision trees
- Self-contained per workflow (no cross-references unless operationally necessary)

### Reference Example

The SDD toolkit's `run-tests/SKILL.md` serves as the gold standard:
- Frontmatter with name and description
- Overview of capabilities
- Core workflow with phases
- Decision trees (when to use, when not to use)
- CLI commands with examples
- Troubleshooting guidance

**Location:** `~/.claude/plugins/cache/sdd-toolkit/skills/run-tests/SKILL.md`

## 2. Workflow Documentation Analysis

### Source Files Analyzed

1. **docs/workflows/ARGUMENT.md** (23,723 bytes)
2. **docs/workflows/IDEATE.md** (28,558 bytes)
3. **docs/workflows/RESEARCH.md** (31,597 bytes)

### Common Structure (16 Sections)

These are **user documentation** files, NOT the format for SKILL.md:

1. Title - Workflow name
2. Tagline - One-line description
3. Quick Reference - Metadata table
4. What It Does - Detailed explanation
5. When to Use - Use cases with bash examples
6. When NOT to Use - Anti-patterns table
7. CLI Usage - Comprehensive bash examples
   - Basic Invocation
   - With Provider Selection
   - With File Context
   - With Continuation (Threading)
   - With Custom Configuration
8. Python API Usage - Code examples (NOT for SKILL.md)
9. Advanced Features
10. Configuration Options - Parameter tables
11. Best Practices - Numbered tips
12. Common Use Patterns - Templates
13. Troubleshooting - Issue/Solution format
14. Comparison with Other Workflows - Tables
15. Real-World Examples - Complete examples
16. See Also - Links to related docs

### Formatting Conventions

- Headers: `##` for main sections, `###` for subsections
- Code blocks: ```bash and ```python
- Tables: Markdown tables for comparisons
- Examples: Real command-line examples with --flags
- Emphasis: **Bold** for terms, `code` for parameters
- Lists: `-` for bullets, numbers for sequences
- Dividers: `---` between major sections

### Key Insight for SKILL.md

**Use sections 4-7 ONLY** (What It Does, When to Use, When NOT to Use, CLI Usage) and adapt to AI agent instruction format. Drop Python API, comparisons, and extensive troubleshooting.

## 3. Workflow Implementation Patterns

### Source Analyzed

**File:** `modelchorus/src/modelchorus/workflows/argument/argument_workflow.py`
**Lines:** 681 total

### Base Architecture

All workflows inherit from `BaseWorkflow` and follow this pattern:

```python
@WorkflowRegistry.register("workflow-name")
class WorkflowName(BaseWorkflow):
    def __init__(self, provider, config=None, conversation_memory=None)
    async def run(self, prompt, continuation_id=None, files=None, **kwargs)
```

### Constructor Pattern

```python
def __init__(
    self,
    provider: ModelProvider,              # Required
    config: Optional[Dict[str, Any]] = None,
    conversation_memory: Optional[ConversationMemory] = None
):
    super().__init__(
        name="WorkflowName",
        description="Description",
        config=config,
        conversation_memory=conversation_memory
    )
    self.provider = provider
```

### run() Method Pattern

```python
async def run(
    self,
    prompt: str,                          # Required: main query
    continuation_id: Optional[str] = None, # Thread ID for conversation
    files: Optional[List[str]] = None,    # Context files
    **kwargs                              # temperature, max_tokens, etc.
) -> WorkflowResult:
```

### WorkflowResult Structure

```python
WorkflowResult(
    success: bool,           # Execution succeeded?
    synthesis: str,          # Combined output
    steps: List[WorkflowStep],  # Individual step outputs
    metadata: Dict,          # thread_id, model, conversation_length, etc.
    error: Optional[str]     # Error message if failed
)
```

### Role-Based Orchestration

Workflows like ARGUMENT use `RoleOrchestrator`:

```python
# Create roles
creator_role = ModelRole(role="creator", stance="for", ...)
skeptic_role = ModelRole(role="skeptic", stance="against", ...)
moderator_role = ModelRole(role="moderator", stance="neutral", ...)

# Execute orchestration
orchestrator = RoleOrchestrator(
    roles=[creator_role, skeptic_role, moderator_role],
    provider_map={provider.provider_name: provider},
    pattern=OrchestrationPattern.SEQUENTIAL,
)
result = await orchestrator.execute(base_prompt=full_prompt)
```

### Key Behaviors

1. **Conversation Threading**: via `continuation_id` parameter
2. **File Context**: Files read and prepended to prompts
3. **History Management**: `_build_prompt_with_history()` method
4. **Provider Abstraction**: Works with any `ModelProvider`
5. **Error Handling**: Graceful failures with error messages
6. **Logging**: Comprehensive logging throughout

## 4. CLI Command Signatures

### Source Analyzed

**File:** `modelchorus/src/modelchorus/cli/main.py`
**Framework:** Typer with Rich console output

### Common Options (All Workflows)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--provider` | `-p` | str | `"claude"` | AI provider to use |
| `--continue` | `-c` | str | `None` | Thread ID for conversation continuation |
| `--file` | `-f` | list | `None` | File paths for context (repeatable) |
| `--system` | | str | `None` | System prompt for additional context |
| `--temperature` | `-t` | float | varies | Creativity level (0.0-1.0) |
| `--max-tokens` | | int | `None` | Maximum response length |
| `--output` | `-o` | path | `None` | Save result to JSON file |
| `--verbose` | `-v` | bool | `False` | Show detailed execution info |

### Workflow-Specific Options

**CHAT:**
- Default temperature: 0.7
- Simple conversation, no special options

**ARGUMENT:**
- Default temperature: 0.7
- Three-role dialectical analysis

**IDEATE:**
- `--num-ideas, -n` (int, default: 5): Number of ideas to generate
- Default temperature: 0.9 (higher for creativity)

**RESEARCH:**
- `--depth` (str): Research depth level (shallow, moderate, thorough)
- `--citations` (str): Citation style (informal, academic, technical)
- Default temperature: 0.5 (lower for factual accuracy)

**THINKDEEP:**
- Has `thinkdeep-status` command for checking investigation status
- Supports multi-step investigation with hypothesis tracking

**CONSENSUS:**
- Requires multiple providers specified
- `--strategy, -s` (str): Synthesis strategy (synthesize, vote, etc.)
- Multiple `-p` flags to specify providers

### Command Examples

```bash
# Basic usage
modelchorus chat "What is quantum computing?" -p claude

# With continuation
modelchorus chat "Tell me more" --continue thread-id-123

# Multiple files
modelchorus argument "Should we use GraphQL?" -f api.yaml -f requirements.md

# Creative brainstorming
modelchorus ideate "New app features" --num-ideas 10 --temperature 0.9

# Research with citations
modelchorus research "API best practices" --depth thorough --citations academic
```

### Default Values and Parameter Constraints

**Common Parameters (All Workflows):**

| Parameter | Default | Constraints | Notes |
|-----------|---------|-------------|-------|
| `--provider, -p` | `"claude"` | `"claude"`, `"gemini"`, `"codex"`, `"cursor-agent"` | Single provider selection |
| `--continue, -c` | `None` | Any valid thread ID string | For conversation threading |
| `--file, -f` | `None` | Valid file paths (repeatable) | Must exist on filesystem |
| `--system` | `None` | Any string | Additional system prompt |
| `--temperature, -t` | Varies by workflow | `0.0` to `1.0` (float) | Creativity control |
| `--max-tokens` | `None` (provider default) | Positive integer | Maximum response length |
| `--output, -o` | `None` | Valid file path | JSON output destination |
| `--verbose, -v` | `False` | Boolean flag | Show detailed execution info |

**Workflow-Specific Defaults:**

**CHAT:**
- Temperature: `0.7` (balanced)
- No workflow-specific parameters

**ARGUMENT:**
- Temperature: `0.7` (balanced)
- No workflow-specific parameters
- Uses 3 fixed roles: Creator, Skeptic, Moderator

**IDEATE:**
- `--num-ideas, -n`: Default `5`, Constraint: Positive integer
- Temperature: `0.9` (higher creativity)
- Generates structured brainstorming ideas

**RESEARCH:**
- `--citation-style`: Default `"informal"`, Valid: `["informal", "academic", "technical"]`
- `--depth, -d`: Default `"thorough"`, Valid: `["shallow", "moderate", "thorough", "comprehensive"]`
- Temperature: `0.5` (lower for factual accuracy)

**CONSENSUS:**
- `--provider, -p`: Default `["claude", "gemini"]`, Constraint: List of valid provider names (repeatable)
- `--strategy, -s`: Default `"all_responses"`, Valid: `["all_responses", "first_valid", "majority", "weighted", "synthesize"]`
- `--timeout`: Default `120.0`, Constraint: Float (seconds), timeout per provider
- Temperature: `0.7` (balanced)
- NOTE: Does NOT support `--continue` (no conversation threading)

**THINKDEEP:**
- `--expert, -e`: Default `None`, Constraint: Valid provider name (different from primary)
- `--disable-expert`: Default `False`, Boolean flag
- Temperature: `0.7` (balanced)
- Supports hypothesis tracking and confidence progression

**THINKDEEP-STATUS:**
- `--steps`: Default `False`, Boolean flag to show all investigation steps
- `--files`: Default `False`, Boolean flag to show all examined files
- NOTE: This is a read-only inspection command, no execution parameters

**Parameter Validation:**

1. **File Existence:**
   - All `--file` paths validated before execution
   - Errors halt execution with clear message

2. **Provider Validation:**
   - Provider names validated against available providers
   - Case-insensitive matching
   - Unknown provider triggers helpful error message

3. **Enum Validation:**
   - Strategy (CONSENSUS): Validated against `ConsensusStrategy` enum
   - Citation style (RESEARCH): Validated against hardcoded list
   - Research depth (RESEARCH): Validated against hardcoded list

4. **Type Constraints:**
   - `temperature`: Must be float between 0.0 and 1.0
   - `max-tokens`: Must be positive integer or None
   - `num-ideas`: Must be positive integer
   - `timeout`: Must be positive float

**Thread ID Format:**

Thread IDs are generated by `ConversationMemory` and follow UUID format:
- Pattern: `thread-{uuid4}`
- Example: `thread-123e4567-e89b-12d3-a456-426614174000`
- Valid for any workflow that supports `--continue`
- NOT supported by CONSENSUS (stateless execution)

## 5. Key Patterns for SKILL.md Creation

### Structure Template

```markdown
---
name: workflow-name
description: Brief description for AI agent
---

# Workflow Name

## Overview
What this workflow does and when to use it.

## When to Use
- Use case 1
- Use case 2
- Use case 3

## When NOT to Use
- Anti-pattern 1 → Use X instead
- Anti-pattern 2 → Use Y instead

## Basic Usage

### Simple Example
\`\`\`bash
modelchorus workflow-name "prompt here"
\`\`\`

### Common Options
- `--provider, -p`: Choose AI provider
- `--continue, -c`: Continue conversation
- (workflow-specific options)

## Advanced Usage

### With Files
\`\`\`bash
modelchorus workflow-name "prompt" -f file1.txt -f file2.md
\`\`\`

### With Custom Settings
\`\`\`bash
modelchorus workflow-name "prompt" --temperature 0.8 --max-tokens 2000
\`\`\`

## Workflow-Specific Features
(Document unique capabilities)

## Decision Guide
When to use which options/approaches

## Tips
- Tip 1
- Tip 2
```

### Content Guidelines

**DO:**
- Write for AI agents as the audience
- Focus on CLI commands and practical usage
- Include decision trees (when to use X vs Y)
- Provide concrete command examples
- Explain workflow-specific parameters
- Keep it self-contained per workflow

**DON'T:**
- Include Python API examples
- Compare extensively to other workflows
- Write comprehensive user documentation
- Include internal implementation details
- Add extensive troubleshooting (basic only)
- Cross-reference other workflows (unless operationally needed)

## 6. Workflow-Specific Characteristics

### CHAT
- Single-model conversation
- Simple back-and-forth dialogue
- Conversation threading support
- General-purpose query answering

### ARGUMENT
- Three-role dialectical analysis
- Creator (for) → Skeptic (against) → Moderator (synthesis)
- Structured debate format
- Sequential role execution

### IDEATE
- Creative brainstorming
- Configurable number of ideas (--num-ideas)
- Higher default temperature (0.9)
- Structured idea generation

### RESEARCH
- Systematic information gathering
- Depth control (shallow, moderate, thorough)
- Citation formatting (informal, academic, technical)
- Source management and references

### THINKDEEP
- Hypothesis-driven investigation
- Multi-step reasoning
- Confidence tracking
- Status checking command

### CONSENSUS
- Multi-model consultation
- Strategy-based synthesis
- Parallel model execution
- No conversation threading (stateless)

## 7. Implementation Priorities

### Phase 2: Core Workflows (Priority 1)
1. **CHAT** - Simplest, foundational
2. **CONSENSUS** - Multi-model coordination
3. **THINKDEEP** - Investigation workflow

### Phase 3: Advanced Workflows (Priority 2)
4. **ARGUMENT** - Role-based orchestration
5. **IDEATE** - Creative generation
6. **RESEARCH** - Systematic research

## 8. References

### Source Files
- User docs: `docs/workflows/*.md`
- Workflow implementation: `modelchorus/src/modelchorus/workflows/`
- CLI implementation: `modelchorus/src/modelchorus/cli/main.py`
- Example SKILL.md: `~/.claude/plugins/cache/sdd-toolkit/skills/run-tests/SKILL.md`

### Spec Information
- **Spec ID:** modelchorus-skill-instructions-2025-11-07-001
- **Location:** `specs/active/` (moved from pending)
- **Total Tasks:** 39
- **Completion:** 12% (5 tasks completed in Phase 1)

## 9. Next Steps

1. **Complete Phase 1** (3 tasks remaining):
   - task-1-3-2: Document MCP tool calling patterns
   - task-1-3-3: Document continuation/threading behavior
   - task-1-4: Create consistency framework template

2. **Begin Phase 2** (Create Core Workflow SKILL.md files):
   - CHAT skill instructions
   - CONSENSUS skill instructions
   - THINKDEEP skill instructions

3. **Phase 3** (Create Advanced Workflow SKILL.md files):
   - ARGUMENT skill instructions
   - IDEATE skill instructions
   - RESEARCH skill instructions

## 10. Recommendations

### For SKILL.md Creation

1. **Start with CHAT** - Simplest workflow, establishes baseline pattern
2. **Use template structure** - Consistent format across all workflows
3. **Focus on CLI only** - No Python API examples
4. **Keep self-contained** - Minimal cross-workflow references
5. **Write for AI agents** - Instruction manual, not user documentation
6. **Test commands** - Verify all CLI examples work as documented

### Quality Criteria

Each SKILL.md should:
- [ ] Include frontmatter with name and description
- [ ] Explain WHEN to use the workflow
- [ ] Explain WHEN NOT to use it
- [ ] Document all CLI options with examples
- [ ] Include workflow-specific features
- [ ] Provide decision guidance
- [ ] Be self-contained (minimal external references)
- [ ] Focus on AI agent needs (not user documentation)

---

**Research completed:** 2025-11-07
**Document version:** 1.0
**Author:** Claude (Autonomous Mode)
