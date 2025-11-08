# ModelChorus SKILL.md Template

**Purpose:** This template defines the standard structure for all ModelChorus workflow SKILL.md files.

**Audience:** AI agents using Claude Code or similar AI development tools

**Focus:** CLI command usage and practical execution guidance

---

```markdown
---
name: [workflow-name]
description: [One-line description of what this workflow does for AI agents]
---

# [Workflow Name]

## Overview

[2-3 sentences explaining what this workflow does and its primary purpose]

**Key Capabilities:**
- [Capability 1]
- [Capability 2]
- [Capability 3]

**Use Cases:**
- [Use case 1]
- [Use case 2]
- [Use case 3]

## When to Use

Use this workflow when you need to:

- **[Scenario 1]** - [Brief explanation]
- **[Scenario 2]** - [Brief explanation]
- **[Scenario 3]** - [Brief explanation]

## When NOT to Use

Avoid this workflow when:

| Situation | Use Instead |
|-----------|-------------|
| [Anti-pattern 1] | [Alternative workflow/approach] |
| [Anti-pattern 2] | [Alternative workflow/approach] |
| [Anti-pattern 3] | [Alternative workflow/approach] |

## Basic Usage

### Simple Example

```bash
model-chorus [workflow-name] "[prompt text here]"
```

**Expected Output:**
[Brief description of what the command returns]

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `claude` | AI provider to use (`claude`, `gemini`, `codex`, `cursor-agent`) |
| `--continue` | `-c` | None | Thread ID to continue conversation [IF SUPPORTED] |
| `--file` | `-f` | None | File paths for context (repeatable) |
| `--system` | | None | Additional system prompt |
| `--temperature` | `-t` | [X.X] | Creativity level (0.0-1.0) |
| `--max-tokens` | | None | Maximum response length |
| `--output` | `-o` | None | Save result to JSON file |
| `--verbose` | `-v` | False | Show detailed execution info |
| [Workflow-specific options] | | | |

## Advanced Usage

### With Provider Selection

```bash
# Use specific provider
model-chorus [workflow-name] "[prompt]" -p gemini

# [Additional provider-specific notes if relevant]
```

### With File Context

```bash
# Include single file
model-chorus [workflow-name] "[prompt]" -f path/to/file.txt

# Include multiple files
model-chorus [workflow-name] "[prompt]" -f file1.txt -f file2.md -f config.json
```

**File Handling:**
- All file paths must exist before execution
- Files are read and included as context
- [Any workflow-specific file handling notes]

### With Conversation Threading

[ONLY INCLUDE IF WORKFLOW SUPPORTS --continue]

```bash
# Start new conversation
model-chorus [workflow-name] "[initial prompt]"
# Output includes: Thread ID: thread-xxx-xxx-xxx

# Continue conversation
model-chorus [workflow-name] "[follow-up prompt]" --continue thread-xxx-xxx-xxx
```

**Threading Notes:**
- Thread IDs persist across sessions
- Conversation history is maintained
- [Any workflow-specific threading behavior]

[IF WORKFLOW DOES NOT SUPPORT THREADING, USE THIS INSTEAD:]

**Note:** This workflow does not support conversation threading (`--continue`). Each invocation is stateless.

### Adjusting Creativity

```bash
# Lower temperature for factual/precise output
model-chorus [workflow-name] "[prompt]" --temperature 0.3

# Higher temperature for creative output
model-chorus [workflow-name] "[prompt]" --temperature 0.9

# Default balanced setting
model-chorus [workflow-name] "[prompt]" --temperature [default value]
```

### Saving Results

```bash
# Save output to JSON file
model-chorus [workflow-name] "[prompt]" --output results.json

# Output file contains:
# - Prompt
# - Response/synthesis
# - Metadata (model, tokens, etc.)
# - [Workflow-specific output fields]
```

## Workflow-Specific Features

[THIS SECTION VARIES BY WORKFLOW - Document unique capabilities]

### [Feature 1 Name]

[Description and CLI usage]

```bash
[Example command demonstrating feature]
```

### [Feature 2 Name]

[Description and CLI usage]

```bash
[Example command demonstrating feature]
```

## Decision Guide

### Choosing Parameters

**Provider Selection (`-p`):**
- `claude`: [When to use Claude - capabilities, strengths]
- `gemini`: [When to use Gemini - capabilities, strengths]
- `codex`: [When to use Codex - capabilities, strengths]
- `cursor-agent`: [When to use Cursor Agent - capabilities, strengths]

**Temperature (`-t`):**
- `0.0-0.3`: Deterministic, factual, precise responses
- `0.4-0.7`: Balanced creativity and accuracy (default range)
- `0.8-1.0`: Maximum creativity, exploratory, brainstorming

[Add workflow-specific decision guidance]

### Common Patterns

**Pattern 1: [Pattern Name]**
```bash
[Example command showing the pattern]
```
[When to use this pattern]

**Pattern 2: [Pattern Name]**
```bash
[Example command showing the pattern]
```
[When to use this pattern]

## Output Format

### Standard Output

[Describe the output structure for this workflow]

**Sections:**
1. [Output section 1] - [What it contains]
2. [Output section 2] - [What it contains]
3. [Output section 3] - [What it contains]

### Verbose Output

When using `--verbose` flag:
- [Additional information shown]
- [Debug/execution details]
- [Performance metrics]

## Error Handling

**Common Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| File not found | Invalid file path in `-f` | Verify file paths exist |
| Provider initialization failed | Invalid provider name | Check available providers with `model-chorus list-providers` |
| [Workflow-specific error] | [Cause] | [Solution] |

## Best Practices

1. **[Practice 1]** - [Explanation and rationale]

2. **[Practice 2]** - [Explanation and rationale]

3. **[Practice 3]** - [Explanation and rationale]

4. **[Practice 4]** - [Explanation and rationale]

5. **[Practice 5]** - [Explanation and rationale]

## Examples

### Example 1: [Use Case Name]

**Scenario:** [Describe the situation]

**Command:**
```bash
model-chorus [workflow-name] "[specific prompt]" [relevant options]
```

**Expected Outcome:** [What the command accomplishes]

---

### Example 2: [Use Case Name]

**Scenario:** [Describe the situation]

**Command:**
```bash
model-chorus [workflow-name] "[specific prompt]" [relevant options]
```

**Expected Outcome:** [What the command accomplishes]

---

### Example 3: [Use Case Name]

**Scenario:** [Describe the situation]

**Command:**
```bash
model-chorus [workflow-name] "[specific prompt]" [relevant options]
```

**Expected Outcome:** [What the command accomplishes]

## Troubleshooting

### Issue: [Common Issue 1]

**Symptoms:** [What you observe]

**Cause:** [Why it happens]

**Solution:**
```bash
[Command or fix to resolve]
```

---

### Issue: [Common Issue 2]

**Symptoms:** [What you observe]

**Cause:** [Why it happens]

**Solution:**
```bash
[Command or fix to resolve]
```

## Related Workflows

- **[Related Workflow 1]** - [When to use instead/in combination]
- **[Related Workflow 2]** - [When to use instead/in combination]
- **[Related Workflow 3]** - [When to use instead/in combination]

[ONLY include if workflows have operational relationships - minimize cross-references]

---

**See Also:**
- ModelChorus Documentation: `/docs/workflows/[WORKFLOW_NAME].md`
- Provider Information: `model-chorus list-providers`
- General CLI Help: `model-chorus --help`
```

---

## Template Usage Guidelines

### DO:

1. **Write for AI Agents** - Assume the reader is an AI that will execute commands, not a human user
2. **Focus on CLI** - All examples should be command-line invocations of `model-chorus`
3. **Be Specific** - Use concrete examples with real parameters
4. **Document Defaults** - Always specify default values clearly
5. **Include Decision Trees** - Help AI agents choose the right options
6. **Keep Self-Contained** - Each SKILL.md should standalone without requiring other docs
7. **Use Tables** - For parameter lists, comparisons, and error references
8. **Show Expected Output** - Describe what commands return
9. **Validate Examples** - Test all CLI commands before documenting

### DON'T:

1. **Don't Include Python API** - SKILL.md is CLI-only, no code examples
2. **Don't Cross-Reference Heavily** - Minimize links to other workflows unless operationally necessary
3. **Don't Write User Documentation** - This is instruction manual for AI agents
4. **Don't Include Implementation Details** - Focus on usage, not internals
5. **Don't Add Extensive Troubleshooting** - Keep it to common issues only
6. **Don't Compare Exhaustively** - Brief workflow comparisons only when essential
7. **Don't Use Marketing Language** - Be technical and precise
8. **Don't Assume Context** - Each section should be understandable independently

### Section-Specific Guidance

**Frontmatter:**
- `name`: Exact workflow name as used in CLI (lowercase, matching command)
- `description`: One sentence, agent-focused, describes WHAT the workflow does

**Overview:**
- 2-3 sentences maximum
- Focus on core capability and purpose
- List 3-5 key capabilities as bullets

**When to Use / When NOT to Use:**
- Concrete scenarios, not abstract concepts
- "When NOT to Use" should always suggest alternatives
- Use table format for "When NOT to Use" section

**Basic Usage:**
- Start with absolute simplest invocation
- Build complexity gradually
- Always show expected output or outcome

**Advanced Usage:**
- Cover all common option combinations
- File handling patterns
- Threading behavior (if supported)
- Temperature/creativity tuning
- Output saving

**Workflow-Specific Features:**
- THIS IS WHERE WORKFLOWS DIFFER MOST
- Document unique parameters (e.g., `--num-ideas` for IDEATE)
- Explain special behaviors (e.g., role-based execution for ARGUMENT)
- Show workflow-specific output formats

**Decision Guide:**
- Help AI agent choose between options
- When to use which provider, temperature, etc.
- Common usage patterns with rationale

**Examples:**
- 3-5 realistic, end-to-end examples
- Cover different use cases
- Show complete commands with expected outcomes

**Best Practices:**
- 5-7 practical tips
- Focus on optimization and effectiveness
- Include rationale for each practice

## Validation Checklist

Before finalizing any SKILL.md file, verify:

- [ ] Frontmatter is complete and accurate
- [ ] All CLI examples use correct syntax
- [ ] Default values are documented
- [ ] Parameter constraints are specified
- [ ] "When NOT to Use" suggests alternatives
- [ ] Conversation threading behavior is clear
- [ ] Workflow-specific features are documented
- [ ] Examples are concrete and realistic
- [ ] No Python API code examples
- [ ] No MCP or Zen server references
- [ ] Minimal cross-workflow references
- [ ] Tables are properly formatted
- [ ] File paths use correct format
- [ ] Error handling covers common issues
- [ ] Output format is described

## Example: CHAT Workflow Template

See `docs/examples/CHAT_SKILL_EXAMPLE.md` for a completed example following this template.

---

**Version:** 1.0
**Last Updated:** 2025-11-07
**Maintained By:** ModelChorus Development Team
