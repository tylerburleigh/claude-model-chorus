---
name: chat
description: Single-model conversational interaction with threading support for general-purpose queries
---

# CHAT

## Overview

The CHAT workflow provides simple, straightforward consultation with a single AI model while maintaining conversation continuity through threading. This is ModelChorus's foundational workflow for basic conversational interactions, ideal for quick consultations and iterative refinement without the complexity of multi-model orchestration.

**Key Capabilities:**
- Single-model conversation with configured AI
- Conversation threading for multi-turn interactions with full history preservation
- File context integration for grounding conversations in specific documents
- Simple request/response pattern with automatic history management

**Use Cases:**
- Quick second opinions or consultations from an AI model
- Iterative conversation development with follow-up questions
- Code explanation with the ability to ask clarifying questions
- Brainstorming sessions where you build on previous responses
- General-purpose queries that benefit from conversational context

## When to Use

Use the CHAT workflow when you need to:

- **Simple consultation** - Ask questions or get feedback from a single AI model without needing multiple perspectives
- **Conversational refinement** - Build on previous exchanges with follow-up questions, clarifications, or iterations
- **Context-aware dialogue** - Maintain conversation history so the model remembers earlier context and builds on it naturally
- **Quick interactions** - Get fast responses without the overhead of multi-model orchestration or specialized workflows
- **File-based consultation** - Discuss specific documents by including them as context in your conversation

## When NOT to Use

Avoid the CHAT workflow when:

| Situation | Use Instead |
|-----------|-------------|
| You need multiple AI perspectives or consensus | **CONSENSUS** - Multi-model consultation with synthesis strategies |
| You need structured debate or dialectical analysis | **ARGUMENT** - Three-role creator/skeptic/moderator analysis |
| You need systematic investigation with hypothesis tracking | **THINKDEEP** - Extended reasoning with confidence progression |
| You need creative brainstorming with many ideas | **IDEATE** - Structured idea generation workflow |

## Basic Usage

### Simple Example

```bash
modelchorus chat "What is quantum computing?"
```

**Expected Output:**
The command returns a conversational response explaining the topic. The response includes a thread ID for conversation continuation.

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--continue` | `-c` | None | Thread ID to continue conversation |
| `--file` | `-f` | None | File paths for context (repeatable) |
| `--system` | | None | Additional system prompt |
| `--output` | `-o` | None | Save result to JSON file |
| `--verbose` | `-v` | False | Show detailed execution info |

## Technical Contract

### Parameters

**Required:**
- `prompt` (string): The conversation prompt or question to send to the AI model

**Optional:**
- `--continue, -c` (string): Thread ID to continue an existing conversation - Format: `thread-{uuid}` - Maintains full conversation history
- `--file, -f` (string, repeatable): File paths to include as context - Can be specified multiple times - Files must exist before execution
- `--system` (string): Additional system prompt to customize model behavior
- `--output, -o` (string): Path to save JSON output file - Creates or overwrites file at specified path
- `--verbose, -v` (boolean): Enable detailed execution information - Default: false - Shows provider details and timing

### Return Format

The CHAT workflow returns a JSON object with the following structure:

```json
{
  "result": "The model's conversational response text...",
  "session_id": "thread-abc-123-def-456",
  "metadata": {
    "provider": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "timestamp": "2025-11-07T10:30:00Z"
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | The conversational response from the AI model |
| `session_id` | string | Thread ID for continuing this conversation (format: `thread-{uuid}`) |
| `metadata.provider` | string | The AI provider that processed the request (`claude`, `gemini`, `codex`, `cursor-agent`) |
| `metadata.model` | string | Specific model version used by the provider |
| `metadata.timestamp` | string | ISO 8601 timestamp of when the request was processed |

**Usage Notes:**
- Save the `session_id` value to continue conversations using `--continue`
- The `result` field contains the complete response text suitable for display or further processing

## Advanced Usage

### With File Context

```bash
# Include single file for discussion
modelchorus chat "Explain this code" -f src/main.py

# Include multiple files for broader context
modelchorus chat "Review these implementations" -f api.py -f models.py -f tests.py
```

**File Handling:**
- All file paths must exist before execution
- Files are read and included as context for the conversation
- Multiple `-f` flags can be used to include multiple files
- Large files may be truncated based on provider token limits

### With Conversation Threading

```bash
# Start new conversation
modelchorus chat "What is quantum computing?"
# Output includes: Session ID: thread-abc-123-def-456

# Continue conversation with follow-up
modelchorus chat "How does it differ from classical computing?" --continue thread-abc-123-def-456

# Further continuation in same thread
modelchorus chat "Give me a practical example" -c thread-abc-123-def-456
```

**Threading Notes:**
- Thread IDs persist across sessions
- Conversation history is maintained and automatically included
- Each message builds on the full conversation context
- Thread IDs follow format: `thread-{uuid}`
- Use short flag `-c` or long flag `--continue`


### Saving Results

```bash
# Save output to JSON file
modelchorus chat "Analyze this architecture" -f design.md --output analysis.json
```

**Output file contains:**
- Original prompt
- Model response/synthesis
- Metadata (model name, token usage, timestamp)
- Thread ID for continuation
- Provider information

## Best Practices

1. **Use threading for multi-turn conversations** - Always save and reuse thread IDs when building on previous context. This ensures the model has full conversation history and provides more coherent responses.

2. **Include relevant files as context** - When discussing code, documentation, or specific content, use `-f` flags to include files rather than copying content into prompts.

3. **Keep prompts clear and specific** - Even though CHAT is conversational, specific prompts yield better results than vague questions.

## Examples

### Example 1: Code Review Conversation

**Scenario:** You want to discuss code quality and get iterative feedback.

**Command:**
```bash
# Start conversation with code file
modelchorus chat "Review this implementation for potential issues" -f src/auth.py

# Follow up with specific question (using thread ID from previous output)
modelchorus chat "How would you refactor the login method?" -c thread-abc-123

# Continue with implementation details
modelchorus chat "Show me an example of the improved version" -c thread-abc-123
```

**Expected Outcome:** Multi-turn conversation where the model builds on previous context to provide detailed, contextual code review and refactoring suggestions.

---

### Example 2: Learning Session with Follow-ups

**Scenario:** You're learning a new concept and want to ask progressively deeper questions.

**Command:**
```bash
# Initial query
modelchorus chat "Explain Rust's ownership system"

# Follow up (saves thread ID: thread-xyz-789)
modelchorus chat "How does borrowing work with mutable references?" -c thread-xyz-789

# Ask for practical example
modelchorus chat "Show me a common mistake beginners make" -c thread-xyz-789
```

**Expected Outcome:** Educational conversation where each response builds on previous explanations, creating a coherent learning path.

---

### Example 3: Multi-file Analysis

**Scenario:** You need to understand how multiple components interact.

**Command:**
```bash
# Include multiple related files
modelchorus chat "Explain how these components work together" -f src/models/user.py -f src/services/auth.py -f src/api/routes.py --output analysis.json
```

**Expected Outcome:** Comprehensive analysis of component relationships with results saved to JSON for later reference.

## Troubleshooting

### Issue: Thread ID not found

**Symptoms:** Error message "Thread ID not found" or "Invalid session_id"

**Cause:** The thread ID doesn't exist in conversation memory, either due to typo or expired/cleared memory

**Solution:**
```bash
# Verify thread ID format (should be thread-{uuid})
# Start a new conversation if thread is unavailable
modelchorus chat "Your prompt here"
```

---

### Issue: File not found error

**Symptoms:** Error message indicating a file path doesn't exist

**Cause:** File path provided to `-f` flag is invalid or file doesn't exist

**Solution:**
```bash
# Verify file exists before running command
ls -l path/to/file.py

# Use correct absolute or relative path
modelchorus chat "Review this code" -f ./src/main.py
```

---

## Progress Reporting

The CHAT workflow automatically displays progress updates to stderr as it executes. You will see messages like:

```
Starting chat workflow...
✓ chat workflow complete
```

**Important for Claude Code:** Progress updates are emitted automatically - do NOT use `BashOutput` to poll for progress. Simply invoke the command and wait for completion. All progress information streams automatically to stderr without interfering with stdout.

## Related Workflows

- **CONSENSUS** - When you need multiple AI perspectives on the same question instead of a single model's view
- **THINKDEEP** - When the question requires deep investigation with hypothesis testing rather than conversational exchange

---

**See Also:**
- ModelChorus Documentation: `/docs/WORKFLOWS.md`
- General CLI Help: `modelchorus --help`
