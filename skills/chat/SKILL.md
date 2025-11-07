---
name: chat
description: Single-model conversational interaction with threading support for general-purpose queries
---

# CHAT

## Overview

The CHAT workflow provides simple, straightforward consultation with a single AI model while maintaining conversation continuity through threading. This is ModelChorus's foundational workflow for basic conversational interactions, ideal for quick consultations and iterative refinement without the complexity of multi-model orchestration.

**Key Capabilities:**
- Single-model conversation with any configured provider (Claude, Gemini, Codex, Cursor Agent)
- Conversation threading for multi-turn interactions with full history preservation
- File context integration for grounding conversations in specific documents
- Simple request/response pattern with automatic history management
- Flexible temperature control for adjusting response creativity

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
| You need comprehensive research with citations | **RESEARCH** - Systematic information gathering with source management |

## Basic Usage

### Simple Example

```bash
modelchorus chat "What is quantum computing?"
```

**Expected Output:**
The command returns a conversational response from the default provider (Claude) explaining the topic. The response includes a thread ID for conversation continuation.

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `claude` | AI provider to use (`claude`, `gemini`, `codex`, `cursor-agent`) |
| `--continue` | `-c` | None | Thread ID to continue conversation |
| `--file` | `-f` | None | File paths for context (repeatable) |
| `--system` | | None | Additional system prompt |
| `--temperature` | `-t` | `0.7` | Creativity level (0.0-1.0) |
| `--max-tokens` | | None | Maximum response length |
| `--output` | `-o` | None | Save result to JSON file |
| `--verbose` | `-v` | False | Show detailed execution info |

## Advanced Usage

### With Provider Selection

```bash
# Use specific provider
modelchorus chat "Explain neural networks" -p gemini

# Use Codex for code-focused questions
modelchorus chat "How does this algorithm work?" -p codex
```

**Provider Selection Tips:**
- `claude`: Best for general conversation, reasoning, and nuanced responses
- `gemini`: Strong for factual queries and technical explanations
- `codex`: Optimized for code-related questions and programming tasks
- `cursor-agent`: Ideal for development-focused consultations

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
# Output includes: Thread ID: thread-abc-123-def-456

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

### Adjusting Creativity

```bash
# Lower temperature for factual/precise output
modelchorus chat "What are the Python PEP 8 guidelines?" --temperature 0.3

# Higher temperature for creative/exploratory output
modelchorus chat "Brainstorm app features for a fitness tracker" --temperature 0.9

# Default balanced setting (0.7)
modelchorus chat "Explain dependency injection" --temperature 0.7
```

**Temperature Guide:**
- `0.0-0.3`: Deterministic, factual, precise (documentation, facts, specs)
- `0.4-0.7`: Balanced creativity and accuracy (general conversation, explanations)
- `0.8-1.0`: Maximum creativity, exploratory (brainstorming, ideation)

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

2. **Choose appropriate temperature** - Use lower temperatures (0.3-0.5) for factual queries and higher temperatures (0.7-0.9) for creative or exploratory conversations.

3. **Include relevant files as context** - When discussing code, documentation, or specific content, use `-f` flags to include files rather than copying content into prompts.

4. **Keep prompts clear and specific** - Even though CHAT is conversational, specific prompts yield better results than vague questions.

5. **Select the right provider** - Match provider strengths to your task: Claude for reasoning, Gemini for facts, Codex for code.

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
# Initial query with lower temperature for accuracy
modelchorus chat "Explain Rust's ownership system" --temperature 0.5

# Follow up (saves thread ID: thread-xyz-789)
modelchorus chat "How does borrowing work with mutable references?" -c thread-xyz-789 -t 0.5

# Ask for practical example
modelchorus chat "Show me a common mistake beginners make" -c thread-xyz-789 -t 0.5
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

**Symptoms:** Error message "Thread ID not found" or "Invalid continuation_id"

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

### Issue: Provider initialization failed

**Symptoms:** Error about provider not being available or initialization failure

**Cause:** Selected provider is not configured or invalid provider name

**Solution:**
```bash
# Check available providers
modelchorus list-providers

# Use a valid provider name
modelchorus chat "Your prompt" -p claude
```

## Related Workflows

- **CONSENSUS** - When you need multiple AI perspectives on the same question instead of a single model's view
- **THINKDEEP** - When the question requires deep investigation with hypothesis testing rather than conversational exchange

---

**See Also:**
- ModelChorus Documentation: `/docs/WORKFLOWS.md`
- Provider Information: `modelchorus list-providers`
- General CLI Help: `modelchorus --help`
