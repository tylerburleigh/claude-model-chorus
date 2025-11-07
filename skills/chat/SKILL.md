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
