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
