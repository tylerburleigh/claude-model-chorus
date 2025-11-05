---
name: model-chorus:chat
description: General chat and collaborative thinking partner for brainstorming, development discussion, getting second opinions, and exploring ideas
---

# ModelChorus: Chat

Conversational AI assistant for collaborative thinking, brainstorming, and exploring complex ideas with external AI models.

## When to Use

Use this skill when you need:
- **Second opinions** on technical decisions or approaches
- **Brainstorming** sessions for new features or solutions
- **Collaborative thinking** to explore different perspectives
- **Explanations** of complex concepts or code
- **Validation** of ideas before implementation
- **Quick consultations** on development questions

## How It Works

This skill provides access to external AI models for conversational interactions. You can:
- Ask questions and get detailed responses
- Share code or file paths for analysis
- Continue multi-turn conversations with context
- Choose specific models for different needs
- Adjust thinking depth and creativity

## Usage

Invoke the chat skill with your question or discussion topic:

```
Use the chat skill to discuss:
[Your question or topic]

Optional context:
- Files: [relevant file paths]
- Previous discussion ID: [continuation_id from earlier]
- Preferred model: [model name]
```

## Parameters

- **prompt** (required): Your question or discussion topic
- **working_directory_absolute_path** (required): Absolute path to project directory
- **model** (optional): Specific model to use (e.g., "gemini-2.5-pro", "gpt-5-pro")
- **absolute_file_paths** (optional): List of relevant code files
- **images** (optional): Screenshots or diagrams for visual context
- **continuation_id** (optional): Continue previous conversation
- **temperature** (optional): 0=deterministic, 1=creative (default: balanced)
- **thinking_mode** (optional): minimal, low, medium, high, max

## Examples

### Example 1: Quick Technical Question

```
Use the chat skill to ask:
"What's the best approach for caching API responses in Python?
Compare in-memory vs Redis vs file-based caching with pros/cons."

Working directory: /home/user/project
Model: gemini-2.5-pro
Thinking mode: medium
```

### Example 2: Code Review Discussion

```
Use the chat skill to review this authentication implementation:

Files:
- /home/user/project/src/auth/middleware.py
- /home/user/project/src/auth/tokens.py

Question: "Review this JWT authentication middleware. Are there security
issues? How can we improve the error handling?"

Model: gpt-5-codex
Thinking mode: high
```

### Example 3: Architecture Brainstorming

```
Use the chat skill to brainstorm:
"I'm designing a real-time notification system. Should I use WebSockets,
Server-Sent Events, or long polling? Consider: 100k concurrent users,
need bi-directional communication, running on AWS."

Model: gemini-2.5-pro
Temperature: 0.8
Thinking mode: max
```

### Example 4: Multi-Turn Conversation

```
First call:
Use chat skill: "Explain dependency injection in Python"
Working directory: /home/user/project

[Gets response and continuation_id: "abc123"]

Follow-up:
Use chat skill: "Show me an example using that pattern in FastAPI"
Continuation ID: abc123
Working directory: /home/user/project
```

## Model Selection

The chat skill supports multiple AI models:

**Top Models:**
- `gemini-2.5-pro` - Best for analysis, thinking, code generation (1M context)
- `gpt-5-pro` - Best for complex reasoning, planning (400K context)
- `gpt-5-codex` - Best for code-specific discussions (400K context)
- `gpt-5` - Balanced model for general use (400K context)

See full model list with `/listmodels` or the `mcp__zen__listmodels` tool.

## Best Practices

**DO:**
- ✅ Be specific about your question or goal
- ✅ Provide relevant code files for context
- ✅ Use continuation_id for related follow-ups
- ✅ Choose appropriate thinking mode for complexity
- ✅ Specify models when you have preferences

**DON'T:**
- ❌ Paste large code blocks in the prompt (use absolute_file_paths instead)
- ❌ Ask overly broad questions without context
- ❌ Use max thinking mode for simple questions
- ❌ Forget to specify working directory

## Integration with ModelChorus

This skill is part of the ModelChorus plugin and uses the underlying `mcp__zen__chat` tool from the Zen MCP server. It provides a streamlined interface for conversational AI interactions within your development workflow.

## See Also

- **thinkdeep** - For systematic investigation and complex problem analysis
- **debug** - For debugging and root cause analysis
- **codereview** - For comprehensive code reviews
- **planner** - For complex project planning
