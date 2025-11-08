# Conversation Infrastructure API Reference

Complete API documentation for ModelChorus conversation system including multi-turn conversations, state management, and custom workflow integration.

---

## Table of Contents

- [Overview](#overview)
- [Core Classes](#core-classes)
  - [ConversationMemory](#conversationmemory)
  - [ConversationThread](#conversationthread)
  - [ConversationMessage](#conversationmessage)
  - [ConversationState](#conversationstate)
- [State Management](#state-management)
- [Custom Workflow Integration](#custom-workflow-integration)
- [Advanced Patterns](#advanced-patterns)
- [Best Practices](#best-practices)

---

## Overview

ModelChorus provides robust conversation infrastructure that enables:

- **Multi-turn conversations** with automatic history management
- **State persistence** across conversation turns
- **Thread-safe operations** with file-based locking
- **Conversation branching** for exploring alternatives
- **Automatic cleanup** of expired conversations
- **Custom workflow state** preservation

**Storage Location:** `~/.model-chorus/conversations/`

**File Format:** JSON with file locking

**TTL:** 72 hours (configurable)

---

## Core Classes

### ConversationMemory

Thread-safe conversation storage with file-based persistence.

**Import:**
```python
from model_chorus.core.conversation import ConversationMemory
```

**Constructor:**
```python
memory = ConversationMemory(
    conversations_dir=None,  # Optional: defaults to ~/.model-chorus/conversations
    ttl_hours=72,           # Optional: time-to-live for threads
    max_messages=100        # Optional: max messages per thread before truncation
)
```

**Parameters:**
- `conversations_dir` (Path, optional): Directory for conversation storage. Defaults to `~/.model-chorus/conversations/`
- `ttl_hours` (int, optional): Hours until thread expires. Default: 72
- `max_messages` (int, optional): Maximum messages per thread. Default: 100

#### Methods

##### create_thread()

Create a new conversation thread.

```python
thread_id = memory.create_thread(
    workflow_name: str,
    initial_context: Optional[Dict[str, Any]] = None,
    parent_thread_id: Optional[str] = None
) -> str
```

**Parameters:**
- `workflow_name` (str): Name of workflow creating the thread (e.g., "chat", "thinkdeep")
- `initial_context` (dict, optional): Initial context data for the thread
- `parent_thread_id` (str, optional): Parent thread ID for conversation branching

**Returns:** Thread ID (UUID string)

**Example:**
```python
# Create new thread
thread_id = memory.create_thread(
    workflow_name="chat",
    initial_context={"topic": "API design", "user_id": "user123"}
)

# Create branched thread
branch_id = memory.create_thread(
    workflow_name="chat",
    parent_thread_id=main_thread_id,
    initial_context={"branched_from": "message_5", "exploring": "alternative"}
)
```

##### get_thread()

Retrieve a conversation thread.

```python
thread = memory.get_thread(thread_id: str) -> Optional[ConversationThread]
```

**Parameters:**
- `thread_id` (str): Thread ID to retrieve

**Returns:** ConversationThread object or None if not found

**Example:**
```python
thread = memory.get_thread(thread_id)
if thread:
    print(f"Thread status: {thread.status}")
    print(f"Messages: {len(thread.messages)}")
    print(f"Workflow: {thread.workflow_name}")
```

##### get_thread_chain()

Get complete conversation chain (thread + all parents).

```python
chain = memory.get_thread_chain(thread_id: str) -> List[ConversationThread]
```

**Parameters:**
- `thread_id` (str): Thread ID to start from

**Returns:** List of ConversationThread objects from root to current

**Example:**
```python
# Get full conversation chain
chain = memory.get_thread_chain(branched_thread_id)
for i, thread in enumerate(chain):
    print(f"Level {i}: {thread.thread_id} ({len(thread.messages)} messages)")
```

##### add_message()

Add message to conversation thread.

```python
memory.add_message(
    thread_id: str,
    message: ConversationMessage
) -> None
```

**Parameters:**
- `thread_id` (str): Thread to add message to
- `message` (ConversationMessage): Message to add

**Example:**
```python
from model_chorus.core.models import ConversationMessage

message = ConversationMessage(
    role="user",
    content="Explain async/await",
    timestamp="2025-11-06T10:00:00Z",
    files=["examples/async_example.py"],
    metadata={"source": "cli"}
)

memory.add_message(thread_id, message)
```

##### get_messages()

Retrieve messages from thread.

```python
messages = memory.get_messages(
    thread_id: str,
    limit: Optional[int] = None,
    include_system: bool = True
) -> List[ConversationMessage]
```

**Parameters:**
- `thread_id` (str): Thread ID to get messages from
- `limit` (int, optional): Maximum number of recent messages to return
- `include_system` (bool, optional): Include system messages. Default: True

**Returns:** List of ConversationMessage objects

**Example:**
```python
# Get all messages
all_messages = memory.get_messages(thread_id)

# Get last 10 messages
recent_messages = memory.get_messages(thread_id, limit=10)

# Get messages without system messages
user_messages = memory.get_messages(thread_id, include_system=False)
```

##### build_conversation_history()

Build conversation history formatted for model consumption.

```python
history = memory.build_conversation_history(
    thread_id: str,
    max_turns: Optional[int] = None,
    include_system: bool = True,
    format: str = "messages"  # "messages" or "text"
) -> Union[List[Dict], str]
```

**Parameters:**
- `thread_id` (str): Thread ID
- `max_turns` (int, optional): Maximum conversation turns to include
- `include_system` (bool, optional): Include system messages. Default: True
- `format` (str, optional): Output format - "messages" (list of dicts) or "text" (string). Default: "messages"

**Returns:** Formatted conversation history

**Example:**
```python
# Get as message list (for API calls)
history = memory.build_conversation_history(
    thread_id,
    max_turns=10,
    format="messages"
)
# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

# Get as text (for prompts)
history_text = memory.build_conversation_history(
    thread_id,
    max_turns=10,
    format="text"
)
# Returns: "User: ...\nAssistant: ...\n..."
```

##### get_context_summary()

Get summary of thread context for display.

```python
summary = memory.get_context_summary(thread_id: str) -> Dict[str, Any]
```

**Parameters:**
- `thread_id` (str): Thread ID

**Returns:** Dictionary with context summary

**Example:**
```python
summary = memory.get_context_summary(thread_id)
print(f"Messages: {summary['message_count']}")
print(f"Files referenced: {summary['files_referenced']}")
print(f"Total tokens (approx): {summary['estimated_tokens']}")
print(f"Last updated: {summary['last_updated']}")
```

##### complete_thread()

Mark thread as completed.

```python
memory.complete_thread(thread_id: str) -> None
```

**Parameters:**
- `thread_id` (str): Thread ID to mark complete

**Example:**
```python
# Mark conversation complete
memory.complete_thread(thread_id)

# Thread status is now "completed"
thread = memory.get_thread(thread_id)
assert thread.status == "completed"
```

##### archive_thread()

Archive completed thread.

```python
memory.archive_thread(thread_id: str) -> None
```

**Parameters:**
- `thread_id` (str): Thread ID to archive

**Example:**
```python
# Archive old thread
memory.archive_thread(old_thread_id)

# Thread status is now "archived"
thread = memory.get_thread(old_thread_id)
assert thread.status == "archived"
```

##### cleanup_expired_threads()

Remove expired threads based on TTL.

```python
removed_count = memory.cleanup_expired_threads(
    max_age_hours: Optional[int] = None
) -> int
```

**Parameters:**
- `max_age_hours` (int, optional): Override default TTL. If None, uses instance TTL

**Returns:** Number of threads removed

**Example:**
```python
# Clean up threads older than 72 hours (default TTL)
removed = memory.cleanup_expired_threads()
print(f"Removed {removed} expired threads")

# Clean up threads older than 24 hours
removed = memory.cleanup_expired_threads(max_age_hours=24)
```

##### cleanup_archived_threads()

Remove archived threads.

```python
removed_count = memory.cleanup_archived_threads(
    keep_recent_hours: int = 168  # 7 days
) -> int
```

**Parameters:**
- `keep_recent_hours` (int, optional): Keep archived threads from this many hours ago. Default: 168 (7 days)

**Returns:** Number of archived threads removed

**Example:**
```python
# Remove archived threads older than 7 days
removed = memory.cleanup_archived_threads()

# Remove all archived threads older than 24 hours
removed = memory.cleanup_archived_threads(keep_recent_hours=24)
```

---

### ConversationThread

Complete conversation context for a thread.

**Import:**
```python
from model_chorus.core.models import ConversationThread
```

**Attributes:**

```python
class ConversationThread:
    thread_id: str                          # UUID identifying thread
    parent_thread_id: Optional[str]         # Parent thread for branches
    created_at: str                         # ISO timestamp of creation
    last_updated_at: str                    # ISO timestamp of last update
    workflow_name: str                      # Workflow that created thread
    messages: List[ConversationMessage]     # All messages in chronological order
    state: Optional[ConversationState]      # Workflow-specific state
    initial_context: Dict[str, Any]         # Original request parameters
    status: str                             # "active", "completed", "archived"
    branch_point: Optional[str]             # Message ID where branch occurred
    sibling_threads: List[str]              # Other threads branched from same point
```

**Example:**
```python
thread = memory.get_thread(thread_id)

print(f"Thread ID: {thread.thread_id}")
print(f"Workflow: {thread.workflow_name}")
print(f"Status: {thread.status}")
print(f"Messages: {len(thread.messages)}")
print(f"Created: {thread.created_at}")

# Access workflow state
if thread.state:
    print(f"State workflow: {thread.state.workflow_name}")
    print(f"State data: {thread.state.data}")

# Check for branches
if thread.parent_thread_id:
    print(f"Branched from: {thread.parent_thread_id}")
if thread.sibling_threads:
    print(f"Sibling threads: {thread.sibling_threads}")
```

---

### ConversationMessage

Single message in a conversation thread.

**Import:**
```python
from model_chorus.core.models import ConversationMessage
```

**Attributes:**

```python
class ConversationMessage:
    role: str                                # "user" or "assistant"
    content: str                             # Message content
    timestamp: str                           # ISO format timestamp
    files: Optional[List[str]]               # File paths referenced
    workflow_name: Optional[str]             # Workflow that generated (assistant only)
    model_provider: Optional[str]            # Provider type: "cli", "api", "mcp"
    model_name: Optional[str]                # Specific model: "claude-3-opus", "gpt-5"
    metadata: Dict[str, Any]                 # Additional metadata
```

**Example:**
```python
from model_chorus.core.models import ConversationMessage
from datetime import datetime, timezone

# User message
user_msg = ConversationMessage(
    role="user",
    content="Explain async/await in Python",
    timestamp=datetime.now(timezone.utc).isoformat(),
    files=["examples/async_code.py"],
    metadata={"source": "cli", "command": "chat"}
)

# Assistant message
assistant_msg = ConversationMessage(
    role="assistant",
    content="Async/await in Python allows...",
    timestamp=datetime.now(timezone.utc).isoformat(),
    workflow_name="chat",
    model_provider="cli",
    model_name="claude-3-opus",
    metadata={
        "tokens": 250,
        "latency": 1.5,
        "temperature": 0.7
    }
)
```

---

### ConversationState

Generic state container for workflow-specific data.

**Import:**
```python
from model_chorus.core.models import ConversationState
```

**Attributes:**

```python
class ConversationState:
    workflow_name: str          # Workflow this state belongs to
    data: Dict[str, Any]        # Arbitrary workflow-specific state
    schema_version: str         # State schema version
    created_at: str            # ISO timestamp of creation
    updated_at: str            # ISO timestamp of last update
```

**Example:**
```python
from model_chorus.core.models import ConversationState
from datetime import datetime, timezone

# Create state for ThinkDeep workflow
state = ConversationState(
    workflow_name="thinkdeep",
    data={
        "hypotheses": [
            {"text": "Race condition", "status": "active", "evidence": []}
        ],
        "steps": [],
        "current_confidence": "exploring",
        "relevant_files": ["src/auth.py"]
    },
    schema_version="1.0",
    created_at=datetime.now(timezone.utc).isoformat(),
    updated_at=datetime.now(timezone.utc).isoformat()
)

# Update state
state.data["current_confidence"] = "medium"
state.data["steps"].append({
    "step_number": 1,
    "findings": "Found concurrent writes",
    "confidence": "medium"
})
state.updated_at = datetime.now(timezone.utc).isoformat()
```

---

## State Management

### Workflow State Persistence

Workflows can maintain custom state across conversation turns using the `ConversationState` model.

**Pattern:**

```python
from model_chorus.core.base_workflow import BaseWorkflow
from model_chorus.core.models import ConversationState
from datetime import datetime, timezone

class MyWorkflow(BaseWorkflow):
    def __init__(self, provider, conversation_memory=None):
        super().__init__(
            name="myworkflow",
            description="My custom workflow",
            conversation_memory=conversation_memory
        )
        self.provider = provider

    async def run(self, prompt, continuation_id=None, **kwargs):
        # Get or create thread
        if continuation_id:
            thread = self.get_thread(continuation_id)
            if not thread:
                raise ValueError(f"Thread {continuation_id} not found")
        else:
            thread_id = self.conversation_memory.create_thread(
                workflow_name=self.name,
                initial_context={"prompt": prompt}
            )
            thread = self.get_thread(thread_id)

        # Get or initialize state
        if thread.state:
            state_data = thread.state.data
        else:
            state_data = self._initialize_state()

        # Process with state
        result = await self._process_with_state(prompt, state_data)

        # Update state
        updated_state = ConversationState(
            workflow_name=self.name,
            data=state_data,
            schema_version="1.0",
            created_at=thread.state.created_at if thread.state else datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )

        # Save state to thread
        thread.state = updated_state
        self.conversation_memory._save_thread(thread)

        return result

    def _initialize_state(self):
        """Initialize workflow-specific state."""
        return {
            "step": 1,
            "findings": [],
            "context": {}
        }

    async def _process_with_state(self, prompt, state):
        """Process prompt using state."""
        # Use state to inform processing
        step = state["step"]
        findings = state["findings"]

        # ... workflow logic ...

        # Update state
        state["step"] += 1
        state["findings"].append("New finding")

        return result
```

### State Access Patterns

**Get current state:**
```python
thread = workflow.get_thread(thread_id)
if thread.state:
    state_data = thread.state.data
    print(f"Current step: {state_data['step']}")
```

**Update state:**
```python
thread = workflow.get_thread(thread_id)
if thread.state:
    thread.state.data["new_field"] = "value"
    thread.state.updated_at = datetime.now(timezone.utc).isoformat()
    workflow.conversation_memory._save_thread(thread)
```

**Check state version:**
```python
thread = workflow.get_thread(thread_id)
if thread.state and thread.state.schema_version != "2.0":
    # Migrate state to new version
    migrated_data = migrate_state(thread.state.data)
    thread.state.data = migrated_data
    thread.state.schema_version = "2.0"
```

---

## Custom Workflow Integration

### Creating a State-Aware Workflow

Complete example of integrating conversation infrastructure:

```python
from model_chorus.core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import ConversationMessage, ConversationState
from model_chorus.providers import GenerationRequest
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import uuid

class CustomAnalysisWorkflow(BaseWorkflow):
    """Custom workflow with state management."""

    def __init__(self, provider, conversation_memory=None):
        super().__init__(
            name="custom_analysis",
            description="Custom analysis workflow with multi-turn support",
            conversation_memory=conversation_memory
        )
        self.provider = provider

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[list] = None,
        **kwargs
    ) -> WorkflowResult:
        """Execute workflow with conversation continuity."""

        # Handle conversation threading
        if continuation_id:
            thread = self.get_thread(continuation_id)
            if not thread:
                raise ValueError(f"Thread {continuation_id} not found")
            thread_id = continuation_id
        else:
            thread_id = self.conversation_memory.create_thread(
                workflow_name=self.name,
                initial_context={"files": files or []}
            )
            thread = self.get_thread(thread_id)

        # Get or initialize state
        state_data = self._get_or_create_state(thread)

        # Add user message
        self.add_message(thread_id, ConversationMessage(
            role="user",
            content=prompt,
            timestamp=datetime.now(timezone.utc).isoformat(),
            files=files or [],
            metadata={"step": state_data["analysis_step"]}
        ))

        # Build context from conversation history
        history = self.conversation_memory.build_conversation_history(
            thread_id,
            max_turns=10,
            format="text"
        )

        # Create analysis prompt with state context
        analysis_prompt = self._build_analysis_prompt(
            prompt=prompt,
            history=history,
            state=state_data,
            files=files
        )

        # Execute analysis
        request = GenerationRequest(
            prompt=analysis_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        response = await self.provider.generate(request)

        # Extract findings from response
        findings = self._extract_findings(response.content)

        # Update state
        state_data["analysis_step"] += 1
        state_data["findings"].extend(findings)
        state_data["files_examined"] = list(set(state_data["files_examined"] + (files or [])))

        # Save updated state
        self._save_state(thread, state_data)

        # Add assistant message
        self.add_message(thread_id, ConversationMessage(
            role="assistant",
            content=response.content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            workflow_name=self.name,
            model_provider="cli",
            model_name=self.provider.provider_name,
            metadata={
                "step": state_data["analysis_step"],
                "findings_count": len(findings)
            }
        ))

        # Create result
        result = WorkflowResult(
            synthesis=response.content,
            metadata={
                "thread_id": thread_id,
                "analysis_step": state_data["analysis_step"],
                "total_findings": len(state_data["findings"]),
                "files_examined": len(state_data["files_examined"])
            }
        )

        # Add step
        result.add_step(WorkflowStep(
            name=f"Analysis Step {state_data['analysis_step']}",
            result=response.content,
            metadata={"findings": findings}
        ))

        return result

    def _get_or_create_state(self, thread) -> Dict[str, Any]:
        """Get existing state or create new one."""
        if thread.state:
            return thread.state.data
        else:
            return {
                "analysis_step": 0,
                "findings": [],
                "files_examined": [],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

    def _save_state(self, thread, state_data: Dict[str, Any]):
        """Save updated state to thread."""
        thread.state = ConversationState(
            workflow_name=self.name,
            data=state_data,
            schema_version="1.0",
            created_at=state_data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        self.conversation_memory._save_thread(thread)

    def _build_analysis_prompt(
        self,
        prompt: str,
        history: str,
        state: Dict[str, Any],
        files: Optional[list]
    ) -> str:
        """Build analysis prompt with context."""
        context_parts = [
            f"Analysis Step: {state['analysis_step'] + 1}",
            f"Previous Findings: {len(state['findings'])}",
        ]

        if state['findings']:
            context_parts.append("Recent Findings:")
            for finding in state['findings'][-3:]:
                context_parts.append(f"- {finding}")

        if files:
            context_parts.append(f"Files to analyze: {', '.join(files)}")

        if history:
            context_parts.append(f"\nConversation History:\n{history}")

        context = "\n".join(context_parts)

        return f"{context}\n\nCurrent Request: {prompt}"

    def _extract_findings(self, content: str) -> list:
        """Extract findings from response content."""
        # Simple extraction - could be more sophisticated
        findings = []
        for line in content.split("\n"):
            if line.strip().startswith("-") or line.strip().startswith("*"):
                findings.append(line.strip().lstrip("-*").strip())
        return findings

    def get_analysis_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get current analysis state for a thread."""
        thread = self.get_thread(thread_id)
        if thread and thread.state:
            return thread.state.data
        return None

# Usage
async def main():
    from model_chorus.providers import ClaudeProvider

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = CustomAnalysisWorkflow(provider, conversation_memory=memory)

    # First analysis
    result1 = await workflow.run(
        "Analyze code quality",
        files=["src/main.py"]
    )
    thread_id = result1.metadata['thread_id']

    # Continue analysis
    result2 = await workflow.run(
        "Check for security issues",
        continuation_id=thread_id,
        files=["src/auth.py"]
    )

    # Get analysis state
    state = workflow.get_analysis_state(thread_id)
    print(f"Total findings: {len(state['findings'])}")
    print(f"Files examined: {state['files_examined']}")
```

---

## Advanced Patterns

### Pattern 1: Conversation Branching

Explore alternatives without losing main conversation:

```python
async def explore_alternatives():
    memory = ConversationMemory()
    chat = ChatWorkflow(provider, conversation_memory=memory)

    # Main conversation
    result1 = await chat.run("Design authentication system")
    main_thread = result1.metadata['thread_id']

    result2 = await chat.run("Use JWT tokens", continuation_id=main_thread)

    # Branch to explore OAuth
    oauth_thread = memory.create_thread(
        workflow_name="chat",
        parent_thread_id=main_thread,
        initial_context={"exploring": "OAuth alternative"}
    )

    result3 = await chat.run(
        "What if we use OAuth 2.0 instead?",
        continuation_id=oauth_thread
    )

    # Branch to explore session-based
    session_thread = memory.create_thread(
        workflow_name="chat",
        parent_thread_id=main_thread,
        initial_context={"exploring": "Session-based alternative"}
    )

    result4 = await chat.run(
        "What about session-based authentication?",
        continuation_id=session_thread
    )

    # Compare branches
    for thread_id in [main_thread, oauth_thread, session_thread]:
        thread = memory.get_thread(thread_id)
        print(f"\nThread: {thread.initial_context.get('exploring', 'main')}")
        print(f"Messages: {len(thread.messages)}")
```

### Pattern 2: State Migration

Handle state schema evolution:

```python
def migrate_state_v1_to_v2(old_data: Dict) -> Dict:
    """Migrate state from v1 to v2."""
    new_data = old_data.copy()

    # v2 adds confidence tracking
    if "confidence" not in new_data:
        new_data["confidence"] = "exploring"

    # v2 restructures findings
    if "findings" in new_data and isinstance(new_data["findings"], list):
        new_data["findings"] = {
            "all": new_data["findings"],
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }

    return new_data

# Use in workflow
thread = workflow.get_thread(thread_id)
if thread.state:
    if thread.state.schema_version == "1.0":
        # Migrate
        migrated_data = migrate_state_v1_to_v2(thread.state.data)
        thread.state.data = migrated_data
        thread.state.schema_version = "2.0"
        thread.state.updated_at = datetime.now(timezone.utc).isoformat()
        workflow.conversation_memory._save_thread(thread)
```

### Pattern 3: Context Window Management

Manage conversation history to fit context limits:

```python
def get_relevant_history(memory, thread_id, max_tokens=4000):
    """Get conversation history that fits in context window."""
    # Get all messages
    messages = memory.get_messages(thread_id)

    # Estimate tokens (rough: 1 token ≈ 4 characters)
    def estimate_tokens(text):
        return len(text) // 4

    # Build history from most recent, staying under limit
    relevant_messages = []
    current_tokens = 0

    for message in reversed(messages):
        msg_tokens = estimate_tokens(message.content)
        if current_tokens + msg_tokens > max_tokens:
            break
        relevant_messages.insert(0, message)
        current_tokens += msg_tokens

    return relevant_messages
```

### Pattern 4: Multi-Workflow Collaboration

Share state across workflows:

```python
async def collaborative_analysis():
    memory = ConversationMemory()
    chat = ChatWorkflow(claude_provider, conversation_memory=memory)
    thinkdeep = ThinkDeepWorkflow(gemini_provider, conversation_memory=memory)

    # Start with chat for quick exploration
    result1 = await chat.run("Analyze this bug")
    chat_thread = result1.metadata['thread_id']

    # Get findings from chat
    chat_state = memory.get_thread(chat_thread)
    findings = chat_state.messages[-1].content

    # Escalate to thinkdeep for systematic investigation
    result2 = await thinkdeep.run(
        f"Systematic investigation of: {findings}",
        files=["src/bug.py"]
    )
    deep_thread = result2.metadata['thread_id']

    # Share context between workflows
    deep_state = memory.get_thread(deep_thread)
    if deep_state.state:
        # Reference chat findings in deep investigation
        deep_state.initial_context["from_chat"] = chat_thread
        memory._save_thread(deep_state)
```

---

## Best Practices

### 1. Thread Management

**DO:**
- ✅ Create new threads for distinct topics
- ✅ Use descriptive `initial_context`
- ✅ Complete threads when done
- ✅ Archive old threads periodically
- ✅ Clean up expired threads regularly

**DON'T:**
- ❌ Reuse threads for unrelated topics
- ❌ Leave threads open indefinitely
- ❌ Skip thread completion
- ❌ Ignore TTL settings

### 2. State Management

**DO:**
- ✅ Version your state schemas
- ✅ Provide state migration functions
- ✅ Keep state focused and minimal
- ✅ Document state structure
- ✅ Validate state on load

**DON'T:**
- ❌ Store large data in state
- ❌ Change state structure without versioning
- ❌ Skip state validation
- ❌ Use state for temporary data

### 3. Message Handling

**DO:**
- ✅ Include relevant metadata
- ✅ Add file references
- ✅ Use ISO timestamps
- ✅ Track token usage
- ✅ Log model information

**DON'T:**
- ❌ Omit timestamps
- ❌ Skip file tracking
- ❌ Ignore metadata
- ❌ Store sensitive data in metadata

### 4. Context Window

**DO:**
- ✅ Limit conversation history
- ✅ Estimate token usage
- ✅ Truncate when necessary
- ✅ Prioritize recent messages
- ✅ Use summary for old context

**DON'T:**
- ❌ Send full history every time
- ❌ Ignore token limits
- ❌ Exceed model context window
- ❌ Skip message pruning

### 5. File-Based Storage

**DO:**
- ✅ Use file locking
- ✅ Handle file errors gracefully
- ✅ Validate JSON on load
- ✅ Clean up regularly
- ✅ Back up important conversations

**DON'T:**
- ❌ Skip file locking
- ❌ Ignore file corruption
- ❌ Let storage grow unbounded
- ❌ Store sensitive data unencrypted

---

## See Also

- [DOCUMENTATION.md](DOCUMENTATION.md) - Complete API reference
- [WORKFLOWS.md](WORKFLOWS.md) - Workflow selection guide
- [../examples/](../examples/) - Code examples
- [../README.md](../README.md) - Getting started
