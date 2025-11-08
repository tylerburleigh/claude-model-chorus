# STUDY Workflow - Memory System and Investigation Lifecycle

Detailed documentation of how the STUDY workflow manages conversation memory, investigation state, and the complete lifecycle of investigations from creation through continuation.

## Table of Contents

1. [Memory System Overview](#memory-system-overview)
2. [Conversation Memory Structure](#conversation-memory-structure)
3. [Investigation Lifecycle](#investigation-lifecycle)
4. [Thread Management](#thread-management)
5. [Persistence and Storage](#persistence-and-storage)
6. [State Management](#state-management)
7. [Memory Retrieval Patterns](#memory-retrieval-patterns)
8. [Continuation Behavior](#continuation-behavior)
9. [Memory Limits and Cleanup](#memory-limits-and-cleanup)
10. [API Reference](#api-reference)

---

## Memory System Overview

### What Gets Remembered?

The STUDY workflow maintains comprehensive memory across investigation sessions:

1. **Conversation History**
   - All user prompts
   - All persona responses
   - System messages and metadata
   - Timestamps for each message

2. **Investigation State**
   - Current phase of investigation
   - Personas involved
   - Key findings and insights
   - Investigation context

3. **Persona Contributions**
   - Messages from each persona
   - Persona-specific metadata
   - Perspective contributions
   - Role-specific analysis

4. **Investigation Metadata**
   - Thread ID (unique identifier)
   - Creation timestamp
   - Last update timestamp
   - Investigation parameters
   - Provider information

### Architecture

```
┌─────────────────────────────────────────┐
│    STUDY Workflow Memory System          │
├─────────────────────────────────────────┤
│                                          │
│  ConversationMemory (Abstract Interface) │
│    ↓                                     │
│  ConversationThread (Data Structure)     │
│    ├── Thread ID                         │
│    ├── Messages[] (Conversation History) │
│    ├── Metadata (Investigation State)    │
│    └── Created/Updated Timestamps        │
│    ↓                                     │
│  Persistent Storage (FileSystem)         │
│    ~/.model-chorus/conversations/        │
│    └── {thread-id}.json                  │
│                                          │
└─────────────────────────────────────────┘
```

---

## Conversation Memory Structure

### ConversationThread

The core data structure representing a complete investigation session.

```python
@dataclass
class ConversationThread:
    """Represents a single investigation conversation thread."""

    # Unique identifier
    thread_id: str

    # Conversation history
    messages: List[ConversationMessage]

    # Investigation metadata
    metadata: Dict[str, Any]

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Optional context
    initial_context: Optional[Dict[str, Any]]
```

### ConversationMessage

Individual message in the conversation history.

```python
@dataclass
class ConversationMessage:
    """Single message in conversation."""

    # Message content
    content: str

    # Sender role (user, assistant, system)
    role: str

    # Message timestamp
    timestamp: datetime

    # Additional metadata
    metadata: Dict[str, Any] = {
        'persona': 'Researcher',        # Which persona sent this
        'workflow': 'study',             # Which workflow
        'phase': 'discovery',            # Investigation phase
        'model': 'claude-3-opus',        # Model used
        'token_count': 1250,             # Tokens in message
        'context_files': [...]           # Associated files
    }
```

### Investigation Metadata

Stored in `ConversationThread.metadata`:

```python
metadata = {
    # Investigation parameters
    'initial_scenario': 'Research question or topic',
    'personas': ['SecurityExpert', 'Architect', 'Developer'],
    'provider': 'claude',
    'temperature': 0.7,
    'max_tokens': 4000,

    # State information
    'current_phase': 'discovery',  # or 'deepening', 'synthesis'
    'investigation_round': 3,      # How many rounds completed
    'total_messages': 15,          # Total messages in thread

    # Statistics
    'total_tokens_used': 12500,
    'personas_active': ['SecurityExpert', 'Architect'],
    'findings_count': 8,

    # Files and context
    'context_files': [
        'src/auth.py',
        'src/middleware.ts'
    ],

    # Continuation tracking
    'is_continuation': False,
    'continuation_from': None,  # Thread ID if this is continuation
    'continuation_count': 0,    # How many continuations
}
```

---

## Investigation Lifecycle

### Phases

Investigation progresses through distinct phases:

```
Phase 1: Discovery
├── Objective: Initial exploration and information gathering
├── Personas: All personas contribute
├── Messages: User prompt → Persona responses → Initial synthesis
└── Duration: 1-2 rounds

Phase 2: Deepening (Optional)
├── Objective: Deeper analysis and hypothesis exploration
├── Personas: Relevant personas selected by router
├── Messages: Follow-up questions → Detailed responses
└── Duration: 1+ rounds

Phase 3: Synthesis
├── Objective: Combine findings into comprehensive conclusion
├── Personas: Synthesis engine combines all perspectives
├── Messages: Final synthesis message
└── Duration: 1 round

Phase 4: Continuation (Optional)
├── Objective: Build on previous findings
├── Personas: Specified or default
├── Messages: New questions → Responses building on context
└── Duration: 1+ rounds
```

### Complete Lifecycle Timeline

```
T0: Investigation Creation
├── User initiates: model-chorus study start --scenario "..."
├── System creates new ConversationThread
├── Assigns unique thread_id (UUID)
└── Stores metadata with creation timestamp

T1: Initial Processing
├── Scenario parsed into investigation
├── Default personas loaded (or user-specified)
├── Context files read if provided
└── System prompt constructed

T2: Initial Investigation
├── User message added to thread
├── PersonaRouter selects relevant personas
├── Each persona generates response
├── Responses added to conversation history
└── Synthesis generated from responses

T3: Investigation Complete
├── Results returned to user
├── Thread saved to persistent storage
├── Thread ID provided for continuation
└── Investigation summary shown

T4+: Continuation (Optional)
├── User calls: model-chorus study start --continue <thread-id>
├── Previous thread loaded from storage
├── Conversation history retrieved
├── New scenario interpreted in context of history
├── Additional investigation continues
├── New findings integrated with previous findings
└── Updated thread saved

T5+: Investigation Review
├── User calls: model-chorus study view --investigation <thread-id>
├── Thread retrieved from storage
├── History displayed (optionally filtered by persona)
└── User can analyze findings at any time
```

### Example Lifecycle

```
Creation: 2025-11-08 14:30:00
├── Created thread: 550e8400-e29b-41d4-a716-446655440000

Investigation: 2025-11-08 14:30:15
├── Scenario: "Analyze OAuth 2.0 security patterns"
├── Personas: SecurityExpert, Architect, Developer
├── Result: 3 persona responses + synthesis

Review: 2025-11-08 14:35:00
├── User views investigation
├── Reads all findings

Continuation 1: 2025-11-08 15:00:00
├── Scenario: "Deep dive into refresh token rotation"
├── Builds on previous findings
├── 3 more persona responses
├── Updated synthesis

Continuation 2: 2025-11-08 16:30:00
├── Scenario: "Compare with SAML security"
├── Uses full conversation history
├── Cross-references previous findings
├── Final comprehensive synthesis

Review 2: 2025-11-08 17:00:00
├── Views complete investigation history
├── 15 total messages
├── Complete analysis from start to finish
```

---

## Thread Management

### Thread ID

Unique identifier for each investigation.

**Format**: UUID (Universally Unique Identifier)

**Example**: `550e8400-e29b-41d4-a716-446655440000`

**Generated**: Automatically when investigation starts

**Used for**: Continuation, viewing, analysis

### Thread Creation

```python
from model_chorus.core.conversation import ConversationMemory

memory = ConversationMemory()

# Create new thread
thread_id = memory.create_thread(
    workflow_name="study",
    initial_context={
        "scenario": "Analyze authentication patterns",
        "personas": ["SecurityExpert", "Architect"],
        "provider": "claude"
    }
)
# Returns: '550e8400-e29b-41d4-a716-446655440000'
```

### Thread Retrieval

```python
# Get complete thread
thread = memory.get_thread(thread_id)

# Get just messages
messages = memory.get_messages(thread_id)

# Get metadata
metadata = thread.metadata if thread else None
```

### Thread Listing

```python
# List all threads (future enhancement)
all_threads = memory.list_threads()

# Filter threads
study_threads = [t for t in all_threads if t.metadata.get('workflow') == 'study']
```

---

## Persistence and Storage

### Storage Location

Investigations stored in user's home directory:

```
~/.model-chorus/conversations/
├── 550e8400-e29b-41d4-a716-446655440000.json
├── a1b2c3d4-e5f6-7890-1234-567890abcdef.json
├── [thread-id].json
└── [thread-id].json
```

**Platform-specific paths**:
- **Linux/macOS**: `$HOME/.model-chorus/conversations/`
- **Windows**: `%USERPROFILE%\.model-chorus\conversations\`

### File Format

Each thread stored as JSON:

```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "workflow_name": "study",
  "created_at": "2025-11-08T14:30:00Z",
  "updated_at": "2025-11-08T16:30:00Z",
  "metadata": {
    "initial_scenario": "Analyze OAuth 2.0 security patterns",
    "personas": ["SecurityExpert", "Architect", "Developer"],
    "provider": "claude",
    "is_continuation": false,
    "continuation_count": 2
  },
  "messages": [
    {
      "role": "user",
      "content": "Analyze OAuth 2.0 security patterns",
      "timestamp": "2025-11-08T14:30:15Z",
      "metadata": {
        "workflow": "study",
        "is_continuation": false
      }
    },
    {
      "role": "assistant",
      "content": "From a security perspective, OAuth 2.0...",
      "timestamp": "2025-11-08T14:30:45Z",
      "metadata": {
        "workflow": "study",
        "persona": "SecurityExpert",
        "model": "claude-3-opus"
      }
    }
  ]
}
```

### File Size

Typical investigation file sizes:

- **Initial investigation**: 10-30 KB
- **After 1 continuation**: 20-50 KB
- **After 3+ continuations**: 50-100 KB

### Access Patterns

```python
# Read thread from storage
import json
from pathlib import Path

thread_path = Path.home() / ".model-chorus" / "conversations" / f"{thread_id}.json"
with open(thread_path) as f:
    data = json.load(f)

# Write thread to storage
with open(thread_path, 'w') as f:
    json.dump(thread_data, f, indent=2)
```

---

## State Management

### Investigation State

Current state of investigation tracked in metadata:

```python
class InvestigationState(Enum):
    """State of an investigation."""

    ACTIVE = "active"              # Currently being worked on
    PAUSED = "paused"              # Paused between sessions
    COMPLETED = "completed"        # Finished investigation
    ARCHIVED = "archived"          # Old investigation
```

### State Transitions

```
ACTIVE ──→ PAUSED (user stops)
  ↑          ↓
  └─ continued ──→ COMPLETED (user finishes)
```

### Tracking Current State

```python
# In metadata
metadata = {
    'investigation_state': 'active',
    'last_activity': '2025-11-08T16:30:00Z',
    'activity_count': 3,  # How many times resumed
}

# Determine if investigation is "recent"
from datetime import datetime, timedelta
last_activity = datetime.fromisoformat(metadata['last_activity'])
is_recent = (datetime.utcnow() - last_activity) < timedelta(days=7)
```

---

## Memory Retrieval Patterns

### Pattern 1: Load Investigation for Continuation

```python
async def resume_investigation(thread_id: str, workflow: StudyWorkflow):
    """Resume a previous investigation."""

    # Load thread from memory
    memory = workflow.conversation_memory
    thread = memory.get_thread(thread_id)

    if not thread:
        raise ValueError(f"Thread not found: {thread_id}")

    # Extract history
    history = thread.messages
    metadata = thread.metadata

    # Prepare context
    context = {
        'previous_findings': extract_synthesis(history),
        'personas_used': metadata.get('personas', []),
        'investigation_phase': metadata.get('current_phase', 'discovery'),
        'continuations': metadata.get('continuation_count', 0)
    }

    return context, history
```

### Pattern 2: Extract Persona Contributions

```python
def get_persona_contributions(thread: ConversationThread, persona: str):
    """Get all messages from specific persona."""

    contributions = []
    for message in thread.messages:
        if message.metadata.get('persona') == persona:
            contributions.append({
                'content': message.content,
                'timestamp': message.timestamp,
                'phase': message.metadata.get('phase')
            })

    return contributions
```

### Pattern 3: Build Investigation Summary

```python
def summarize_investigation(thread: ConversationThread):
    """Create summary of investigation findings."""

    summary = {
        'thread_id': thread.thread_id,
        'scenario': thread.metadata.get('initial_scenario'),
        'personas': thread.metadata.get('personas', []),
        'total_messages': len(thread.messages),
        'continuations': thread.metadata.get('continuation_count', 0),
        'findings': [],
        'key_insights': []
    }

    # Extract assistant messages as findings
    for msg in thread.messages:
        if msg.role == 'assistant':
            persona = msg.metadata.get('persona', 'Unknown')
            summary['findings'].append({
                'persona': persona,
                'content': msg.content[:200] + '...'  # Truncate
            })

    return summary
```

### Pattern 4: Find Related Investigations

```python
def find_related_investigations(keyword: str):
    """Find investigations related to keyword."""

    from pathlib import Path
    conv_dir = Path.home() / '.model-chorus' / 'conversations'

    related = []
    for thread_file in conv_dir.glob('*.json'):
        with open(thread_file) as f:
            data = json.load(f)
            if keyword.lower() in data.get('metadata', {}).get('initial_scenario', '').lower():
                related.append(data['thread_id'])

    return related
```

---

## Continuation Behavior

### How Continuation Works

When continuing an investigation:

1. **Load Previous Thread**
   ```python
   thread = memory.get_thread(continuation_id)
   history = thread.messages
   ```

2. **Preserve Context**
   - All previous messages kept
   - Persona roles maintained
   - Investigation metadata preserved

3. **Add New Interaction**
   ```python
   memory.add_message(
       thread_id=continuation_id,
       role='user',
       content='New investigation question',
       metadata={'is_continuation': True}
   )
   ```

4. **Generate Response in Context**
   - PersonaRouter aware of conversation history
   - Personas build on previous findings
   - Synthesis incorporates previous insights

5. **Save Updated Thread**
   - New messages appended to thread
   - Metadata updated (continuation count, last activity)
   - Thread file updated in storage

### Example Continuation Flow

```python
# Initial investigation
result1 = await workflow.run(
    prompt="OAuth 2.0 analysis",
    continuation_id=None  # New investigation
)
thread_id = result1.metadata['thread_id']

# Later continuation
result2 = await workflow.run(
    prompt="Deep dive into refresh tokens",
    continuation_id=thread_id  # Continue previous
)

# Thread now contains:
# - Original question and responses
# - New question and responses
# - Synthesis building on all findings
```

### Continuation Detection

The workflow detects continuation context:

```python
if continuation_id:
    # Load existing thread
    thread = memory.get_thread(continuation_id)
    if thread:
        # Mark as continuation
        metadata['is_continuation'] = True
        metadata['continuation_count'] += 1
        metadata['previous_findings'] = extract_synthesis(thread.messages)
```

---

## Memory Limits and Cleanup

### Storage Limits

No hard limits imposed by system:

- **Per thread**: Up to 10,000 messages (in practice)
- **Total storage**: Limited by disk space
- **Message size**: Limited by JSON size limits

### Practical Limits

- **Very long investigations** (100+ messages) may see performance degradation
- **Large context files** can increase message size significantly
- **Token usage** grows linearly with message count

### Cleanup Strategy

For managing storage:

```python
# List all threads
import json
from pathlib import Path
conv_dir = Path.home() / '.model-chorus' / 'conversations'

# Find old threads
from datetime import datetime, timedelta
old_date = datetime.utcnow() - timedelta(days=90)

for thread_file in conv_dir.glob('*.json'):
    with open(thread_file) as f:
        data = json.load(f)
        updated = datetime.fromisoformat(data['updated_at'])
        if updated < old_date:
            print(f"Old thread: {data['thread_id']}")
            # Optionally archive or delete
```

### Manual Cleanup

```bash
# View conversation storage
ls -lh ~/.model-chorus/conversations/

# Backup before cleanup
cp -r ~/.model-chorus/conversations/ backup_conversations/

# Remove specific thread
rm ~/.model-chorus/conversations/{thread-id}.json

# Archive threads
mkdir archived_conversations
mv ~/.model-chorus/conversations/*.json archived_conversations/
```

---

## API Reference

### ConversationMemory Class

```python
class ConversationMemory:
    """Manages conversation threads and message persistence."""

    def create_thread(
        self,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new conversation thread.

        Returns:
            Thread ID (UUID string)
        """
        pass

    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Get complete conversation thread.

        Returns:
            ConversationThread or None if not found
        """
        pass

    def get_messages(self, thread_id: str) -> Optional[List[ConversationMessage]]:
        """Get just messages from thread.

        Returns:
            List of ConversationMessage or None if thread not found
        """
        pass

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add message to thread.

        Args:
            thread_id: Thread ID
            role: 'user' or 'assistant'
            content: Message text
            metadata: Additional metadata

        Returns:
            True if successful
        """
        pass

    def list_threads(self) -> List[ConversationThread]:
        """List all conversation threads.

        Returns:
            List of ConversationThread objects
        """
        pass

    def delete_thread(self, thread_id: str) -> bool:
        """Delete conversation thread.

        Returns:
            True if successful
        """
        pass
```

### Usage Example

```python
from model_chorus.workflows.study import StudyWorkflow
from model_chorus.core.conversation import ConversationMemory
from model_chorus.providers import ClaudeProvider

# Create memory and provider
memory = ConversationMemory()
provider = ClaudeProvider()

# Create workflow
workflow = StudyWorkflow(provider, conversation_memory=memory)

# Start investigation
result = await workflow.run(
    prompt="Analyze authentication patterns"
)
thread_id = result.metadata['thread_id']

# Later, continue investigation
result2 = await workflow.run(
    prompt="Deep dive into OAuth 2.0",
    continuation_id=thread_id
)

# View thread
thread = memory.get_thread(thread_id)
print(f"Messages: {len(thread.messages)}")
print(f"Continuations: {thread.metadata.get('continuation_count', 0)}")
```

---

## Advanced Topics

### Performance Optimization

For large investigations:

1. **Lazy Loading**: Load messages on demand
2. **Compression**: Compress old messages
3. **Archiving**: Move completed investigations to archive
4. **Indexing**: Build keyword index for faster search

### Memory Analysis

```python
def analyze_investigation(thread_id: str):
    """Analyze memory usage of investigation."""

    thread = memory.get_thread(thread_id)

    analysis = {
        'thread_id': thread_id,
        'total_messages': len(thread.messages),
        'messages_by_role': {
            'user': len([m for m in thread.messages if m.role == 'user']),
            'assistant': len([m for m in thread.messages if m.role == 'assistant']),
        },
        'messages_by_persona': {},
        'total_characters': sum(len(m.content) for m in thread.messages),
        'estimated_tokens': sum(
            len(m.metadata.get('token_count', 0))
            for m in thread.messages
        ),
        'continuations': thread.metadata.get('continuation_count', 0),
        'time_span': {
            'start': thread.created_at,
            'end': thread.updated_at,
            'duration': (thread.updated_at - thread.created_at).total_seconds()
        }
    }

    # Count by persona
    for msg in thread.messages:
        persona = msg.metadata.get('persona', 'Unknown')
        analysis['messages_by_persona'][persona] = \
            analysis['messages_by_persona'].get(persona, 0) + 1

    return analysis
```

---

## See Also

- [STUDY Workflow README](./README.md)
- [CLI Reference](./CLI_REFERENCE.md)
- [ConversationMemory Implementation](../../core/conversation.py)
- [BaseWorkflow Documentation](../../core/base_workflow.py)
