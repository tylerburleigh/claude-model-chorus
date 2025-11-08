# STUDY Workflow Architecture Guide - Key Insights for Specification

## Quick Navigation

This guide provides the essential information needed to create an accurate specification for the STUDY workflow (persona orchestration).

---

## 1. Directory Structure (What to Know)

**Core Locations:**
- **Workflow code**: `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/workflows/`
- **Core abstractions**: `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/`
- **CLI commands**: `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/cli/main.py`
- **Configuration**: `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/config.py`
- **Conversations storage**: `~/.model-chorus/conversations/` (user home directory)

**Where to Add STUDY Files:**
- `model_chorus/src/model_chorus/workflows/study.py` - Main workflow class
- Optional: `model_chorus/src/model_chorus/workflows/study/` - Subdirectory if needing multiple files
- Extension to `model_chorus/src/model_chorus/core/models.py` - New data models
- Extension to `model_chorus/src/model_chorus/cli/main.py` - New CLI command

---

## 2. Existing Workflows - Use as Templates

### ARGUMENT Workflow (Best Template for Personas)
**File**: `workflows/argument/argument_workflow.py`

Uses RoleOrchestrator for multi-role execution:
```python
@WorkflowRegistry.register("argument")
class ArgumentWorkflow(BaseWorkflow):
    async def run(self, prompt: str, continuation_id: Optional[str] = None, files: Optional[List[str]] = None, **config):
        # 1. Define roles (Creator, Skeptic, Moderator)
        # 2. Call RoleOrchestrator.execute() with roles
        # 3. Return WorkflowResult with steps (one per role)
```

**Pattern**: Define personas as ModelRole objects → Execute via RoleOrchestrator → Return structured results

### THINKDEEP Workflow (Best Template for State/Memory)
**File**: `workflows/thinkdeep.py`

Maintains investigation state across conversation turns:
```python
class ThinkDeepWorkflow(BaseWorkflow):
    async def run(self, step: str, step_number: int, total_steps: int, ..., continuation_id: Optional[str], ...):
        # 1. Resume state from conversation thread if continuation_id provided
        # 2. Update investigation state
        # 3. Generate response
        # 4. Persist state back to conversation thread
        # 5. Return WorkflowResult with metadata
    
    def get_investigation_state(self, thread_id: str) -> ThinkDeepState:
        # Retrieve current investigation state
        pass
```

**Pattern**: Use ConversationMemory + custom state model → Persist state in thread → Retrieve on continuation

### CHAT Workflow (Simplest Template)
**File**: `workflows/chat.py`

Single-provider with threading:
```python
class ChatWorkflow(BaseWorkflow):
    def __init__(self, provider: ModelProvider, conversation_memory: Optional[ConversationMemory]):
        # Single provider (not multi-model)
        # Simple continuation via thread ID
        pass
    
    async def run(self, prompt: str, continuation_id: Optional[str] = None, ...):
        # Simple: get history → generate → save to thread
        pass
```

---

## 3. Key Abstractions You Must Inherit/Use

### BaseWorkflow (Abstract Class)
**File**: `core/base_workflow.py`

Every workflow must inherit and implement:
```python
class MyWorkflow(BaseWorkflow):
    def __init__(self, name, description, config=None, conversation_memory=None):
        super().__init__(name, description, config, conversation_memory)
        # Your initialization
    
    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        # Must implement: return WorkflowResult with success, steps, synthesis, error, metadata
        pass
```

**Inherited Methods Available:**
- `_execute_with_fallback()` - Automatic provider fallback
- `check_provider_availability()` - Test providers before running
- `get_thread()`, `add_message()`, `resume_conversation()` - Conversation support

### WorkflowResult Structure
```python
@dataclass
class WorkflowResult:
    success: bool                      # True if execution succeeded
    steps: List[WorkflowStep] = []    # Individual steps/persona outputs
    synthesis: Optional[str] = None    # Final combined result
    error: Optional[str] = None        # Error message if failed
    metadata: Dict[str, Any] = {}      # thread_id, model, roles_executed, etc.

@dataclass
class WorkflowStep:
    step_number: int                   # 1, 2, 3...
    content: str                       # The actual output
    model: Optional[str] = None        # Which model produced this
    metadata: Dict[str, Any] = {}      # Additional info
```

### RoleOrchestrator (For Personas)
**File**: `core/role_orchestration.py`

Perfect for your persona system:
```python
class ModelRole(BaseModel):
    role: str                    # "researcher", "analyst", "critic"
    model: str                   # "gpt-5", "claude-opus"
    stance: Optional[str]        # "for", "against", "neutral", or custom
    stance_prompt: Optional[str] # Additional prompt for this persona
    system_prompt: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    metadata: Dict[str, Any]

class RoleOrchestrator:
    async def execute(
        self,
        prompt: str,
        roles: List[ModelRole],
        pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
        synthesis: SynthesisStrategy = SynthesisStrategy.CONCATENATE
    ) -> OrchestrationResult

# Patterns available:
class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"     # One persona at a time
    PARALLEL = "parallel"         # All personas simultaneously
    HYBRID = "hybrid"             # Mix of both

# Synthesis strategies:
class SynthesisStrategy(Enum):
    NONE = "none"                 # Raw responses only
    CONCATENATE = "concatenate"   # Join with labels
    AI_SYNTHESIZE = "ai_synthesize"  # Use AI to merge
    STRUCTURED = "structured"     # Dict with role keys
```

### ConversationMemory (For State Persistence)
**File**: `core/conversation.py`

```python
class ConversationMemory:
    def create_thread(self, workflow_name: str, initial_context: Optional[Dict] = None) -> str:
        # Returns thread_id
        pass
    
    def add_message(self, thread_id: str, role: str, content: str, **kwargs) -> bool:
        # Add user/assistant message to thread
        pass
    
    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        # Retrieve entire thread
        pass
    
    def get_messages(self, thread_id: str) -> Optional[List[ConversationMessage]]:
        # Get just messages for context
        pass

# Messages are stored as JSON in ~/.model-chorus/conversations/{thread_id}.json
# Supports metadata: files, workflow_name, model_provider, model_name, etc.
```

---

## 4. Data Models to Define (in core/models.py)

Based on existing patterns, you'll need these new models:

```python
# Persona definition
class PersonaState(BaseModel):
    persona_id: str
    name: str
    role: str
    description: str
    perspective: str              # What this persona focuses on
    memory: List[str]            # Key facts this persona remembers
    expertise_areas: List[str]
    reasoning_style: str
    metadata: Dict[str, Any]

# Study-specific investigation state
class StudyState(BaseModel):
    thread_id: str
    workflow_name: str = "study"
    personas: List[PersonaState]         # Active personas
    current_focus: str                   # What's being studied
    investigation_steps: List[Dict]      # Step history
    persona_memories: Dict[str, List[str]]  # Persona-specific memories
    cross_persona_insights: List[str]    # Connections between personas
    created_at: datetime
    updated_at: datetime

# Persona routing decision
class PersonaRouting(BaseModel):
    step_number: int
    input_topic: str
    selected_personas: List[str]         # Which personas to query
    routing_reason: str
    timestamp: datetime

# Memory entry with persona context
class PersonaMemoryEntry(BaseModel):
    persona_id: str
    content: str
    importance: float              # How important (0.0-1.0)
    tags: List[str]
    related_personas: List[str]    # Cross-references
    timestamp: datetime
```

---

## 5. CLI Integration Pattern (in cli/main.py)

Follow this structure (similar to existing commands):

```python
@app.command()
def study(
    prompt: str = typer.Argument(..., help="Topic to study from multiple perspectives"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Primary provider"),
    num_personas: int = typer.Option(3, "--personas", "-n", help="Number of personas"),
    continuation_id: Optional[str] = typer.Option(None, "--continue", "-c", help="Resume study"),
    files: Optional[List[str]] = typer.Option(None, "--file", "-f", help="Context files"),
    routing_strategy: str = typer.Option("adaptive", "--routing", help="How to route queries"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_provider_check: bool = typer.Option(False, "--skip-provider-check"),
):
    """
    Study a topic through coordinated persona exploration.
    
    Multiple AI personas with different perspectives investigate a topic,
    with intelligent routing to select relevant personas for each question.
    """
    try:
        # 1. Get config defaults
        config = get_config()
        if provider is None:
            provider = config.get_default_provider('study', 'claude')
        
        # 2. Create provider instance
        provider_instance = get_provider_by_name(provider)
        
        # 3. Create conversation memory
        memory = ConversationMemory()
        
        # 4. Create workflow
        workflow = StudyWorkflow(
            provider=provider_instance,
            conversation_memory=memory,
            config={'num_personas': num_personas, 'routing_strategy': routing_strategy}
        )
        
        # 5. Run async workflow
        result = asyncio.run(
            workflow.run(
                prompt=prompt,
                continuation_id=continuation_id,
                files=files,
                skip_provider_check=skip_provider_check,
                **config
            )
        )
        
        # 6. Display results
        if result.success:
            console.print(f"[bold green]✓ Study completed[/bold green]")
            # Show persona responses
            for i, step in enumerate(result.steps):
                console.print(f"\n[bold cyan]{step.metadata.get('persona_name', f'Persona {i+1}')}:[/bold cyan]")
                console.print(step.content)
            # Show synthesis
            if result.synthesis:
                console.print(f"\n[bold magenta]Cross-Persona Insights:[/bold magenta]")
                console.print(result.synthesis)
        else:
            console.print(f"[red]✗ Study failed: {result.error}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
```

---

## 6. Workflow Registration (Automatic)

In your `study.py` file, use the decorator:

```python
@WorkflowRegistry.register("study")
class StudyWorkflow(BaseWorkflow):
    # Implementation
    pass
```

This automatically makes it discoverable via:
- `WorkflowRegistry.get("study")`
- `WorkflowRegistry.list_workflows()`
- CLI help text (if documented)

---

## 7. Configuration Addition (in core/config.py)

Update `WorkflowConfig` validation:

```python
class WorkflowConfig(BaseModel):
    # ... existing fields ...
    
    # Study-specific
    num_personas: Optional[int] = Field(None, ge=1, le=10)
    routing_strategy: Optional[str] = Field(None, pattern=r"^(adaptive|round_robin|expertise_based|diversity)$")
    persona_temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
```

Update `ModelChorusConfig.validate_workflow_names()` to include 'study':
```python
valid_workflows = ['chat', 'consensus', 'thinkdeep', 'argument', 'ideate', 'research', 'study']
```

Add to default config template in `_config_init()`:
```yaml
workflows:
  study:
    provider: claude
    num_personas: 3
    routing_strategy: adaptive
    temperature: 0.7
```

---

## 8. Key Implementation Decisions to Make in Spec

### Decision 1: Provider Model
- **Single provider**: Uses one model for all personas (memory efficient, deterministic)
- **Multi-provider**: Different providers for different personas (diverse perspectives)
- **Hybrid**: Primary + fallbacks per persona

**Recommendation**: Start with single-provider, personas are logical roles not separate models

### Decision 2: Persona Definition
- **Hardcoded personas**: Fixed set (e.g., "researcher", "critic", "synthesizer")
- **Generated personas**: AI-creates personas based on topic
- **User-specified personas**: Users define their own

**Recommendation**: Start with predefined personas + allow customization

### Decision 3: Routing Strategy
- **Sequential**: All personas answer all questions
- **Round-robin**: Rotate which persona answers each question
- **Adaptive**: System decides which personas are relevant (LLM-based routing)
- **Expertise-based**: Route to personas with matching expertise

**Recommendation**: Start with sequential → add adaptive routing as extension

### Decision 4: Memory System
- **Shallow**: Conversation history only, no persona-specific memory
- **Deep**: Each persona has separate memory + shared context
- **Hybrid**: Shared working memory + persona specialization

**Recommendation**: Hybrid approach (extend ConversationMemory with persona tracking)

### Decision 5: State Persistence
- **Lightweight**: Just thread history, reconstruct personas each run
- **Full**: Persist persona states, memories, reasoning history
- **Selective**: Persist only essential state

**Recommendation**: Full persistence (like ThinkDeep) for long investigations

---

## 9. Extension Points to Consider

1. **Persona Creation**
   - Hook for custom persona definition
   - Support for persona templates

2. **Memory Management**
   - Implement PersonaMemory class extending ConversationMemory
   - Support persona-specific context windows

3. **Routing Intelligence**
   - PersonaRouter class for intelligent question routing
   - Cost optimization (fewer calls to relevant personas)

4. **Synthesis Strategies**
   - Different synthesis approaches (consensus, voting, ranking)
   - Persona conflict resolution

5. **Monitoring & Metrics**
   - Track which personas contribute most
   - Measure persona diversity/redundancy

---

## 10. Testing Strategy

Based on codebase patterns:

```python
# test_study_workflow.py

@pytest.mark.asyncio
async def test_study_workflow_creation():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = StudyWorkflow(provider, conversation_memory=memory)
    assert workflow.name == "Study"

@pytest.mark.asyncio
async def test_study_workflow_run():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = StudyWorkflow(provider, conversation_memory=memory)
    
    result = await workflow.run(
        prompt="What is AI safety?",
        skip_provider_check=True,
        num_personas=3
    )
    
    assert result.success
    assert len(result.steps) == 3  # Three personas
    assert result.synthesis is not None
    assert 'thread_id' in result.metadata
    assert 'personas_used' in result.metadata

@pytest.mark.asyncio
async def test_study_workflow_continuation():
    # Test resuming a study via continuation_id
    pass

@pytest.mark.asyncio
async def test_persona_routing():
    # Test intelligent routing decisions
    pass
```

---

## 11. Files to Review Before Writing Spec

**Must Read** (essential patterns):
1. `core/base_workflow.py` - Lines 58-112 (BaseWorkflow class)
2. `core/role_orchestration.py` - Lines 94-150 (ModelRole definition)
3. `core/conversation.py` - Lines 36-100 (ConversationMemory)
4. `workflows/argument/argument_workflow.py` - Lines 30-96 (Workflow doc structure)
5. `workflows/thinkdeep.py` - Lines 28-82 (State persistence example)

**Reference Implementations**:
1. `cli/main.py` - Lines 314-524 (argument command pattern) or Lines 528-743 (ideate pattern)
2. `workflows/argument/argument_workflow.py` - Full implementation (role-based)
3. `workflows/thinkdeep.py` - Full implementation (stateful)

**Configuration**:
1. `core/config.py` - Lines 33-103 (WorkflowConfig validation)

---

## 12. Quick Summary for Your Spec

**STUDY Workflow is:**
- A role-based investigation workflow combining personas
- Built on BaseWorkflow + RoleOrchestrator + ConversationMemory
- Uses single provider (or optionally multiple)
- Supports conversation threading for long investigations
- Includes intelligent routing to select relevant personas
- Maintains persona memory across turns
- Produces cross-persona synthesis

**Key Components to Specify:**
1. Persona definitions (structure, capabilities, memory)
2. Routing strategy (which personas for which questions)
3. Memory system (shared vs persona-specific)
4. State persistence (what to save between turns)
5. CLI interface (commands and options)
6. Output format (individual personas + synthesis)
7. Configuration options (defaults, customization)

---

## Absolute Paths Reference

```
Main Workflow File:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/workflows/argument/argument_workflow.py

Base Classes:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/base_workflow.py
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/role_orchestration.py
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/conversation.py

CLI Entry:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/cli/main.py

Config:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/config.py

Data Models:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/models.py

Registry:
/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/registry.py

Similar Workflows:
- Single + State: /home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/workflows/thinkdeep.py
- Roles: /home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/workflows/argument/argument_workflow.py
- Simple: /home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/workflows/chat.py
```

