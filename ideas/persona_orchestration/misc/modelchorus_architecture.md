# ModelChorus Architecture Overview

## Project Structure

### Root Directory Layout
```
/home/tyler/Documents/GitHub/claude-model-chorus/
├── model_chorus/                    # Main package directory
│   ├── src/model_chorus/           # Source code root
│   ├── tests/                      # Test suite
│   ├── pyproject.toml              # Project configuration
│   └── examples/                   # Example usage scripts
├── docs/                           # Documentation
├── ideas/                          # Concept explorations (including persona_orchestration/)
├── agents/                         # Agent definitions
├── README.md                       # Main documentation
└── CHANGELOG.md                    # Version history
```

### Source Code Organization (model_chorus/src/model_chorus/)

```
model_chorus/
├── __init__.py                     # Package initialization
├── cli/                            # CLI module
│   ├── main.py                     # CLI entry point (Typer app)
│   ├── setup.py                    # CLI setup command
│   └── __init__.py
├── core/                           # Core abstractions & infrastructure
│   ├── base_workflow.py            # BaseWorkflow abstract class
│   ├── models.py                   # Pydantic data models
│   ├── conversation.py             # ConversationMemory for multi-turn
│   ├── role_orchestration.py       # Role-based execution framework
│   ├── registry.py                 # Workflow registry/plugin system
│   ├── config.py                   # Configuration management
│   ├── state.py                    # State management
│   ├── progress.py                 # Progress tracking
│   ├── gap_analysis.py             # Gap analysis utilities
│   ├── clustering.py               # Response clustering
│   ├── contradiction.py            # Contradiction detection
│   └── __init__.py
├── providers/                      # Provider implementations
│   ├── base_provider.py            # ModelProvider abstract class
│   ├── cli_provider.py             # CLIProvider base for CLI-based models
│   ├── claude_provider.py          # Claude CLI provider
│   ├── gemini_provider.py          # Gemini CLI provider
│   ├── codex_provider.py           # OpenAI Codex CLI provider
│   ├── cursor_agent_provider.py    # Cursor Agent CLI provider
│   └── __init__.py
├── workflows/                      # Workflow implementations
│   ├── __init__.py                 # Workflow exports
│   ├── chat.py                     # ChatWorkflow (single-model)
│   ├── consensus.py                # ConsensusWorkflow (multi-model)
│   ├── thinkdeep.py                # ThinkDeepWorkflow (investigation)
│   ├── argument/                   # ArgumentWorkflow package
│   │   ├── argument_workflow.py    # Main argument workflow
│   │   ├── semantic.py             # Semantic analysis utilities
│   │   └── __init__.py
│   └── ideate/                     # IdeateWorkflow package
│       ├── ideate_workflow.py      # Main ideate workflow
│       └── __init__.py
└── utils/                          # Utilities
    ├── citation_formatter.py       # Citation formatting
    └── __init__.py
```

---

## Key Architecture Patterns

### 1. BaseWorkflow Pattern

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/base_workflow.py`

All workflows inherit from `BaseWorkflow` abstract class:

```python
class BaseWorkflow(ABC):
    def __init__(self, name, description, config=None, conversation_memory=None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.conversation_memory = conversation_memory
    
    @abstractmethod
    async def run(self, prompt: str, **kwargs) -> WorkflowResult
```

**Key Features:**
- `WorkflowResult` dataclass contains:
  - `success: bool`
  - `steps: List[WorkflowStep]` - Individual execution steps
  - `synthesis: Optional[str]` - Final synthesis
  - `error: Optional[str]` - Error message if failed
  - `metadata: Dict[str, Any]` - Execution metadata
  
- Built-in methods:
  - `_execute_with_fallback()` - Provider fallback handling
  - `check_provider_availability()` - Availability checking
  - `get_thread()`, `add_message()`, `resume_conversation()` - Conversation support

**Registration:** Workflows use `@WorkflowRegistry.register("name")` decorator

---

### 2. Provider Architecture

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/providers/`

**Base Classes:**
1. `ModelProvider` (abstract)
   - Methods: `generate()`, `supports_vision()`, `get_available_models()`
   - Models are wrapped as `ModelConfig` dataclass

2. `CLIProvider` (extends ModelProvider)
   - Executes CLI commands asynchronously
   - Handles timeouts, retries, error parsing
   - Methods: `check_availability()`, `build_command()`, `parse_response()`

**Concrete Providers:**
- `ClaudeProvider` - Anthropic Claude via `claude` CLI
- `GeminiProvider` - Google Gemini via `gemini` CLI
- `CodexProvider` - OpenAI Codex via CLI
- `CursorAgentProvider` - Cursor Agent via CLI

**Key Data Structures:**
```python
@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    continuation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResponse:
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### 3. Conversation Memory & Threading

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/conversation.py`

**Features:**
- File-based persistence: `~/.model-chorus/conversations/{thread_id}.json`
- Thread-safe with file locking via `filelock`
- TTL-based cleanup (default: 3 hours)
- Max messages per thread (default: 50)

**Core Models:**
```python
class ConversationMessage:
    role: str  # "user" or "assistant"
    content: str
    workflow_name: Optional[str]
    model_provider: Optional[str]
    model_name: Optional[str]
    files: Optional[List[str]]
    timestamp: datetime

class ConversationThread:
    thread_id: str
    workflow_name: str
    created_at: datetime
    parent_thread_id: Optional[str]
    messages: List[ConversationMessage]

class ConversationMemory:
    conversations_dir: Path = ~/.model-chorus/conversations
    ttl_hours: int = 3
    max_messages: int = 50
```

**Methods:**
- `create_thread()` - Create new thread, returns thread_id
- `add_message()` - Add message to thread
- `get_thread()` - Retrieve thread by ID
- `list_threads()` - List all threads
- `cleanup_expired()` - Remove TTL-expired threads

---

### 4. Role Orchestration Framework

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/role_orchestration.py`

**Purpose:** Enable multi-role workflows (ARGUMENT, IDEATE, etc.)

**Enums:**
```python
class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"    # One role at a time, in order
    PARALLEL = "parallel"        # All roles simultaneously
    HYBRID = "hybrid"            # Mix of sequential & parallel

class SynthesisStrategy(Enum):
    NONE = "none"                # Return raw responses
    CONCATENATE = "concatenate"  # Simple text joining
    AI_SYNTHESIZE = "ai_synthesize"  # Use AI to merge
    STRUCTURED = "structured"    # Dict with role keys
```

**Core Classes:**
```python
class ModelRole(BaseModel):
    role: str              # e.g., "creator", "skeptic", "moderator"
    model: str             # e.g., "gpt-5", "claude-opus"
    stance: Optional[str]  # "for", "against", "neutral"
    stance_prompt: Optional[str]
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
```

---

## Existing Workflow Implementations

### 1. ChatWorkflow (Single-Provider)

**Location:** `workflows/chat.py`

**Pattern:**
- Single provider (not multi-model)
- Conversation threading via `continuation_id`
- Simple request/response flow

**Key Methods:**
```python
class ChatWorkflow(BaseWorkflow):
    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> WorkflowResult
```

---

### 2. ConsensusWorkflow (Multi-Provider)

**Location:** `workflows/consensus.py`

**Pattern:**
- Multiple providers (parallel execution)
- No roles - just "get all opinions"
- Configurable synthesis strategies

**Strategy Options:**
```python
class ConsensusStrategy(Enum):
    ALL_RESPONSES = "all_responses"      # Return all responses
    FIRST_VALID = "first_valid"         # Use first successful
    MAJORITY = "majority"               # Most common answer
    WEIGHTED = "weighted"               # Weight by model quality
    SYNTHESIZE = "synthesize"           # AI-synthesized result
```

**Key Methods:**
```python
class ConsensusWorkflow:
    async def execute(
        self,
        request: GenerationRequest,
        strategy: ConsensusStrategy = ConsensusStrategy.ALL_RESPONSES
    ) -> ConsensusResult
```

---

### 3. ThinkDeepWorkflow (Investigation)

**Location:** `workflows/thinkdeep.py`

**Pattern:**
- Single provider with extended reasoning
- Multi-step investigation with hypothesis tracking
- Confidence progression across steps
- State persistence via conversation threading

**Unique Features:**
- Maintains `ThinkDeepState`:
  - `hypotheses: List[Hypothesis]` - Tracked theories
  - `steps: List[InvestigationStep]` - Investigation history
  - `current_confidence: ConfidenceLevel` - Confidence tracking
  - `relevant_files: List[str]` - Examined files

**Confidence Levels:**
```python
class ConfidenceLevel(str, Enum):
    EXPLORING = "exploring"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    ALMOST_CERTAIN = "almost_certain"
    CERTAIN = "certain"
```

**Key Methods:**
```python
class ThinkDeepWorkflow(BaseWorkflow):
    async def run(
        self,
        step: str,
        step_number: int,
        total_steps: int,
        next_step_required: bool,
        findings: str,
        hypothesis: Optional[str] = None,
        confidence: str = "exploring",
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult
    
    def get_investigation_state(self, thread_id: str) -> ThinkDeepState
    def get_investigation_summary(self, thread_id: str) -> Dict[str, Any]
```

---

### 4. ArgumentWorkflow (Role-Based)

**Location:** `workflows/argument/argument_workflow.py`

**Pattern:**
- Uses `RoleOrchestrator` for sequential role execution
- Three roles: Creator, Skeptic, Moderator
- Dialectical reasoning framework

**Roles:**
1. **Creator** (Stance: FOR) - Generates thesis supporting position
2. **Skeptic** (Stance: AGAINST) - Provides critical rebuttal
3. **Moderator** (Stance: NEUTRAL) - Synthesizes into balanced analysis

**Key Methods:**
```python
@WorkflowRegistry.register("argument")
class ArgumentWorkflow(BaseWorkflow):
    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        **config
    ) -> WorkflowResult
```

**Output Structure:**
- `result.steps[0]` = Creator's thesis
- `result.steps[1]` = Skeptic's rebuttal
- `result.steps[2]` = Moderator's synthesis
- `result.metadata['roles_executed']` = List of executed roles

---

### 5. IdeateWorkflow (Creative)

**Location:** `workflows/ideate/ideate_workflow.py`

**Pattern:**
- Single provider with high temperature (creativity)
- Multi-round idea generation
- Conversation threading for iterative refinement

**Key Methods:**
```python
@WorkflowRegistry.register("ideate")
class IdeateWorkflow(BaseWorkflow):
    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        num_ideas: int = 5,
        temperature: float = 0.9,
        **kwargs
    ) -> WorkflowResult
```

---

## CLI Structure

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/cli/main.py`

**Framework:** Typer (FastAPI-like for CLI)

**Entry Point:** `model-chorus` command (defined in `pyproject.toml`)

**Command Registration Pattern:**
```python
app = typer.Typer(name="model-chorus")

@app.command()
def chat(prompt: str, provider: Optional[str], ...):
    """Chat with a single AI model"""
    # Implementation

@app.command()
def consensus(prompt: str, providers: Optional[List[str]], ...):
    """Run consensus workflow"""
    # Implementation

@app.command()
def argument(prompt: str, provider: Optional[str], ...):
    """Analyze arguments through dialectical reasoning"""
    # Implementation

@app.command()
def ideate(prompt: str, provider: Optional[str], ...):
    """Generate creative ideas"""
    # Implementation

@app.command()
def thinkdeep(step: str, step_number: int, ...):
    """Start ThinkDeep investigation"""
    # Implementation
```

**Common Options Pattern:**
```
--provider / -p          : Provider to use
--continuation-id / -c   : Thread ID for continuation
--file / -f              : Files to include
--system                 : System prompt
--temperature / -t       : Temperature (0.0-1.0)
--max-tokens             : Max tokens to generate
--output / -o            : Output file (JSON)
--verbose / -v           : Verbose output
--skip-provider-check    : Skip availability check
```

---

## Configuration Management

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/config.py`

**Config File:** `.model-chorusrc` (YAML or JSON)

**Search Path:** Searches up directory tree from cwd

**Supported Filenames:**
- `.model-chorusrc`
- `.model-chorusrc.yaml`
- `.model-chorusrc.yml`
- `.model-chorusrc.json`

**Schema:**
```yaml
default_provider: claude

generation:
  temperature: 0.7
  max_tokens: 2000
  timeout: 120.0

workflows:
  chat:
    provider: claude
    temperature: 0.7
    
  consensus:
    providers:
      - claude
      - gemini
    strategy: synthesize
    temperature: 0.7
  
  thinkdeep:
    provider: claude
    thinking_mode: medium
    temperature: 0.6
  
  argument:
    provider: claude
  
  ideate:
    provider: claude
    temperature: 0.9
```

**Validation:**
- Provider names: claude, gemini, codex, cursor-agent
- Workflow names: chat, consensus, thinkdeep, argument, ideate, research
- Temperature: 0.0-1.0
- Enum patterns for strategy, thinking_mode, citation_style, depth

---

## Dependencies

**Location:** `model_chorus/pyproject.toml`

**Core Dependencies:**
```
anthropic>=0.25.0          # Anthropic API (fallback)
openai>=1.0.0              # OpenAI API (fallback)
google-generativeai>=0.3.0 # Google Generative AI (fallback)
pydantic>=2.0.0            # Data validation
httpx>=0.24.0              # HTTP client
rich>=13.0.0               # Terminal formatting
typer>=0.16.0,<0.20        # CLI framework
```

**Notable Omissions:**
- No database dependencies (file-based conversations)
- No caching library (in-memory only)
- No memory/vector storage (future extension point)

**Dev Dependencies:**
```
pytest>=7.0.0              # Testing
pytest-asyncio>=0.21.0     # Async testing
black>=23.0.0              # Code formatting
ruff>=0.1.0                # Linting
mypy>=1.0.0                # Type checking
```

---

## Data Models

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/models.py` (64KB file)

**Key Models:**
```python
class WorkflowRequest(BaseModel):
    prompt: str
    models: List[str]
    config: Dict[str, Any]
    system_prompt: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    images: Optional[List[str]]
    metadata: Dict[str, Any]

class WorkflowResponse(BaseModel):
    result: str
    success: bool
    workflow_name: str
    steps: int
    models_used: List[str]
    error: Optional[str]
    metadata: Dict[str, Any]

class ThinkDeepState(BaseModel):
    thread_id: str
    workflow_name: str
    hypotheses: List[Hypothesis]
    steps: List[InvestigationStep]
    current_confidence: ConfidenceLevel
    relevant_files: List[str]
    created_at: datetime
    updated_at: datetime

class Hypothesis(BaseModel):
    hypothesis: str
    status: Literal["active", "validated", "disproven"]
    evidence: List[str]
    introduced_at_step: int
    last_updated_at_step: int

class InvestigationStep(BaseModel):
    step_number: int
    step_description: str
    findings: str
    confidence: ConfidenceLevel
    files_checked: List[str]
    timestamp: datetime
```

---

## Workflow Registry & Plugin System

**Location:** `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/src/model_chorus/core/registry.py`

**Pattern:** Decorator-based registration

**Usage:**
```python
@WorkflowRegistry.register("myworkflow")
class MyWorkflow(BaseWorkflow):
    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        # Implementation
        pass

# Later, retrieve:
WorkflowClass = WorkflowRegistry.get("myworkflow")
workflow = WorkflowClass("name", "description")

# List all:
all_workflows = WorkflowRegistry.list_workflows()
```

**Methods:**
- `register(name)` - Decorator to register workflow
- `register_workflow(name, class)` - Programmatic registration
- `get(name)` - Retrieve workflow class
- `list_workflows()` - Get all registered
- `is_registered(name)` - Check if registered
- `unregister(name)` - Remove from registry
- `clear()` - Clear all (for testing)

---

## Testing Structure

**Location:** Multiple test files across codebase

**Test Locations:**
- `/home/tyler/Documents/GitHub/claude-model-chorus/tests/` - Top-level tests
- `/home/tyler/Documents/GitHub/claude-model-chorus/model_chorus/tests/` - Package tests

**Test Types:**
- Unit tests (individual components)
- Integration tests (multi-component workflows)
- Provider tests
- Conversation threading tests
- Contradiction detection tests
- Gap analysis tests

**Test Files Identified:**
- `test_chat_workflow.py`
- `test_consensus_workflow.py`
- `test_argument_workflow.py`
- `test_ideate_workflow.py`
- `test_thinkdeep_workflow.py`
- `test_conversation.py`
- `test_claude_provider.py`
- `test_gemini_integration.py`
- `test_role_orchestration.py`
- `test_cli_integration.py`

---

## Key Architectural Patterns for STUDY Workflow

### Pattern 1: Single-Provider with State
Similar to ThinkDeepWorkflow - maintain investigation state across turns

### Pattern 2: Role-Based Execution
Use RoleOrchestrator framework from ARGUMENT workflow for persona assignment

### Pattern 3: Memory System
Leverage ConversationMemory for:
- Conversation history
- Thread persistence
- Context management

### Pattern 4: Workflow Registration
Use `@WorkflowRegistry.register("study")` decorator to integrate

### Pattern 5: CLI Integration
Follow Typer command pattern in `main.py` with standard options:
- `--provider`, `--continuation-id`, `--files`, `--output`, etc.

---

## Extension Points for STUDY Workflow

1. **New Data Models** (in `core/models.py`):
   - PersonaState - Define persona characteristics
   - StudyState - Investigation state with personas
   - PersonaMemory - Persona-specific conversation history

2. **New Workflow Class** (new file: `workflows/study.py`):
   - Extends BaseWorkflow
   - Uses RoleOrchestrator for persona roles
   - Integrates ConversationMemory for multi-turn

3. **New CLI Command** (in `cli/main.py`):
   - `@app.command()`
   - `def study(...)` function
   - Follow existing command patterns

4. **Configuration** (in `core/config.py`):
   - Add WorkflowConfig section for "study" workflow
   - Define persona-specific parameters

5. **Registry Integration**:
   - Decorate class with `@WorkflowRegistry.register("study")`
   - Automatically discoverable

---

## Important Files Summary

| File | Purpose |
|------|---------|
| `base_workflow.py` | Abstract base class all workflows inherit from |
| `models.py` | All Pydantic data models (conversation, investigation state, etc.) |
| `role_orchestration.py` | Multi-role execution framework |
| `conversation.py` | Thread-based conversation persistence |
| `registry.py` | Workflow plugin system |
| `config.py` | Configuration file loading |
| `cli/main.py` | CLI command definitions (Typer app) |
| `providers/base_provider.py` | Provider abstraction |
| `providers/cli_provider.py` | CLI-based provider base class |
| `workflows/chat.py` | Simple single-model workflow |
| `workflows/consensus.py` | Multi-provider consensus workflow |
| `workflows/thinkdeep.py` | Investigation with hypothesis tracking |
| `workflows/argument/*.py` | Role-based dialectical reasoning |
| `workflows/ideate/*.py` | Creative brainstorming workflow |

---

## Summary

ModelChorus is a well-structured, extensible framework for multi-model AI orchestration:

✓ **Clear abstractions** - BaseWorkflow, ModelProvider, RoleOrchestrator
✓ **Plugin architecture** - WorkflowRegistry for easy extension
✓ **Conversation threading** - ConversationMemory for multi-turn interactions
✓ **Role orchestration** - Framework for multi-role workflows
✓ **CLI & Python API** - Both available via Typer
✓ **Configuration driven** - YAML/JSON .model-chorusrc files
✓ **Provider flexibility** - Supports 4+ AI providers
✓ **Error handling** - Fallback providers, availability checking

**Ready for STUDY workflow implementation** with personas, memory system, and routing using existing patterns.

