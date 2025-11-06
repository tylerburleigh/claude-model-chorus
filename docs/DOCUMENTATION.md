# claude-model-chorus Documentation

**Version:** 1.0.0
**Generated:** 2025-11-06 12:26:35

---

## 📊 Project Statistics

- **Total Files:** 51
- **Total Lines:** 18542
- **Total Classes:** 95
- **Total Functions:** 67
- **Avg Complexity:** 4.61
- **Max Complexity:** 36
- **High Complexity Functions:**
  - thinkdeep (36)
  - chat (19)
  - thinkdeep_status (18)
  - consensus (16)
  - validate_citation (14)



## 🏛️ Classes

### `BaseWorkflow`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:50`

**Description:**
> Abstract base class for all ModelChorus workflows.

All workflow implementations (thinkdeep, debug, consensus, etc.) must inherit
from this class and implement the run() method.

Attributes:
    name: Human-readable name of the workflow
    description: Brief description of what this workflow does
    config: Configuration dictionary for the workflow
    conversation_memory: Optional ConversationMemory instance for multi-turn conversations

**Methods:**
- `__init__()`
- `run()`
- `synthesize()`
- `get_result()`
- `validate_config()`
- `get_thread()`
- `add_message()`
- `resume_conversation()`
- `__repr__()`

---

### `CLIProvider`

**Language:** python
**Inherits from:** `ModelProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/cli_provider.py:24`

**Description:**
> Base class for CLI-based model providers.

This class extends ModelProvider to handle interactions with AI models
via CLI tools, providing common functionality for:
- Building CLI commands
- Executing commands asynchronously via subprocess
- Parsing CLI output
- Error handling and retries

Subclasses must implement:
- build_command(): Construct the CLI command for a request
- parse_response(): Parse CLI output into GenerationResponse

Attributes:
    cli_command: Base CLI command (e.g., "claude", "gemini")
    timeout: Command timeout in seconds
    retry_limit: Maximum number of retry attempts

**Methods:**
- `__init__()`
- `_load_conversation_context()`
- `build_command()`
- `parse_response()`
- `execute_command()`
- `generate()`
- `supports_vision()`
- `__repr__()`

---

### `ChatWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `modelchorus/src/modelchorus/workflows/chat.py:21`

**Description:**
> Simple single-model chat workflow with conversation continuity.

This workflow provides straightforward peer consultation with a single AI model,
supporting conversation threading for multi-turn interactions. Unlike multi-model
workflows like Consensus, Chat focuses on simplicity and conversational flow.

Key features:
- Single provider (not multi-model)
- Conversation threading via continuation_id
- Inherits conversation support from BaseWorkflow
- Automatic conversation history management
- Simple request/response pattern

The ChatWorkflow is ideal for:
- Quick second opinions from an AI model
- Iterative conversations and refinement
- Simple consultations without orchestration overhead
- Building conversational applications

Example:
    >>> from modelchorus.providers import ClaudeProvider
    >>> from modelchorus.workflows import ChatWorkflow
    >>> from modelchorus.core.conversation import ConversationMemory
    >>>
    >>> # Create provider and conversation memory
    >>> provider = ClaudeProvider()
    >>> memory = ConversationMemory()
    >>>
    >>> # Create workflow
    >>> workflow = ChatWorkflow(provider, conversation_memory=memory)
    >>>
    >>> # First message (creates new conversation)
    >>> result1 = await workflow.run("What is quantum computing?")
    >>> thread_id = result1.metadata.get('thread_id')
    >>> print(result1.synthesis)
    >>>
    >>> # Follow-up message (continues conversation)
    >>> result2 = await workflow.run(
    ...     "How does it differ from classical computing?",
    ...     continuation_id=thread_id
    ... )
    >>> print(result2.synthesis)
    >>>
    >>> # Check conversation history
    >>> thread = workflow.get_thread(thread_id)
    >>> print(f"Total messages: {len(thread.messages)}")

**Methods:**
- `__init__()`
- `run()`
- `_build_prompt_with_history()`
- `_get_conversation_length()`
- `get_provider()`
- `clear_conversation()`
- `get_message_count()`
- `validate_config()`
- `__repr__()`

---

### `Citation`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:823`

**Description:**
> Citation model for tracking sources in ARGUMENT workflow.

Tracks the source of information, its location, and confidence level
for evidence-based argumentation and research workflows.

Attributes:
    source: The source identifier (URL, file path, document ID, etc.)
    location: Specific location within source (page, line, section, timestamp)
    confidence: Confidence level in the citation accuracy (0.0-1.0)
    snippet: Optional text snippet from the source
    metadata: Additional citation metadata (author, date, context, etc.)

---

### `CitationMap`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:883`

**Description:**
> Maps claims to their supporting citations for evidence tracking.

Used in ARGUMENT workflow to maintain bidirectional mapping between
claims/arguments and their source citations, enabling verification
and citation analysis.

Attributes:
    claim_id: Unique identifier for the claim being supported
    claim_text: The actual claim or argument text
    citations: List of Citation objects supporting this claim
    strength: Overall strength of citation support (0.0-1.0)
    metadata: Additional mapping metadata (argument_type, verification_status, etc.)

---

### `CitationStyle`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:15`

**Description:**
> Supported citation formatting styles.

---

### `ClaudeProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/claude_provider.py:22`

**Description:**
> Provider for Anthropic's Claude models via the `claude` CLI tool.

This provider wraps the `claude` command-line interface to enable
text generation with Claude models (Opus, Sonnet, Haiku).

Supported features:
- Text generation with customizable prompts
- System prompts
- Temperature and token control
- Vision capabilities (model-dependent)
- Thinking mode (extended reasoning)

Example:
    >>> provider = ClaudeProvider()
    >>> request = GenerationRequest(
    ...     prompt="Explain quantum computing",
    ...     temperature=0.7,
    ...     max_tokens=1000
    ... )
    >>> response = await provider.generate(request)
    >>> print(response.content)

**Methods:**
- `__init__()`
- `_initialize_models()`
- `build_command()`
- `parse_response()`
- `supports_vision()`
- `supports_thinking()`

---

### `CodexProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/codex_provider.py:22`

**Description:**
> Provider for OpenAI's Codex models via the `codex` CLI tool.

This provider wraps the `codex` command-line interface to enable
code generation and completion with Codex models (GPT-4, GPT-3.5, etc.).

Supported features:
- Text and code generation
- System prompts
- Temperature and token control
- Function calling capabilities
- Vision support (model-dependent)

Example:
    >>> provider = CodexProvider()
    >>> request = GenerationRequest(
    ...     prompt="Write a Python function to sort a list",
    ...     temperature=0.5,
    ...     max_tokens=500
    ... )
    >>> response = await provider.generate(request)
    >>> print(response.content)

**Methods:**
- `__init__()`
- `_initialize_models()`
- `build_command()`
- `parse_response()`
- `supports_vision()`
- `supports_function_calling()`

---

### `ConfidenceLevel`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:16`

**Description:**
> Confidence level enum for investigation workflows.

Used in Thinkdeep workflow to track the investigator's confidence
in their hypothesis as evidence accumulates. Levels progress from
initial exploration through to complete certainty.

Values:
    EXPLORING: Just starting investigation, no clear hypothesis yet
    LOW: Early investigation with initial hypothesis forming
    MEDIUM: Some supporting evidence found
    HIGH: Strong evidence supporting hypothesis
    VERY_HIGH: Very strong evidence, high confidence
    ALMOST_CERTAIN: Near complete confidence, comprehensive evidence
    CERTAIN: 100% confidence, hypothesis validated beyond reasonable doubt

---

### `ConsensusConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:334`

**Description:**
> Configuration for consensus-building workflows.

Specifies how multiple models should be consulted and how their
responses should be synthesized into a consensus.

Attributes:
    mode: Consensus mode ("debate", "vote", "synthesis")
    stances: Optional list of stances to assign to models (e.g., ["for", "against", "neutral"])
    temperature: Temperature for model responses
    min_agreement: Minimum agreement threshold (0.0-1.0) for consensus
    synthesis_model: Optional model to use for final synthesis
    max_rounds: Maximum number of consensus rounds

---

### `ConsensusResult`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/consensus.py:45`

**Description:**
> Result from a consensus workflow execution.

---

### `ConsensusStrategy`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `modelchorus/src/modelchorus/workflows/consensus.py:24`

**Description:**
> Strategy for reaching consensus among multiple model responses.

---

### `ConsensusWorkflow`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/consensus.py:56`

**Description:**
> Workflow for coordinating multiple AI models to reach consensus.

This workflow allows querying multiple models simultaneously and applying
various strategies to synthesize or select from their responses.

Key features:
- Parallel execution across multiple providers
- Configurable consensus strategies
- Error handling and fallback mechanisms
- Response weighting and prioritization
- Timeout management per provider

Example:
    >>> from modelchorus.providers import ClaudeProvider, GeminiProvider
    >>> from modelchorus.workflows import ConsensusWorkflow
    >>>
    >>> # Create providers
    >>> claude = ClaudeProvider()
    >>> gemini = GeminiProvider()
    >>>
    >>> # Create workflow with providers
    >>> workflow = ConsensusWorkflow([claude, gemini])
    >>>
    >>> # Execute consensus query
    >>> request = GenerationRequest(prompt="Explain quantum computing")
    >>> result = await workflow.execute(request)
    >>>
    >>> # Access consensus or individual responses
    >>> print(result.consensus_response)
    >>> for provider, response in result.provider_results.items():
    ...     print(f"{provider}: {response.content}")

**Methods:**
- `__init__()`
- `add_provider()`
- `_execute_provider()`
- `execute()`
- `_apply_strategy()`
- `get_provider_count()`
- `get_providers()`
- `set_strategy()`
- `__repr__()`

---

### `Contradiction`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:42`

**Description:**
> Model for tracking contradictions between claims in ARGUMENT workflow.

Represents a detected contradiction between two claims, including
severity assessment, confidence in detection, and resolution suggestions.
Used to identify conflicts in evidence and maintain argument coherence.

Attributes:
    contradiction_id: Unique identifier for this contradiction
    claim_1_id: Identifier of the first conflicting claim
    claim_2_id: Identifier of the second conflicting claim
    claim_1_text: Full text of the first claim
    claim_2_text: Full text of the second claim
    severity: Severity level of the contradiction
    confidence: Confidence in contradiction detection (0.0-1.0)
    explanation: Detailed explanation of why claims contradict
    resolution_suggestion: Optional suggestion for resolving the contradiction
    metadata: Additional metadata (detection_method, timestamp, etc.)

**Methods:**
- `validate_confidence()`
- `validate_different_claims()`

---

### `ContradictionSeverity`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:17`

**Description:**
> Severity levels for contradictions between claims.

Classifies contradictions by their importance and impact on
argument validity. Higher severity indicates more significant
conflicts requiring immediate attention.

Values:
    MINOR: Slight inconsistency, may be due to different perspectives
           or temporal differences. Low impact on argument validity.
    MODERATE: Notable contradiction that should be investigated.
             May indicate measurement differences or scope variations.
    MAJOR: Significant contradiction that undermines argument coherence.
          Requires careful analysis and resolution.
    CRITICAL: Direct, irreconcilable contradiction that invalidates
             one or both claims. Immediate attention required.

---

### `ConversationMemory`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/conversation.py:36`

**Description:**
> Manages conversation threads with file-based persistence.

Provides thread-safe storage and retrieval of conversation history,
enabling multi-turn conversations across workflow executions.

Architecture:
    - Each thread stored as JSON file: ~/.modelchorus/conversations/{thread_id}.json
    - File locking prevents concurrent access corruption
    - TTL-based cleanup removes expired threads
    - Supports conversation chains via parent_thread_id

Attributes:
    conversations_dir: Directory where conversation files are stored
    ttl_hours: Time-to-live for conversation threads in hours
    max_messages: Maximum messages per thread before truncation

**Methods:**
- `__init__()`
- `create_thread()`
- `get_thread()`
- `get_thread_chain()`
- `add_message()`
- `get_messages()`
- `build_conversation_history()`
- `get_context_summary()`
- `complete_thread()`
- `archive_thread()`
- `cleanup_expired_threads()`
- `cleanup_archived_threads()`
- `_save_thread()`
- `_delete_thread()`

---

### `ConversationMessage`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:407`

**Description:**
> Single message in a conversation thread.

Based on Zen MCP's ConversationTurn but adapted for CLI-based architecture.
Tracks who said what, when, and with what context (files, models, workflow).

Attributes:
    role: Message role - 'user' or 'assistant'
    content: The actual message text/content
    timestamp: ISO format timestamp of when message was created
    files: Optional list of file paths referenced in this message
    workflow_name: Optional workflow that generated this message (for assistant messages)
    model_provider: Optional provider type used (cli, api, mcp)
    model_name: Optional specific model identifier (e.g., claude-3-opus, gpt-5)
    metadata: Additional message metadata (tokens, latency, cost, etc.)

---

### `ConversationState`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:763`

**Description:**
> Generic state container for workflow-specific conversation data.

Provides type-safe structure for storing arbitrary workflow state
while maintaining serializability for file-based persistence.
Includes versioning to support schema evolution.

Attributes:
    workflow_name: Workflow this state belongs to
    data: Arbitrary workflow-specific state data
    schema_version: State schema version for compatibility
    created_at: ISO timestamp of state creation
    updated_at: ISO timestamp of last state update

---

### `ConversationThread`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:479`

**Description:**
> Complete conversation context for a thread.

Adapted from Zen MCP's ThreadContext with enhancements for CLI orchestration:
- Provider-agnostic design supporting multiple CLI providers
- Explicit lifecycle management (status field)
- Support for conversation branching (future enhancement)
- Workflow-specific state persistence

Attributes:
    thread_id: UUID identifying this conversation thread
    parent_thread_id: Optional parent thread ID for conversation chains
    created_at: ISO timestamp of thread creation
    last_updated_at: ISO timestamp of last update
    workflow_name: Workflow that created this thread
    messages: All messages in chronological order
    state: Workflow-specific state data (persisted across turns)
    initial_context: Original request parameters
    status: Thread lifecycle status (active, completed, archived)
    branch_point: Optional message ID where branch occurred
    sibling_threads: Other thread IDs branched from same point

---

### `CursorAgentProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/cursor_agent_provider.py:22`

**Description:**
> Provider for Cursor's AI agent via the `cursor-agent` CLI tool.

This provider wraps the `cursor-agent` command-line interface to enable
code-focused AI interactions with Cursor's agent models.

Supported features:
- Code generation and completion
- Natural language to code conversion
- Code explanation and documentation
- System prompts for context
- Temperature control
- Working directory support for context-aware code generation

Example:
    >>> provider = CursorAgentProvider()
    >>> request = GenerationRequest(
    ...     prompt="Write a function to validate email addresses",
    ...     temperature=0.3,
    ...     max_tokens=500,
    ...     metadata={"working_directory": "/path/to/project"}
    ... )
    >>> response = await provider.generate(request)
    >>> print(response.content)

**Methods:**
- `__init__()`
- `_initialize_models()`
- `build_command()`
- `parse_response()`
- `supports_vision()`
- `supports_code_generation()`

---

### `ExampleProvider`

**Language:** python
**Inherits from:** `ModelProvider`
**Defined in:** `modelchorus/examples/provider_integration.py:18`

**Description:**
> Example provider implementation.

This is a placeholder showing how to implement a custom provider
for ModelChorus. In a real implementation, this would integrate
with an actual AI provider's API.

**Methods:**
- `__init__()`
- `generate()`
- `supports_vision()`

---

### `ExampleWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `modelchorus/examples/basic_workflow.py:14`

**Description:**
> Example workflow that demonstrates basic workflow structure.

This is a placeholder implementation showing the minimal structure
needed to create a working workflow.

**Methods:**
- `run()`

---

### `GeminiProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/gemini_provider.py:22`

**Description:**
> Provider for Google's Gemini models via the `gemini` CLI tool.

This provider wraps the `gemini` command-line interface to enable
text generation with Gemini models (Pro, Flash, Ultra).

Supported features:
- Text generation with customizable prompts
- System instructions
- Temperature and token control
- Vision capabilities (multimodal support)
- Function calling
- Thinking/reasoning mode

Example:
    >>> provider = GeminiProvider()
    >>> request = GenerationRequest(
    ...     prompt="Explain machine learning",
    ...     temperature=0.7,
    ...     max_tokens=1000
    ... )
    >>> response = await provider.generate(request)
    >>> print(response.content)

**Methods:**
- `__init__()`
- `_initialize_models()`
- `build_command()`
- `parse_response()`
- `supports_vision()`
- `supports_thinking()`

---

### `GenerationRequest`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:37`

**Description:**
> Request for text generation.

---

### `GenerationResponse`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:51`

**Description:**
> Response from text generation.

---

### `Hypothesis`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:589`

**Description:**
> Model for tracking hypotheses in investigation workflows.

Used in Thinkdeep workflow to track hypothesis evolution during
systematic investigation, including the hypothesis text, supporting
evidence, and current validation status.

Attributes:
    hypothesis: The hypothesis text/statement being investigated
    evidence: List of evidence items supporting or refuting this hypothesis
    status: Current validation status (active, disproven, validated)

---

### `InvestigationStep`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:634`

**Description:**
> Model for a single investigation step in Thinkdeep workflow.

Captures the details of one step in a systematic investigation,
including what was found, which files were examined, and the
current confidence level in the hypothesis.

Attributes:
    step_number: Sequential step number (1-indexed)
    findings: Key findings and insights discovered in this step
    files_checked: List of files examined during this step
    confidence: Current confidence level after this step

---

### `ModelCapability`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:15`

**Description:**
> Enumeration of model capabilities.

---

### `ModelConfig`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:26`

**Description:**
> Configuration for a model.

---

### `ModelProvider`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:61`

**Description:**
> Abstract base class for all model providers.

All provider implementations (Anthropic, OpenAI, Google, etc.) must inherit
from this class and implement the required methods.

Attributes:
    provider_name: Name of the provider (e.g., "anthropic", "openai")
    api_key: API key for authentication
    config: Provider-specific configuration

**Methods:**
- `__init__()`
- `generate()`
- `supports_vision()`
- `get_available_models()`
- `supports_capability()`
- `validate_api_key()`
- `set_model_list()`
- `__repr__()`

---

### `ModelResponse`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:286`

**Description:**
> Model for a response from a single model.

Represents the output from querying a specific model, used in
multi-model workflows to track individual model contributions.

Attributes:
    model: Identifier of the model that generated this response
    content: The response content/text
    role: Optional role this model played (e.g., "for", "against", "neutral")
    metadata: Additional response metadata (tokens, latency, etc.)

---

### `ModelSelection`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:184`

**Description:**
> Model for specifying model selection criteria.

Used to configure which models should be used for specific
workflow steps or roles.

Attributes:
    model_id: The model identifier (e.g., "gpt-4", "claude-3-opus")
    role: Optional role for this model (e.g., "analyzer", "synthesizer")
    config: Optional model-specific configuration

---

### `ProviderConfig`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/consensus.py:35`

**Description:**
> Configuration for a provider in the consensus workflow.

---

### `StateManager`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/state.py:32`

**Description:**
> Thread-safe state persistence manager for workflows.

Manages workflow execution state separately from conversation history.
Provides in-memory storage with optional file-based persistence.

Workflow state typically includes:
- Current step/phase information
- Intermediate results
- Configuration and settings
- Workflow-specific metadata

Attributes:
    state_dir: Directory for file-based state persistence
    enable_file_persistence: Whether to persist state to disk
    _state_store: Thread-safe in-memory state storage
    _lock: Lock for thread-safe operations

**Methods:**
- `__init__()`
- `set_state()`
- `get_state()`
- `get_state_object()`
- `update_state()`
- `delete_state()`
- `list_workflows()`
- `clear_all()`
- `serialize_state()`
- `deserialize_state()`
- `export_state()`
- `import_state()`
- `_save_to_file()`
- `_delete_file()`
- `load_from_disk()`
- `load_all_from_disk()`
- `sync_to_disk()`

---

### `TestArchitecturalDecisionScenarios`

**Language:** python
**Defined in:** `tests/test_thinkdeep_complex.py:26`

**Description:**
> Test suite for architectural decision making scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_architectural_decision_rest_vs_graphql()`
- `test_architectural_decision_database_selection()`

---

### `TestBatchSimilarity`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:252`

**Description:**
> Test batch similarity computation.

**Methods:**
- `test_compute_claim_similarity_batch_shape()`
- `test_compute_claim_similarity_batch_diagonal()`
- `test_compute_claim_similarity_batch_symmetric()`
- `test_compute_claim_similarity_batch_range()`

---

### `TestBugInvestigationScenarios`

**Language:** python
**Defined in:** `tests/test_thinkdeep_complex.py:250`

**Description:**
> Test suite for systematic bug investigation scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_bug_investigation_api_slowness()`
- `test_bug_investigation_intermittent_crash()`

---

### `TestCLIProvidersImplementInterface`

**Language:** python
**Defined in:** `tests/test_providers/test_cli_interface.py:27`

**Description:**
> Test that all CLI providers implement the ModelProvider interface.

**Methods:**
- `providers()`
- `test_all_providers_inherit_from_base()`
- `test_all_providers_inherit_from_cli_provider()`
- `test_all_providers_have_generate_method()`
- `test_all_providers_have_build_command_method()`
- `test_all_providers_have_parse_response_method()`
- `test_all_providers_have_supports_vision_method()`
- `test_all_providers_have_get_available_models_method()`
- `test_all_providers_return_models()`
- `test_all_providers_have_provider_name()`
- `test_all_providers_have_cli_command()`
- `test_build_command_returns_list()`
- `test_build_command_includes_cli_command()`
- `test_supports_vision_returns_bool()`

---

### `TestChatErrorHandling`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:180`

**Description:**
> Test error handling in chat workflow.

**Methods:**
- `test_invalid_continuation_id()`
- `test_empty_prompt()`
- `test_very_long_conversation()`

---

### `TestChatThreadManagement`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:234`

**Description:**
> Test conversation thread management.

**Methods:**
- `test_multiple_concurrent_threads()`
- `test_thread_retrieval()`

---

### `TestChatWorkflowInitialization`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:52`

**Description:**
> Test ChatWorkflow initialization.

**Methods:**
- `test_initialization_with_provider()`
- `test_initialization_without_memory()`
- `test_initialization_without_provider_raises_error()`
- `test_validate_config()`
- `test_get_provider()`

---

### `TestCitationIntegration`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:308`

**Description:**
> Test integration with Citation model.

**Methods:**
- `test_add_similarity_to_citation_with_snippet()`
- `test_add_similarity_to_citation_without_snippet()`
- `test_add_similarity_preserves_existing_metadata()`

---

### `TestClaimSimilarity`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:123`

**Description:**
> Test claim-to-claim similarity computation.

**Methods:**
- `test_compute_claim_similarity_identical()`
- `test_compute_claim_similarity_similar()`
- `test_compute_claim_similarity_different()`
- `test_compute_claim_similarity_symmetric()`

---

### `TestClaudeProvider`

**Language:** python
**Defined in:** `modelchorus/tests/test_claude_provider.py:13`

**Description:**
> Test suite for ClaudeProvider.

**Methods:**
- `test_initialization()`
- `test_build_command_basic()`
- `test_build_command_with_model()`
- `test_build_command_without_system_prompt()`
- `test_parse_response_success()`
- `test_parse_response_failure()`
- `test_parse_response_invalid_json()`
- `test_generate_success()`
- `test_generate_with_retry()`
- `test_generate_all_retries_fail()`
- `test_supports_vision()`
- `test_supports_thinking()`

---

### `TestClusterRepresentative`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:665`

**Description:**
> Test cluster representative selection.

**Methods:**
- `test_get_cluster_representative_basic()`
- `test_get_cluster_representative_single_item()`
- `test_get_cluster_representative_empty_cluster()`

---

### `TestClusterStatistics`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:717`

**Description:**
> Test cluster statistics computation.

**Methods:**
- `test_compute_cluster_statistics_basic()`
- `test_compute_cluster_statistics_empty()`
- `test_compute_cluster_statistics_single_item_clusters()`

---

### `TestClusteringIntegration`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:768`

**Description:**
> Test end-to-end clustering workflows.

**Methods:**
- `test_kmeans_to_statistics_pipeline()`
- `test_hierarchical_to_statistics_pipeline()`

---

### `TestCodexProvider`

**Language:** python
**Defined in:** `modelchorus/tests/test_codex_provider.py:13`

**Description:**
> Test suite for CodexProvider.

**Methods:**
- `test_initialization()`
- `test_build_command_basic()`
- `test_build_command_with_model()`
- `test_build_command_with_images()`
- `test_parse_response_success()`
- `test_parse_response_failure()`
- `test_parse_response_invalid_jsonl()`
- `test_parse_response_missing_agent_message()`
- `test_generate_success()`
- `test_generate_with_retry()`
- `test_generate_all_retries_fail()`
- `test_supports_vision()`
- `test_supports_function_calling()`

---

### `TestComplexMultiStepReasoning`

**Language:** python
**Defined in:** `tests/test_thinkdeep_complex.py:597`

**Description:**
> Test suite for complex multi-step reasoning scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_long_investigation_with_hypothesis_pivots()`
- `test_investigation_with_multiple_evidence_types()`

---

### `TestConcurrentConversationHandling`

**Language:** python
**Defined in:** `tests/test_concurrent_conversations.py:26`

**Description:**
> Test suite for concurrent conversation handling.

Validates that ConversationMemory and workflows can handle high concurrency
scenarios with proper thread isolation and performance.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_100_concurrent_chat_conversations()`
- `test_concurrent_multi_turn_conversations()`
- `test_mixed_workflow_concurrent_execution()`
- `test_performance_scalability()`

---

### `TestConfidenceLevel`

**Language:** python
**Defined in:** `modelchorus/tests/test_thinkdeep_models.py:25`

**Description:**
> Test suite for ConfidenceLevel enum.

**Methods:**
- `test_confidence_level_values()`
- `test_confidence_level_count()`
- `test_confidence_level_progression()`
- `test_confidence_level_string_representation()`

---

### `TestConfidenceProgression`

**Language:** python
**Defined in:** `tests/test_thinkdeep_workflow.py:1135`

**Description:**
> Test suite for confidence level progression in ThinkDeepWorkflow.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_initial_confidence_level()`
- `test_update_confidence_level()`
- `test_get_confidence_level()`
- `test_confidence_progression_across_steps()`
- `test_confidence_tracked_in_metadata()`
- `test_invalid_confidence_level_rejected()`
- `test_confidence_complete_progression()`
- `test_investigation_completion_criteria()`
- `test_investigation_summary_includes_confidence()`
- `test_confidence_cannot_decrease()`
- `test_confidence_persistence_across_turns()`

---

### `TestConsensusThinkDeepChatChaining`

**Language:** python
**Defined in:** `tests/test_workflow_integration_chaining.py:22`

**Description:**
> Test suite for consensus → thinkdeep → chat workflow integration.

This pattern demonstrates using multiple orchestration strategies in sequence:
1. Consensus: Gather multi-model opinions on a decision
2. ThinkDeep: Investigate specific concerns raised
3. Chat: Refine understanding and get practical recommendations

**Methods:**
- `mock_provider()`
- `mock_provider_2()`
- `conversation_memory()`
- `test_consensus_to_thinkdeep_to_chat_workflow()`
- `test_workflow_chain_context_isolation()`
- `test_consensus_without_continuation_support()`

---

### `TestConsensusWorkflow`

**Language:** python
**Defined in:** `modelchorus/tests/test_consensus_workflow.py:12`

**Description:**
> Test suite for ConsensusWorkflow.

**Methods:**
- `test_initialization()`
- `test_initialization_multiple_providers()`
- `test_initialization_with_strategy()`
- `test_execute_all_responses_strategy()`
- `test_execute_first_valid_strategy()`
- `test_execute_with_parameters()`
- `test_execute_all_providers_fail()`

---

### `TestConversationContinuation`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:140`

**Description:**
> Test conversation continuation functionality.

**Methods:**
- `test_continuation_uses_same_thread_id()`
- `test_continuation_includes_history()`
- `test_continuation_tracks_message_count()`
- `test_multiple_continuations()`

---

### `TestConversationInitiation`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:88`

**Description:**
> Test conversation creation and initiation.

**Methods:**
- `test_new_conversation_creates_thread_id()`
- `test_new_conversation_has_response()`
- `test_new_conversation_without_memory()`
- `test_provider_generate_called_with_correct_params()`

---

### `TestConversationMemory`

**Language:** python
**Defined in:** `modelchorus/tests/test_conversation.py:22`

**Description:**
> Test suite for ConversationMemory class.

**Methods:**
- `test_create_thread_generates_valid_uuid()`
- `test_create_thread_unique_ids()`
- `test_create_thread_with_initial_context()`
- `test_create_thread_persists_to_file()`
- `test_create_thread_with_parent()`
- `test_add_message_to_thread()`
- `test_add_multiple_messages()`
- `test_add_message_with_metadata()`
- `test_get_messages_returns_chronological_order()`
- `test_message_persistence_across_instances()`
- `test_thread_context_window_management()`
- `test_get_thread_context_includes_state()`

---

### `TestConversationTracking`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:303`

**Description:**
> Test conversation history and tracking.

**Methods:**
- `test_get_thread_retrieves_conversation()`
- `test_get_thread_returns_none_for_invalid_id()`
- `test_get_thread_without_memory_returns_none()`
- `test_get_message_count()`

---

### `TestCosineSimilarity`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:81`

**Description:**
> Test cosine similarity computation.

**Methods:**
- `test_cosine_similarity_identical()`
- `test_cosine_similarity_range()`
- `test_cosine_similarity_similar_text()`
- `test_cosine_similarity_different_text()`

---

### `TestDuplicateDetection`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:358`

**Description:**
> Test duplicate claim detection.

**Methods:**
- `test_find_duplicate_claims_basic()`
- `test_find_duplicate_claims_high_threshold()`
- `test_find_duplicate_claims_no_duplicates()`
- `test_find_duplicate_claims_empty_list()`
- `test_find_duplicate_claims_single_item()`

---

### `TestEdgeCases`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:469`

**Description:**
> Test edge cases and error handling.

**Methods:**
- `test_empty_claim_text()`
- `test_very_long_claim()`
- `test_special_characters()`
- `test_unicode_characters()`

---

### `TestEmbeddingComputation`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:27`

**Description:**
> Test embedding computation and caching.

**Methods:**
- `test_compute_embedding_returns_array()`
- `test_compute_embedding_normalization()`
- `test_compute_embedding_caching()`
- `test_compute_embedding_case_insensitive()`
- `test_compute_embedding_empty_string()`

---

### `TestEndToEndIntegration`

**Language:** python
**Defined in:** `tests/test_thinkdeep_workflow.py:1613`

**Description:**
> End-to-end integration tests for complete investigation scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_five_step_investigation_with_hypothesis_evolution()`
- `test_complete_investigation_workflow()`

---

### `TestErrorHandling`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:348`

**Description:**
> Test error handling in ChatWorkflow.

**Methods:**
- `test_provider_error_handled()`
- `test_get_result_returns_last_result()`

---

### `TestExpertProviderIntegration`

**Language:** python
**Defined in:** `tests/test_thinkdeep_expert_validation.py:27`

**Description:**
> Test suite for expert provider integration in ThinkDeepWorkflow.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_enabled_with_expert_provider()`
- `test_expert_validation_disabled_without_expert_provider()`
- `test_expert_validation_explicit_disable_via_config()`
- `test_expert_validation_explicit_enable_via_config()`

---

### `TestExpertValidationErrorHandling`

**Language:** python
**Defined in:** `tests/test_thinkdeep_expert_validation.py:428`

**Description:**
> Test suite for error handling in expert validation.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_failure_does_not_crash_investigation()`
- `test_expert_validation_timeout_handling()`
- `test_expert_validation_with_empty_response()`

---

### `TestExpertValidationResultHandling`

**Language:** python
**Defined in:** `tests/test_thinkdeep_expert_validation.py:314`

**Description:**
> Test suite for handling expert validation results.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_result_included_in_metadata()`
- `test_expert_validation_conversation_history_updated()`

---

### `TestExpertValidationTriggering`

**Language:** python
**Defined in:** `tests/test_thinkdeep_expert_validation.py:116`

**Description:**
> Test suite for expert validation triggering logic.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_triggered_at_medium_confidence()`
- `test_expert_validation_not_triggered_at_exploring_confidence()`
- `test_expert_validation_triggered_at_high_confidence()`
- `test_expert_validation_not_triggered_when_disabled()`

---

### `TestExpertValidationWithHypotheses`

**Language:** python
**Defined in:** `tests/test_thinkdeep_expert_validation.py:579`

**Description:**
> Test suite for expert validation interaction with hypotheses.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_validates_hypothesis()`
- `test_expert_validation_with_multiple_hypotheses()`

---

### `TestFileContext`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:222`

**Description:**
> Test file context handling.

**Methods:**
- `test_file_context_included_in_prompt()`
- `test_multiple_files_included()`
- `test_file_not_found_handled_gracefully()`
- `test_file_references_stored_in_conversation()`

---

### `TestFindSimilarClaims`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:163`

**Description:**
> Test finding similar claims in citation maps.

**Methods:**
- `sample_citation_maps()`
- `test_find_similar_claims_basic()`
- `test_find_similar_claims_threshold()`
- `test_find_similar_claims_top_k()`
- `test_find_similar_claims_sorted()`
- `test_find_similar_claims_empty_list()`

---

### `TestGeminiIntegration`

**Language:** python
**Defined in:** `modelchorus/tests/test_gemini_integration.py:13`

**Description:**
> Integration tests for Gemini provider.

**Methods:**
- `provider()`
- `simple_request()`
- `test_gemini_cli_available()`
- `test_build_command_basic()`
- `test_build_command_with_model()`
- `test_generate_simple_query()`
- `test_parse_response_format()`
- `test_parse_response_error_handling()`
- `test_supports_vision()`
- `test_supports_thinking()`

---

### `TestHierarchicalClustering`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:597`

**Description:**
> Test hierarchical clustering functionality.

**Methods:**
- `sample_maps()`
- `test_cluster_claims_hierarchical_basic()`
- `test_cluster_claims_hierarchical_empty_list()`
- `test_cluster_claims_hierarchical_linkage_methods()`
- `test_cluster_claims_hierarchical_too_many_clusters()`

---

### `TestHypothesis`

**Language:** python
**Defined in:** `modelchorus/tests/test_thinkdeep_models.py:69`

**Description:**
> Test suite for Hypothesis model.

**Methods:**
- `test_hypothesis_creation()`
- `test_hypothesis_default_values()`
- `test_hypothesis_all_statuses()`
- `test_hypothesis_invalid_status()`
- `test_hypothesis_empty_hypothesis_text()`
- `test_hypothesis_with_multiple_evidence()`
- `test_hypothesis_serialization()`
- `test_hypothesis_json_serialization()`
- `test_hypothesis_from_dict()`

---

### `TestHypothesisEvolution`

**Language:** python
**Defined in:** `tests/test_thinkdeep_workflow.py:598`

**Description:**
> Test suite for hypothesis evolution in ThinkDeepWorkflow.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_add_hypothesis_to_investigation()`
- `test_update_hypothesis_with_evidence()`
- `test_validate_hypothesis()`
- `test_disprove_hypothesis()`
- `test_multiple_hypothesis_evolution()`
- `test_get_active_hypotheses()`
- `test_get_all_hypotheses()`
- `test_hypothesis_persistence_across_turns()`
- `test_hypothesis_update_with_status_change()`
- `test_hypothesis_not_found_handling()`
- `test_hypothesis_metadata_tracking()`

---

### `TestIntegration`

**Language:** python
**Defined in:** `modelchorus/tests/test_integration.py:17`

**Description:**
> Integration test suite.

**Methods:**
- `test_end_to_end_consensus()`
- `test_provider_initialization_and_generation()`
- `test_error_handling_across_workflow()`
- `test_multiple_strategy_comparison()`
- `test_concurrent_provider_execution()`

---

### `TestInvestigationStep`

**Language:** python
**Defined in:** `modelchorus/tests/test_thinkdeep_models.py:180`

**Description:**
> Test suite for InvestigationStep model.

**Methods:**
- `test_investigation_step_creation()`
- `test_investigation_step_default_files()`
- `test_investigation_step_multiple_files()`
- `test_investigation_step_number_validation()`
- `test_investigation_step_empty_findings()`
- `test_investigation_step_empty_confidence()`
- `test_investigation_step_serialization()`
- `test_investigation_step_json_roundtrip()`

---

### `TestInvestigationStepExecution`

**Language:** python
**Defined in:** `tests/test_thinkdeep_workflow.py:29`

**Description:**
> Test suite for investigation step execution in ThinkDeepWorkflow.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_single_investigation_step_execution()`
- `test_investigation_step_creates_investigation_step_object()`
- `test_multi_step_investigation_progression()`
- `test_investigation_step_tracks_files_checked()`
- `test_investigation_step_with_no_files()`
- `test_investigation_step_confidence_tracking()`
- `test_investigation_step_with_hypothesis_integration()`
- `test_investigation_step_findings_extraction()`
- `test_investigation_step_with_expert_validation()`
- `test_investigation_step_metadata_completeness()`
- `test_investigation_step_error_handling()`
- `test_investigation_without_conversation_memory()`

---

### `TestKMeansClustering`

**Language:** python
**Defined in:** `modelchorus/tests/test_semantic_similarity.py:500`

**Description:**
> Test K-means clustering functionality.

**Methods:**
- `diverse_citation_maps()`
- `test_cluster_claims_kmeans_basic()`
- `test_cluster_claims_kmeans_empty_list()`
- `test_cluster_claims_kmeans_single_cluster()`
- `test_cluster_claims_kmeans_too_many_clusters()`
- `test_cluster_claims_kmeans_reproducibility()`

---

### `TestLongConversations`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:288`

**Description:**
> Test handling of long multi-turn conversations (20+ turns).

**Methods:**
- `test_20_turn_conversation()`
- `test_25_turn_conversation_with_context_retention()`
- `test_conversation_length_tracking()`
- `test_long_conversation_with_file_references()`
- `test_conversation_stability_under_load()`

---

### `TestMemoryManagement`

**Language:** python
**Defined in:** `tests/test_memory_management.py:23`

**Description:**
> Test suite for memory management with long conversations.

Validates that ConversationMemory properly manages memory when dealing
with long-running conversations with many messages.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_long_conversation_memory_stability()`
- `test_multiple_long_conversations_memory_isolation()`
- `test_memory_efficiency_with_large_messages()`
- `test_concurrent_long_conversations_memory()`

---

### `TestModelIntegration`

**Language:** python
**Defined in:** `modelchorus/tests/test_thinkdeep_models.py:592`

**Description:**
> Test integration scenarios using multiple models together.

**Methods:**
- `test_hypothesis_lifecycle()`
- `test_investigation_progression()`
- `test_multiple_hypothesis_tracking()`

---

### `TestMultiProviderChat`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:71`

**Description:**
> Test chat functionality across multiple providers.

**Methods:**
- `test_basic_conversation()`
- `test_multi_turn_conversation()`
- `test_conversation_with_file_context()`
- `test_conversation_persistence()`

---

### `TestStateManager`

**Language:** python
**Defined in:** `modelchorus/tests/test_state.py:23`

**Description:**
> Test suite for StateManager class.

**Methods:**
- `test_set_and_get_state()`
- `test_get_state_object_with_metadata()`
- `test_update_state_existing()`
- `test_update_state_nonexistent()`
- `test_delete_state()`
- `test_list_workflows()`
- `test_clear_all()`
- `test_concurrent_set_state()`
- `test_concurrent_read_write()`
- `test_state_isolation()`
- `test_serialize_state()`
- `test_deserialize_state()`
- `test_roundtrip_serialization()`
- `test_export_state()`
- `test_import_state()`
- `test_deserialize_invalid_json()`
- `test_file_persistence_on_set()`
- `test_load_from_disk()`
- `test_load_all_from_disk()`
- `test_sync_to_disk()`
- `test_delete_removes_file()`
- `test_persistence_disabled_no_files()`
- `test_get_nonexistent_workflow()`
- `test_delete_nonexistent_workflow()`
- `test_serialize_nonexistent()`
- `test_import_malformed_json()`
- `test_state_timestamps_update()`
- `test_schema_version_preservation()`
- `test_get_default_state_manager()`

---

### `TestStateManagerExportImportRoundtrip`

**Language:** python
**Defined in:** `modelchorus/tests/test_state.py:609`

**Description:**
> Test complete export/import workflow.

**Methods:**
- `test_export_import_roundtrip()`

---

### `TestStateManagerFileRecovery`

**Language:** python
**Defined in:** `modelchorus/tests/test_state.py:647`

**Description:**
> Test state recovery after simulated process restart.

**Methods:**
- `test_process_restart_recovery()`

---

### `TestThinkDeepState`

**Language:** python
**Defined in:** `modelchorus/tests/test_thinkdeep_models.py:314`

**Description:**
> Test suite for ThinkDeepState model.

**Methods:**
- `test_thinkdeep_state_creation()`
- `test_thinkdeep_state_default_values()`
- `test_thinkdeep_state_multiple_hypotheses()`
- `test_thinkdeep_state_multiple_steps()`
- `test_thinkdeep_state_file_accumulation()`
- `test_thinkdeep_state_complex_scenario()`
- `test_thinkdeep_state_serialization()`
- `test_thinkdeep_state_json_roundtrip()`
- `test_thinkdeep_state_nested_validation()`

---

### `ThinkDeepState`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:688`

**Description:**
> State model for Thinkdeep workflow multi-turn conversations.

Maintains the complete investigation state across conversation turns,
tracking hypothesis evolution, investigation steps, confidence progression,
and files examined.

Attributes:
    hypotheses: List of all hypotheses tracked during investigation
    steps: List of all investigation steps completed
    current_confidence: Current overall confidence level
    relevant_files: All files identified as relevant to the investigation

---

### `ThinkDeepWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `modelchorus/src/modelchorus/workflows/thinkdeep.py:27`

**Description:**
> Extended reasoning workflow with systematic investigation and hypothesis tracking.

This workflow provides multi-step investigation capabilities where hypotheses
are formed, tested, and refined across conversation turns. It maintains state
including hypothesis evolution, investigation steps, confidence levels, and
relevant files examined.

Key features:
- Single provider with extended reasoning
- Hypothesis tracking and evolution
- Investigation step progression
- Confidence level tracking
- File examination history
- State persistence across turns via conversation threading

The ThinkDeepWorkflow is ideal for:
- Complex problem analysis requiring systematic investigation
- Debugging scenarios with hypothesis testing
- Architecture decisions with evidence-based reasoning
- Security analysis with confidence tracking
- Any task requiring methodical, step-by-step investigation

Example:
    >>> from modelchorus.providers import ClaudeProvider
    >>> from modelchorus.workflows import ThinkDeepWorkflow
    >>> from modelchorus.core.conversation import ConversationMemory
    >>>
    >>> # Create provider and conversation memory
    >>> provider = ClaudeProvider()
    >>> memory = ConversationMemory()
    >>>
    >>> # Create workflow
    >>> workflow = ThinkDeepWorkflow(provider, conversation_memory=memory)
    >>>
    >>> # First step (creates new investigation)
    >>> result1 = await workflow.run(
    ...     "Why is authentication failing intermittently?",
    ...     files=["src/auth.py", "tests/test_auth.py"]
    ... )
    >>> thread_id = result1.metadata.get('thread_id')
    >>>
    >>> # Follow-up investigation (continues thread with state)
    >>> result2 = await workflow.run(
    ...     "Check if it's related to async/await patterns",
    ...     continuation_id=thread_id,
    ...     files=["src/services/user.py"]
    ... )
    >>>
    >>> # Check investigation state
    >>> state = workflow.get_investigation_state(thread_id)
    >>> print(f"Hypotheses: {len(state.hypotheses)}")
    >>> print(f"Confidence: {state.current_confidence}")

**Methods:**
- `__init__()`
- `run()`
- `_get_or_create_state()`
- `_save_state()`
- `_build_investigation_prompt()`
- `_extract_findings()`
- `_perform_expert_validation()`
- `_build_expert_validation_prompt()`
- `get_investigation_state()`
- `add_hypothesis()`
- `update_hypothesis()`
- `validate_hypothesis()`
- `disprove_hypothesis()`
- `get_active_hypotheses()`
- `get_all_hypotheses()`
- `update_confidence()`
- `get_confidence()`
- `is_investigation_complete()`
- `get_investigation_summary()`
- `get_provider()`
- `validate_config()`
- `__repr__()`

---

### `WorkflowRegistry`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/registry.py:13`

**Description:**
> Registry for workflow implementations.

Provides a plugin system for registering and retrieving workflow classes
dynamically. Workflows can be registered using the @register decorator or
programmatically via register_workflow().

Example:
    ```python
    @WorkflowRegistry.register("thinkdeep")
    class ThinkDeepWorkflow(BaseWorkflow):
        async def run(self, prompt: str, **kwargs):
            # Implementation
            pass

    # Later, retrieve the workflow
    workflow_class = WorkflowRegistry.get("thinkdeep")
    workflow = workflow_class("My Workflow", "Description")
    ```

**Methods:**
- `register()`
- `register_workflow()`
- `get()`
- `list_workflows()`
- `is_registered()`
- `unregister()`
- `clear()`

---

### `WorkflowRequest`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:43`

**Description:**
> Request model for workflow execution.

Encapsulates all the information needed to execute a workflow,
including the prompt, model specifications, and configuration.

Attributes:
    prompt: The main input prompt/task for the workflow
    models: List of model identifiers to use (e.g., ["gpt-4", "claude-3"])
    config: Optional workflow-specific configuration dictionary
    system_prompt: Optional system prompt for model context
    temperature: Optional temperature setting (0.0-1.0)
    max_tokens: Optional maximum tokens for generation
    images: Optional list of image paths or URLs for vision-capable models
    metadata: Additional metadata for the workflow execution

---

### `WorkflowResponse`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:117`

**Description:**
> Response model for workflow execution.

Contains the results of a workflow execution along with metadata
about the execution process.

Attributes:
    result: The main result/output from the workflow
    success: Whether the workflow executed successfully
    workflow_name: Name of the workflow that was executed
    steps: Number of steps executed in the workflow
    models_used: List of models that were used
    error: Error message if the workflow failed
    metadata: Additional metadata about the execution

---

### `WorkflowResult`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:29`

**Description:**
> Result of a workflow execution.

**Methods:**
- `add_step()`

---

### `WorkflowStep`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:18`

**Description:**
> Represents a single step in a workflow execution.

---

### `WorkflowStep`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:224`

**Description:**
> Model for a single workflow execution step.

Represents one step in a multi-step workflow, capturing what
was done, which model was used, and the results.

Attributes:
    step_number: Sequential step number (1-indexed)
    description: Human-readable description of this step
    model: Model that executed this step
    prompt: The prompt used for this step
    response: The response from this step
    metadata: Additional step metadata

---


## ⚡ Functions

### `_compute_embedding_cached(text_hash, text, model_name) -> np.ndarray`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:57`
**Complexity:** 1

**Decorators:** `@lru_cache(maxsize=1000)`

**Description:**
> Internal cached embedding computation.

Uses text hash as cache key to enable LRU caching while avoiding
unhashable numpy arrays as cache keys.

Args:
    text_hash: SHA256 hash of the normalized text (for cache key)
    text: Actual text to compute embedding for
    model_name: Model to use for embedding

Returns:
    Embedding vector as numpy array

**Parameters:**
- `text_hash`: str
- `text`: str
- `model_name`: str

---

### `_format_apa(citation) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:58`
**Complexity:** 6

**Description:**
> Format citation in APA style.

APA format: Author(s). (Year). Title. Source.

Args:
    citation: The Citation object to format

Returns:
    APA-formatted citation string

**Parameters:**
- `citation`: 'Citation'

---

### `_format_chicago(citation) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:138`
**Complexity:** 6

**Description:**
> Format citation in Chicago style.

Chicago format: Author(s). "Title." Source (Year): Location.

Args:
    citation: The Citation object to format

Returns:
    Chicago-formatted citation string

**Parameters:**
- `citation`: 'Citation'

---

### `_format_mla(citation) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:98`
**Complexity:** 6

**Description:**
> Format citation in MLA style.

MLA format: Author(s). "Title." Source, Year. Location.

Args:
    citation: The Citation object to format

Returns:
    MLA-formatted citation string

**Parameters:**
- `citation`: 'Citation'

---

### `_get_model(model_name) -> SentenceTransformer`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:24`
**Complexity:** 2

**Description:**
> Get or initialize the sentence transformer model.

Uses lazy loading to avoid loading the model until needed.
Model is cached globally to avoid repeated loading.

Args:
    model_name: Name of the sentence transformer model to use

Returns:
    SentenceTransformer instance

**Parameters:**
- `model_name`: str

---

### `_import_citation_map() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:212`
**Complexity:** 2

**Description:**
> Import CitationMap model.

---

### `_import_semantic_functions() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:197`
**Complexity:** 2

**Description:**
> Import semantic similarity functions from workflows.argument.semantic.

---

### `_normalize_text(text) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:43`
**Complexity:** 1

**Description:**
> Normalize text for consistent embedding computation.

Args:
    text: Input text to normalize

Returns:
    Normalized text (lowercase, stripped whitespace)

**Parameters:**
- `text`: str

---

### `add_similarity_to_citation(citation, reference_claim, model_name) -> Citation`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:278`
**Complexity:** 2

**Description:**
> Add semantic similarity score to a citation's metadata.

Computes similarity between the citation's snippet (if available)
and a reference claim, storing the result in citation.metadata.

Args:
    citation: Citation object to enhance
    reference_claim: Claim text to compare against
    model_name: Sentence transformer model to use

Returns:
    Enhanced Citation object with similarity_score in metadata

Example:
    >>> citation = Citation(
    ...     source="paper.pdf",
    ...     confidence=0.9,
    ...     snippet="ML improves accuracy by 23%"
    ... )
    >>> enhanced = add_similarity_to_citation(
    ...     citation,
    ...     "Machine learning enhances precision"
    ... )
    >>> print(enhanced.metadata["similarity_score"])
    0.847

**Parameters:**
- `citation`: Citation
- `reference_claim`: str
- `model_name`: str

---

### `assess_contradiction_severity(semantic_similarity, has_polarity_opposition, polarity_confidence) -> ContradictionSeverity`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:310`
**Complexity:** 7

**Description:**
> Assess the severity of a contradiction based on multiple factors.

Combines semantic similarity and polarity opposition to classify
contradiction severity.

Args:
    semantic_similarity: Cosine similarity between claims (0.0-1.0)
    has_polarity_opposition: Whether claims have opposing polarity
    polarity_confidence: Confidence in polarity detection (0.0-1.0)

Returns:
    ContradictionSeverity enum value

Severity Rules:
    - High similarity (>0.7) + strong polarity opposition = CRITICAL
    - High similarity (>0.7) + weak polarity opposition = MAJOR
    - Moderate similarity (0.5-0.7) + polarity opposition = MODERATE
    - Low similarity (<0.5) + polarity opposition = MINOR

Example:
    >>> severity = assess_contradiction_severity(
    ...     semantic_similarity=0.85,
    ...     has_polarity_opposition=True,
    ...     polarity_confidence=0.8
    ... )
    >>> print(severity)
    ContradictionSeverity.CRITICAL

**Parameters:**
- `semantic_similarity`: float
- `has_polarity_opposition`: bool
- `polarity_confidence`: float

---

### `async basic_chat_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/chat_example.py:19`
**Complexity:** 2

**Description:**
> Basic chat conversation without continuation.

---

### `async basic_investigation_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:27`
**Complexity:** 2

**Description:**
> Basic single-step investigation.

---

### `calculate_citation_confidence(citation) -> Dict[str, Any]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:279`
⚠️ **Complexity:** 11 (High)

**Description:**
> Calculate a detailed confidence score for a citation's reliability.

Evaluates multiple factors:
- Base confidence score (from citation.confidence)
- Metadata completeness (author, year, title presence)
- Source quality (URL vs file, academic domains)
- Location specificity (page numbers, sections)

Args:
    citation: The Citation object to score

Returns:
    Dictionary with:
    - overall_confidence: Final confidence score (0.0-1.0)
    - base_confidence: Original confidence value
    - metadata_score: Completeness score for metadata (0.0-1.0)
    - source_quality_score: Quality score for source type (0.0-1.0)
    - location_score: Specificity score for location (0.0-1.0)
    - factors: Detailed breakdown of scoring factors

Example:
    >>> scores = calculate_citation_confidence(citation)
    >>> print(f"Overall confidence: {scores['overall_confidence']:.2f}")
    >>> print(f"Metadata completeness: {scores['metadata_score']:.2f}")

**Parameters:**
- `citation`: 'Citation'

---

### `calculate_citation_map_confidence(citation_map) -> Dict[str, Any]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:367`
**Complexity:** 2

**Description:**
> Calculate aggregate confidence scores for a CitationMap.

Evaluates the overall quality of citations supporting a claim.

Args:
    citation_map: The CitationMap object to score

Returns:
    Dictionary with:
    - overall_confidence: Aggregate confidence for the claim (0.0-1.0)
    - citation_count: Number of citations
    - average_citation_confidence: Mean confidence across citations
    - min_confidence: Lowest confidence citation
    - max_confidence: Highest confidence citation
    - strength: Original strength value from CitationMap
    - individual_scores: List of confidence scores per citation

Example:
    >>> scores = calculate_citation_map_confidence(citation_map)
    >>> print(f"Claim supported by {scores['citation_count']} citations")
    >>> print(f"Overall confidence: {scores['overall_confidence']:.2f}")

**Parameters:**
- `citation_map`: 'CitationMap'

---

### `chat(prompt, provider, continuation_id, files, system, temperature, max_tokens, output, verbose) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:55`
⚠️ **Complexity:** 19 (High)

**Decorators:** `@app.command()`

**Description:**
> Chat with a single AI model with conversation continuity.

Example:
    # Start new conversation
    modelchorus chat "What is quantum computing?" -p claude

    # Continue conversation
    modelchorus chat "Give me an example" --continue thread-id-123

    # Include files
    modelchorus chat "Review this code" -f src/main.py -f tests/test_main.py

**Parameters:**
- `prompt`: str
- `provider`: str
- `continuation_id`: Optional[str]
- `files`: Optional[List[str]]
- `system`: Optional[str]
- `temperature`: float
- `max_tokens`: Optional[int]
- `output`: Optional[Path]
- `verbose`: bool

---

### `async chat_with_file_context_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/chat_example.py:106`
**Complexity:** 2

**Description:**
> Chat with file context included.

---

### `chat_workflow(provider, conversation_memory) -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:62`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ChatWorkflow instance with real provider.

**Parameters:**
- `provider`: None
- `conversation_memory`: None

---

### `chat_workflow(mock_provider, conversation_memory) -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:44`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ChatWorkflow instance for testing.

**Parameters:**
- `mock_provider`: None
- `conversation_memory`: None

---

### `cluster_claims_hierarchical(citation_maps, n_clusters, model_name, linkage_method) -> List[List[CitationMap]]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:458`
**Complexity:** 4

**Description:**
> Cluster claims using hierarchical clustering on semantic embeddings.

Uses agglomerative hierarchical clustering to group claims,
building a tree-based hierarchy and cutting at specified level.

Args:
    citation_maps: List of CitationMap objects to cluster
    n_clusters: Number of clusters to create (default: 3)
    model_name: Sentence transformer model to use
    linkage_method: Linkage method ('ward', 'complete', 'average', 'single')

Returns:
    List of clusters, where each cluster is a list of CitationMaps

Raises:
    ValueError: If n_clusters > len(citation_maps)

Example:
    >>> maps = [cm1, cm2, cm3, cm4, cm5]
    >>> clusters = cluster_claims_hierarchical(
    ...     maps,
    ...     n_clusters=2,
    ...     linkage_method="ward"
    ... )
    >>> print(f"Found {len(clusters)} clusters")
    Found 2 clusters

**Parameters:**
- `citation_maps`: List[CitationMap]
- `n_clusters`: int
- `model_name`: str
- `linkage_method`: str

---

### `cluster_claims_kmeans(citation_maps, n_clusters, model_name, random_state) -> List[List[CitationMap]]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:397`
**Complexity:** 4

**Description:**
> Cluster claims using K-means algorithm on semantic embeddings.

Groups claims into k clusters based on semantic similarity,
useful for organizing large collections of claims by topic.

Args:
    citation_maps: List of CitationMap objects to cluster
    n_clusters: Number of clusters to create (default: 3)
    model_name: Sentence transformer model to use
    random_state: Random seed for reproducibility

Returns:
    List of clusters, where each cluster is a list of CitationMaps

Raises:
    ValueError: If n_clusters > len(citation_maps)

Example:
    >>> maps = [cm1, cm2, cm3, cm4, cm5]
    >>> clusters = cluster_claims_kmeans(maps, n_clusters=2)
    >>> print(f"Found {len(clusters)} clusters")
    Found 2 clusters
    >>> for i, cluster in enumerate(clusters):
    ...     print(f"Cluster {i}: {len(cluster)} claims")
    Cluster 0: 3 claims
    Cluster 1: 2 claims

**Parameters:**
- `citation_maps`: List[CitationMap]
- `n_clusters`: int
- `model_name`: str
- `random_state`: int

---

### `compute_claim_similarity(claim1, claim2, model_name) -> float`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:142`
**Complexity:** 1

**Description:**
> Compute semantic similarity between two claim texts.

Args:
    claim1: First claim text
    claim2: Second claim text
    model_name: Sentence transformer model to use

Returns:
    Similarity score (0.0 to 1.0)
    - >= 0.9: Very similar (likely duplicates)
    - 0.7-0.9: Similar (related claims)
    - 0.5-0.7: Moderate similarity (overlapping topics)
    - < 0.5: Different claims

Example:
    >>> score = compute_claim_similarity(
    ...     "AI improves software quality",
    ...     "Artificial intelligence enhances code quality"
    ... )
    >>> print(f"Similarity: {score:.3f}")
    Similarity: 0.875

**Parameters:**
- `claim1`: str
- `claim2`: str
- `model_name`: str

---

### `compute_claim_similarity_batch(claims, model_name) -> np.ndarray`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:235`
**Complexity:** 1

**Description:**
> Compute pairwise similarities for a batch of claims.

More efficient than calling compute_claim_similarity repeatedly,
as it batches the embedding computation.

Args:
    claims: List of claim texts
    model_name: Sentence transformer model to use

Returns:
    NxN similarity matrix where result[i][j] = similarity(claims[i], claims[j])

Example:
    >>> claims = [
    ...     "AI improves quality",
    ...     "ML enhances accuracy",
    ...     "Weather is sunny"
    ... ]
    >>> sim_matrix = compute_claim_similarity_batch(claims)
    >>> print(sim_matrix.shape)
    (3, 3)
    >>> print(f"Similarity[0,1]: {sim_matrix[0,1]:.3f}")
    Similarity[0,1]: 0.847

**Parameters:**
- `claims`: List[str]
- `model_name`: str

---

### `compute_cluster_statistics(clusters, model_name) -> Dict[str, Any]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:669`
**Complexity:** 6

**Description:**
> Compute statistics and quality metrics for clusters.

Provides insights into cluster quality, including size distribution,
intra-cluster similarity, and representative claims.

Args:
    clusters: List of clusters (each cluster is a list of CitationMaps)
    model_name: Sentence transformer model to use

Returns:
    Dictionary with cluster statistics:
    - num_clusters: Number of clusters
    - cluster_sizes: List of cluster sizes
    - avg_cluster_size: Average cluster size
    - representatives: List of representative claims (one per cluster)
    - intra_cluster_similarities: Average similarity within each cluster

Example:
    >>> stats = compute_cluster_statistics(clusters)
    >>> print(f"Number of clusters: {stats['num_clusters']}")
    >>> print(f"Average cluster size: {stats['avg_cluster_size']:.1f}")
    >>> print(f"Representatives: {stats['representatives']}")

**Parameters:**
- `clusters`: List[List[CitationMap]]
- `model_name`: str

---

### `compute_embedding(text, model_name, normalize) -> np.ndarray`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:77`
**Complexity:** 2

**Description:**
> Compute semantic embedding for text using sentence transformers.

Embeddings are cached using LRU cache to avoid recomputing for
duplicate text. Text is normalized before embedding computation
for consistency.

Args:
    text: Text to compute embedding for
    model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
    normalize: Whether to normalize text before embedding (default: True)

Returns:
    Normalized embedding vector as numpy array (unit length if model normalizes)

Example:
    >>> emb1 = compute_embedding("Machine learning improves accuracy")
    >>> emb2 = compute_embedding("ML enhances precision")
    >>> similarity = cosine_similarity(emb1, emb2)
    >>> print(f"Similarity: {similarity:.3f}")
    Similarity: 0.847

**Parameters:**
- `text`: str
- `model_name`: str
- `normalize`: bool

---

### `async confidence_progression_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:365`
**Complexity:** 5

**Description:**
> Demonstrate confidence progression through investigation steps.

Shows how confidence should naturally increase as evidence accumulates
and hypotheses are validated.

---

### `consensus(prompt, providers, strategy, system, temperature, max_tokens, timeout, output, verbose) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:232`
⚠️ **Complexity:** 16 (High)

**Decorators:** `@app.command()`

**Description:**
> Run consensus workflow across multiple AI models.

Example:
    modelchorus consensus "Explain quantum computing" -p claude -p gemini -s synthesize

**Parameters:**
- `prompt`: str
- `providers`: List[str]
- `strategy`: str
- `system`: Optional[str]
- `temperature`: float
- `max_tokens`: Optional[int]
- `timeout`: float
- `output`: Optional[Path]
- `verbose`: bool

---

### `conversation_memory() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:35`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ConversationMemory instance for testing.

---

### `conversation_memory() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:38`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ConversationMemory instance for testing.

---

### `async conversation_tracking_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/chat_example.py:152`
**Complexity:** 5

**Description:**
> Demonstrate conversation history tracking.

---

### `cosine_similarity(embedding1, embedding2) -> float`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:113`
**Complexity:** 1

**Description:**
> Compute cosine similarity between two embedding vectors.

Assumes embeddings are already normalized (unit length).
If normalized, this reduces to simple dot product.

Args:
    embedding1: First embedding vector
    embedding2: Second embedding vector

Returns:
    Cosine similarity score (0.0 to 1.0)
    - 1.0 = identical/very similar
    - 0.0 = completely dissimilar
    - 0.5 = moderate similarity

Example:
    >>> emb1 = compute_embedding("hello world")
    >>> emb2 = compute_embedding("hello world")
    >>> cosine_similarity(emb1, emb2)
    1.0

**Parameters:**
- `embedding1`: np.ndarray
- `embedding2`: np.ndarray

---

### `detect_contradiction(claim_1_id, claim_1_text, claim_2_id, claim_2_text, similarity_threshold, model_name) -> Optional[Contradiction]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:366`
**Complexity:** 9

**Description:**
> Detect contradiction between two claims.

Main entry point for contradiction detection. Analyzes semantic
similarity and polarity opposition to determine if claims contradict.

Args:
    claim_1_id: Identifier for first claim
    claim_1_text: Text of first claim
    claim_2_id: Identifier for second claim
    claim_2_text: Text of second claim
    similarity_threshold: Minimum similarity to consider (default: 0.3)
    model_name: Sentence transformer model to use

Returns:
    Contradiction object if contradiction detected, None otherwise

Detection Logic:
    1. Compute semantic similarity
    2. If similarity < threshold: likely unrelated, return None
    3. Detect polarity opposition
    4. If no polarity opposition and low similarity: return None
    5. Assess severity
    6. Generate explanation
    7. Return Contradiction object

Example:
    >>> contra = detect_contradiction(
    ...     "claim-1", "AI improves accuracy by 23%",
    ...     "claim-2", "AI reduces accuracy by 15%"
    ... )
    >>> if contra:
    ...     print(f"Severity: {contra.severity}")
    ...     print(f"Confidence: {contra.confidence}")
    Severity: ContradictionSeverity.CRITICAL
    Confidence: 0.87

**Parameters:**
- `claim_1_id`: str
- `claim_1_text`: str
- `claim_2_id`: str
- `claim_2_text`: str
- `similarity_threshold`: float
- `model_name`: str

---

### `detect_contradictions_batch(claims, similarity_threshold, model_name) -> List[Contradiction]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:503`
**Complexity:** 4

**Description:**
> Detect contradictions in a batch of claims.

Efficiently compares all pairs of claims to find contradictions.
Useful for analyzing collections of claims from multiple sources.

Args:
    claims: List of (claim_id, claim_text) tuples
    similarity_threshold: Minimum similarity to consider (default: 0.3)
    model_name: Sentence transformer model to use

Returns:
    List of detected Contradiction objects

Example:
    >>> claims = [
    ...     ("claim-1", "AI improves accuracy"),
    ...     ("claim-2", "AI reduces accuracy"),
    ...     ("claim-3", "Weather is sunny"),
    ... ]
    >>> contradictions = detect_contradictions_batch(claims)
    >>> print(f"Found {len(contradictions)} contradictions")
    Found 1 contradictions

**Parameters:**
- `claims`: List[Tuple[str, str]]
- `similarity_threshold`: float
- `model_name`: str

---

### `detect_polarity_opposition(claim_text_1, claim_text_2) -> Tuple[bool, float]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/contradiction.py:238`
**Complexity:** 10

**Description:**
> Detect if two claims have opposing polarity (positive vs negative).

Uses keyword-based analysis to identify claims that make opposite
assertions about the same topic.

Args:
    claim_text_1: First claim text
    claim_text_2: Second claim text

Returns:
    Tuple of (has_opposition, confidence)
    - has_opposition: True if claims have opposing polarity
    - confidence: Confidence in polarity detection (0.0-1.0)

Example:
    >>> has_opp, conf = detect_polarity_opposition(
    ...     "AI improves accuracy by 23%",
    ...     "AI reduces accuracy by 15%"
    ... )
    >>> print(f"Opposition: {has_opp}, Confidence: {conf:.2f}")
    Opposition: True, Confidence: 0.80

**Parameters:**
- `claim_text_1`: str
- `claim_text_2`: str

---

### `find_duplicate_claims(citation_maps, threshold, model_name) -> List[List[CitationMap]]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:329`
**Complexity:** 8

**Description:**
> Detect groups of duplicate or near-duplicate claims.

Uses high similarity threshold to identify claims that are
essentially the same despite different wording.

Args:
    citation_maps: List of CitationMap objects to check for duplicates
    threshold: Minimum similarity to consider duplicates (default: 0.9)
    model_name: Sentence transformer model to use

Returns:
    List of duplicate groups, where each group is a list of similar CitationMaps

Example:
    >>> maps = [citation_map1, citation_map2, citation_map3]
    >>> duplicates = find_duplicate_claims(maps, threshold=0.9)
    >>> for group in duplicates:
    ...     print(f"Found {len(group)} duplicates:")
    ...     for cm in group:
    ...         print(f"  - {cm.claim_text}")
    Found 2 duplicates:
      - AI improves accuracy
      - Artificial intelligence enhances precision

**Parameters:**
- `citation_maps`: List[CitationMap]
- `threshold`: float
- `model_name`: str

---

### `find_similar_claims(query_claim, citation_maps, threshold, top_k, model_name) -> List[Tuple[CitationMap, float]]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:175`
**Complexity:** 4

**Description:**
> Find citation maps with claims similar to the query claim.

Uses semantic similarity to identify related claims across a
collection of citation maps. Useful for:
- Detecting duplicate claims
- Finding supporting evidence for new claims
- Clustering related arguments

Args:
    query_claim: Claim text to search for
    citation_maps: List of CitationMap objects to search within
    threshold: Minimum similarity score to include (default: 0.7)
    top_k: Optional limit on number of results (returns top-k most similar)
    model_name: Sentence transformer model to use

Returns:
    List of (CitationMap, similarity_score) tuples, sorted by similarity (descending)

Example:
    >>> maps = [citation_map1, citation_map2, citation_map3]
    >>> results = find_similar_claims(
    ...     "Machine learning improves accuracy",
    ...     maps,
    ...     threshold=0.7,
    ...     top_k=5
    ... )
    >>> for cm, score in results:
    ...     print(f"{score:.3f}: {cm.claim_text}")
    0.892: ML models enhance prediction accuracy
    0.745: AI systems improve results

**Parameters:**
- `query_claim`: str
- `citation_maps`: List[CitationMap]
- `threshold`: float
- `top_k`: Optional[int]
- `model_name`: str

---

### `format_citation(citation, style) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:23`
**Complexity:** 4

**Description:**
> Format a Citation object according to the specified style.

Args:
    citation: The Citation object to format
    style: The citation style to use (APA, MLA, or Chicago)

Returns:
    Formatted citation string according to the specified style

Example:
    >>> from modelchorus.core.models import Citation
    >>> from modelchorus.utils.citation_formatter import format_citation, CitationStyle
    >>> c = Citation(
    ...     source="https://arxiv.org/abs/2401.12345",
    ...     confidence=0.95,
    ...     metadata={"author": "Smith, J.", "year": "2024", "title": "Machine Learning"}
    ... )
    >>> format_citation(c, CitationStyle.APA)
    'Smith, J. (2024). Machine Learning. Retrieved from https://arxiv.org/abs/2401.12345'

**Parameters:**
- `citation`: 'Citation'
- `style`: CitationStyle

---

### `format_citation_map(citation_map, style, include_claim) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:180`
**Complexity:** 4

**Description:**
> Format a CitationMap object with all its citations.

Args:
    citation_map: The CitationMap object to format
    style: The citation style to use (APA, MLA, or Chicago)
    include_claim: Whether to include the claim text in the output

Returns:
    Formatted string with claim and all citations

Example:
    >>> formatted = format_citation_map(cm, CitationStyle.APA, include_claim=True)
    >>> print(formatted)
    Claim: Machine learning improves accuracy by 23%

    Citations:
    1. Smith, J. (2024). ML Research. Retrieved from https://arxiv.org/abs/2401.12345
    2. Doe, A. (2024). AI Studies. Retrieved from paper2.pdf

**Parameters:**
- `citation_map`: 'CitationMap'
- `style`: CitationStyle
- `include_claim`: bool

---

### `generate_cluster_name(cluster, model_name, max_words) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:565`
**Complexity:** 3

**Description:**
> Generate a concise name for a cluster based on representative claims.

Uses extractive approach: identifies key terms from the cluster
representative claim and constructs a short descriptive name.

Args:
    cluster: List of CitationMaps in the cluster
    model_name: Sentence transformer model to use
    max_words: Maximum words in generated name (default: 5)

Returns:
    Concise cluster name (e.g., "AI Quality Improvement")

Example:
    >>> cluster = [cm1, cm2, cm3]  # Claims about "AI improves quality"
    >>> name = generate_cluster_name(cluster, max_words=4)
    >>> print(name)
    AI Quality Improvement

**Parameters:**
- `cluster`: List[CitationMap]
- `model_name`: str
- `max_words`: int

---

### `get_cluster_representative(cluster, model_name) -> CitationMap`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:522`
**Complexity:** 3

**Description:**
> Find the most representative claim in a cluster (centroid).

Computes the claim closest to the cluster centroid in embedding space,
useful for summarizing or labeling clusters.

Args:
    cluster: List of CitationMaps in the cluster
    model_name: Sentence transformer model to use

Returns:
    CitationMap closest to cluster centroid

Example:
    >>> cluster = [cm1, cm2, cm3]
    >>> representative = get_cluster_representative(cluster)
    >>> print(f"Representative claim: {representative.claim_text}")

**Parameters:**
- `cluster`: List[CitationMap]
- `model_name`: str

---

### `get_default_state_manager() -> StateManager`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/state.py:517`
**Complexity:** 2

**Description:**
> Get singleton default state manager instance.

Returns:
    Default StateManager instance

Example:
    >>> from modelchorus.core.state import get_default_state_manager
    >>> manager = get_default_state_manager()
    >>> manager.set_state("my_workflow", {"step": 1})

---

### `get_provider_by_name(name) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:36`
**Complexity:** 2

**Description:**
> Get provider instance by name.

**Parameters:**
- `name`: str

---

### `async hypothesis_management_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:275`
**Complexity:** 6

**Description:**
> Demonstrate manual hypothesis management and state inspection.

Shows how to programmatically add, update, and track hypotheses
during an investigation.

---

### `async investigation_with_expert_validation() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:195`
**Complexity:** 4

**Description:**
> Investigation with expert validation from a different model.

Demonstrates how ThinkDeep can use a second model for validation
and additional insights when confidence hasn't reached "certain" level.

---

### `is_provider_available(provider_class) -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:18`
**Complexity:** 2

**Description:**
> Check if a provider is available (API key configured).

**Parameters:**
- `provider_class`: None

---

### `list_providers() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:789`
**Complexity:** 3

**Decorators:** `@app.command()`

**Description:**
> List all available providers and their models.

---

### `async main() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/basic_workflow.py:56`
**Complexity:** 2

**Description:**
> Main entry point for the example.

---

### `async main() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/chat_example.py:196`
**Complexity:** 1

**Description:**
> Run all examples.

---

### `async main() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/provider_integration.py:101`
**Complexity:** 2

**Description:**
> Main entry point for the provider example.

---

### `async main() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:431`
**Complexity:** 1

**Description:**
> Run all examples.

---

### `main() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:827`
**Complexity:** 1

**Description:**
> Main entry point for CLI.

---

### `mock_claude_response() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:10`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock response from Claude CLI --output-format json.

---

### `mock_codex_response() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:35`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock JSONL response from Codex CLI --json.

---

### `mock_provider() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_workflow.py:18`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock ModelProvider for testing.

---

### `mock_subprocess_run() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:45`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock subprocess.run for CLI command execution.

---

### `async multi_step_investigation_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/thinkdeep_example.py:63`
**Complexity:** 10

**Description:**
> Multi-step investigation showing hypothesis evolution and confidence progression.

This example demonstrates a systematic investigation of a performance issue,
showing how hypotheses are formed, tested, and confidence evolves across steps.

---

### `async multi_turn_conversation_example() -> None`

**Language:** python
**Defined in:** `modelchorus/examples/chat_example.py:48`
**Complexity:** 4

**Description:**
> Multi-turn conversation with continuation.

---

### `provider(provider_name) -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:51`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create provider instance based on provider name.

**Parameters:**
- `provider_name`: None

---

### `provider_name(request) -> None`

**Language:** python
**Defined in:** `modelchorus/tests/test_chat_integration.py:45`
**Complexity:** 1

**Decorators:** `@pytest.fixture(params=[pytest.param('claude', marks=pytest.mark.skipif(not CLAUDE_AVAILABLE, reason='Claude API not configured')), pytest.param('gemini', marks=pytest.mark.skipif(not GEMINI_AVAILABLE, reason='Gemini API not configured')), pytest.param('codex', marks=pytest.mark.skipif(not CODEX_AVAILABLE, reason='Codex API not configured'))])`

**Description:**
> Parameterized fixture for provider names.

**Parameters:**
- `request`: None

---

### `sample_generation_request() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:55`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Sample GenerationRequest for testing.

---

### `score_cluster_coherence(cluster, model_name) -> float`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:747`
**Complexity:** 4

**Description:**
> Measure how tightly grouped claims are within a cluster.

Computes average pairwise similarity between claims in the cluster.
Higher scores indicate more coherent/similar claims.

Args:
    cluster: List of CitationMaps in the cluster
    model_name: Sentence transformer model to use

Returns:
    Coherence score (0.0 to 1.0)
    - 1.0 = perfect coherence (all claims identical)
    - 0.8-1.0 = high coherence (very similar claims)
    - 0.5-0.8 = moderate coherence
    - < 0.5 = low coherence (diverse claims)

Example:
    >>> cluster = [cm1, cm2, cm3]
    >>> coherence = score_cluster_coherence(cluster)
    >>> print(f"Coherence: {coherence:.3f}")
    Coherence: 0.847

**Parameters:**
- `cluster`: List[CitationMap]
- `model_name`: str

---

### `score_cluster_separation(clusters, model_name) -> float`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:794`
**Complexity:** 5

**Description:**
> Measure how distinct clusters are from each other.

Computes average distance between cluster centroids.
Higher scores indicate better separation between clusters.

Args:
    clusters: List of clusters (each cluster is a list of CitationMaps)
    model_name: Sentence transformer model to use

Returns:
    Separation score (0.0 to 1.0)
    - 1.0 = perfect separation (clusters completely distinct)
    - 0.7-1.0 = high separation (well-separated clusters)
    - 0.5-0.7 = moderate separation
    - < 0.5 = low separation (overlapping clusters)

Example:
    >>> clusters = [[cm1, cm2], [cm3, cm4]]
    >>> separation = score_cluster_separation(clusters)
    >>> print(f"Separation: {separation:.3f}")
    Separation: 0.723

**Parameters:**
- `clusters`: List[List[CitationMap]]
- `model_name`: str

---

### `score_clustering_quality(clusters, model_name) -> Dict[str, Any]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:864`
**Complexity:** 8

**Description:**
> Compute comprehensive quality metrics for clustering results.

Combines multiple metrics to provide overall assessment of
clustering quality, including coherence, separation, and
interpretability measures.

Args:
    clusters: List of clusters (each cluster is a list of CitationMaps)
    model_name: Sentence transformer model to use

Returns:
    Dictionary with quality metrics:
    - coherence_scores: List of coherence scores (one per cluster)
    - avg_coherence: Average coherence across all clusters
    - separation: Inter-cluster separation score
    - silhouette_score: Sklearn silhouette coefficient (-1 to 1)
    - quality_score: Overall quality (0.0 to 1.0)
    - num_clusters: Number of clusters
    - cluster_sizes: List of cluster sizes
    - interpretability: Named clusters and summaries

Example:
    >>> clusters = [[cm1, cm2], [cm3, cm4]]
    >>> quality = score_clustering_quality(clusters)
    >>> print(f"Quality: {quality['quality_score']:.3f}")
    >>> print(f"Coherence: {quality['avg_coherence']:.3f}")
    >>> print(f"Separation: {quality['separation']:.3f}")
    Quality: 0.812
    Coherence: 0.847
    Separation: 0.723

**Parameters:**
- `clusters`: List[List[CitationMap]]
- `model_name`: str

---

### `summarize_cluster(cluster, model_name, max_length) -> str`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/workflows/argument/semantic.py:614`
**Complexity:** 5

**Description:**
> Generate a detailed summary of cluster themes.

Analyzes claims in cluster to identify common patterns
and generates a descriptive summary.

Args:
    cluster: List of CitationMaps in the cluster
    model_name: Sentence transformer model to use
    max_length: Maximum characters in summary (default: 150)

Returns:
    Cluster summary (1-2 sentences)

Example:
    >>> cluster = [cm1, cm2, cm3]
    >>> summary = summarize_cluster(cluster)
    >>> print(summary)
    This cluster focuses on AI quality improvement claims.
    All claims discuss machine learning enhancing accuracy.

**Parameters:**
- `cluster`: List[CitationMap]
- `model_name`: str
- `max_length`: int

---

### `thinkdeep(prompt, provider, expert_provider, continuation_id, files, system, temperature, max_tokens, disable_expert, output, verbose) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:417`
⚠️ **Complexity:** 36 (High)

**Decorators:** `@app.command()`

**Description:**
> Start a ThinkDeep investigation for systematic problem analysis.

ThinkDeep provides extended reasoning with hypothesis tracking, confidence
progression, and optional expert validation.

Example:
    # Start new investigation
    modelchorus thinkdeep "Why is authentication failing?" -p claude

    # Continue investigation
    modelchorus thinkdeep "Check async patterns" --continue thread-id-123

    # Include files and expert validation
    modelchorus thinkdeep "Analyze bug" -f src/auth.py -e gemini

**Parameters:**
- `prompt`: str
- `provider`: str
- `expert_provider`: Optional[str]
- `continuation_id`: Optional[str]
- `files`: Optional[List[str]]
- `system`: Optional[str]
- `temperature`: float
- `max_tokens`: Optional[int]
- `disable_expert`: bool
- `output`: Optional[Path]
- `verbose`: bool

---

### `thinkdeep_status(thread_id, show_steps, show_files, verbose) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:664`
⚠️ **Complexity:** 18 (High)

**Decorators:** `@app.command(name='thinkdeep-status')`

**Description:**
> Inspect the state of an ongoing ThinkDeep investigation.

Shows current hypotheses, confidence level, investigation progress,
and optionally all steps and examined files.

Example:
    # View basic status
    modelchorus thinkdeep-status thread-id-123

    # View with all steps
    modelchorus thinkdeep-status thread-id-123 --steps

    # View with files
    modelchorus thinkdeep-status thread-id-123 --files --verbose

**Parameters:**
- `thread_id`: str
- `show_steps`: bool
- `show_files`: bool
- `verbose`: bool

---

### `validate_citation(citation) -> Tuple[bool, List[str]]`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/utils/citation_formatter.py:227`
⚠️ **Complexity:** 14 (High)

**Description:**
> Validate a Citation object for completeness and quality.

Args:
    citation: The Citation object to validate

Returns:
    Tuple of (is_valid, issues) where:
    - is_valid: True if citation meets minimum requirements
    - issues: List of validation issue messages

Example:
    >>> is_valid, issues = validate_citation(citation)
    >>> if not is_valid:
    ...     print(f"Validation issues: {', '.join(issues)}")

**Parameters:**
- `citation`: 'Citation'

---

### `version() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:816`
**Complexity:** 1

**Decorators:** `@app.command()`

**Description:**
> Show version information.

---


## 📦 Dependencies

### `modelchorus/examples/basic_workflow.py`

- `asyncio`
- `modelchorus.core.BaseWorkflow`
- `modelchorus.core.WorkflowRegistry`
- `modelchorus.core.WorkflowRequest`
- `modelchorus.core.WorkflowResult`

### `modelchorus/examples/chat_example.py`

- `asyncio`
- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.ClaudeProvider`
- `modelchorus.workflows.ChatWorkflow`
- `pathlib.Path`

### `modelchorus/examples/provider_integration.py`

- `asyncio`
- `modelchorus.providers.GenerationRequest`
- `modelchorus.providers.GenerationResponse`
- `modelchorus.providers.ModelCapability`
- `modelchorus.providers.ModelConfig`
- `modelchorus.providers.ModelProvider`

### `modelchorus/examples/thinkdeep_example.py`

- `asyncio`
- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.ClaudeProvider`
- `modelchorus.providers.GeminiProvider`
- `modelchorus.workflows.ConfidenceLevel`
- `modelchorus.workflows.Hypothesis`
- `modelchorus.workflows.InvestigationStep`
- `modelchorus.workflows.ThinkDeepState`
- `modelchorus.workflows.ThinkDeepWorkflow`
- `pathlib.Path`

### `modelchorus/src/modelchorus/cli/__init__.py`

- `main.app`
- `main.main`

### `modelchorus/src/modelchorus/cli/main.py`

- `asyncio`
- `core.conversation.ConversationMemory`
- `json`
- `pathlib.Path`
- `providers.ClaudeProvider`
- `providers.CodexProvider`
- `providers.CursorAgentProvider`
- `providers.GeminiProvider`
- `providers.GenerationRequest`
- `rich.console.Console`
- `rich.print`
- `rich.table.Table`
- `sys`
- `typer`
- `typing.List`
- `typing.Optional`
- `workflows.ChatWorkflow`
- `workflows.ConsensusStrategy`
- `workflows.ConsensusWorkflow`
- `workflows.ThinkDeepWorkflow`

### `modelchorus/src/modelchorus/core/__init__.py`

- `base_workflow.BaseWorkflow`
- `base_workflow.WorkflowResult`
- `base_workflow.WorkflowStep`
- `conversation.ConversationMemory`
- `models.ConfidenceLevel`
- `models.ConsensusConfig`
- `models.ConversationMessage`
- `models.ConversationState`
- `models.ConversationThread`
- `models.Hypothesis`
- `models.InvestigationStep`
- `models.ModelResponse`
- `models.ModelSelection`
- `models.ThinkDeepState`
- `models.WorkflowRequest`
- `models.WorkflowResponse`
- `models.WorkflowStep`
- `registry.WorkflowRegistry`

### `modelchorus/src/modelchorus/core/base_workflow.py`

- `abc.ABC`
- `abc.abstractmethod`
- `conversation.ConversationMemory`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `models.ConversationMessage`
- `models.ConversationThread`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/core/contradiction.py`

- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.field_validator`
- `re`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`

### `modelchorus/src/modelchorus/core/conversation.py`

- `datetime.datetime`
- `datetime.timedelta`
- `datetime.timezone`
- `filelock`
- `json`
- `logging`
- `models.ConversationMessage`
- `models.ConversationState`
- `models.ConversationThread`
- `pathlib.Path`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`
- `uuid`

### `modelchorus/src/modelchorus/core/models.py`

- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Literal`
- `typing.Optional`

### `modelchorus/src/modelchorus/core/registry.py`

- `base_workflow.BaseWorkflow`
- `inspect`
- `typing.Callable`
- `typing.Dict`
- `typing.Optional`
- `typing.Type`

### `modelchorus/src/modelchorus/core/state.py`

- `datetime.datetime`
- `datetime.timezone`
- `json`
- `logging`
- `models.ConversationState`
- `pathlib.Path`
- `threading`
- `typing.Any`
- `typing.Dict`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/__init__.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `base_provider.ModelProvider`
- `claude_provider.ClaudeProvider`
- `cli_provider.CLIProvider`
- `codex_provider.CodexProvider`
- `cursor_agent_provider.CursorAgentProvider`
- `gemini_provider.GeminiProvider`

### `modelchorus/src/modelchorus/providers/base_provider.py`

- `abc.ABC`
- `abc.abstractmethod`
- `dataclasses.dataclass`
- `dataclasses.field`
- `enum.Enum`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/claude_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/cli_provider.py`

- `abc.abstractmethod`
- `asyncio`
- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelProvider`
- `json`
- `logging`
- `pathlib.Path`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/codex_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/cursor_agent_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/providers/gemini_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/utils/__init__.py`

- `modelchorus.utils.citation_formatter.CitationStyle`
- `modelchorus.utils.citation_formatter.calculate_citation_confidence`
- `modelchorus.utils.citation_formatter.calculate_citation_map_confidence`
- `modelchorus.utils.citation_formatter.format_citation`
- `modelchorus.utils.citation_formatter.format_citation_map`
- `modelchorus.utils.citation_formatter.validate_citation`

### `modelchorus/src/modelchorus/utils/citation_formatter.py`

- `enum.Enum`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`

### `modelchorus/src/modelchorus/workflows/__init__.py`

- `chat.ChatWorkflow`
- `consensus.ConsensusResult`
- `consensus.ConsensusStrategy`
- `consensus.ConsensusWorkflow`
- `consensus.ProviderConfig`
- `core.models.ConfidenceLevel`
- `core.models.Hypothesis`
- `core.models.InvestigationStep`
- `core.models.ThinkDeepState`
- `thinkdeep.ThinkDeepWorkflow`

### `modelchorus/src/modelchorus/workflows/argument/__init__.py`

- `modelchorus.workflows.argument.semantic.cluster_claims_hierarchical`
- `modelchorus.workflows.argument.semantic.cluster_claims_kmeans`
- `modelchorus.workflows.argument.semantic.compute_claim_similarity`
- `modelchorus.workflows.argument.semantic.compute_cluster_statistics`
- `modelchorus.workflows.argument.semantic.compute_embedding`
- `modelchorus.workflows.argument.semantic.cosine_similarity`
- `modelchorus.workflows.argument.semantic.find_similar_claims`
- `modelchorus.workflows.argument.semantic.get_cluster_representative`

### `modelchorus/src/modelchorus/workflows/argument/semantic.py`

- `functools.lru_cache`
- `hashlib`
- `modelchorus.core.models.Citation`
- `modelchorus.core.models.CitationMap`
- `numpy`
- `sentence_transformers.SentenceTransformer`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`

### `modelchorus/src/modelchorus/workflows/chat.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.base_workflow.WorkflowStep`
- `core.conversation.ConversationMemory`
- `core.models.ConversationMessage`
- `logging`
- `providers.GenerationRequest`
- `providers.GenerationResponse`
- `providers.ModelProvider`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `uuid`

### `modelchorus/src/modelchorus/workflows/consensus.py`

- `asyncio`
- `dataclasses.dataclass`
- `dataclasses.field`
- `enum.Enum`
- `logging`
- `providers.GenerationRequest`
- `providers.GenerationResponse`
- `providers.ModelProvider`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/workflows/thinkdeep.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.base_workflow.WorkflowStep`
- `core.conversation.ConversationMemory`
- `core.models.ConfidenceLevel`
- `core.models.ConversationMessage`
- `core.models.Hypothesis`
- `core.models.InvestigationStep`
- `core.models.ThinkDeepState`
- `logging`
- `providers.GenerationRequest`
- `providers.GenerationResponse`
- `providers.ModelProvider`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `uuid`

### `modelchorus/tests/conftest.py`

- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

### `modelchorus/tests/test_chat_integration.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.ClaudeProvider`
- `modelchorus.providers.CodexProvider`
- `modelchorus.providers.GeminiProvider`
- `modelchorus.workflows.ChatWorkflow`
- `os`
- `pathlib.Path`
- `pytest`

### `modelchorus/tests/test_chat_workflow.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.ChatWorkflow`
- `pathlib.Path`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`
- `uuid`

### `modelchorus/tests/test_claude_provider.py`

- `json`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.claude_provider.ClaudeProvider`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`

### `modelchorus/tests/test_codex_provider.py`

- `json`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.codex_provider.CodexProvider`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`

### `modelchorus/tests/test_consensus_workflow.py`

- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.consensus.ConsensusStrategy`
- `modelchorus.workflows.consensus.ConsensusWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `modelchorus/tests/test_conversation.py`

- `datetime.datetime`
- `datetime.timezone`
- `json`
- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.core.models.ConversationMessage`
- `modelchorus.core.models.ConversationThread`
- `pathlib.Path`
- `pytest`
- `uuid`

### `modelchorus/tests/test_gemini_integration.py`

- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.gemini_provider.GeminiProvider`
- `pytest`
- `subprocess`

### `modelchorus/tests/test_integration.py`

- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.providers.claude_provider.ClaudeProvider`
- `modelchorus.providers.codex_provider.CodexProvider`
- `modelchorus.workflows.consensus.ConsensusStrategy`
- `modelchorus.workflows.consensus.ConsensusWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`

### `modelchorus/tests/test_semantic_similarity.py`

- `modelchorus.core.models.Citation`
- `modelchorus.core.models.CitationMap`
- `modelchorus.workflows.argument.semantic.add_similarity_to_citation`
- `modelchorus.workflows.argument.semantic.cluster_claims_hierarchical`
- `modelchorus.workflows.argument.semantic.cluster_claims_kmeans`
- `modelchorus.workflows.argument.semantic.compute_claim_similarity`
- `modelchorus.workflows.argument.semantic.compute_claim_similarity_batch`
- `modelchorus.workflows.argument.semantic.compute_cluster_statistics`
- `modelchorus.workflows.argument.semantic.compute_embedding`
- `modelchorus.workflows.argument.semantic.cosine_similarity`
- `modelchorus.workflows.argument.semantic.find_duplicate_claims`
- `modelchorus.workflows.argument.semantic.find_similar_claims`
- `modelchorus.workflows.argument.semantic.get_cluster_representative`
- `numpy`
- `pytest`

### `modelchorus/tests/test_state.py`

- `datetime.datetime`
- `datetime.timezone`
- `json`
- `modelchorus.core.models.ConversationState`
- `modelchorus.core.state.StateManager`
- `modelchorus.core.state.get_default_state_manager`
- `pathlib.Path`
- `pytest`
- `threading`
- `time`

### `modelchorus/tests/test_thinkdeep_models.py`

- `json`
- `modelchorus.core.models.ConfidenceLevel`
- `modelchorus.core.models.Hypothesis`
- `modelchorus.core.models.InvestigationStep`
- `modelchorus.core.models.ThinkDeepState`
- `pydantic.ValidationError`
- `pytest`

### `tests/test_concurrent_conversations.py`

- `asyncio`
- `concurrent.futures.ThreadPoolExecutor`
- `datetime.datetime`
- `datetime.timezone`
- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.core.models.ConversationMessage`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.chat.ChatWorkflow`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `time`
- `unittest.mock.AsyncMock`
- `uuid`

### `tests/test_memory_management.py`

- `asyncio`
- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.chat.ChatWorkflow`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `sys`
- `unittest.mock.AsyncMock`
- `uuid`

### `tests/test_providers/test_cli_interface.py`

- `modelchorus.providers.CLIProvider`
- `modelchorus.providers.ClaudeProvider`
- `modelchorus.providers.CodexProvider`
- `modelchorus.providers.CursorAgentProvider`
- `modelchorus.providers.GeminiProvider`
- `modelchorus.providers.GenerationRequest`
- `modelchorus.providers.ModelProvider`
- `pathlib.Path`
- `pytest`
- `sys`

### `tests/test_thinkdeep_complex.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.core.models.ConfidenceLevel`
- `modelchorus.core.models.Hypothesis`
- `modelchorus.core.models.InvestigationStep`
- `modelchorus.core.models.ThinkDeepState`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`
- `uuid`

### `tests/test_thinkdeep_expert_validation.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.core.models.ConfidenceLevel`
- `modelchorus.core.models.Hypothesis`
- `modelchorus.core.models.InvestigationStep`
- `modelchorus.core.models.ThinkDeepState`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`
- `uuid`

### `tests/test_thinkdeep_workflow.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.core.models.ConfidenceLevel`
- `modelchorus.core.models.Hypothesis`
- `modelchorus.core.models.InvestigationStep`
- `modelchorus.core.models.ThinkDeepState`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`
- `uuid`

### `tests/test_workflow_integration_chaining.py`

- `modelchorus.core.conversation.ConversationMemory`
- `modelchorus.providers.base_provider.GenerationRequest`
- `modelchorus.providers.base_provider.GenerationResponse`
- `modelchorus.workflows.chat.ChatWorkflow`
- `modelchorus.workflows.consensus.ConsensusStrategy`
- `modelchorus.workflows.consensus.ConsensusWorkflow`
- `modelchorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`
- `uuid`
