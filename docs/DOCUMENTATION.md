# claude-model-chorus Documentation

**Version:** 1.0.0
**Generated:** 2025-11-21 09:51:34

---

## 📊 Project Statistics

- **Total Files:** 113
- **Total Lines:** 50581
- **Total Classes:** 293
- **Total Functions:** 214
- **Avg Complexity:** 4.43
- **Max Complexity:** 44
- **High Complexity Functions:**
  - start (44)
  - thinkdeep (41)
  - _create_smart_mock_provider (37)
  - study_next (35)
  - study_view (31)



## 🏛️ Classes

### `ArgumentMap`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1430`

**Description:**
> Structured output from ARGUMENT workflow containing all perspectives.

ArgumentMap provides a comprehensive view of the dialectical analysis,
including the thesis (Creator), rebuttal (Skeptic), and synthesis (Moderator).
This structured format enables programmatic access to different perspectives
and supports downstream analysis, visualization, and decision-making.

Attributes:
    topic: The argument topic or claim being analyzed
    perspectives: List of perspectives (Creator, Skeptic, Moderator)
    synthesis: Final balanced synthesis from Moderator
    metadata: Additional workflow metadata (thread_id, model, timestamps, etc.)

Example:
    >>> arg_map = ArgumentMap(
    ...     topic="Universal basic income would reduce poverty",
    ...     perspectives=[creator_perspective, skeptic_perspective, moderator_perspective],
    ...     synthesis="After examining both perspectives...",
    ...     metadata={"thread_id": "abc123", "model": "claude-sonnet-4"}
    ... )
    >>> print(arg_map.perspectives[0].role)  # 'creator'
    >>> print(arg_map.perspectives[1].role)  # 'skeptic'
    >>> print(arg_map.perspectives[2].role)  # 'moderator'

**Methods:**
- `get_perspective()`
- `to_dict()`
- `from_dict()`

---

### `ArgumentPerspective`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1369`

**Description:**
> Represents a single perspective in an argument analysis.

Each perspective captures one role's analysis (Creator, Skeptic, or Moderator)
with their stance, reasoning, and key points.

Attributes:
    role: The role name (creator, skeptic, moderator)
    stance: The perspective's stance (for, against, neutral)
    content: Full response content from this perspective
    key_points: List of key points or arguments
    model: Model used for this perspective
    metadata: Additional perspective metadata

Example:
    >>> perspective = ArgumentPerspective(
    ...     role="creator",
    ...     stance="for",
    ...     content="Universal basic income would reduce poverty...",
    ...     key_points=["Ensures basic needs", "Reduces wealth gap"],
    ...     model="claude-sonnet-4"
    ... )

---

### `ArgumentWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/argument_workflow.py:31`

**Description:**
> Role-based dialectical reasoning workflow using RoleOrchestrator.

This workflow implements structured argument analysis through role-based
orchestration, where different AI roles (Creator, Skeptic, Moderator) examine
an argument from multiple perspectives to produce balanced dialectical analysis.

Architecture:
- Uses RoleOrchestrator for sequential role execution
- Creator role: Generates strong thesis advocating FOR the position (Step 1)
- Skeptic role: Provides critical rebuttal AGAINST the position (Step 2)
- Moderator role: Synthesizes perspectives into balanced analysis (Step 3)

Current Implementation Status:
- Step 1 (Creator): ✓ Implemented - Generates thesis with supporting arguments
- Step 2 (Skeptic): ✓ Implemented - Provides critical rebuttal and counter-arguments
- Step 3 (Moderator): ✓ Implemented - Synthesizes both perspectives into balanced analysis

Key Features:
- Role-based orchestration using RoleOrchestrator
- Sequential execution pattern (roles build on each other's outputs)
- Stance-driven prompts (for/against/neutral)
- Conversation threading via continuation_id
- Inherits conversation support from BaseWorkflow
- Structured dialectical reasoning

The ArgumentWorkflow is ideal for:
- Analyzing the strength of arguments and claims
- Debate preparation and research
- Critical thinking and decision-making support
- Examining multiple perspectives systematically
- Identifying both strengths and weaknesses in reasoning

Workflow Steps (when complete):
1. **Creator Role (Thesis Generation)**: Build strong case FOR the position
2. **Skeptic Role (Critical Rebuttal)**: Challenge with counter-arguments
3. **Moderator Role (Synthesis)**: Integrate perspectives into balanced assessment

Example:
    >>> from model_chorus.providers import ClaudeProvider
    >>> from model_chorus.workflows import ArgumentWorkflow
    >>> from model_chorus.core.conversation import ConversationMemory
    >>>
    >>> # Create provider and conversation memory
    >>> provider = ClaudeProvider()
    >>> memory = ConversationMemory()
    >>>
    >>> # Create workflow
    >>> workflow = ArgumentWorkflow(provider, conversation_memory=memory)
    >>>
    >>> # Analyze an argument (all three roles execute)
    >>> result = await workflow.run(
    ...     "Universal basic income would reduce poverty"
    ... )
    >>> print(result.steps[0].content)  # Creator's thesis
    >>> print(result.steps[1].content)  # Skeptic's rebuttal
    >>> print(result.steps[2].content)  # Moderator's synthesis
    >>> print(result.metadata['roles_executed'])  # ['creator', 'skeptic', 'moderator']
    >>>
    >>> # Continue analysis with follow-up
    >>> result2 = await workflow.run(
    ...     "What about the impact on work incentives?",
    ...     continuation_id=result.metadata.get('thread_id')
    ... )
    >>> print(result2.synthesis)

**Methods:**
- `__init__()`
- `_create_creator_role()`
- `_create_skeptic_role()`
- `_create_moderator_role()`
- `_generate_argument_map()`
- `run()`
- `_build_prompt_with_history()`
- `_get_conversation_length()`
- `get_provider()`
- `validate_config()`
- `__repr__()`

---

### `ArgumentWorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:115`

**Description:**
> Configuration specific to argument workflow.

---

### `BaseWorkflow`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `model_chorus/src/model_chorus/core/base_workflow.py:56`

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
- `get_execution_metrics()`
- `register_telemetry_callback()`
- `_execute_with_fallback()`
- `check_provider_availability()`
- `get_thread()`
- `add_message()`
- `resume_conversation()`
- `__repr__()`

---

### `BinaryFileError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/core/context_ingestion.py:34`

**Description:**
> Raised when attempting to read a binary file as text.

---

### `CLIProvider`

**Language:** python
**Inherits from:** `ModelProvider`
**Defined in:** `model_chorus/src/model_chorus/providers/cli_provider.py:51`

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
- `set_env_overrides()`
- `check_availability()`
- `_load_conversation_context()`
- `build_command()`
- `parse_response()`
- `execute_command()`
- `_build_subprocess_env()`
- `_is_retryable_error()`
- `generate()`
- `supports_vision()`
- `__repr__()`

---

### `ChatWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `model_chorus/src/model_chorus/workflows/chat.py:22`

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
    >>> from model_chorus.providers import ClaudeProvider
    >>> from model_chorus.workflows import ChatWorkflow
    >>> from model_chorus.core.conversation import ConversationMemory
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

### `ChatWorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:126`

**Description:**
> Configuration specific to chat workflow.

---

### `CircuitBreakerConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:418`

**Description:**
> Configuration for circuit breaker behavior.

Attributes:
    failure_threshold: Number of consecutive failures before opening circuit (default: 5)
    success_threshold: Number of consecutive successes in half-open to close circuit (default: 2)
    recovery_timeout: Seconds to wait before trying half-open state (default: 60.0)
    monitored_exceptions: Tuple of exception types that count as failures (default: (Exception,))
    excluded_exceptions: Tuple of exception types that don't count as failures
        (e.g., validation errors that shouldn't trigger circuit)

Examples:
    # Default config: 5 failures, 60s timeout
    config = CircuitBreakerConfig()

    # Aggressive protection
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)

    # Conservative (allow more failures)
    config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=120.0)

---

### `CircuitBreakerMiddleware`

**Language:** python
**Inherits from:** `Middleware`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:447`

**Description:**
> Middleware that implements circuit breaker pattern for fault tolerance.

Protects against cascading failures by tracking provider health and
"opening the circuit" when failures exceed threshold. Gives failing
providers time to recover before allowing requests through again.

State Machine:
    CLOSED (normal) --[failures >= threshold]--> OPEN (rejecting)
    OPEN --[timeout elapsed]--> HALF_OPEN (testing)
    HALF_OPEN --[success >= threshold]--> CLOSED
    HALF_OPEN --[any failure]--> OPEN

Example:
    # Use default config (5 failures, 60s timeout)
    provider = CircuitBreakerMiddleware(ClaudeProvider())

    # Custom config
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
    provider = CircuitBreakerMiddleware(ClaudeProvider(), config)

    # Compose with retry middleware (circuit breaker wraps retry)
    provider = CircuitBreakerMiddleware(
        RetryMiddleware(ClaudeProvider())
    )

    try:
        response = await provider.generate(request)
    except CircuitOpenError as e:
        print(f"Circuit open, retry in {e.recovery_time}s")

**Methods:**
- `__init__()`
- `_transition_to()`
- `_should_attempt_request()`
- `_time_until_recovery()`
- `_is_monitored_exception()`
- `_record_success()`
- `_record_failure()`
- `generate()`

**Properties:**
- `state`
- `failure_count`
- `success_count`

---

### `CircuitOpenError`

**Language:** python
**Inherits from:** `ProviderError`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:393`

**Description:**
> Raised when circuit breaker is open and request is rejected.

Attributes:
    provider_name: Name of the provider with open circuit
    recovery_time: Seconds until circuit will try half-open state

**Methods:**
- `__init__()`

---

### `CircuitState`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:380`

**Description:**
> States of a circuit breaker.

- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests fail immediately
- HALF_OPEN: Testing recovery, limited requests allowed

---

### `Citation`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:936`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:996`

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
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:15`

**Description:**
> Supported citation formatting styles.

---

### `Claim`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1071`

**Description:**
> Represents a factual or arguable statement extracted from model output.

Claims are fundamental units in argument analysis and research workflows.
Each claim captures a specific assertion along with metadata about its source,
location, and confidence level.

Used in workflows like:
- ARGUMENT: Extracting claims from debate responses for contradiction detection
- RESEARCH: Identifying factual claims across multiple sources
- IDEATE: Capturing key assertions from diverse perspectives

Attributes:
    content: The actual claim text (the statement being made)
    source_id: Identifier for the source (role name, document ID, model ID, etc.)
    location: Optional location within source (line number, section, paragraph, etc.)
    confidence: Confidence score for this claim (0.0 = low, 1.0 = high)

Example:
    >>> claim = Claim(
    ...     content="TypeScript reduces runtime errors by 15%",
    ...     source_id="proponent",
    ...     location="paragraph 2",
    ...     confidence=0.8
    ... )
    >>> print(claim)
    [proponent@paragraph 2] (0.80): TypeScript reduces runtime errors by 15%

**Methods:**
- `__str__()`
- `to_dict()`
- `from_dict()`

---

### `ClaudeConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:544`

**Description:**
> Loads and manages configuration from .claude/model_chorus_config.yaml.

**Methods:**
- `__init__()`
- `find_config_file()`
- `load_config()`
- `get_config()`
- `is_provider_enabled()`
- `get_enabled_providers()`
- `get_workflow_providers()`
- `get_workflow_provider_priority()`
- `get_workflow_num_to_consult()`
- `get_default_providers()`
- `get_workflow_default()`
- `get_workflow_default_provider()`
- `get_workflow_fallback_providers()`
- `get_provider_model()`

**Properties:**
- `config_path`

---

### `ClaudeProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `model_chorus/src/model_chorus/providers/claude_provider.py:23`

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

### `ClusterResult`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/clustering.py:22`

**Description:**
> Result of a clustering operation.

Attributes:
    cluster_id: Unique identifier for this cluster
    items: List of item indices belonging to this cluster
    centroid: Cluster centroid in embedding space
    name: Human-readable cluster name/label
    summary: Brief summary of cluster theme
    quality_score: Quality/coherence score (0.0 = poor, 1.0 = excellent)
    metadata: Additional cluster metadata

**Methods:**
- `__repr__()`

---

### `CodexProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `model_chorus/src/model_chorus/providers/codex_provider.py:23`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:17`

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

### `ConfigError`

**Language:** python
**Inherits from:** `ProviderError`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:59`

**Description:**
> Configuration-related errors.

Raised when middleware or provider configuration is invalid,
missing required fields, or contains conflicting settings.

Examples:
    - Invalid retry configuration (negative values, etc.)
    - Missing required circuit breaker thresholds
    - Conflicting rate limit settings

**Methods:**
- `__init__()`

---

### `ConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/config/loader.py:22`

**Description:**
> Loads and manages ModelChorus configuration.

ConfigLoader handles:
- Finding config files in project directories (.model-chorusrc)
- Parsing YAML and JSON config formats
- Validating configuration with Pydantic models
- Providing convenient access to config values with fallback logic

**Methods:**
- `__init__()`
- `find_config_file()`
- `load_config()`
- `_parse_config_content()`
- `get_config()`
- `get_workflow_default()`
- `get_workflow_default_provider()`
- `get_workflow_default_providers()`
- `get_workflow_fallback_providers()`
- `get_provider_model()`

---

### `ConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:129`

**Description:**
> Loads and manages ModelChorus configuration.

**Methods:**
- `__init__()`
- `find_config_file()`
- `load_config()`
- `_parse_config_content()`
- `get_config()`
- `get_workflow_default()`
- `get_default_provider()`
- `get_default_providers()`
- `get_fallback_providers()`
- `get_fallback_providers_excluding()`
- `get_provider_model()`

**Properties:**
- `config_path`

---

### `ConsensusConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:356`

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
**Defined in:** `model_chorus/src/model_chorus/workflows/consensus.py:51`

**Description:**
> Result from a consensus workflow execution.

---

### `ConsensusStrategy`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `model_chorus/src/model_chorus/workflows/consensus.py:30`

**Description:**
> Strategy for reaching consensus among multiple model responses.

---

### `ConsensusWorkflow`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/consensus.py:62`

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
    >>> from model_chorus.providers import ClaudeProvider, GeminiProvider
    >>> from model_chorus.workflows import ConsensusWorkflow
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

### `ConsensusWorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:65`

**Description:**
> Configuration specific to consensus workflow.

---

### `ContextAnalysisInput`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/context_analysis.py:25`

**Description:**
> Input model for context analysis skill.

Defines the investigation context needed to determine which persona
to consult next based on current phase, confidence, and findings.

Attributes:
    current_phase: Current investigation phase (discovery/validation/planning)
    confidence: Current confidence level (0-100 or ConfidenceLevel enum value)
    findings: List of findings/insights discovered so far in the investigation
    unresolved_questions: List of questions that still need investigation
    prior_persona: Name of the previously consulted persona (optional)

**Methods:**
- `validate_phase()`
- `validate_confidence()`

---

### `ContextAnalysisResult`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/context_analysis.py:140`

**Description:**
> Result of context analysis determining next persona to consult.

Contains the recommended persona, reasoning for the selection,
and any context-specific guidance for the persona invocation.

Attributes:
    recommended_persona: Name of the persona to consult next
    reasoning: Explanation for why this persona was selected
    context_summary: Summary of current investigation context
    confidence: Current confidence level from context
    guidance: Specific guidance or focus areas for the persona
    metadata: Additional analysis metadata

**Methods:**
- `to_dict()`
- `to_json()`

---

### `ContextIngestionService`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/context_ingestion.py:40`

**Description:**
> Manages safe file reading for context injection.

Provides controlled access to file contents with size limits
and proper error handling to prevent context window overflow.

Attributes:
    max_file_size_kb: Maximum file size in kilobytes
    warn_file_size_kb: File size threshold for warnings

**Methods:**
- `__init__()`
- `_validate_path()`
- `_detect_encoding()`
- `read_file()`
- `get_file_info()`
- `can_read_file()`
- `read_file_chunked()`
- `read_file_lines()`

---

### `Contradiction`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:53`

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
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:28`

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
**Defined in:** `model_chorus/src/model_chorus/core/conversation.py:35`

**Description:**
> Manages conversation threads with file-based persistence.

Provides thread-safe storage and retrieval of conversation history,
enabling multi-turn conversations across workflow executions.

Architecture:
    - Each thread stored as JSON file: ~/.model-chorus/conversations/{thread_id}.json
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
- `_estimate_tokens()`
- `_is_important_message()`
- `_apply_token_budget()`
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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:429`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:876`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:501`

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

### `CriticPersona`

**Language:** python
**Inherits from:** `Persona`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/critic.py:13`

**Description:**
> Critic persona with challenge and stress-test focus.

This persona specializes in:
- Challenging assumptions and identifying biases
- Finding edge cases and potential problems
- Stress-testing conclusions and hypotheses
- Identifying gaps in reasoning
- Providing constructive skepticism

The Critic persona approaches investigations with healthy skepticism,
seeking to strengthen findings by identifying weaknesses and alternatives.

**Methods:**
- `__init__()`
- `invoke()`

---

### `CursorAgentProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `model_chorus/src/model_chorus/providers/cursor_agent_provider.py:23`

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

### `Evidence`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1195`

**Description:**
> Represents supporting or refuting evidence for claims and hypotheses.

Evidence is a fundamental building block in investigation, research, and
argument workflows. Each piece of evidence captures specific information
from a source along with its type, strength, and relevance.

Used in workflows like:
- THINKDEEP: Tracking evidence that supports or refutes hypotheses
- ARGUMENT: Recording factual evidence for debate claims
- RESEARCH: Documenting findings from multiple sources

Attributes:
    content: The actual evidence text (the observation or fact)
    source_id: Identifier for the source (file path, model ID, document ID, etc.)
    location: Optional location within source (line number, section, page, etc.)
    evidence_type: Type of evidence (supporting, refuting, neutral, contextual)
    strength: Strength/weight of this evidence (0.0 = weak, 1.0 = strong)
    timestamp: Optional ISO timestamp when evidence was collected
    metadata: Additional evidence metadata (tags, categories, analysis, etc.)

Example:
    >>> evidence = Evidence(
    ...     content="Found async def pattern in auth.py line 45",
    ...     source_id="src/services/auth.py",
    ...     location="line 45",
    ...     evidence_type="supporting",
    ...     strength=0.9
    ... )
    >>> print(evidence)
    [src/services/auth.py@line 45] (supporting, 0.90): Found async def pattern in auth.py line 45

**Methods:**
- `__str__()`
- `to_dict()`
- `from_dict()`

---

### `ExampleProvider`

**Language:** python
**Inherits from:** `ModelProvider`
**Defined in:** `model_chorus/examples/provider_integration.py:19`

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
**Defined in:** `model_chorus/examples/basic_workflow.py:20`

**Description:**
> Example workflow that demonstrates basic workflow structure.

This is a placeholder implementation showing the minimal structure
needed to create a working workflow.

**Methods:**
- `run()`

---

### `ExecutionMetrics`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/workflow_runner.py:31`

**Description:**
> Metrics collected during workflow execution.

Attributes:
    started_at: Timestamp when execution started
    completed_at: Timestamp when execution completed (None if still running)
    duration_ms: Execution duration in milliseconds (None if still running)
    provider_attempts: List of (provider_name, success, error) tuples for each attempt
    total_attempts: Total number of provider attempts made
    successful_provider: Name of the provider that succeeded (None if all failed)
    status: Overall execution status (SUCCESS, FAILURE, or PARTIAL)
    input_tokens: Total input tokens used (if available from response)
    output_tokens: Total output tokens generated (if available from response)
    custom_metrics: Dictionary for custom metrics added by telemetry handlers

**Methods:**
- `mark_complete()`
- `add_attempt()`
- `set_token_usage()`
- `add_custom_metric()`

---

### `ExecutionStatus`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/workflow_runner.py:22`

**Description:**
> Status of workflow execution.

---

### `FileTooLargeError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/core/context_ingestion.py:28`

**Description:**
> Raised when a file exceeds the maximum size limit.

---

### `Gap`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:72`

**Description:**
> Model for tracking gaps in arguments.

Represents a detected gap in reasoning, evidence, or support,
including severity assessment, confidence in detection, and
recommendations for improvement.

Attributes:
    gap_id: Unique identifier for this gap
    gap_type: Type of gap (evidence/logical/support/assumption)
    severity: Severity level of the gap
    claim_id: Identifier of the claim with the gap
    claim_text: Full text of the claim
    description: Detailed description of what's missing
    recommendation: Suggestion for addressing the gap
    confidence: Confidence in gap detection (0.0-1.0)
    metadata: Additional metadata (detection_method, context, etc.)

**Methods:**
- `validate_confidence()`

---

### `GapSeverity`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:48`

**Description:**
> Severity levels for gaps in arguments.

Classifies gaps by their impact on argument validity and persuasiveness.
Higher severity indicates more critical gaps requiring immediate attention.

Values:
    MINOR: Small gap with minimal impact on argument strength.
           May improve clarity but not essential.
    MODERATE: Notable gap that weakens the argument.
             Should be addressed to improve persuasiveness.
    MAJOR: Significant gap that undermines argument validity.
          Must be addressed for credible argumentation.
    CRITICAL: Fundamental gap that invalidates the argument.
             Immediate attention required.

---

### `GapType`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:28`

**Description:**
> Types of gaps that can be detected in arguments.

Classifies gaps by their nature to help prioritize remediation
and guide improvement strategies.

Values:
    EVIDENCE: Claim lacks supporting evidence or citations
    LOGICAL: Missing logical steps or reasoning gaps
    SUPPORT: Insufficient supporting arguments for main claim
    ASSUMPTION: Unstated or unjustified assumptions

---

### `GeminiProvider`

**Language:** python
**Inherits from:** `CLIProvider`
**Defined in:** `model_chorus/src/model_chorus/providers/gemini_provider.py:26`

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
- `check_availability()`
- `build_command()`
- `parse_response()`
- `supports_vision()`
- `supports_thinking()`
- `_configure_cli_environment()`
- `_get_forced_cli_home()`
- `_is_home_writable()`
- `_prepare_cli_home()`
- `_copy_existing_gemini_data()`

---

### `GenerationDefaults`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:13`

**Description:**
> Default generation parameters that apply globally.

These defaults can be overridden by workflow-specific settings or
runtime parameters passed to CLI commands.

---

### `GenerationDefaults`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:21`

**Description:**
> Default generation parameters.

---

### `GenerationDefaultsV2`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:454`

**Description:**
> Default generation parameters for .claude/model_chorus_config.yaml.

---

### `GenerationRequest`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:37`

**Description:**
> Request for text generation.

---

### `GenerationResponse`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:159`

**Description:**
> Response from text generation with standardized structure across providers.

This dataclass provides a unified response format for all AI providers (Claude,
Gemini, OpenAI Codex, etc.), supporting conversation continuation via thread_id,
token usage tracking, and debugging capabilities.

Attributes:
    content: The generated text content from the model.
    model: Model identifier that generated this response (e.g., "claude-3-opus",
        "gemini-pro", "gpt-4").
    usage: Token usage information as a TokenUsage dataclass. Supports both
        attribute access (usage.input_tokens) and dict-like access
        (usage['input_tokens']) for backward compatibility.
    stop_reason: Reason generation stopped (e.g., "end_turn", "max_tokens",
        "stop_sequence"). Provider-specific values, may be None.
    metadata: Provider-specific additional metadata (e.g., safety ratings,
        citations, model version details).
    thread_id: Conversation continuation identifier for multi-turn interactions.
        Provider-specific mapping:
        - Claude: Maps from CLI response 'session_id' field
        - Cursor: Maps from CLI response 'session_id' field
        - Codex (OpenAI): Maps from CLI response 'thread_id' field
        - Gemini: Always None (does not support conversation continuation)
        Used to maintain context across multiple generation requests.
    provider: Name of the provider that generated this response. Valid values:
        "claude", "gemini", "codex", "cursor". Useful for multi-provider
        workflows and debugging.
    stderr: Standard error output captured from CLI-based providers. Contains
        warning messages, debug output, or error details. Empty string if no
        errors, None if not captured. Only populated for CLI providers (Claude,
        Gemini, Codex).
    duration_ms: Request duration in milliseconds, measured from request start
        to response completion. Useful for performance monitoring, latency
        analysis, and cost optimization. None if not measured.
    raw_response: Complete raw response from the provider as returned by the
        CLI or API. Useful for debugging, testing provider-specific features,
        and understanding response structure. May contain sensitive data.
        None if not captured.

Example:
    Basic usage::

        response = GenerationResponse(
            content="Hello, world!",
            model="claude-3-opus-20240229",
            provider="claude"
        )
        response.usage['input_tokens'] = 10
        response.usage['output_tokens'] = 5

    With conversation continuation::

        # First turn
        resp1 = GenerationResponse(
            content="Initial response",
            model="gpt-4",
            thread_id="thread_abc123",
            provider="codex"
        )

        # Follow-up turn using same thread_id
        resp2 = GenerationResponse(
            content="Follow-up response",
            model="gpt-4",
            thread_id="thread_abc123",  # Same ID for continuation
            provider="codex"
        )

---

### `Hypothesis`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:614`

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

### `Idea`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1576`

**Description:**
> Represents a single idea extracted from brainstorming.

Used in the IDEATE workflow to track individual creative ideas
that are extracted from multiple perspective-based brainstorming sessions.

Attributes:
    id: Unique identifier for the idea (e.g., "idea-1")
    label: Brief descriptive label for the idea (1-2 words)
    description: Full description of the idea (1-2 sentences)
    perspective: The perspective this idea originated from (practical, innovative, etc.)
    source_model: Model that generated this idea
    metadata: Additional metadata about the idea

Example:
    >>> idea = Idea(
    ...     id="idea-1",
    ...     label="Gamification System",
    ...     description="Add game mechanics like points and badges to improve engagement",
    ...     perspective="innovative",
    ...     source_model="claude"
    ... )

---

### `IdeaCluster`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1645`

**Description:**
> Represents a themed cluster of related ideas.

Used in the IDEATE workflow to group similar ideas into coherent
themes after convergent analysis. Each cluster represents a distinct
approach or solution category.

Attributes:
    id: Unique identifier for the cluster (e.g., "cluster-1")
    theme: Theme name that describes this cluster
    description: Detailed description of the cluster theme
    idea_ids: List of idea IDs belonging to this cluster
    ideas: Optional list of full Idea objects in this cluster
    scores: Evaluation scores for this cluster (feasibility, impact, etc.)
    overall_score: Average score across all criteria (0.0-5.0)
    recommendation: Priority recommendation (High/Medium/Low Priority)
    metadata: Additional metadata about the cluster

Example:
    >>> cluster = IdeaCluster(
    ...     id="cluster-1",
    ...     theme="User Experience Improvements",
    ...     description="Ideas focused on enhancing user interface and usability",
    ...     idea_ids=["idea-1", "idea-3", "idea-5"],
    ...     scores={"feasibility": 4.5, "impact": 4.0, "novelty": 3.5},
    ...     overall_score=4.0,
    ...     recommendation="High Priority"
    ... )

**Methods:**
- `add_idea()`
- `get_idea_count()`

---

### `IdeateWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `model_chorus/src/model_chorus/workflows/ideate/ideate_workflow.py:28`

**Description:**
> Creative ideation workflow for brainstorming and idea generation.

This workflow implements structured brainstorming through multiple rounds
of creative idea generation, helping users explore diverse solutions and
innovative approaches to problems or challenges.

Architecture:
- Single-model creative generation with high temperature
- Multiple ideation rounds for diverse perspectives
- Conversation threading for iterative refinement
- Focus on quantity and creativity over evaluation

Key Features:
- Multi-round idea generation
- Creative prompting strategies
- Conversation threading via continuation_id
- Inherits conversation support from BaseWorkflow
- Structured brainstorming approach

The IdeateWorkflow is ideal for:
- Brainstorming new features or solutions
- Creative problem-solving
- Exploring innovative approaches
- Generating diverse perspectives on challenges
- Early-stage product ideation

Workflow Pattern:
1. **Initial Ideation**: Generate diverse initial ideas
2. **Expansion**: Explore variations and combinations
3. **Refinement**: Develop promising concepts further

Example:
    >>> from model_chorus.providers import ClaudeProvider
    >>> from model_chorus.workflows import IdeateWorkflow
    >>> from model_chorus.core.conversation import ConversationMemory
    >>>
    >>> # Create provider and conversation memory
    >>> provider = ClaudeProvider()
    >>> memory = ConversationMemory()
    >>>
    >>> # Create workflow
    >>> workflow = IdeateWorkflow(provider, conversation_memory=memory)
    >>>
    >>> # Generate ideas
    >>> result = await workflow.run(
    ...     "How can we improve user onboarding?"
    ... )
    >>> print(result.synthesis)
    >>>
    >>> # Refine specific ideas
    >>> result2 = await workflow.run(
    ...     "Expand on the gamification idea",
    ...     continuation_id=result.metadata.get('thread_id')
    ... )

**Methods:**
- `__init__()`
- `run()`
- `_get_ideation_system_prompt()`
- `_frame_ideation_prompt()`
- `_create_brainstormer_role()`
- `run_parallel_brainstorming()`
- `_synthesize_brainstorming_results()`
- `run_convergent_analysis()`
- `_extract_ideas()`
- `_create_extraction_prompt()`
- `_get_extraction_system_prompt()`
- `_parse_extracted_ideas()`
- `_cluster_ideas()`
- `_create_clustering_prompt()`
- `_get_clustering_system_prompt()`
- `_parse_clusters()`
- `_score_ideas()`
- `_create_scoring_prompt()`
- `_get_scoring_system_prompt()`
- `_parse_scores()`
- `_synthesize_convergent_analysis()`
- `_format_scored_cluster()`
- `run_complete_ideation()`
- `run_interactive_selection()`
- `_display_and_select_clusters()`
- `_parse_selection_input()`
- `_synthesize_selection()`
- `run_elaboration()`
- `_elaborate_cluster()`
- `_create_elaboration_prompt()`
- `_get_elaboration_system_prompt()`
- `_parse_outline_sections()`
- `_synthesize_elaborations()`
- `validate_config()`
- `get_provider()`

---

### `IdeateWorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:98`

**Description:**
> Configuration specific to ideate workflow.

---

### `IdeationState`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:1769`

**Description:**
> Represents the complete state of an ideation workflow session.

Tracks the full lifecycle of an IDEATE workflow execution, from initial
brainstorming through convergent analysis, selection, and elaboration.

Attributes:
    session_id: Unique identifier for this ideation session
    topic: The topic or problem being ideated on
    perspectives: List of perspectives used in brainstorming
    ideas: All extracted ideas from brainstorming
    clusters: Thematic clusters of related ideas
    selected_cluster_ids: IDs of clusters selected for elaboration
    elaborations: Detailed outlines for selected clusters
    scoring_criteria: Criteria used for evaluation (feasibility, impact, etc.)
    workflow_metadata: Metadata about workflow execution
    created_at: Timestamp when ideation session started
    updated_at: Timestamp of last update

Example:
    >>> state = IdeationState(
    ...     session_id="ideation-2024-01-15-001",
    ...     topic="How can we improve our API documentation?",
    ...     perspectives=["practical", "innovative", "user-focused"],
    ...     ideas=[idea1, idea2, idea3],
    ...     clusters=[cluster1, cluster2],
    ...     selected_cluster_ids=["cluster-1"],
    ...     scoring_criteria=["feasibility", "impact", "user_value"]
    ... )

**Methods:**
- `add_idea()`
- `add_cluster()`
- `get_cluster_by_id()`
- `get_selected_clusters()`
- `get_idea_count()`
- `get_cluster_count()`
- `to_dict()`
- `from_dict()`

---

### `InvestigationPhase`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:44`

**Description:**
> Investigation phase enum for persona-based research workflows.

Used in Study workflow to track the current phase of collaborative
investigation. Phases progress from initial discovery through to
completion with systematic exploration and validation.

Values:
    DISCOVERY: Initial exploration phase where personas gather information
    VALIDATION: Critical examination phase where findings are validated
    PLANNING: Synthesis phase where insights are organized into actionable plans
    COMPLETE: Investigation concluded with comprehensive findings

---

### `InvestigationResult`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/config.py:82`

**Description:**
> Result from a study workflow investigation.

Contains all findings, persona contributions, and investigation metadata
from a completed investigation session.

Attributes:
    investigation_id: Unique identifier for this investigation
    final_phase: Final investigation phase reached
    final_confidence: Final confidence level achieved
    iteration_count: Number of iterations completed
    persona_findings: Dict mapping persona names to their findings
    synthesis: Final synthesis of all findings
    relevant_files: Files examined during investigation
    metadata: Additional result metadata

---

### `InvestigationStateMachine`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/state_machine.py:15`

**Description:**
> State machine for managing investigation phase transitions.

Enforces valid phase transitions and provides methods for checking
transition validity and progressing through investigation phases.

Valid transitions:
    DISCOVERY → VALIDATION
    VALIDATION → PLANNING or VALIDATION → DISCOVERY (if more discovery needed)
    PLANNING → COMPLETE or PLANNING → DISCOVERY (if gaps found)
    COMPLETE → (terminal state)

Attributes:
    current_phase: Current investigation phase
    state: StudyState instance being managed

**Methods:**
- `__init__()`
- `can_transition()`
- `transition()`
- `get_next_phase()`
- `get_valid_transitions()`
- `is_terminal()`
- `advance_to_next()`
- `reset_to_discovery()`
- `update_confidence()`
- `should_escalate_phase()`
- `get_confidence_threshold()`

---

### `InvestigationStep`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:659`

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

### `LongTermStorage`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/persistence.py:28`

**Description:**
> SQLite-based persistent storage for memory entries.

Provides durable storage for investigation memory entries with support
for complex queries, historical analysis, and cross-investigation retrieval.

Database Schema:
    investigations: Track investigation metadata
        - investigation_id (PK): Unique investigation identifier
        - created_at: Investigation start timestamp
        - completed_at: Investigation completion timestamp
        - persona_count: Number of personas involved
        - entry_count: Number of memory entries
        - metadata_json: Additional investigation metadata

    memory_entries: Store individual memory entries
        - id (PK): Auto-incrementing entry ID
        - entry_id: Unique entry identifier (for external reference)
        - investigation_id (FK): Link to investigations table
        - session_id: Session identifier
        - timestamp: Entry creation timestamp
        - persona: Persona identifier
        - findings: Main content/findings
        - evidence: Supporting evidence
        - confidence_before: Confidence before this step
        - confidence_after: Confidence after this step
        - memory_type: Type of memory entry
        - metadata_json: Additional entry metadata

    memory_references: Track relationships between entries
        - id (PK): Auto-incrementing reference ID
        - source_entry_id: Entry making the reference
        - target_entry_id: Entry being referenced
        - created_at: Reference creation timestamp

Thread Safety:
    SQLite connections are not shared across threads. Each operation
    creates a new connection which is automatically closed.

Example:
    >>> storage = LongTermStorage("investigations.db")
    >>> storage.initialize()
    >>> storage.save(entry_id, memory_entry)
    >>> results = storage.query(MemoryQuery(persona="researcher"))
    >>> storage.close()

**Methods:**
- `__init__()`
- `initialize()`
- `save()`
- `get()`
- `query()`
- `delete()`
- `get_metadata()`
- `close()`
- `_get_connection()`

---

### `MemoryController`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/controller.py:24`

**Description:**
> Unified controller for memory system operations.

Coordinates between ShortTermCache (fast, volatile) and LongTermStorage
(durable, persistent) to provide optimal performance with durability.

The controller implements a write-through caching strategy:
- Writes: Stored in both cache and persistence
- Reads: Check cache first, fall back to persistence
- Cache misses: Automatically promote to cache for future reads

This ensures:
- Fast access to recent/active investigation data (cache)
- Durable storage of all findings (persistence)
- Automatic recovery after cache eviction (promotion)

Attributes:
    cache: ShortTermCache instance for fast access
    storage: LongTermStorage instance for durability
    write_through: If True, writes go to both cache and storage (default)

Example:
    >>> controller = MemoryController(
    ...     cache_size=100,
    ...     db_path="investigations.db"
    ... )
    >>> controller.initialize()
    >>>
    >>> # Store memory entry (goes to cache + persistence)
    >>> entry_id = controller.store(
    ...     investigation_id="inv-123",
    ...     persona="researcher",
    ...     findings="Found important pattern",
    ...     evidence="Analysis of dataset X"
    ... )
    >>>
    >>> # Retrieve entry (cache-first)
    >>> entry = controller.get(entry_id)
    >>>
    >>> # Query entries (searches both cache and persistence)
    >>> results = controller.query(
    ...     MemoryQuery(investigation_id="inv-123", persona="researcher")
    ... )
    >>>
    >>> controller.close()

**Methods:**
- `__init__()`
- `initialize()`
- `store()`
- `get()`
- `query()`
- `delete()`
- `flush_cache_to_storage()`
- `clear_cache()`
- `get_metadata()`
- `close()`

---

### `MemoryEntry`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/models.py:42`

**Description:**
> Single memory entry in the investigation memory system.

A memory entry captures a discrete piece of information from the investigation,
including what was learned, who learned it (persona), when it was learned,
and how confident they are in the finding.

Memory entries are created during investigation steps and can reference other
memory entries to build a knowledge graph of the investigation.

Attributes:
    investigation_id: Unique identifier for the investigation
    session_id: Unique identifier for the current investigation session
    timestamp: ISO format timestamp of when entry was created
    persona: Persona identifier (e.g., 'researcher', 'critic', 'planner')
    findings: The main content/finding of this memory entry
    evidence: Supporting evidence or context for the finding
    confidence_before: Confidence level before this investigation step
    confidence_after: Confidence level after this investigation step
    memory_references: List of memory entry IDs referenced/related to this one
    memory_type: Type of memory entry for categorization
    metadata: Additional metadata (sources, tags, importance score, etc.)

---

### `MemoryMetadata`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/models.py:149`

**Description:**
> Metadata for memory operations and statistics.

Tracks information about memory system usage, performance,
and health metrics for monitoring and optimization.

Attributes:
    total_entries: Total number of memory entries stored
    cache_entries: Number of entries currently in cache
    persisted_entries: Number of entries persisted to long-term storage
    investigation_count: Number of distinct investigations tracked
    last_cleanup: ISO timestamp of last cleanup/pruning operation
    cache_hit_rate: Percentage of cache hits vs misses
    avg_retrieval_time_ms: Average retrieval time in milliseconds
    storage_size_bytes: Total storage size in bytes (for persistence layer)

---

### `MemoryQuery`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/models.py:231`

**Description:**
> Query model for searching and filtering memory entries.

Supports flexible filtering across multiple dimensions to retrieve
relevant memory entries from cache or persistence storage.

Attributes:
    investigation_id: Filter by specific investigation ID
    persona: Filter by persona identifier
    confidence_level: Filter by minimum confidence level
    time_range_start: Filter by start time (ISO format)
    time_range_end: Filter by end time (ISO format)
    memory_type: Filter by memory entry type
    limit: Maximum number of results to return
    offset: Number of results to skip (for pagination)
    sort_by: Field to sort by (timestamp, confidence_after, etc.)
    sort_order: Sort order (asc or desc)

---

### `MemoryType`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/models.py:19`

**Description:**
> Type of memory entry for categorization and retrieval.

Values:
    FINDING: Research findings or discovered facts
    HYPOTHESIS: Proposed hypotheses or theories
    EVIDENCE: Supporting evidence for hypotheses
    CONCLUSION: Final conclusions from investigation
    QUESTION: Open questions requiring investigation
    REFERENCE: References to external sources or context
    PERSONA_NOTE: Persona-specific observations or notes

---

### `Middleware`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:123`

**Description:**
> Abstract base class for provider middleware.

Middleware wraps a ModelProvider to add cross-cutting concerns like
retries, rate limiting, logging, or circuit breakers. Middleware can
be chained together to compose multiple behaviors.

Example:
    provider = RetryMiddleware(
        LoggingMiddleware(
            ClaudeProvider()
        )
    )

**Methods:**
- `__init__()`
- `generate()`

---

### `MockGenerationRequest`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:30`

**Description:**
> Mock GenerationRequest for testing.

---

### `MockGenerationResponse`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:40`

**Description:**
> Mock GenerationResponse for testing.

**Methods:**
- `__post_init__()`

---

### `MockProvider`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:52`

**Description:**
> Mock provider for testing orchestration.

**Methods:**
- `__init__()`
- `generate()`

---

### `ModelCapability`

**Language:** python
**Inherits from:** `Enum`
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:15`

**Description:**
> Enumeration of model capabilities.

---

### `ModelChorusConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:218`

**Description:**
> Root configuration model for ModelChorus.

This is the top-level configuration object that encompasses all settings
including global defaults, provider configurations, and workflow-specific
settings.

**Methods:**
- `validate_default_provider()`
- `validate_provider_names()`
- `get_provider_config()`
- `get_workflow_config()`
- `is_provider_enabled()`

---

### `ModelChorusConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:80`

**Description:**
> Root configuration model for ModelChorus.

**Methods:**
- `validate_default_provider()`
- `validate_provider_names()`
- `validate_workflow_names()`

---

### `ModelChorusConfigV2`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:506`

**Description:**
> Root configuration for .claude/model_chorus_config.yaml.

**Methods:**
- `validate_provider_names()`
- `validate_workflow_names()`

---

### `ModelConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:26`

**Description:**
> Configuration for a model.

---

### `ModelProvider`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:241`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:308`

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

### `ModelRole`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/role_orchestration.py:95`

**Description:**
> Data class defining a model's role, stance, and prompt customization.

Represents a specific role assignment for an AI model in a multi-model
workflow. Includes the role name, optional stance (for/against/neutral),
and prompt customization to guide the model's behavior.

Used in workflows like ARGUMENT (models with different stances),
IDEATE (models with different creative perspectives), and RESEARCH
(models focusing on different aspects of investigation).

Attributes:
    role: Descriptive name for this role (e.g., "proponent", "critic", "synthesizer")
    model: Model identifier to assign to this role (e.g., "gpt-5", "gemini-2.5-pro")
    stance: Optional stance for debate-style workflows ("for", "against", "neutral")
    stance_prompt: Optional additional prompt text to reinforce the stance
    system_prompt: Optional system-level prompt for this role
    temperature: Optional temperature override for this role (0.0-1.0)
    max_tokens: Optional max tokens override for this role
    metadata: Additional metadata for this role (tags, priority, etc.)

Example:
    >>> proponent = ModelRole(
    ...     role="proponent",
    ...     model="gpt-5",
    ...     stance="for",
    ...     stance_prompt="You are advocating FOR the proposal. Present strong supporting arguments."
    ... )
    >>> critic = ModelRole(
    ...     role="critic",
    ...     model="gemini-2.5-pro",
    ...     stance="against",
    ...     stance_prompt="You are critically analyzing AGAINST the proposal. Identify weaknesses and risks."
    ... )

**Methods:**
- `validate_stance()`
- `validate_temperature()`
- `get_full_prompt()`

---

### `ModelSelection`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:206`

**Description:**
> Model for specifying model selection criteria.

Used to configure which models should be used for specific
workflow steps or roles.

Attributes:
    model_id: The model identifier (e.g., "gpt-4", "claude-3-opus")
    role: Optional role for this model (e.g., "analyzer", "synthesizer")
    config: Optional model-specific configuration

---

### `OrchestrationPattern`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/role_orchestration.py:30`

**Description:**
> Execution patterns for multi-model orchestration.

Defines how multiple models with assigned roles are coordinated
during workflow execution. Different patterns enable different
collaboration strategies.

Values:
    SEQUENTIAL: Execute models one at a time in defined order
               (e.g., analyst → critic → synthesizer)
    PARALLEL: Execute all models concurrently, then aggregate
             (e.g., multiple experts providing simultaneous input)
    HYBRID: Mix of sequential and parallel phases
           (e.g., parallel research → sequential debate → parallel voting)

---

### `OrchestrationResult`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/role_orchestration.py:298`

**Description:**
> Result from orchestrating multiple models with assigned roles.

Contains responses from all models, execution metadata, synthesis output,
and error information. Used by RoleOrchestrator to return structured results
after executing a multi-model workflow.

Attributes:
    role_responses: List of (role_name, response) tuples in execution order
    all_responses: List of all GenerationResponse objects
    failed_roles: List of role names that failed to execute
    pattern_used: Orchestration pattern that was executed
    execution_order: List of role names in the order they were executed
    synthesized_output: Optional synthesized/combined output (if synthesis enabled)
    synthesis_strategy: Strategy used for synthesis (if any)
    metadata: Additional execution metadata (timing, context, synthesis_metadata, etc.)

Example:
    >>> result = OrchestrationResult(
    ...     role_responses=[
    ...         ("proponent", GenerationResponse(content="Argument FOR...", model="gpt-5")),
    ...         ("critic", GenerationResponse(content="Argument AGAINST...", model="gemini-2.5-pro")),
    ...     ],
    ...     pattern_used=OrchestrationPattern.SEQUENTIAL,
    ...     execution_order=["proponent", "critic"],
    ...     synthesized_output="After considering both perspectives...",
    ...     synthesis_strategy=SynthesisStrategy.AI_SYNTHESIZE
    ... )

---

### `OutputFormatter`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/primitives.py:317`

**Description:**
> Standardizes console output formatting for workflow commands.

Extracts common patterns for:
1. Displaying workflow start/continuation messages
2. Showing execution parameters (prompt, provider, files, etc.)
3. Consistent formatting and truncation

**Methods:**
- `display_workflow_start()`
- `write_json_output()`

---

### `Persona`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_base.py:32`

**Description:**
> Represents a persona in the STUDY workflow.

A persona is a specialized investigator with specific expertise and
characteristics that guide its contributions to the research process.

Attributes:
    name: The persona's name (e.g., "Researcher", "Critic")
    prompt_template: Template for prompting this persona
    temperature: Temperature setting for generation (controls randomness)
    max_tokens: Maximum tokens in persona's responses

**Methods:**
- `invoke()`

---

### `PersonaConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/config.py:13`

**Description:**
> Configuration for a single persona in the study workflow.

Attributes:
    name: Persona name (e.g., "Researcher", "Critic")
    expertise: Domain expertise description
    role: Role in investigation (e.g., "primary investigator", "critical reviewer")
    system_prompt: Optional custom system prompt for this persona
    temperature: Optional temperature override for this persona
    metadata: Additional persona-specific metadata

---

### `PersonaRegistry`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_base.py:77`

**Description:**
> Registry for managing available personas in the STUDY workflow.

Provides centralized management of persona definitions, allowing
registration, retrieval, and listing of available personas.

**Methods:**
- `__init__()`
- `register()`
- `get()`
- `list_all()`

---

### `PersonaResponse`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_base.py:13`

**Description:**
> Response from a persona invocation.

Contains the persona's findings and any confidence level updates
based on the investigation.

Attributes:
    findings: List of findings or insights from the persona
    confidence_update: Optional confidence level change based on findings
    metadata: Additional response metadata

---

### `PersonaRouter`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_router.py:85`

**Description:**
> Router for determining which persona to consult next in Study workflow.

Uses context analysis skill to intelligently route to appropriate personas
based on investigation phase, confidence level, findings, and prior consultations.

The router integrates the context analysis logic with the persona registry,
providing a complete routing solution from state analysis to persona retrieval.

Attributes:
    registry: PersonaRegistry containing available personas
    routing_history: List of historical routing decisions for analysis

**Methods:**
- `__init__()`
- `_get_fallback_persona()`
- `route_next_persona()`
- `get_available_personas()`
- `get_routing_history()`
- `clear_routing_history()`

---

### `PlannerPersona`

**Language:** python
**Inherits from:** `Persona`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/planner.py:13`

**Description:**
> Planner persona with actionable roadmap focus.

This persona specializes in:
- Synthesizing findings into coherent plans
- Defining actionable next steps
- Creating structured roadmaps
- Prioritizing actions and recommendations
- Translating insights into practical outcomes

The Planner persona approaches investigations with a solution-oriented mindset,
focusing on turning knowledge into actionable strategies.

**Methods:**
- `__init__()`
- `invoke()`

---

### `ProviderConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:40`

**Description:**
> Configuration for a specific AI provider.

Allows customization of provider-specific settings like model selection,
API endpoints, and provider-specific parameters.

---

### `ProviderConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:30`

**Description:**
> Configuration for a specific provider.

---

### `ProviderConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/consensus.py:41`

**Description:**
> Configuration for a provider in the consensus workflow.

---

### `ProviderConfigV2`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:462`

**Description:**
> Configuration for a provider in .claude/model_chorus_config.yaml.

---

### `ProviderDisabledError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:48`

**Description:**
> Raised when attempting to use a disabled provider.

---

### `ProviderDisabledError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/providers/cli_provider.py:25`

**Description:**
> Raised when attempting to use a disabled provider.

---

### `ProviderError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:29`

**Description:**
> Base exception for provider-related errors.

All provider middleware errors should inherit from this base class
to enable consistent error handling across the middleware stack.

Attributes:
    provider_name: Name of the provider where the error occurred
    original_error: The underlying exception that caused this error (if any)

**Methods:**
- `__init__()`

---

### `ProviderResolver`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/primitives.py:19`

**Description:**
> Handles provider initialization and fallback provider setup.

Extracts the common pattern of:
1. Initializing primary provider with error handling
2. Loading and initializing fallback providers from config
3. Providing helpful error messages for disabled/unavailable providers

**Methods:**
- `__init__()`
- `resolve_provider()`
- `resolve_fallback_providers()`

---

### `ProviderUnavailableError`

**Language:** python
**Inherits from:** `Exception`
**Defined in:** `model_chorus/src/model_chorus/providers/cli_provider.py:31`

**Description:**
> Provider CLI is not available or cannot be used.

**Methods:**
- `__init__()`

---

### `ResearcherPersona`

**Language:** python
**Inherits from:** `Persona`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/researcher.py:13`

**Description:**
> Researcher persona with deep analysis focus.

This persona specializes in:
- Systematic investigation and methodical exploration
- Deep dive analysis of complex topics
- Identifying patterns and connections
- Building comprehensive understanding
- Evidence-based reasoning

The Researcher persona approaches investigations with rigor and thoroughness,
seeking to uncover underlying principles and detailed insights.

**Methods:**
- `__init__()`
- `invoke()`

---

### `RetryConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:169`

**Description:**
> Configuration for retry behavior with exponential backoff.

Attributes:
    max_retries: Maximum number of retry attempts (default: 3)
    base_delay: Initial delay between retries in seconds (default: 1.0)
    max_delay: Maximum delay between retries in seconds (default: 60.0)
    backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        - delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
    jitter: Whether to add random jitter to delays (default: True)
        - Helps prevent thundering herd when multiple requests retry simultaneously
    retryable_exceptions: Tuple of exception types to retry (default: (Exception,))
    permanent_error_patterns: List of error message patterns that indicate
        permanent failures that should not be retried (default: common auth/validation errors)

Examples:
    # Default config: 3 retries with exponential backoff
    config = RetryConfig()

    # Aggressive retries for flaky network
    config = RetryConfig(max_retries=5, base_delay=0.5)

    # Conservative with longer delays
    config = RetryConfig(max_retries=2, base_delay=2.0, max_delay=120.0)

---

### `RetryExhaustedError`

**Language:** python
**Inherits from:** `ProviderError`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:90`

**Description:**
> Raised when all retry attempts have been exhausted.

Contains details about the retry attempts and the final error
that prevented success.

Attributes:
    attempts: Number of attempts made (including initial try)
    last_error: The exception from the final attempt
    provider_name: Name of the provider that failed

**Methods:**
- `__init__()`

---

### `RetryMiddleware`

**Language:** python
**Inherits from:** `Middleware`
**Defined in:** `model_chorus/src/model_chorus/providers/middleware.py:219`

**Description:**
> Middleware that adds retry logic with exponential backoff.

Automatically retries failed generation requests with configurable
exponential backoff. Distinguishes between transient errors (retryable)
and permanent errors (fail immediately).

Example:
    # Use default retry config (3 retries, exponential backoff)
    provider = RetryMiddleware(ClaudeProvider())

    # Custom retry config
    config = RetryConfig(max_retries=5, base_delay=2.0)
    provider = RetryMiddleware(ClaudeProvider(), config)

    # Use like a regular provider
    response = await provider.generate(request)

**Methods:**
- `__init__()`
- `_is_retryable_error()`
- `_calculate_delay()`
- `generate()`

---

### `RoleOrchestrator`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/role_orchestration.py:339`

**Description:**
> Coordinator for executing multiple models with assigned roles.

Manages the execution of multi-model workflows where each model has a specific
role (e.g., proponent, critic, synthesizer). Supports both sequential and parallel
execution patterns. Handles provider resolution, prompt customization, and result
aggregation.

Execution Patterns:
    SEQUENTIAL: Roles execute one at a time in defined order
               (e.g., analyst → critic → synthesizer)
    PARALLEL: All roles execute concurrently for independent perspectives
             (e.g., multiple experts providing simultaneous input)
    HYBRID: Not yet implemented (future work)

This orchestrator enables workflows like:
- ARGUMENT: Sequential debate (roles take turns) or parallel perspectives
- IDEATE: Multiple creative perspectives (parallel for diversity)
- RESEARCH: Multi-angle investigation (parallel for breadth)

Attributes:
    roles: List of ModelRole instances defining the workflow
    provider_map: Mapping from model identifiers to provider instances
    pattern: Orchestration pattern (SEQUENTIAL or PARALLEL)
    default_timeout: Default timeout for each model execution (seconds)

Example:
    >>> from model_chorus.core.role_orchestration import ModelRole, RoleOrchestrator
    >>> from model_chorus.providers import ClaudeProvider, GeminiProvider
    >>>
    >>> # Define roles
    >>> roles = [
    ...     ModelRole(role="proponent", model="claude", stance="for"),
    ...     ModelRole(role="critic", model="gemini", stance="against"),
    ... ]
    >>>
    >>> # Create provider map
    >>> providers = {
    ...     "claude": ClaudeProvider(),
    ...     "gemini": GeminiProvider(),
    ... }
    >>>
    >>> # Create orchestrator
    >>> orchestrator = RoleOrchestrator(roles, providers)
    >>>
    >>> # Execute workflow
    >>> result = await orchestrator.execute("Should we adopt this proposal?")
    >>> for role_name, response in result.role_responses:
    ...     print(f"{role_name}: {response.content}")

**Methods:**
- `__init__()`
- `_resolve_provider()`
- `execute()`
- `_execute_sequential()`
- `_execute_parallel()`
- `synthesize()`
- `_build_synthesis_prompt()`

---

### `RoutingDecision`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_router.py:25`

**Description:**
> Result of persona routing decision.

Contains the selected persona instance, reasoning for selection,
and guidance for the persona invocation.

Attributes:
    persona: The Persona instance to consult next (None if investigation complete)
    persona_name: Name of the selected persona
    reasoning: Explanation for why this persona was selected
    confidence: Current confidence level from investigation state
    guidance: Specific guidance or focus areas for the persona
    context_summary: Summary of the investigation context
    metadata: Additional routing metadata
    timestamp: When this routing decision was made (ISO format)

---

### `RoutingHistoryEntry`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/persona_router.py:54`

**Description:**
> Historical record of a routing decision.

Tracks routing decisions over time for analysis and debugging.

Attributes:
    timestamp: When the routing decision was made (ISO format)
    investigation_id: Investigation this routing was for
    phase: Investigation phase at time of routing
    confidence: Confidence level at time of routing
    findings_count: Number of findings at time of routing
    questions_count: Number of unresolved questions
    prior_persona: Previously consulted persona (if any)
    selected_persona: Persona selected by routing decision
    reasoning: Reasoning for the selection
    context_summary: Summary of investigation context

---

### `SemanticClustering`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/clustering.py:54`

**Description:**
> Semantic clustering engine for grouping textual content by theme.

This class provides methods for:
1. Computing semantic similarity between texts using embeddings
2. Clustering texts using K-means or hierarchical methods
3. Naming and summarizing clusters
4. Scoring cluster quality

Example:
    >>> clustering = SemanticClustering(model_name="all-MiniLM-L6-v2")
    >>> texts = ["Python is great", "I love Python", "Java is verbose"]
    >>> clusters = clustering.cluster(texts, n_clusters=2)
    >>> for cluster in clusters:
    ...     print(f"{cluster.name}: {cluster.items}")

**Methods:**
- `__init__()`
- `_load_model()`
- `compute_embeddings()`
- `compute_similarity()`
- `cluster_kmeans()`
- `cluster_hierarchical()`
- `name_cluster()`
- `summarize_cluster()`
- `score_cluster()`
- `cluster()`

---

### `ShortTermCache`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/memory/cache.py:23`

**Description:**
> LRU-based in-memory cache for memory entries.

Provides fast access to recent memory entries with automatic eviction
of least recently used entries when the cache reaches capacity.

The cache uses OrderedDict to maintain insertion/access order and
implements LRU eviction by moving accessed items to the end and
removing items from the beginning when full.

Thread Safety:
    All operations are protected by a threading lock to ensure
    safe concurrent access from multiple personas/threads.

Attributes:
    max_size: Maximum number of entries to cache (default 100)
    cache: OrderedDict storing entry_id -> MemoryEntry mappings
    stats: Statistics tracking cache performance
    lock: Threading lock for thread-safe operations

Example:
    >>> cache = ShortTermCache(max_size=50)
    >>> cache.put(entry_id, memory_entry)
    >>> entry = cache.get(entry_id)
    >>> results = cache.query(MemoryQuery(persona="researcher"))

**Methods:**
- `__init__()`
- `get()`
- `put()`
- `delete()`
- `query()`
- `_matches_query()`
- `clear()`
- `size()`
- `get_metadata()`
- `get_stats()`

---

### `Source`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:2018`

**Description:**
> Represents a research source with metadata and validation.

Used in the RESEARCH workflow to track sources that provide evidence
for research findings. Each source includes credibility assessment,
type classification, and validation status.

Attributes:
    source_id: Unique identifier for this source
    title: Title or description of the source
    url: Optional URL or reference to the source
    source_type: Type of source (article, paper, book, website, etc.)
    credibility: Credibility assessment (high, medium, low, unassessed)
    tags: Optional list of tags for categorization
    validated: Whether source has been validated
    validation_score: Numeric validation score if validated
    validation_notes: List of validation findings
    ingested_at: ISO timestamp when source was added
    metadata: Additional source metadata

Example:
    >>> source = Source(
    ...     source_id="src-001",
    ...     title="Machine Learning Best Practices",
    ...     url="https://example.com/ml-practices",
    ...     source_type="article",
    ...     credibility="high",
    ...     tags=["machine-learning", "best-practices"]
    ... )

---

### `StateManager`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/state.py:30`

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

### `StudyConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/config.py:35`

**Description:**
> Configuration for STUDY workflow execution.

Defines default settings and behavior for persona-based investigations,
including persona configurations, iteration limits, and confidence thresholds.

Attributes:
    personas: List of persona configurations to use
    max_iterations: Maximum number of investigation iterations
    min_confidence_for_completion: Minimum confidence level to complete investigation
    enable_cross_persona_dialogue: Whether personas should interact with each other
    temperature: Default temperature for generation (can be overridden per persona)
    max_tokens: Maximum tokens per persona response
    enable_file_analysis: Whether to enable file examination capabilities
    collaboration_rounds: Number of rounds of cross-persona collaboration
    metadata: Additional workflow-specific metadata

---

### `StudyState`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:788`

**Description:**
> State model for Study workflow multi-persona investigations.

Maintains the complete investigation state across conversation turns,
tracking investigation phase, persona findings, confidence levels,
and collaborative exploration progress.

Attributes:
    investigation_id: Unique identifier for this investigation session
    session_id: Session/thread identifier for conversation continuity
    current_phase: Current phase of investigation (InvestigationPhase)
    confidence: Overall confidence level in findings (ConfidenceLevel)
    iteration_count: Number of investigation iterations completed
    findings: List of all findings from persona investigations
    personas_active: List of personas currently participating
    relevant_files: Files examined during investigation

---

### `StudyWorkflow`

**Language:** python
**Inherits from:** `BaseWorkflow`
**Defined in:** `model_chorus/src/model_chorus/workflows/study/study_workflow.py:26`

**Description:**
> Persona-based collaborative research workflow.

This workflow implements systematic investigation through multiple personas
with distinct expertise, enabling collaborative exploration of complex topics
through role-based orchestration and conversation threading.

Architecture:
- Multi-persona role orchestration with intelligent routing
- PersonaRouter for context-aware persona selection
- Conversation threading and memory
- Systematic hypothesis exploration
- Persona-specific expertise and perspectives

Key Features:
- Intelligent persona routing based on investigation phase and context
- Fallback routing for graceful error handling
- Role-based investigation with distinct personas
- Conversation threading for multi-turn exploration
- Systematic knowledge building
- Collaborative analysis and synthesis
- Inherits conversation support from BaseWorkflow

The StudyWorkflow is ideal for:
- Complex topic exploration requiring multiple perspectives
- Research investigations with specialized knowledge domains
- Collaborative problem analysis
- Systematic learning and knowledge building
- Multi-faceted topic investigation

Workflow Pattern:
1. **Persona Assignment**: Intelligently route to appropriate persona via PersonaRouter
2. **Collaborative Exploration**: Personas investigate from their perspectives
3. **Synthesis**: Combine insights into comprehensive understanding

Attributes:
    provider: Primary ModelProvider for persona invocations
    fallback_providers: Optional fallback providers
    persona_router: PersonaRouter instance for intelligent persona selection

Example:
    >>> from model_chorus.providers import ClaudeProvider
    >>> from model_chorus.workflows.study import StudyWorkflow
    >>> from model_chorus.core.conversation import ConversationMemory
    >>>
    >>> # Create provider and conversation memory
    >>> provider = ClaudeProvider()
    >>> memory = ConversationMemory()
    >>>
    >>> # Create workflow
    >>> workflow = StudyWorkflow(provider, conversation_memory=memory)
    >>>
    >>> # Conduct research
    >>> result = await workflow.run(
    ...     "Explore the implementation patterns of authentication systems"
    ... )
    >>> print(result.synthesis)
    >>>
    >>> # Continue investigation
    >>> result2 = await workflow.run(
    ...     "Dive deeper into OAuth 2.0 flow patterns",
    ...     continuation_id=result.metadata.get('thread_id')
    ... )

**Methods:**
- `__init__()`
- `run()`
- `_setup_personas()`
- `_conduct_investigation()`
- `_synthesize_findings()`
- `get_routing_history()`

---

### `SynthesisStrategy`

**Language:** python
**Inherits from:** `str`, `Enum`
**Defined in:** `model_chorus/src/model_chorus/core/role_orchestration.py:52`

**Description:**
> Strategies for combining multiple role outputs into a unified result.

After roles execute (sequentially or in parallel), their outputs can be
synthesized using different strategies depending on workflow needs.

Values:
    NONE: No synthesis - return raw responses as-is
         Use when: You want to process each response individually

    CONCATENATE: Simple concatenation with role labels
                Use when: You want a readable combined text with clear attribution

    AI_SYNTHESIZE: Use AI model to intelligently combine responses
                  Use when: You want a coherent synthesis that resolves conflicts
                           and integrates multiple perspectives

    STRUCTURED: Combine into structured format (dict with role keys)
               Use when: You need programmatic access to individual responses
                        while keeping them organized

Example:
    >>> # Get raw responses
    >>> result = await orchestrator.execute(prompt, synthesis=SynthesisStrategy.NONE)
    >>>
    >>> # Get simple concatenation
    >>> result = await orchestrator.execute(prompt, synthesis=SynthesisStrategy.CONCATENATE)
    >>>
    >>> # Get AI-synthesized output
    >>> result = await orchestrator.execute(
    ...     prompt,
    ...     synthesis=SynthesisStrategy.AI_SYNTHESIZE,
    ...     synthesis_provider=synthesis_model
    ... )

---

### `TestArchitecturalDecisionScenarios`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_complex.py:23`

**Description:**
> Test suite for architectural decision making scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_architectural_decision_rest_vs_graphql()`
- `test_architectural_decision_database_selection()`

---

### `TestArgumentCommand`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:74`

**Description:**
> Test suite for 'argument' CLI command.

**Methods:**
- `test_argument_basic_invocation()`
- `test_argument_with_provider_option()`
- `test_argument_with_continuation()`
- `test_argument_with_file()`
- `test_argument_with_nonexistent_file()`
- `test_argument_with_temperature()`
- `test_argument_with_output_file()`
- `test_argument_verbose_mode()`
- `test_argument_invalid_provider()`

---

### `TestArgumentMapGeneration`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:113`

**Description:**
> Test ArgumentMap generation.

**Methods:**
- `test_generate_argument_map()`
- `test_argument_map_creator_perspective()`
- `test_argument_map_skeptic_perspective()`
- `test_argument_map_moderator_perspective()`

---

### `TestArgumentWorkflowExecution`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:243`

**Description:**
> Test ArgumentWorkflow execution.

**Methods:**
- `test_workflow_execution_with_mocked_orchestrator()`
- `test_workflow_metadata()`

---

### `TestArgumentWorkflowInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:34`

**Description:**
> Test ArgumentWorkflow initialization.

**Methods:**
- `test_initialization_with_provider()`
- `test_initialization_without_memory()`
- `test_initialization_without_provider_raises_error()`
- `test_validate_config()`
- `test_get_provider()`

---

### `TestBasicIdeation`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:185`

**Description:**
> Test basic ideation methods.

**Methods:**
- `test_run_basic_ideation()`
- `test_run_with_empty_prompt_raises_error()`
- `test_run_with_continuation_id()`
- `test_run_with_custom_temperature()`
- `test_run_updates_conversation_memory()`

---

### `TestBatchContradictionDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:353`

**Description:**
> Test batch contradiction detection.

**Methods:**
- `test_detect_contradictions_in_batch()`
- `test_batch_no_contradictions()`

---

### `TestBatchSimilarity`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:255`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_complex.py:278`

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
**Defined in:** `model_chorus/tests/test_providers/test_cli_interface.py:22`

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

### `TestCanReadFile`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:216`

**Description:**
> Test file readability checks.

**Methods:**
- `test_can_read_small_file()`
- `test_can_read_large_file()`
- `test_can_read_nonexistent_file()`

---

### `TestChatErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:265`

**Description:**
> Test error handling in chat workflow.

**Methods:**
- `test_invalid_continuation_id()`
- `test_empty_prompt()`
- `test_very_long_conversation()`

---

### `TestChatThreadManagement`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:324`

**Description:**
> Test conversation thread management.

**Methods:**
- `test_multiple_concurrent_threads()`
- `test_thread_retrieval()`

---

### `TestChatWorkflowInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_workflow.py:25`

**Description:**
> Test ChatWorkflow initialization.

**Methods:**
- `test_initialization_with_provider()`
- `test_initialization_without_memory()`
- `test_initialization_without_provider_raises_error()`
- `test_validate_config()`
- `test_get_provider()`

---

### `TestCircuitBreakerConfig`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:381`

**Description:**
> Test CircuitBreakerConfig dataclass.

**Methods:**
- `test_default_config()`
- `test_custom_config()`

---

### `TestCircuitBreakerMiddleware`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:414`

**Description:**
> Test CircuitBreakerMiddleware implementation.

**Methods:**
- `test_initialization_default_config()`
- `test_initialization_custom_config()`
- `test_initialization_with_callback()`
- `test_generate_success_in_closed_state()`
- `test_failure_count_increases_on_error()`
- `test_circuit_opens_after_threshold()`
- `test_circuit_open_rejects_requests()`
- `test_circuit_transitions_to_half_open()`
- `test_half_open_closes_after_success_threshold()`
- `test_half_open_reopens_on_failure()`
- `test_excluded_exceptions_not_counted()`
- `test_state_change_callback_invoked()`
- `test_success_resets_failure_count_in_closed()`
- `test_is_monitored_exception()`

---

### `TestCitation`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation.py:20`

**Description:**
> Test suite for Citation model.

**Methods:**
- `test_citation_creation()`
- `test_citation_minimal_creation()`
- `test_citation_empty_source()`
- `test_citation_confidence_bounds()`
- `test_citation_various_source_types()`
- `test_citation_location_formats()`
- `test_citation_metadata_flexibility()`
- `test_citation_serialization()`
- `test_citation_json_serialization()`
- `test_citation_from_dict()`
- `test_citation_json_roundtrip()`

---

### `TestCitationConfidenceScoring`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:358`

**Description:**
> Test citation confidence calculation.

**Methods:**
- `test_calculate_confidence_complete_citation()`
- `test_calculate_confidence_minimal_citation()`
- `test_confidence_academic_source_bonus()`
- `test_confidence_doi_source_bonus()`
- `test_confidence_https_vs_http()`
- `test_confidence_location_specificity_bonus()`
- `test_confidence_weighted_formula()`

---

### `TestCitationFormatting`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:108`

**Description:**
> Test citation formatting in different styles.

**Methods:**
- `test_format_apa_complete()`
- `test_format_apa_minimal()`
- `test_format_mla_complete()`
- `test_format_mla_minimal()`
- `test_format_chicago_complete()`
- `test_format_chicago_minimal()`
- `test_format_file_citation_apa()`
- `test_format_doi_citation_apa()`
- `test_format_unsupported_style_raises_error()`
- `test_year_extraction_from_full_date()`

---

### `TestCitationIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation.py:460`

**Description:**
> Test integration scenarios for citation tracking.

**Methods:**
- `test_claim_evidence_mapping()`
- `test_multiple_claims_same_source()`
- `test_citation_strength_calculation()`
- `test_citation_filtering_by_confidence()`
- `test_argument_workflow_citation_tracking()`

---

### `TestCitationIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:311`

**Description:**
> Test integration with Citation model.

**Methods:**
- `test_add_similarity_to_citation_with_snippet()`
- `test_add_similarity_to_citation_without_snippet()`
- `test_add_similarity_preserves_existing_metadata()`

---

### `TestCitationMap`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation.py:210`

**Description:**
> Test suite for CitationMap model.

**Methods:**
- `test_citation_map_creation()`
- `test_citation_map_minimal_creation()`
- `test_citation_map_empty_claim_id()`
- `test_citation_map_empty_claim_text()`
- `test_citation_map_strength_bounds()`
- `test_citation_map_single_citation()`
- `test_citation_map_multiple_citations()`
- `test_citation_map_metadata_flexibility()`
- `test_citation_map_serialization()`
- `test_citation_map_json_serialization()`
- `test_citation_map_json_roundtrip()`
- `test_citation_map_nested_validation()`

---

### `TestCitationMapConfidenceScoring`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:474`

**Description:**
> Test CitationMap confidence calculation.

**Methods:**
- `test_calculate_map_confidence_complete()`
- `test_calculate_map_confidence_empty()`
- `test_calculate_map_confidence_formula()`
- `test_calculate_map_confidence_count_plateau()`

---

### `TestCitationMapFormatting`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:200`

**Description:**
> Test CitationMap formatting.

**Methods:**
- `test_format_citation_map_with_claim()`
- `test_format_citation_map_without_claim()`
- `test_format_citation_map_mla_style()`
- `test_format_citation_map_empty()`

---

### `TestCitationStyleEnum`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:541`

**Description:**
> Test CitationStyle enum.

**Methods:**
- `test_citation_style_values()`
- `test_citation_style_string_comparison()`

---

### `TestCitationValidation`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:253`

**Description:**
> Test citation validation logic.

**Methods:**
- `test_validate_complete_citation_passes()`
- `test_validate_minimal_citation_has_recommendations()`
- `test_validate_empty_source_fails()`
- `test_validate_whitespace_source_fails()`
- `test_validate_confidence_out_of_range_low()`
- `test_validate_confidence_out_of_range_high()`
- `test_validate_recognized_source_formats()`
- `test_validate_unrecognized_source_format()`

---

### `TestClaimSimilarity`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:123`

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
**Defined in:** `model_chorus/tests/test_claude_provider.py:14`

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
- `test_read_only_mode_allowed_tools()`
- `test_read_only_mode_disallowed_tools()`

---

### `TestClusterRepresentative`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:670`

**Description:**
> Test cluster representative selection.

**Methods:**
- `test_get_cluster_representative_basic()`
- `test_get_cluster_representative_single_item()`
- `test_get_cluster_representative_empty_cluster()`

---

### `TestClusterResult`

**Language:** python
**Defined in:** `model_chorus/tests/test_clustering.py:35`

**Description:**
> Test suite for ClusterResult dataclass.

**Methods:**
- `test_cluster_result_creation()`
- `test_cluster_result_repr()`

---

### `TestClusterStatistics`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:722`

**Description:**
> Test cluster statistics computation.

**Methods:**
- `test_compute_cluster_statistics_basic()`
- `test_compute_cluster_statistics_empty()`
- `test_compute_cluster_statistics_single_item_clusters()`

---

### `TestClusteringIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_clustering.py:455`

**Description:**
> Integration tests that verify clustering with real sentence-transformers (if available).

**Methods:**
- `test_real_clustering_with_semantics()`

---

### `TestClusteringIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:803`

**Description:**
> Test end-to-end clustering workflows.

**Methods:**
- `test_kmeans_to_statistics_pipeline()`
- `test_hierarchical_to_statistics_pipeline()`

---

### `TestCodexProvider`

**Language:** python
**Defined in:** `model_chorus/tests/test_codex_provider.py:13`

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
- `test_read_only_sandbox_mode()`
- `test_non_interactive_approval_mode()`

---

### `TestCommandIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:732`

**Description:**
> Test suite for cross-command integration.

**Methods:**
- `test_all_commands_available()`
- `test_help_shows_all_commands()`
- `test_common_options_work_across_commands()`

---

### `TestCompleteIdeation`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:352`

**Description:**
> Test complete ideation workflow (divergent + convergent).

**Methods:**
- `test_run_complete_ideation()`
- `test_complete_ideation_with_custom_parameters()`

---

### `TestCompleteIdeation`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:713`

**Description:**
> Test complete ideation workflow.

**Methods:**
- `test_run_complete_ideation()`
- `test_complete_ideation_empty_prompt_raises_error()`
- `test_complete_ideation_empty_provider_map_raises_error()`

---

### `TestComplexConfigScenarios`

**Language:** python
**Defined in:** `model_chorus/tests/test_config.py:448`

**Description:**
> Test suite for complex real-world configuration scenarios.

**Methods:**
- `test_full_config_all_workflows()`
- `test_precedence_workflow_overrides_global()`

---

### `TestComplexMultiStepReasoning`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_complex.py:675`

**Description:**
> Test suite for complex multi-step reasoning scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_long_investigation_with_hypothesis_pivots()`
- `test_investigation_with_multiple_evidence_types()`

---

### `TestComprehensiveGapDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:361`

**Description:**
> Test comprehensive gap detection combining all types.

**Methods:**
- `test_detect_multiple_gap_types()`
- `test_detect_no_gaps_complete_argument()`
- `test_gap_detection_with_mixed_quality()`

---

### `TestConcurrentConversationHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_concurrent_conversations.py:24`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_models.py:26`

**Description:**
> Test suite for ConfidenceLevel enum.

**Methods:**
- `test_confidence_level_values()`
- `test_confidence_level_count()`
- `test_confidence_level_progression()`
- `test_confidence_level_string_representation()`

---

### `TestConfidenceLevelProgression`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_state_machine.py:266`

**Description:**
> Test suite for confidence level progression logic.

**Methods:**
- `state_machine()`
- `test_update_confidence()`
- `test_should_escalate_from_discovery_low_confidence()`
- `test_should_escalate_from_discovery_medium_confidence()`
- `test_should_escalate_from_discovery_high_confidence()`
- `test_should_escalate_from_validation_high_confidence()`
- `test_should_escalate_from_validation_medium_confidence()`
- `test_should_escalate_from_planning()`
- `test_should_escalate_from_complete()`
- `test_confidence_threshold_discovery()`
- `test_confidence_threshold_validation()`
- `test_confidence_threshold_planning()`
- `test_confidence_threshold_complete()`
- `test_confidence_threshold_current_phase()`

---

### `TestConfidenceProgression`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_workflow.py:1357`

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

### `TestConfigError`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:91`

**Description:**
> Test ConfigError exception.

**Methods:**
- `test_basic_initialization()`
- `test_full_initialization()`

---

### `TestConfigLoader`

**Language:** python
**Defined in:** `model_chorus/tests/test_config.py:25`

**Description:**
> Test suite for ConfigLoader class.

**Methods:**
- `test_find_config_file_in_current_dir()`
- `test_find_config_file_in_parent_dir()`
- `test_find_config_file_yaml_extension()`
- `test_find_config_file_json_extension()`
- `test_find_config_file_not_found()`
- `test_load_yaml_config_global_defaults()`
- `test_load_yaml_config_workflow_specific()`
- `test_load_json_config()`
- `test_invalid_provider_raises_error()`
- `test_invalid_temperature_raises_error()`
- `test_invalid_workflow_name_raises_error()`
- `test_invalid_consensus_strategy_raises_error()`
- `test_get_default_provider_workflow_specific()`
- `test_get_default_provider_fallback()`
- `test_get_workflow_default_temperature()`
- `test_get_default_providers_multi()`
- `test_load_empty_config()`
- `test_load_minimal_config()`
- `test_generation_defaults_validation()`
- `test_workflow_config_validation()`
- `test_workflow_config_provider_normalization()`
- `test_model_chorus_config_complete()`
- `test_get_config_loader_singleton()`
- `test_load_nonexistent_file_raises_error()`
- `test_load_invalid_yaml_raises_error()`
- `test_load_invalid_json_raises_error()`

---

### `TestConsensusThinkDeepChatChaining`

**Language:** python
**Defined in:** `model_chorus/tests/test_workflow_integration_chaining.py:22`

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
**Defined in:** `model_chorus/tests/test_consensus_workflow.py:13`

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

### `TestContextIngestionServiceInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:22`

**Description:**
> Test service initialization and configuration.

**Methods:**
- `test_default_initialization()`
- `test_custom_initialization()`
- `test_initialization_validation_max_size_zero()`
- `test_initialization_validation_max_size_negative()`
- `test_initialization_validation_warn_exceeds_max()`

---

### `TestContradictionDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:291`

**Description:**
> Test end-to-end contradiction detection.

**Methods:**
- `test_detect_clear_contradiction()`
- `test_no_contradiction_unrelated_claims()`
- `test_no_contradiction_similar_polarity()`

---

### `TestContradictionExplanation`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:121`

**Description:**
> Test contradiction explanation generation.

**Methods:**
- `test_explanation_includes_polarity_info()`
- `test_explanation_severity_critical()`
- `test_explanation_severity_moderate()`
- `test_explanation_severity_minor()`

---

### `TestContradictionModel`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:205`

**Description:**
> Test Contradiction Pydantic model.

**Methods:**
- `test_valid_contradiction_creation()`
- `test_confidence_validation_in_range()`
- `test_confidence_validation_too_high()`
- `test_confidence_validation_too_low()`
- `test_different_claim_ids_validation()`

---

### `TestConvergentAnalysis`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:191`

**Description:**
> Test convergent analysis functionality.

**Methods:**
- `test_convergent_analysis_with_brainstorming_result()`
- `test_convergent_analysis_metadata()`
- `test_convergent_analysis_raises_error_on_empty_result()`
- `test_convergent_analysis_custom_criteria()`

---

### `TestConvergentAnalysis`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:354`

**Description:**
> Test convergent analysis functionality.

**Methods:**
- `mock_brainstorming_result()`
- `test_run_convergent_analysis()`
- `test_convergent_analysis_without_brainstorming_raises_error()`
- `test_idea_extraction()`
- `test_idea_clustering()`
- `test_idea_scoring()`

---

### `TestConversationContinuation`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_workflow.py:120`

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
**Defined in:** `model_chorus/tests/test_chat_workflow.py:64`

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
**Defined in:** `model_chorus/tests/test_conversation.py:18`

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

### `TestConversationThreading`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:375`

**Description:**
> Test conversation threading and continuation.

**Methods:**
- `test_new_conversation_creates_thread()`
- `test_continuation_uses_existing_thread()`

---

### `TestConversationTracking`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_workflow.py:291`

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
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:81`

**Description:**
> Test cosine similarity computation.

**Methods:**
- `test_cosine_similarity_identical()`
- `test_cosine_similarity_range()`
- `test_cosine_similarity_similar_text()`
- `test_cosine_similarity_different_text()`

---

### `TestCriticPersona`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:320`

**Description:**
> Test suite for CriticPersona.

**Methods:**
- `test_critic_init()`
- `test_critic_init_custom_temperature()`
- `test_critic_prompt_template()`
- `test_critic_invoke_returns_response()`
- `test_critic_invoke_includes_findings()`
- `test_critic_invoke_metadata()`
- `test_create_critic_factory()`
- `test_create_critic_factory_with_params()`

---

### `TestCursorAgentProvider`

**Language:** python
**Defined in:** `model_chorus/tests/test_cursor_agent_provider.py:14`

**Description:**
> Test suite for CursorAgentProvider.

**Methods:**
- `test_initialization()`
- `test_build_command_basic()`
- `test_build_command_with_model()`
- `test_build_command_with_system_prompt()`
- `test_build_command_without_system_prompt()`
- `test_parse_response_success()`
- `test_parse_response_failure()`
- `test_parse_response_invalid_json()`
- `test_parse_response_error_result()`
- `test_parse_response_with_session_id()`
- `test_generate_success()`
- `test_generate_with_retry()`
- `test_generate_all_retries_fail()`
- `test_supports_vision()`
- `test_supports_code_generation()`
- `test_read_only_mode_by_default()`

---

### `TestCustomSizeLimits`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:454`

**Description:**
> Test service with custom size limits.

**Methods:**
- `test_custom_max_size_allows_larger_files()`
- `test_custom_warn_threshold()`

---

### `TestDuplicateDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:361`

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
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:472`

**Description:**
> Test edge cases and error handling.

**Methods:**
- `test_empty_claim_text()`
- `test_very_long_claim()`
- `test_special_characters()`
- `test_unicode_characters()`

---

### `TestElaboration`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:612`

**Description:**
> Test elaboration functionality.

**Methods:**
- `mock_selection_result()`
- `test_run_elaboration()`
- `test_elaboration_without_selection_raises_error()`
- `test_elaborate_cluster()`
- `test_parse_outline_sections()`

---

### `TestEmbeddingComputation`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:27`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_workflow.py:1915`

**Description:**
> End-to-end integration tests for complete investigation scenarios.

**Methods:**
- `mock_provider()`
- `conversation_memory()`
- `test_five_step_investigation_with_hypothesis_evolution()`
- `test_complete_investigation_workflow()`

---

### `TestEnums`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:752`

**Description:**
> Test suite for enum classes.

**Methods:**
- `test_orchestration_pattern_values()`
- `test_synthesis_strategy_values()`

---

### `TestErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:455`

**Description:**
> Test error handling.

**Methods:**
- `test_workflow_handles_orchestration_failure()`
- `test_workflow_handles_insufficient_responses()`

---

### `TestErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_workflow.py:336`

**Description:**
> Test error handling in ChatWorkflow.

**Methods:**
- `test_provider_error_handled()`
- `test_get_result_returns_last_result()`

---

### `TestErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:656`

**Description:**
> Test suite for error handling and edge cases.

**Methods:**
- `test_empty_prompt()`
- `test_workflow_failure()`
- `test_keyboard_interrupt()`
- `test_very_long_prompt()`
- `test_special_characters_in_prompt()`

---

### `TestErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:786`

**Description:**
> Test error handling and edge cases.

**Methods:**
- `test_provider_generation_failure()`
- `test_extraction_with_no_brainstorming_steps()`
- `test_clustering_with_no_extracted_ideas()`
- `test_scoring_with_no_clusters()`

---

### `TestExpertProviderIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_expert_validation.py:24`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_expert_validation.py:492`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_expert_validation.py:354`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_expert_validation.py:114`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_expert_validation.py:672`

**Description:**
> Test suite for expert validation interaction with hypotheses.

**Methods:**
- `mock_provider()`
- `mock_expert_provider()`
- `conversation_memory()`
- `test_expert_validation_validates_hypothesis()`
- `test_expert_validation_with_multiple_hypotheses()`

---

### `TestFallbackRouting`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_routing.py:159`

**Description:**
> Test fallback routing when context analysis fails.

**Methods:**
- `test_fallback_routing_on_exception()`
- `test_fallback_for_all_phases()`
- `test_fallback_provides_valid_guidance()`

---

### `TestFileContext`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_workflow.py:202`

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
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:163`

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

### `TestGapModel`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:23`

**Description:**
> Test Gap Pydantic model.

**Methods:**
- `test_valid_gap_creation()`
- `test_confidence_validation_in_range()`
- `test_confidence_validation_too_high()`
- `test_confidence_validation_too_low()`

---

### `TestGapRecommendations`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:158`

**Description:**
> Test gap recommendation generation.

**Methods:**
- `test_evidence_recommendation_critical()`
- `test_evidence_recommendation_minor()`
- `test_logical_recommendation()`
- `test_support_recommendation()`
- `test_assumption_recommendation()`

---

### `TestGeminiIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_gemini_integration.py:23`

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

### `TestGetFileInfo`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:171`

**Description:**
> Test file metadata retrieval.

**Methods:**
- `test_get_file_info_small_file()`
- `test_get_file_info_large_file()`
- `test_get_file_info_nonexistent_file()`

---

### `TestHierarchicalClustering`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:604`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_models.py:75`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_workflow.py:782`

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

### `TestIdeaClustering`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:293`

**Description:**
> Test idea clustering functionality.

**Methods:**
- `test_cluster_ideas()`

---

### `TestIdeaExtraction`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:258`

**Description:**
> Test idea extraction from brainstorming results.

**Methods:**
- `test_extract_ideas_from_brainstorming()`
- `test_extract_ideas_preserves_perspectives()`

---

### `TestIdeaScoring`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:320`

**Description:**
> Test idea scoring functionality.

**Methods:**
- `test_score_ideas()`

---

### `TestIdeateCommand`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:304`

**Description:**
> Test suite for 'ideate' CLI command.

**Methods:**
- `test_ideate_basic_invocation()`
- `test_ideate_with_num_ideas()`
- `test_ideate_with_high_temperature()`
- `test_ideate_with_continuation()`
- `test_ideate_with_files()`
- `test_ideate_with_output()`

---

### `TestIdeateWorkflowInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:166`

**Description:**
> Test IdeateWorkflow initialization.

**Methods:**
- `test_initialization_with_provider()`
- `test_initialization_without_provider_raises_error()`
- `test_validate_config()`
- `test_get_provider()`

---

### `TestIdeateWorkflowInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:148`

**Description:**
> Test IdeateWorkflow initialization.

**Methods:**
- `test_initialization_with_provider()`
- `test_initialization_without_memory()`
- `test_initialization_without_provider_raises_error()`
- `test_validate_config()`
- `test_get_provider()`

---

### `TestIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_integration.py:18`

**Description:**
> Integration test suite.

**Methods:**
- `test_end_to_end_consensus()`
- `test_provider_initialization_and_generation()`
- `test_error_handling_across_workflow()`
- `test_multiple_strategy_comparison()`
- `test_concurrent_provider_execution()`

---

### `TestInteractiveSelection`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:522`

**Description:**
> Test interactive selection functionality.

**Methods:**
- `mock_convergent_result()`
- `test_parse_selection_input_single_number()`
- `test_parse_selection_input_comma_separated()`
- `test_parse_selection_input_range()`
- `test_parse_selection_input_all()`
- `test_parse_selection_input_none()`
- `test_parse_selection_input_invalid_range()`
- `test_parse_selection_input_with_max_selections()`

---

### `TestInvestigationStateMachineTransitions`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_state_machine.py:14`

**Description:**
> Test suite for state machine phase transitions.

**Methods:**
- `initial_state()`
- `state_machine()`
- `test_discovery_to_validation_transition()`
- `test_validation_to_planning_transition()`
- `test_planning_to_complete_transition()`
- `test_validation_back_to_discovery()`
- `test_planning_back_to_discovery()`
- `test_invalid_discovery_to_planning()`
- `test_invalid_discovery_to_complete()`
- `test_invalid_transition_from_complete()`
- `test_get_next_phase()`
- `test_get_valid_transitions()`
- `test_advance_to_next()`
- `test_reset_to_discovery()`
- `test_reset_from_complete_raises_error()`
- `test_reset_from_discovery_is_noop()`
- `test_transition_with_reason()`

---

### `TestInvestigationStep`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_models.py:182`

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
**Defined in:** `model_chorus/tests/test_thinkdeep_workflow.py:27`

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
- `test_relevant_files_merge_into_state_and_metadata()`
- `test_investigation_step_with_no_files()`
- `test_investigation_step_confidence_tracking()`
- `test_investigation_step_with_hypothesis_integration()`
- `test_investigation_step_findings_extraction()`
- `test_investigation_step_with_expert_validation()`
- `test_investigation_step_metadata_completeness()`
- `test_investigation_step_error_handling()`
- `test_empty_provider_response_reports_error()`
- `test_investigation_without_conversation_memory()`

---

### `TestKMeansClustering`

**Language:** python
**Defined in:** `model_chorus/tests/test_semantic_similarity.py:503`

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

### `TestLogicalGapDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:261`

**Description:**
> Test detection of logical gaps.

**Methods:**
- `test_detect_logical_gap_conclusion_without_support()`
- `test_no_logical_gap_with_support()`
- `test_no_logical_gap_non_conclusion_claim()`
- `test_conclusion_indicators_detected()`

---

### `TestMemoryManagement`

**Language:** python
**Defined in:** `model_chorus/tests/test_memory_management.py:21`

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

### `TestMiddlewareChaining`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:706`

**Description:**
> Test chaining multiple middleware together.

**Methods:**
- `test_retry_with_circuit_breaker()`
- `test_retry_exhausted_counts_as_circuit_failure()`

---

### `TestMissingEvidenceDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:203`

**Description:**
> Test detection of missing evidence gaps.

**Methods:**
- `test_detect_missing_evidence_no_citations()`
- `test_detect_missing_evidence_insufficient_citations()`
- `test_no_gap_sufficient_citations()`
- `test_evidence_gap_metadata()`

---

### `TestModelIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_models.py:553`

**Description:**
> Test integration scenarios using multiple models together.

**Methods:**
- `test_hypothesis_lifecycle()`
- `test_investigation_progression()`
- `test_multiple_hypothesis_tracking()`

---

### `TestModelRole`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:100`

**Description:**
> Test suite for ModelRole class.

**Methods:**
- `test_model_role_creation_minimal()`
- `test_model_role_creation_full()`
- `test_stance_validation_valid()`
- `test_stance_validation_invalid()`
- `test_temperature_validation_valid()`
- `test_temperature_validation_invalid()`
- `test_max_tokens_validation_valid()`
- `test_max_tokens_validation_invalid()`
- `test_get_full_prompt_base_only()`
- `test_get_full_prompt_with_system()`
- `test_get_full_prompt_with_stance()`
- `test_get_full_prompt_with_all()`
- `test_role_length_validation()`
- `test_model_name_validation()`

---

### `TestMultiProviderChat`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:121`

**Description:**
> Test chat functionality across multiple providers.

**Methods:**
- `test_basic_conversation()`
- `test_multi_turn_conversation()`
- `test_conversation_with_file_context()`
- `test_conversation_persistence()`

---

### `TestOrchestrationResult`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:245`

**Description:**
> Test suite for OrchestrationResult dataclass.

**Methods:**
- `test_orchestration_result_creation_minimal()`
- `test_orchestration_result_creation_full()`

---

### `TestOutputFormatter`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_primitives.py:629`

**Description:**
> Test suite for OutputFormatter class.

**Methods:**
- `test_display_workflow_start_minimal()`
- `test_display_workflow_start_with_provider()`
- `test_display_workflow_start_continuation()`
- `test_display_workflow_start_with_files()`
- `test_display_workflow_start_truncates_long_prompt()`
- `test_display_workflow_start_does_not_truncate_short_prompt()`
- `test_display_workflow_start_with_kwargs()`
- `test_display_workflow_start_skips_none_kwargs()`
- `test_write_json_output_creates_file()`
- `test_write_json_output_creates_parent_dirs()`
- `test_write_json_output_shows_confirmation()`
- `test_write_json_output_handles_string_path()`
- `test_write_json_output_formatted_with_indent()`

---

### `TestParallelBrainstorming`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:277`

**Description:**
> Test parallel brainstorming functionality.

**Methods:**
- `test_run_parallel_brainstorming()`
- `test_parallel_brainstorming_empty_prompt_raises_error()`
- `test_parallel_brainstorming_empty_provider_map_raises_error()`

---

### `TestPathValidation`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:424`

**Description:**
> Test path validation and normalization.

**Methods:**
- `test_validate_path_with_string()`
- `test_validate_path_with_pathlib()`

---

### `TestPersona`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:79`

**Description:**
> Test suite for Persona base class.

**Methods:**
- `test_persona_init()`
- `test_persona_init_custom_temperature()`
- `test_persona_init_custom_max_tokens()`
- `test_persona_invoke_returns_response()`
- `test_persona_invoke_with_context()`
- `test_persona_invoke_includes_metadata()`

---

### `TestPersonaFactories`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:452`

**Description:**
> Test suite for persona factory functions.

**Methods:**
- `test_create_default_personas()`
- `test_create_default_personas_types()`
- `test_create_default_personas_independent()`
- `test_get_default_registry()`
- `test_get_default_registry_has_researcher()`
- `test_get_default_registry_has_critic()`
- `test_get_default_registry_has_planner()`
- `test_get_default_registry_independent()`

---

### `TestPersonaIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:529`

**Description:**
> Integration tests for persona system.

**Methods:**
- `test_registry_with_all_personas()`
- `test_persona_responses_structure()`
- `test_different_personas_different_findings()`
- `test_persona_temperature_affects_generation()`
- `test_persona_max_tokens_configuration()`

---

### `TestPersonaRegistry`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:135`

**Description:**
> Test suite for PersonaRegistry.

**Methods:**
- `test_registry_init()`
- `test_registry_register_persona()`
- `test_registry_register_multiple_personas()`
- `test_registry_register_duplicate_raises_error()`
- `test_registry_get_persona()`
- `test_registry_get_nonexistent_persona()`
- `test_registry_list_all()`

---

### `TestPersonaResponse`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:31`

**Description:**
> Test suite for PersonaResponse dataclass.

**Methods:**
- `test_persona_response_init()`
- `test_persona_response_with_findings()`
- `test_persona_response_with_confidence_update()`
- `test_persona_response_with_metadata()`
- `test_persona_response_full_initialization()`

---

### `TestPlannerPersona`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:393`

**Description:**
> Test suite for PlannerPersona.

**Methods:**
- `test_planner_init()`
- `test_planner_init_custom_temperature()`
- `test_planner_prompt_template()`
- `test_planner_invoke_returns_response()`
- `test_planner_invoke_includes_findings()`
- `test_create_planner_factory()`
- `test_create_planner_factory_with_params()`

---

### `TestPolarityOpposition`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:22`

**Description:**
> Test polarity opposition detection.

**Methods:**
- `test_clear_positive_negative_opposition()`
- `test_negation_opposition()`
- `test_no_opposition_similar_polarity()`
- `test_numerical_opposition_strengthens_confidence()`

---

### `TestPromptFraming`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:892`

**Description:**
> Test prompt framing methods.

**Methods:**
- `test_frame_ideation_prompt_initial()`
- `test_frame_ideation_prompt_continuation()`

---

### `TestProviderError`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:72`

**Description:**
> Test ProviderError base exception.

**Methods:**
- `test_basic_initialization()`
- `test_full_initialization()`

---

### `TestProviderResolver`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_primitives.py:29`

**Description:**
> Test suite for ProviderResolver class.

**Methods:**
- `test_init_stores_dependencies()`
- `test_init_defaults_verbose_false()`
- `test_resolve_provider_success()`
- `test_resolve_provider_casts_timeout_to_int()`
- `test_resolve_provider_verbose_output()`
- `test_resolve_provider_disabled_error()`
- `test_resolve_provider_unavailable_error()`
- `test_resolve_provider_generic_error()`
- `test_resolve_fallback_providers_success()`
- `test_resolve_fallback_providers_empty_list()`
- `test_resolve_fallback_providers_skips_disabled()`
- `test_resolve_fallback_providers_handles_errors()`
- `test_resolve_fallback_providers_verbose_output()`

---

### `TestReadFile`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:55`

**Description:**
> Test basic file reading functionality.

**Methods:**
- `test_read_small_text_file()`
- `test_read_file_with_unicode()`
- `test_read_file_with_multiple_lines()`
- `test_read_file_not_found()`
- `test_read_directory_raises_error()`
- `test_read_file_too_large()`
- `test_read_file_exceeds_warning_threshold()`
- `test_read_binary_file_raises_error()`

---

### `TestReadFileChunked`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:255`

**Description:**
> Test chunked file reading.

**Methods:**
- `test_read_file_chunked_single_chunk()`
- `test_read_file_chunked_multiple_chunks()`
- `test_read_file_chunked_with_max_chunks()`
- `test_read_file_chunked_invalid_chunk_size()`
- `test_read_file_chunked_binary_file()`

---

### `TestReadFileLines`

**Language:** python
**Defined in:** `model_chorus/tests/test_context_ingestion.py:340`

**Description:**
> Test line-by-line file reading.

**Methods:**
- `test_read_file_lines_all_lines()`
- `test_read_file_lines_with_max_lines()`
- `test_read_file_lines_skip_empty()`
- `test_read_file_lines_keep_empty()`
- `test_read_file_lines_too_large()`

---

### `TestRealisticScenarios`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:402`

**Description:**
> Test realistic contradiction scenarios.

**Methods:**
- `test_medical_accuracy_contradiction()`
- `test_performance_metric_contradiction()`

---

### `TestRealisticScenarios`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:451`

**Description:**
> Test realistic gap detection scenarios.

**Methods:**
- `test_policy_argument_with_gaps()`
- `test_well_supported_scientific_claim()`
- `test_argumentative_essay_analysis()`

---

### `TestReconciliationSuggestions`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:174`

**Description:**
> Test reconciliation suggestion generation.

**Methods:**
- `test_critical_suggestion_mentions_reliability()`
- `test_major_suggestion_mentions_context()`
- `test_moderate_suggestion_mentions_differences()`
- `test_minor_no_suggestion()`

---

### `TestResearcherPersona`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_personas.py:216`

**Description:**
> Test suite for ResearcherPersona.

**Methods:**
- `test_researcher_init()`
- `test_researcher_init_custom_temperature()`
- `test_researcher_prompt_template()`
- `test_researcher_invoke_returns_response()`
- `test_researcher_invoke_includes_findings()`
- `test_researcher_invoke_discovery_phase()`
- `test_researcher_invoke_validation_phase()`
- `test_researcher_invoke_metadata()`
- `test_researcher_invoke_includes_prompt_in_findings()`
- `test_create_researcher_factory()`
- `test_create_researcher_factory_with_params()`

---

### `TestRetryConfig`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:137`

**Description:**
> Test RetryConfig dataclass.

**Methods:**
- `test_default_config()`
- `test_custom_config()`
- `test_permanent_error_patterns()`

---

### `TestRetryExhaustedError`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:113`

**Description:**
> Test RetryExhaustedError exception.

**Methods:**
- `test_initialization()`

---

### `TestRetryMiddleware`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:186`

**Description:**
> Test RetryMiddleware implementation.

**Methods:**
- `test_initialization_default_config()`
- `test_initialization_custom_config()`
- `test_is_retryable_error_transient()`
- `test_is_retryable_error_permanent()`
- `test_is_retryable_error_non_exception_type()`
- `test_calculate_delay_exponential_backoff()`
- `test_calculate_delay_max_cap()`
- `test_calculate_delay_with_jitter()`
- `test_generate_success_first_attempt()`
- `test_generate_success_after_retries()`
- `test_generate_permanent_error_no_retry()`
- `test_generate_all_retries_exhausted()`
- `test_generate_respects_backoff_delay()`

---

### `TestRoleCreation`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:73`

**Description:**
> Test role creation methods.

**Methods:**
- `test_create_creator_role()`
- `test_create_skeptic_role()`
- `test_create_moderator_role()`

---

### `TestRoleCreation`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:245`

**Description:**
> Test brainstormer role creation.

**Methods:**
- `test_create_brainstormer_role_practical()`
- `test_create_brainstormer_role_innovative()`
- `test_create_brainstormer_role_user_focused()`

---

### `TestRoleOrchestrator`

**Language:** python
**Defined in:** `model_chorus/tests/test_role_orchestration.py:286`

**Description:**
> Test suite for RoleOrchestrator class.

**Methods:**
- `simple_roles()`
- `simple_providers()`
- `test_orchestrator_initialization_minimal()`
- `test_orchestrator_initialization_custom()`
- `test_orchestrator_initialization_empty_roles()`
- `test_orchestrator_initialization_unsupported_pattern()`
- `test_resolve_provider_exact_match()`
- `test_resolve_provider_case_variation()`
- `test_resolve_provider_hyphen_variation()`
- `test_resolve_provider_not_found()`
- `test_execute_sequential_success()`
- `test_execute_sequential_with_context()`
- `test_execute_sequential_partial_failure()`
- `test_execute_parallel_success()`
- `test_execute_parallel_partial_failure()`
- `test_execute_parallel_maintains_order()`
- `test_synthesize_none_strategy()`
- `test_synthesize_concatenate_strategy()`
- `test_synthesize_structured_strategy()`
- `test_synthesize_ai_strategy_default_provider()`
- `test_synthesize_ai_strategy_custom_provider()`
- `test_synthesize_ai_strategy_custom_prompt()`
- `test_synthesize_ai_strategy_fallback_on_failure()`
- `test_synthesize_ai_strategy_no_responses()`
- `test_synthesize_unknown_strategy()`
- `test_role_prompt_customization()`
- `test_build_synthesis_prompt()`

---

### `TestRoutingHistory`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_routing.py:270`

**Description:**
> Test routing history tracking.

**Methods:**
- `test_routing_history_recorded()`
- `test_routing_history_filtering()`
- `test_routing_history_limit()`

---

### `TestRoutingSkillInvocation`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_routing.py:20`

**Description:**
> Test routing skill invocation and JSON output.

**Methods:**
- `test_routing_skill_invocation()`
- `test_routing_different_phases()`
- `test_routing_with_findings()`
- `test_routing_complete_phase()`

---

### `TestSemanticClustering`

**Language:** python
**Defined in:** `model_chorus/tests/test_clustering.py:75`

**Description:**
> Test suite for SemanticClustering class.

**Methods:**
- `clustering()`
- `mock_model()`
- `test_initialization()`
- `test_lazy_model_loading()`
- `test_compute_embeddings_basic()`
- `test_compute_embeddings_caching()`
- `test_compute_embeddings_no_cache()`
- `test_compute_similarity_cosine()`
- `test_compute_similarity_euclidean()`
- `test_compute_similarity_dot()`
- `test_compute_similarity_invalid_metric()`
- `test_cluster_kmeans_basic()`
- `test_cluster_hierarchical_basic()`
- `test_cluster_hierarchical_linkages()`
- `test_name_cluster_basic()`
- `test_name_cluster_empty()`
- `test_name_cluster_long_text()`
- `test_summarize_cluster_basic()`
- `test_summarize_cluster_empty()`
- `test_summarize_cluster_truncation()`
- `test_score_cluster_basic()`
- `test_score_cluster_empty()`
- `test_cluster_end_to_end()`
- `test_cluster_with_hierarchical()`
- `test_cluster_empty_texts()`
- `test_cluster_more_clusters_than_texts()`
- `test_cluster_invalid_method()`
- `test_cluster_single_text()`
- `test_cluster_reproducibility()`

---

### `TestSeverityAssessment`

**Language:** python
**Defined in:** `model_chorus/tests/test_contradiction.py:67`

**Description:**
> Test contradiction severity assessment.

**Methods:**
- `test_critical_severity_high_similarity_strong_polarity()`
- `test_major_severity_high_similarity_weak_polarity()`
- `test_moderate_severity_medium_similarity()`
- `test_minor_severity_low_similarity()`
- `test_moderate_severity_high_similarity_no_opposition()`

---

### `TestSeverityAssessment`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:92`

**Description:**
> Test gap severity assessment logic.

**Methods:**
- `test_evidence_gap_critical_no_citations()`
- `test_evidence_gap_major_no_citations_low_expectation()`
- `test_evidence_gap_moderate_insufficient_citations()`
- `test_logical_gap_major_no_support()`
- `test_logical_gap_moderate_with_support()`
- `test_support_gap_moderate()`
- `test_assumption_gap_moderate_no_logic()`
- `test_assumption_gap_minor_with_logic()`

---

### `TestStandardization`

**Language:** python
**Defined in:** `model_chorus/tests/test_standardization.py:8`

**Methods:**
- `test_claude_standardization()`
- `test_gemini_standardization()`

---

### `TestStateManager`

**Language:** python
**Defined in:** `model_chorus/tests/test_state.py:22`

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
**Defined in:** `model_chorus/tests/test_state.py:599`

**Description:**
> Test complete export/import workflow.

**Methods:**
- `test_export_import_roundtrip()`

---

### `TestStateManagerFileRecovery`

**Language:** python
**Defined in:** `model_chorus/tests/test_state.py:634`

**Description:**
> Test state recovery after simulated process restart.

**Methods:**
- `test_process_restart_recovery()`

---

### `TestStateTransitionIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_state_machine.py:378`

**Description:**
> Integration tests for state transitions with confidence.

**Methods:**
- `state_machine()`
- `test_full_investigation_flow()`
- `test_investigation_with_backtrack()`
- `test_multiple_confidence_updates_in_phase()`

---

### `TestStudyWorkflowConversationHandling`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:289`

**Description:**
> Test suite for conversation memory integration in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow_with_memory()`
- `workflow_without_memory()`
- `test_run_creates_thread_in_memory()`
- `test_run_stores_messages_in_memory()`
- `test_run_without_memory_still_works()`
- `test_run_reuses_thread_on_continuation()`

---

### `TestStudyWorkflowErrorHandling`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:494`

**Description:**
> Test suite for error handling in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_run_error_returns_false_success()`
- `test_run_captures_error_message()`

---

### `TestStudyWorkflowInitialization`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:22`

**Description:**
> Test suite for StudyWorkflow initialization.

**Methods:**
- `test_init_with_provider()`
- `test_init_with_fallback_providers()`
- `test_init_provider_none_raises_error()`
- `test_init_with_conversation_memory()`
- `test_init_persona_router_ready()`
- `test_init_with_config()`

---

### `TestStudyWorkflowIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_routing.py:345`

**Description:**
> Test PersonaRouter integration with StudyWorkflow.

**Methods:**
- `test_workflow_has_router()`
- `test_workflow_routing_history_access()`

---

### `TestStudyWorkflowIntegration`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:570`

**Description:**
> Integration tests for StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow_with_memory()`
- `test_full_workflow_execution()`
- `test_conversation_continuation_flow()`
- `test_workflow_with_custom_personas()`
- `test_router_persona_count()`

---

### `TestStudyWorkflowInvestigation`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:368`

**Description:**
> Test suite for investigation flow in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_investigation_returns_steps()`
- `test_investigation_steps_have_metadata()`
- `test_investigation_includes_available_personas()`
- `test_investigation_empty_personas()`

---

### `TestStudyWorkflowPersonaSetup`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:224`

**Description:**
> Test suite for persona setup in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_setup_personas_default()`
- `test_setup_personas_default_structure()`
- `test_setup_personas_custom()`
- `test_setup_personas_empty_list()`
- `test_run_personas_in_metadata()`

---

### `TestStudyWorkflowRoutingHistory`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:533`

**Description:**
> Test suite for routing history access in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_get_routing_history_available()`
- `test_get_routing_history_with_limit()`
- `test_get_routing_history_with_investigation_id()`

---

### `TestStudyWorkflowRun`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:91`

**Description:**
> Test suite for StudyWorkflow.run() method.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_run_with_valid_prompt()`
- `test_run_empty_prompt_raises_error()`
- `test_run_whitespace_prompt_raises_error()`
- `test_run_returns_workflow_result()`
- `test_run_result_metadata_structure()`
- `test_run_result_steps_is_list()`
- `test_run_result_steps_are_workflow_steps()`
- `test_run_creates_thread_id()`
- `test_run_with_continuation_id()`
- `test_run_without_continuation_is_not_continuation()`
- `test_run_metadata_timestamp_format()`
- `test_run_synthesis_is_string()`

---

### `TestStudyWorkflowSynthesis`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/test_study_workflow.py:435`

**Description:**
> Test suite for synthesis in StudyWorkflow.

**Methods:**
- `mock_provider()`
- `workflow()`
- `test_synthesize_findings_returns_string()`
- `test_synthesize_findings_empty_steps()`
- `test_synthesize_findings_multiple_steps()`
- `test_synthesize_includes_step_count()`

---

### `TestSystemPrompts`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:850`

**Description:**
> Test system prompt generation.

**Methods:**
- `test_get_ideation_system_prompt()`
- `test_get_extraction_system_prompt()`
- `test_get_clustering_system_prompt()`
- `test_get_scoring_system_prompt()`
- `test_get_elaboration_system_prompt()`

---

### `TestThinkDeepCommand`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:466`

**Description:**
> Test suite for 'thinkdeep' CLI command.

**Methods:**
- `test_thinkdeep_missing_legacy_file_warns_and_continues()`
- `test_thinkdeep_remaps_legacy_file()`
- `test_thinkdeep_relevant_files_option()`
- `test_thinkdeep_relevant_files_missing_errors()`

---

### `TestThinkDeepState`

**Language:** python
**Defined in:** `model_chorus/tests/test_thinkdeep_models.py:300`

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

### `TestUnsupportedClaimsDetection`

**Language:** python
**Defined in:** `model_chorus/tests/test_gap_analysis.py:315`

**Description:**
> Test batch detection of unsupported claims.

**Methods:**
- `test_detect_multiple_unsupported_claims()`
- `test_detect_no_gaps_all_supported()`
- `test_custom_minimum_citations()`

---

### `TestWorkflowConfigValidation`

**Language:** python
**Defined in:** `model_chorus/tests/test_config.py:401`

**Description:**
> Test suite for workflow-specific configuration validation.

**Methods:**
- `test_thinkdeep_thinking_mode_validation()`
- `test_research_citation_style_validation()`
- `test_research_depth_validation()`
- `test_consensus_strategy_validation()`

---

### `TestWorkflowContext`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_primitives.py:351`

**Description:**
> Test suite for WorkflowContext class.

**Methods:**
- `test_init_stores_dependencies()`
- `test_validate_prompt_from_positional_arg()`
- `test_validate_prompt_from_flag()`
- `test_validate_prompt_missing_both()`
- `test_validate_prompt_both_provided()`
- `test_resolve_config_defaults_all_provided()`
- `test_resolve_config_defaults_from_config()`
- `test_resolve_config_defaults_custom_default_provider()`
- `test_resolve_config_defaults_disabled_provider()`
- `test_resolve_config_defaults_partial_override()`
- `test_create_memory()`
- `test_prepare_prompt_with_files_no_files()`
- `test_prepare_prompt_with_files_with_files()`

---

### `ThinkDeepState`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:713`

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
**Defined in:** `model_chorus/src/model_chorus/workflows/thinkdeep.py:27`

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
    >>> from model_chorus.providers import ClaudeProvider
    >>> from model_chorus.workflows import ThinkDeepWorkflow
    >>> from model_chorus.core.conversation import ConversationMemory
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
- `_merge_file_lists()`
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

### `ThinkDeepWorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:81`

**Description:**
> Configuration specific to thinkdeep workflow.

---

### `TokenUsage`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/providers/base_provider.py:51`

**Description:**
> Token usage information with explicit fields for type safety.

This dataclass provides a standardized way to track token consumption
across different AI providers, with support for caching and provider-
specific metadata.

Supports both attribute access (usage.input_tokens) and dict-like access
(usage['input_tokens']) for backward compatibility.

Attributes:
    input_tokens: Number of tokens in the input prompt/context.
    output_tokens: Number of tokens in the generated response.
    cached_input_tokens: Number of input tokens retrieved from cache
        (provider-dependent, e.g., OpenAI prompt caching).
    total_tokens: Total token count, typically input + output tokens.
        May or may not include cached tokens depending on provider.
    metadata: Provider-specific additional usage information (e.g.,
        cost, rate limits, model-specific metrics).

**Methods:**
- `__getitem__()`
- `__setitem__()`
- `get()`
- `keys()`
- `values()`
- `items()`

---

### `WorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/config/models.py:137`

**Description:**
> Configuration for a specific workflow.

Provides workflow-specific overrides for providers, generation parameters,
and workflow-specific settings.

**Methods:**
- `validate_provider()`
- `validate_providers()`

---

### `WorkflowConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:36`

**Description:**
> Configuration for a specific workflow.

**Methods:**
- `validate_provider()`
- `validate_providers()`

---

### `WorkflowConfigV2`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/config.py:470`

**Description:**
> Workflow configuration for .claude/model_chorus_config.yaml.

**Methods:**
- `validate_provider()`
- `validate_providers()`

---

### `WorkflowContext`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/primitives.py:170`

**Description:**
> Manages workflow execution context including config, prompt validation, and memory setup.

Extracts common patterns of:
1. Prompt validation (arg vs flag)
2. Config resolution with workflow defaults
3. Conversation memory initialization
4. File context ingestion

**Methods:**
- `__init__()`
- `validate_and_get_prompt()`
- `resolve_config_defaults()`
- `create_memory()`
- `prepare_prompt_with_files()`

---

### `WorkflowRegistry`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/registry.py:14`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:65`

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
**Defined in:** `model_chorus/src/model_chorus/core/models.py:139`

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
**Defined in:** `model_chorus/src/model_chorus/core/base_workflow.py:37`

**Description:**
> Result of a workflow execution.

**Methods:**
- `add_step()`

---

### `WorkflowRunner`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/workflow_runner.py:103`

**Description:**
> Orchestrates workflow execution with provider fallback and telemetry.

The WorkflowRunner extracts provider fallback logic from BaseWorkflow
and adds structured error handling, execution metrics, and extension hooks
for telemetry/observability.

Key Features:
- Automatic provider fallback with configurable retry logic
- Execution metrics collection (timing, attempts, success/failure)
- Extension hooks for telemetry integration
- Structured error handling with detailed context

Example:
    >>> runner = WorkflowRunner()
    >>> response, metrics = await runner.execute(
    ...     request,
    ...     primary_provider,
    ...     fallback_providers=[backup1, backup2]
    ... )
    >>> print(f"Used {metrics.successful_provider} after {metrics.total_attempts} attempts")
    >>> print(f"Duration: {metrics.duration_ms}ms")

**Methods:**
- `__init__()`
- `register_telemetry_callback()`
- `unregister_telemetry_callback()`
- `execute()`
- `_on_execution_complete()`
- `_on_execution_failed()`

---

### `WorkflowStep`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/base_workflow.py:26`

**Description:**
> Represents a single step in a workflow execution.

---

### `WorkflowStep`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `model_chorus/src/model_chorus/core/models.py:246`

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

### `_check_sentence_transformers_available() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_clustering.py:25`
**Complexity:** 2

**Description:**
> Check if sentence-transformers is available.

---

### `_compute_embedding_cached(text_hash, text, model_name) -> np.ndarray`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:59`
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

### `_config_init(verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1874`
**Complexity:** 4

**Description:**
> Initialize a sample .model-chorusrc file.

**Parameters:**
- `verbose`: bool

---

### `_config_show(verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1778`
⚠️ **Complexity:** 15 (High)

**Description:**
> Show current effective configuration.

**Parameters:**
- `verbose`: bool

---

### `_config_validate(verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1841`
**Complexity:** 4

**Description:**
> Validate .model-chorusrc file.

**Parameters:**
- `verbose`: bool

---

### `_create_smart_mock_provider(provider_name, model_name, stop_reason) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:183`
⚠️ **Complexity:** 37 (High)

**Description:**
> Create a smart mock provider that understands conversation context and file contents.

This mock provider can:
- Extract and echo back information from the prompt
- Detect and respond to file contents in the prompt
- Generate contextually appropriate responses for common test queries

**Parameters:**
- `provider_name`: str
- `model_name`: str
- `stop_reason`: str

---

### `_find_project_root(start_path) -> Path`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:62`
**Complexity:** 4

**Description:**
> Traverse upward from the starting path to locate the project root.

**Parameters:**
- `start_path`: Path

---

### `_format_apa(citation) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:58`
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
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:142`
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
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:100`
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

### `_format_path_for_display(path) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:80`
**Complexity:** 2

**Description:**
> Return a repository-relative path when possible for cleaner display.

**Parameters:**
- `path`: Path

---

### `_get_model(model_name) -> SentenceTransformer`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:23`
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
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:227`
**Complexity:** 2

**Description:**
> Import CitationMap model.

---

### `_import_citation_map() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:192`
**Complexity:** 2

**Description:**
> Import CitationMap model.

---

### `_import_semantic_functions() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:211`
**Complexity:** 2

**Description:**
> Import semantic similarity functions from workflows.argument.semantic.

---

### `_make_provider(name, response_text) -> AsyncMock`

**Language:** python
**Defined in:** `model_chorus/tests/test_consensus_provider_models.py:13`
**Complexity:** 1

**Parameters:**
- `name`: str
- `response_text`: str

---

### `_normalize_text(text) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:45`
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

### `_select_persona_by_phase_and_state(phase, findings_count, has_questions, prior_persona) -> tuple[str, str, list[str]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/context_analysis.py:197`
**Complexity:** 10

**Description:**
> Select persona based on investigation phase and current state.

Phase-Based Selection Strategy:
- DISCOVERY: Start with Researcher for initial exploration
- VALIDATION: Use Critic to challenge assumptions and findings
- PLANNING: Use Planner to synthesize into actionable roadmap
- COMPLETE: No further persona needed (investigation complete)

Args:
    phase: Current investigation phase
    findings_count: Number of existing findings
    has_questions: Whether there are unresolved questions
    prior_persona: Previously consulted persona (to avoid repetition)

Returns:
    Tuple of (persona_name, reasoning, guidance_list)

**Parameters:**
- `phase`: str
- `findings_count`: int
- `has_questions`: bool
- `prior_persona`: str | None

---

### `add_permissions(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:856`
**Complexity:** 9

**Description:**
> Add ModelChorus permissions to .claude/settings.local.json.

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with result

**Parameters:**
- `project_root`: Path | None

---

### `add_similarity_to_citation(citation, reference_claim, model_name) -> Citation`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:280`
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

### `add_to_gitignore(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:789`
**Complexity:** 10

**Description:**
> Add .model-chorusrc to project .gitignore.

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with result

**Parameters:**
- `project_root`: Path | None

---

### `analyze_context(context_input) -> ContextAnalysisResult`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/context_analysis.py:321`
**Complexity:** 2

**Description:**
> Analyze investigation context and determine next persona to consult.

This is the main entry point for the context analysis skill. Based on
the current investigation phase, confidence level, findings, and
unresolved questions, it selects the most appropriate persona to
consult next.

Selection Strategy:
- DISCOVERY phase: Alternate between Researcher (exploration) and Critic (challenge)
- VALIDATION phase: Primarily Critic (testing), with Researcher to fill gaps
- PLANNING phase: Primarily Planner (synthesis), with fallback to Researcher/Critic if needed
- COMPLETE phase: No further persona needed

The logic also considers:
- Prior persona to avoid repetition
- Presence of findings (do we have material to work with?)
- Presence of unresolved questions (do we need more investigation?)

Args:
    context_input: Validated context analysis input

Returns:
    ContextAnalysisResult with recommended persona and guidance

**Parameters:**
- `context_input`: ContextAnalysisInput

---

### `argument(prompt_arg, prompt_flag, provider, continuation_id, files, system, temperature, timeout, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:465`
⚠️ **Complexity:** 13 (High)

**Decorators:** `@app.command()`

**Description:**
> Analyze arguments through structured dialectical reasoning.

The argument workflow uses role-based orchestration to examine claims from
multiple perspectives: Creator (thesis), Skeptic (critique), and Moderator (synthesis).

Example:
    # Analyze an argument (positional prompt)
    model-chorus argument "Universal basic income would reduce poverty"

    # Analyze an argument (flag prompt)
    model-chorus argument --prompt "Universal basic income would reduce poverty"

    # Continue analysis
    model-chorus argument "What about inflation?" --continue thread-id-123

    # Include supporting files
    model-chorus argument "Review this proposal" -f proposal.md -f data.csv

**Parameters:**
- `prompt_arg`: str | None
- `prompt_flag`: str | None
- `provider`: str | None
- `continuation_id`: str | None
- `files`: list[str] | None
- `system`: str | None
- `temperature`: float | None
- `timeout`: float | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `argument_workflow(mock_provider, conversation_memory) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_argument_workflow.py:26`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ArgumentWorkflow instance for testing.

**Parameters:**
- `mock_provider`: None
- `conversation_memory`: None

---

### `assess_contradiction_severity(semantic_similarity, has_polarity_opposition, polarity_confidence) -> ContradictionSeverity`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:357`
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

### `assess_gap_severity(gap_type, citation_count, expected_citations, has_supporting_logic) -> GapSeverity`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:202`
**Complexity:** 10

**Description:**
> Assess the severity of a gap based on type and context.

Combines gap type, citation coverage, and logical support to
classify gap severity.

Args:
    gap_type: Type of gap detected
    citation_count: Number of citations supporting the claim
    expected_citations: Expected number of citations for this type of claim
    has_supporting_logic: Whether logical support/reasoning is present

Returns:
    GapSeverity enum value

Severity Rules:
    - EVIDENCE gap with 0 citations = MAJOR or CRITICAL
    - LOGICAL gap without supporting logic = MAJOR
    - SUPPORT gap = MODERATE
    - ASSUMPTION gap = MINOR to MODERATE

Example:
    >>> severity = assess_gap_severity(
    ...     gap_type=GapType.EVIDENCE,
    ...     citation_count=0,
    ...     expected_citations=2
    ... )
    >>> print(severity)
    GapSeverity.MAJOR

**Parameters:**
- `gap_type`: GapType
- `citation_count`: int
- `expected_citations`: int
- `has_supporting_logic`: bool

---

### `async basic_chat_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/chat_example.py:19`
**Complexity:** 2

**Description:**
> Basic chat conversation without continuation.

---

### `async basic_investigation_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/thinkdeep_example.py:24`
**Complexity:** 2

**Description:**
> Basic single-step investigation.

---

### `cache() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:18`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create a ShortTermCache instance with small size for testing.

---

### `calculate_citation_confidence(citation) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:285`
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

### `calculate_citation_map_confidence(citation_map) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:377`
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

### `chat(prompt_arg, prompt_flag, provider, continuation_id, files, system, timeout, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:253`
⚠️ **Complexity:** 11 (High)

**Decorators:** `@app.command()`

**Description:**
> Chat with a single AI model with conversation continuity.

Example:
    # Start new conversation (positional prompt)
    model-chorus chat "What is quantum computing?" --provider claude

    # Start new conversation (flag prompt)
    model-chorus chat --prompt "What is quantum computing?" --provider claude

    # Continue conversation
    model-chorus chat "Give me an example" --continue thread-id-123

    # Include files
    model-chorus chat "Review this code" -f src/main.py -f tests/test_main.py

**Parameters:**
- `prompt_arg`: str | None
- `prompt_flag`: str | None
- `provider`: str | None
- `continuation_id`: str | None
- `files`: list[str] | None
- `system`: str | None
- `timeout`: float | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `async chat_with_file_context_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/chat_example.py:104`
**Complexity:** 2

**Description:**
> Chat with file context included.

---

### `chat_workflow(provider, conversation_memory) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:112`
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
**Defined in:** `model_chorus/tests/test_chat_workflow.py:17`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ChatWorkflow instance for testing.

**Parameters:**
- `mock_provider`: None
- `conversation_memory`: None

---

### `check_available_providers() -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:174`
**Complexity:** 5

**Description:**
> Check which CLI providers are available on the system.

Returns:
    Dict with available provider names and details

---

### `check_config_exists(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:228`
**Complexity:** 4

**Description:**
> Check if .model-chorusrc config file exists.

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with config file status

**Parameters:**
- `project_root`: Path | None

---

### `check_package_installed() -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:23`
**Complexity:** 5

**Description:**
> Check if model-chorus package is installed.

Returns:
    Dict with installation status and details

---

### `check_permissions(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:751`
**Complexity:** 4

**Description:**
> Check if Claude Code permissions are configured.

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with permissions status

**Parameters:**
- `project_root`: Path | None

---

### `check_version_compatibility() -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:89`
**Complexity:** 9

**Description:**
> Check if installed package version matches plugin version.

Compares the version from .claude-plugin/plugin.json with the installed
package version. If package version is lower, recommends reinstall.

Returns:
    Dict with compatibility status and version details

---

### `citation_map_complete(complete_citation, file_citation) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:81`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> CitationMap with multiple citations.

**Parameters:**
- `complete_citation`: None
- `file_citation`: None

---

### `citation_map_empty() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:93`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> CitationMap with no citations.

---

### `cli_runner() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:27`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create Typer CLI runner for testing.

---

### `cluster_claims_hierarchical(citation_maps, n_clusters, model_name, linkage_method) -> list[list[CitationMap]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:506`
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
- `citation_maps`: list[CitationMap]
- `n_clusters`: int
- `model_name`: str
- `linkage_method`: str

---

### `cluster_claims_kmeans(citation_maps, n_clusters, model_name, random_state, max_iterations) -> list[list[CitationMap]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:399`
⚠️ **Complexity:** 11 (High)

**Description:**
> Cluster claims using K-means algorithm on semantic embeddings.

Groups claims into k clusters based on semantic similarity,
useful for organizing large collections of claims by topic.

Args:
    citation_maps: List of CitationMap objects to cluster
    n_clusters: Number of clusters to create (default: 3)
    model_name: Sentence transformer model to use
    random_state: Random seed for reproducibility
    max_iterations: Maximum iterations for centroid refinement

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
- `citation_maps`: list[CitationMap]
- `n_clusters`: int
- `model_name`: str
- `random_state`: int
- `max_iterations`: int

---

### `complete_citation() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:26`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Citation with all metadata fields populated.

---

### `compute_claim_similarity(claim1, claim2, model_name) -> float`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:144`
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
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:237`
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
- `claims`: list[str]
- `model_name`: str

---

### `compute_cluster_statistics(clusters, model_name) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:721`
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
- `clusters`: list[list[CitationMap]]
- `model_name`: str

---

### `compute_embedding(text, model_name, normalize) -> np.ndarray`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:79`
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
**Defined in:** `model_chorus/examples/thinkdeep_example.py:381`
**Complexity:** 5

**Description:**
> Demonstrate confidence progression through investigation steps.

Shows how confidence should naturally increase as evidence accumulates
and hypotheses are validated.

---

### `config_cmd(subcommand, verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1731`
**Complexity:** 7

**Decorators:** `@app.command(name='config')`

**Description:**
> Manage ModelChorus configuration.

Subcommands:
    show     - Display current effective configuration
    validate - Validate .model-chorusrc file
    init     - Generate sample .model-chorusrc file

Examples:
    model-chorus config show
    model-chorus config validate
    model-chorus config init

**Parameters:**
- `subcommand`: str
- `verbose`: bool

---

### `consensus(prompt_arg, prompt_flag, num_to_consult, strategy, files, system, timeout, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:957`
⚠️ **Complexity:** 26 (High)

**Decorators:** `@app.command()`

**Description:**
> Run consensus workflow with priority-based provider selection.

Tries providers in priority order until num_to_consult successful responses
are obtained. If a provider fails, automatically falls back to the next
provider in the priority list.

Example:
    # Positional prompt
    model-chorus consensus "Explain quantum computing" --num-to-consult 2

    # Flag prompt
    model-chorus consensus --prompt "Explain quantum computing" --num-to-consult 2

**Parameters:**
- `prompt_arg`: str | None
- `prompt_flag`: str | None
- `num_to_consult`: int | None
- `strategy`: str | None
- `files`: list[str] | None
- `system`: str | None
- `timeout`: float | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `construct_prompt_with_files(prompt, files) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:898`
**Complexity:** 9

**Description:**
> Construct a prompt by prepending the content of files using ContextIngestionService.

NOTE: This function is retained as a convenience wrapper for backward compatibility.
It uses ContextIngestionService internally per architectural requirements while
maintaining a simple API for CLI command handlers. Direct use of ContextIngestionService
is preferred for new code requiring more control over file reading behavior.

**Parameters:**
- `prompt`: str
- `files`: list[str] | None

---

### `conversation_memory() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:354`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ConversationMemory instance for testing.

Provides a fresh conversation memory instance for each test.

---

### `conversation_memory(tmp_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:44`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create ConversationMemory instance for testing.

Uses a temporary directory for each test to ensure isolation.
Uses a high max_messages limit (100) to allow long conversation tests
to run without hitting the truncation limit.

**Parameters:**
- `tmp_path`: None

---

### `async conversation_tracking_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/chat_example.py:149`
**Complexity:** 5

**Description:**
> Demonstrate conversation history tracking.

---

### `cosine_similarity(embedding1, embedding2) -> float`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:115`
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

### `create_claude_config(project_root, enabled_providers, auto_detect) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:535`
⚠️ **Complexity:** 27 (High)

**Description:**
> Create .claude/model_chorus_config.yaml configuration file.

Args:
    project_root: Project root directory (defaults to cwd)
    enabled_providers: List of providers to enable (if None and auto_detect=False, all disabled)
    auto_detect: If True, auto-detect available providers and enable them

Returns:
    Dict with creation result

**Parameters:**
- `project_root`: Path | None
- `enabled_providers`: list[str] | None
- `auto_detect`: bool

---

### `create_config_file(project_root, default_provider, timeout, available_providers, workflows) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:255`
⚠️ **Complexity:** 16 (High)

**Description:**
> Create .model-chorusrc configuration file.

Args:
    project_root: Project root directory (defaults to cwd)
    default_provider: Default AI provider
    timeout: Default timeout in seconds
    available_providers: List of available providers for model config
    workflows: Workflow-specific configurations (optional)

Returns:
    Dict with creation result

**Parameters:**
- `project_root`: Path | None
- `default_provider`: str
- `timeout`: float
- `available_providers`: list[str] | None
- `workflows`: dict[str, Any] | None

---

### `create_critic(temperature, max_tokens) -> CriticPersona`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/critic.py:126`
**Complexity:** 1

**Description:**
> Factory function to create a Critic persona instance.

Args:
    temperature: Generation temperature (default: 0.6 for focused critique)
    max_tokens: Maximum tokens per response (default: 4096)

Returns:
    Configured CriticPersona instance

**Parameters:**
- `temperature`: float
- `max_tokens`: int

---

### `create_default_personas() -> list`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/__init__.py:25`
**Complexity:** 1

**Description:**
> Create the default set of personas for STUDY workflow.

Returns:
    List of default persona instances (Researcher, Critic, Planner)

---

### `create_express_config(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:354`
**Complexity:** 8

**Description:**
> Create Express (zero-question) .model-chorusrc configuration.

Auto-detects available providers and configures with smart defaults:
- Primary provider: first available (claude → gemini → codex → cursor-agent)
- Fallbacks: all other available providers
- Default models: configured for each available provider
- Timeout: 120s (standard)
- All workflows configured with balanced settings

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with creation result

**Parameters:**
- `project_root`: Path | None

---

### `create_planner(temperature, max_tokens) -> PlannerPersona`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/planner.py:127`
**Complexity:** 1

**Description:**
> Factory function to create a Planner persona instance.

Args:
    temperature: Generation temperature (default: 0.7 for creative planning)
    max_tokens: Maximum tokens per response (default: 4096)

Returns:
    Configured PlannerPersona instance

**Parameters:**
- `temperature`: float
- `max_tokens`: int

---

### `create_researcher(temperature, max_tokens) -> ResearcherPersona`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/researcher.py:115`
**Complexity:** 1

**Description:**
> Factory function to create a Researcher persona instance.

Args:
    temperature: Generation temperature (default: 0.7)
    max_tokens: Maximum tokens per response (default: 4096)

Returns:
    Configured ResearcherPersona instance

**Parameters:**
- `temperature`: float
- `max_tokens`: int

---

### `create_tiered_config(project_root, tier, default_provider, consensus_providers, consensus_strategy, thinkdeep_thinking_mode, ideate_providers, workflow_overrides) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:437`
⚠️ **Complexity:** 13 (High)

**Description:**
> Create tiered .model-chorusrc configuration file.

Args:
    project_root: Project root directory (defaults to cwd)
    tier: Configuration tier (quick, standard, advanced)
    default_provider: Default AI provider
    consensus_providers: Providers for consensus workflow (standard+)
    consensus_strategy: Strategy for consensus workflow (standard+)
    thinkdeep_thinking_mode: Thinking mode for thinkdeep (standard+)
    ideate_providers: Providers for ideate workflow (standard+)
    workflow_overrides: Additional workflow overrides (advanced)

Returns:
    Dict with creation result

**Parameters:**
- `project_root`: Path | None
- `tier`: str
- `default_provider`: str
- `consensus_providers`: list | None
- `consensus_strategy`: str
- `thinkdeep_thinking_mode`: str
- `ideate_providers`: list | None
- `workflow_overrides`: dict[str, dict[str, Any]] | None

---

### `detect_contradiction(claim_1_id, claim_1_text, claim_2_id, claim_2_text, similarity_threshold, model_name) -> Contradiction | None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:506`
**Complexity:** 3

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

### `detect_contradictions_batch(claims, similarity_threshold, model_name) -> list[Contradiction]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:612`
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
- `claims`: list[tuple[str, str]]
- `similarity_threshold`: float
- `model_name`: str

---

### `detect_gaps(claims, min_citations_per_claim) -> list[Gap]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:540`
**Complexity:** 4

**Description:**
> Main entry point for comprehensive gap detection.

Analyzes a collection of claims to identify evidence gaps, logical gaps,
and other weaknesses in argument structure.

Args:
    claims: List of claim dictionaries with keys:
        - claim_id: Unique identifier
        - claim_text: Claim text
        - citations: Optional list of citations
        - supporting_claims: Optional list of supporting claim texts
    min_citations_per_claim: Minimum expected citations per claim

Returns:
    List of all detected Gap objects

Example:
    >>> claims = [
    ...     {
    ...         "claim_id": "claim-1",
    ...         "claim_text": "Universal basic income reduces poverty",
    ...         "citations": [],
    ...         "supporting_claims": []
    ...     }
    ... ]
    >>> gaps = detect_gaps(claims)
    >>> for gap in gaps:
    ...     print(f"{gap.gap_type}: {gap.description}")

**Parameters:**
- `claims`: list[dict[str, Any]]
- `min_citations_per_claim`: int

---

### `detect_logical_gaps(claim_id, claim_text, supporting_claims) -> Gap | None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:416`
**Complexity:** 4

**Description:**
> Detect logical gaps or missing reasoning steps.

Analyzes whether a claim has adequate logical support or if there
are missing steps in the reasoning chain.

Args:
    claim_id: Identifier for the claim
    claim_text: Text of the claim to analyze
    supporting_claims: Optional list of supporting claim texts

Returns:
    Gap object if logical gap detected, None otherwise

Example:
    >>> gap = detect_logical_gaps(
    ...     "claim-1",
    ...     "Therefore, we should implement policy X",
    ...     supporting_claims=[]
    ... )
    >>> if gap:
    ...     print(f"Detected: {gap.description}")

**Parameters:**
- `claim_id`: str
- `claim_text`: str
- `supporting_claims`: list[str] | None

---

### `detect_missing_evidence(claim_id, claim_text, citations, expected_citation_count) -> Gap | None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:332`
**Complexity:** 3

**Description:**
> Detect if a claim lacks adequate evidential support.

Analyzes citation coverage to identify claims that need more
empirical support or references.

Args:
    claim_id: Identifier for the claim
    claim_text: Text of the claim to analyze
    citations: Optional list of Citation objects supporting this claim
    expected_citation_count: Minimum expected citations for this claim type

Returns:
    Gap object if evidence gap detected, None otherwise

Example:
    >>> gap = detect_missing_evidence(
    ...     "claim-1",
    ...     "AI reduces diagnostic errors by 40%",
    ...     citations=[]
    ... )
    >>> if gap:
    ...     print(f"Gap type: {gap.gap_type}, Severity: {gap.severity}")
    Gap type: GapType.EVIDENCE, Severity: GapSeverity.MAJOR

**Parameters:**
- `claim_id`: str
- `claim_text`: str
- `citations`: list[Any] | None
- `expected_citation_count`: int

---

### `detect_polarity_opposition(claim_text_1, claim_text_2) -> tuple[bool, float]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:285`
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

### `detect_unsupported_claims(claims, min_citations_per_claim) -> list[Gap]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:500`
**Complexity:** 3

**Description:**
> Detect unsupported claims in a collection.

Batch analysis to identify all claims lacking adequate citation support.

Args:
    claims: List of (claim_id, claim_text, citations) tuples
    min_citations_per_claim: Minimum expected citations per claim

Returns:
    List of Gap objects for unsupported claims

Example:
    >>> claims = [
    ...     ("claim-1", "AI improves accuracy", []),
    ...     ("claim-2", "Studies show benefits", [citation1, citation2]),
    ... ]
    >>> gaps = detect_unsupported_claims(claims)
    >>> print(f"Found {len(gaps)} unsupported claims")

**Parameters:**
- `claims`: list[tuple[str, str, list[Any] | None]]
- `min_citations_per_claim`: int

---

### `doi_citation() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:66`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Citation with DOI reference.

---

### `emit_progress(message, prefix, style) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:35`
**Complexity:** 3

**Description:**
> Emit a progress update to stderr.

Progress messages are sent to stderr so they don't interfere with
stdout output (which may be parsed as JSON or redirected).

Args:
    message: Progress message to display
    prefix: Optional prefix (e.g., role name, stage name)
    style: Rich style for the message (default: cyan)

Example:
    >>> emit_progress("Analyzing argument...", prefix="Creator")
    [Creator] Analyzing argument...

    >>> emit_progress("Generating response...")
    Generating response...

**Parameters:**
- `message`: str
- `prefix`: str | None
- `style`: str

---

### `emit_provider_complete(provider, duration) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:93`
**Complexity:** 2

**Description:**
> Emit progress for provider execution completion.

Args:
    provider: Provider name
    duration: Optional execution duration in seconds

Example:
    >>> emit_provider_complete("claude", 2.3)
    [claude] Complete (2.3s)

**Parameters:**
- `provider`: str
- `duration`: float | None

---

### `emit_provider_start(provider) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:79`
**Complexity:** 1

**Description:**
> Emit progress for provider execution start.

Args:
    provider: Provider name

Example:
    >>> emit_provider_start("claude")
    [claude] Executing...

**Parameters:**
- `provider`: str

---

### `emit_stage(stage) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:65`
**Complexity:** 1

**Description:**
> Emit a workflow stage indicator.

Args:
    stage: Stage name (e.g., "Creator", "Skeptic", "Moderator")

Example:
    >>> emit_stage("Creator")
    [Creator] Generating initial perspective...

**Parameters:**
- `stage`: str

---

### `emit_workflow_complete(workflow) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:133`
**Complexity:** 1

**Description:**
> Emit workflow completion.

Args:
    workflow: Workflow name

Example:
    >>> emit_workflow_complete("argument")
    ✓ argument workflow complete

**Parameters:**
- `workflow`: str

---

### `emit_workflow_start(workflow, estimated_duration) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:113`
**Complexity:** 2

**Description:**
> Emit workflow start with optional time estimate.

Args:
    workflow: Workflow name
    estimated_duration: Optional human-readable duration estimate

Example:
    >>> emit_workflow_start("argument", "15-30s")
    Starting argument workflow (estimated: 15-30s)...

**Parameters:**
- `workflow`: str
- `estimated_duration`: str | None

---

### `async example_argument_basic() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:37`
**Complexity:** 4

**Description:**
> Example 1: Basic argument analysis.

Demonstrates the simplest use case - analyzing a single argument/claim
through dialectical reasoning (Creator → Skeptic → Moderator).

---

### `async example_argument_continuation() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:134`
**Complexity:** 3

**Description:**
> Example 3: Continuing an argument analysis.

Demonstrates how to use conversation threading to continue
a previous analysis with follow-up questions or new angles.

---

### `async example_argument_custom_config() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:176`
**Complexity:** 2

**Description:**
> Example 4: Argument analysis with custom configuration.

Shows how to customize the argument workflow with different
parameters like temperature, max_tokens, and system prompts.

---

### `async example_argument_with_files() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:85`
**Complexity:** 2

**Description:**
> Example 2: Argument analysis with file context.

Shows how to provide supporting documents/context files to enrich
the argument analysis with specific data or background information.

---

### `async example_error_handling() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:361`
**Complexity:** 3

**Description:**
> Example 13: Proper error handling across workflows.

Demonstrates best practices for handling workflow failures.

---

### `async example_ideate_basic() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:212`
**Complexity:** 4

**Description:**
> Example 5: Basic ideation/brainstorming.

Demonstrates simple creative idea generation with default parameters.

---

### `async example_ideate_high_creativity() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:250`
**Complexity:** 2

**Description:**
> Example 6: High-creativity brainstorming.

Uses maximum temperature for highly creative, unconventional ideas.

---

### `async example_ideate_refine() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:315`
**Complexity:** 3

**Description:**
> Example 8: Refining specific ideas through continuation.

Shows how to use threading to drill down into specific ideas
and develop them further.

---

### `async example_ideate_with_constraints() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:278`
**Complexity:** 2

**Description:**
> Example 7: Ideation with constraints/criteria.

Demonstrates how to guide idea generation with specific constraints
or requirements via system prompts.

---

### `async example_output_management() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:401`
**Complexity:** 2

**Description:**
> Example 14: Managing workflow outputs.

Shows how to save, load, and process workflow results.

---

### `async example_provider_comparison() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:453`
**Complexity:** 4

**Description:**
> Example 15: Comparing results across providers.

Demonstrates running the same query with different providers
to compare outputs.

---

### `file_citation() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:51`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Citation referencing a file.

---

### `find_duplicate_claims(citation_maps, threshold, model_name) -> list[list[CitationMap]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:331`
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
- `citation_maps`: list[CitationMap]
- `threshold`: float
- `model_name`: str

---

### `find_similar_claims(query_claim, citation_maps, threshold, top_k, model_name) -> list[tuple[CitationMap, float]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:177`
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
- `citation_maps`: list[CitationMap]
- `threshold`: float
- `top_k`: int | None
- `model_name`: str

---

### `fix_multiline_call(content) -> None`

**Language:** python
**Defined in:** `model_chorus/fix_thinkdeep_calls.py:44`
**Complexity:** 4

**Description:**
> Fix multi-line workflow.run calls.

**Parameters:**
- `content`: None

---

### `fix_simple_call(match) -> None`

**Language:** python
**Defined in:** `model_chorus/fix_thinkdeep_calls.py:23`
**Complexity:** 1

**Description:**
> Fix a simple single-line prompt call.

**Parameters:**
- `match`: None

---

### `format_citation(citation, style) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:23`
**Complexity:** 4

**Description:**
> Format a Citation object according to the specified style.

Args:
    citation: The Citation object to format
    style: The citation style to use (APA, MLA, or Chicago)

Returns:
    Formatted citation string according to the specified style

Example:
    >>> from model_chorus.core.models import Citation
    >>> from model_chorus.utils.citation_formatter import format_citation, CitationStyle
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
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:186`
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
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:617`
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
- `cluster`: list[CitationMap]
- `model_name`: str
- `max_words`: int

---

### `generate_contradiction_explanation(severity, semantic_similarity, has_polarity_opposition, polarity_confidence) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:413`
**Complexity:** 4

**Description:**
> Generate human-readable explanation for a detected contradiction.

Creates a detailed explanation describing why two claims contradict,
including semantic similarity scores and polarity analysis.

Args:
    severity: Assessed severity level of the contradiction
    semantic_similarity: Cosine similarity between claims (0.0-1.0)
    has_polarity_opposition: Whether claims have opposing polarity
    polarity_confidence: Confidence in polarity detection (0.0-1.0)

Returns:
    Formatted explanation string describing the contradiction

Example:
    >>> explanation = generate_contradiction_explanation(
    ...     severity=ContradictionSeverity.CRITICAL,
    ...     semantic_similarity=0.85,
    ...     has_polarity_opposition=True,
    ...     polarity_confidence=0.8
    ... )
    >>> print(explanation)
    Claims have opposing polarity (confidence: 0.80). Semantic similarity: 0.85. Claims are highly related but present contradictory assertions

**Parameters:**
- `severity`: ContradictionSeverity
- `semantic_similarity`: float
- `has_polarity_opposition`: bool
- `polarity_confidence`: float

---

### `generate_gap_recommendation(gap_type, severity, claim_text) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/gap_analysis.py:273`
**Complexity:** 6

**Description:**
> Generate actionable recommendation for addressing a gap.

Provides specific guidance based on gap type and severity to help
improve argument completeness and validity.

Args:
    gap_type: Type of gap detected
    severity: Severity level of the gap
    claim_text: Text of the claim with the gap

Returns:
    Recommendation string with actionable guidance

Example:
    >>> rec = generate_gap_recommendation(
    ...     GapType.EVIDENCE,
    ...     GapSeverity.MAJOR,
    ...     "AI improves accuracy"
    ... )
    >>> print(rec)
    Add empirical evidence with specific citations. Find peer-reviewed studies or data supporting this claim.

**Parameters:**
- `gap_type`: GapType
- `severity`: GapSeverity
- `claim_text`: str

---

### `generate_reconciliation_suggestion(severity) -> str | None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/contradiction.py:469`
**Complexity:** 4

**Description:**
> Generate reconciliation suggestion based on contradiction severity.

Provides actionable guidance for investigating and resolving contradictions.
Higher severity contradictions receive more urgent recommendations.

Args:
    severity: Severity level of the contradiction

Returns:
    Suggestion string for CRITICAL/MAJOR/MODERATE, None for MINOR

Example:
    >>> suggestion = generate_reconciliation_suggestion(ContradictionSeverity.CRITICAL)
    >>> print(suggestion)
    Investigate source reliability and experimental conditions. One or both claims may be incorrect.

**Parameters:**
- `severity`: ContradictionSeverity

---

### `get_claude_config() -> ModelChorusConfigV2`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:895`
**Complexity:** 1

**Description:**
> Get the current Claude configuration (convenience function).

Returns:
    Current configuration from .claude/model_chorus_config.yaml

---

### `get_claude_config_loader() -> ClaudeConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:887`
**Complexity:** 2

**Description:**
> Get the global Claude config loader instance.

---

### `get_cluster_representative(cluster, model_name) -> CitationMap`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:572`
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
- `cluster`: list[CitationMap]
- `model_name`: str

---

### `get_config() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:175`
**Complexity:** 3

**Description:**
> Get the config loader instance, initializing if needed.

---

### `get_config() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:42`
**Complexity:** 3

**Description:**
> Get the config loader instance, initializing if needed.

---

### `get_config() -> ModelChorusConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:439`
**Complexity:** 1

**Description:**
> Get the current configuration (convenience function).

Returns:
    Current configuration

---

### `get_config_loader() -> ConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/config/loader.py:326`
**Complexity:** 2

**Description:**
> Get the global ConfigLoader instance.

Returns:
    Singleton ConfigLoader instance

---

### `get_config_loader() -> ConfigLoader`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:418`
**Complexity:** 2

**Description:**
> Get the global config loader instance.

---

### `get_default_registry() -> PersonaRegistry`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/study/personas/__init__.py:39`
**Complexity:** 2

**Description:**
> Get a registry pre-populated with default personas.

Returns:
    PersonaRegistry with Researcher, Critic, and Planner registered

---

### `get_default_state_manager() -> StateManager`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/state.py:511`
**Complexity:** 2

**Description:**
> Get singleton default state manager instance.

Returns:
    Default StateManager instance

Example:
    >>> from model_chorus.core.state import get_default_state_manager
    >>> manager = get_default_state_manager()
    >>> manager.set_state("my_workflow", {"step": 1})

---

### `get_install_command(provider) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:188`
**Complexity:** 1

**Description:**
> Get installation command for a provider CLI.

Args:
    provider: Provider name (claude, gemini, codex, cursor-agent)

Returns:
    Installation command string

**Parameters:**
- `provider`: str

---

### `get_install_command(provider) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:55`
**Complexity:** 1

**Description:**
> Get installation command for a provider CLI.

Args:
    provider: Provider name (claude, gemini, codex, cursor-agent)

Returns:
    Installation command string

**Parameters:**
- `provider`: str

---

### `get_provider_by_name(name, timeout) -> ClaudeProvider | CodexProvider | GeminiProvider | CursorAgentProvider`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:206`
**Complexity:** 3

**Description:**
> Get provider instance by name.

Args:
    name: Provider name (claude, gemini, codex, cursor-agent)
    timeout: Timeout in seconds for provider operations (default: 120)

Returns:
    Provider instance

Raises:
    ProviderDisabledError: If provider is disabled in config
    typer.Exit: If provider is unknown

**Parameters:**
- `name`: str
- `timeout`: int

---

### `get_provider_by_name(name, timeout) -> ClaudeProvider | CodexProvider | GeminiProvider | CursorAgentProvider`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:73`
**Complexity:** 2

**Description:**
> Get provider instance by name.

Args:
    name: Provider name (claude, gemini, codex, cursor-agent)
    timeout: Timeout in seconds for provider operations (default: 120)

Returns:
    Provider instance

**Parameters:**
- `name`: str
- `timeout`: int

---

### `get_read_only_system_prompt() -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/prompts.py:9`
**Complexity:** 1

**Description:**
> Generate system prompt informing AI of read-only tool constraints.

This prompt should be prepended to workflow-specific system prompts
to set expectations about available tool capabilities when operating
via external CLI tools.

The constraints are enforced at the CLI provider level (no --yolo flag
for Gemini, etc.), but this prompt ensures the AI understands the
limitations upfront, preventing wasted API tokens on failed attempts.

Returns:
    System prompt text describing read-only constraints

---

### `get_run_kwargs(provider_name, prompt) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:25`
**Complexity:** 3

**Description:**
> Get kwargs for chat_workflow.run() that are compatible with the provider.

Filters out unsupported parameters (e.g., temperature for Gemini).
Note: Fast models are automatically injected by the provider fixture.

**Parameters:**
- `provider_name`: str
- `prompt`: str

---

### `async hypothesis_management_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/thinkdeep_example.py:290`
**Complexity:** 6

**Description:**
> Demonstrate manual hypothesis management and state inspection.

Shows how to programmatically add, update, and track hypotheses
during an investigation.

---

### `ideate(prompt_arg, prompt_flag, provider, continuation_id, files, num_ideas, system, temperature, timeout, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:679`
⚠️ **Complexity:** 13 (High)

**Decorators:** `@app.command()`

**Description:**
> Generate creative ideas through structured brainstorming.

The ideate workflow uses enhanced creative prompting to generate diverse
and innovative ideas for any topic or problem.

Example:
    # Generate ideas (positional prompt)
    model-chorus ideate "New features for a task management app"

    # Generate ideas (flag prompt)
    model-chorus ideate --prompt "New features for a task management app"

    # Control creativity and quantity
    model-chorus ideate "Marketing campaign ideas" -n 10

    # Continue brainstorming
    model-chorus ideate "Refine the third idea" --continue thread-id-123

**Parameters:**
- `prompt_arg`: str | None
- `prompt_flag`: str | None
- `provider`: str | None
- `continuation_id`: str | None
- `files`: list[str] | None
- `num_ideas`: int
- `system`: str | None
- `temperature`: float | None
- `timeout`: float | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `ideate_workflow(mock_provider) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:123`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create IdeateWorkflow instance for testing.

**Parameters:**
- `mock_provider`: None

---

### `ideate_workflow(mock_provider, conversation_memory) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:140`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create IdeateWorkflow instance for testing.

**Parameters:**
- `mock_provider`: None
- `conversation_memory`: None

---

### `install_package(dev_mode) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:54`
**Complexity:** 4

**Description:**
> Install model-chorus package.

Args:
    dev_mode: If True, install in editable mode with -e flag

Returns:
    Dict with installation result

**Parameters:**
- `dev_mode`: bool

---

### `async investigation_with_expert_validation() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/thinkdeep_example.py:202`
**Complexity:** 4

**Description:**
> Investigation with expert validation from a different model.

Demonstrates how ThinkDeep can use a second model for validation
and additional insights when confidence hasn't reached "certain" level.

---

### `is_cli_available(cli_command) -> bool`

**Language:** python
**Defined in:** `model_chorus/tests/test_helpers.py:64`
**Complexity:** 4

**Description:**
> Check if a CLI command is available in PATH and working.

Args:
    cli_command: The CLI command to check (e.g., 'claude', 'gemini')

Returns:
    bool: True if CLI is available and responds to --version

**Parameters:**
- `cli_command`: str

---

### `is_progress_enabled() -> bool`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:30`
**Complexity:** 1

**Description:**
> Check if progress output is enabled.

---

### `is_provider_available(provider_name, cli_command, model_chorus_config) -> bool`

**Language:** python
**Defined in:** `model_chorus/tests/test_helpers.py:90`
**Complexity:** 2

**Description:**
> Combined check: provider must be enabled in config AND have CLI available.

Args:
    provider_name: Name of the provider (e.g., 'claude', 'gemini', 'codex')
    cli_command: CLI command for the provider
    model_chorus_config: The loaded model_chorus_config providers section

Returns:
    bool: True if both config-enabled and CLI available

**Parameters:**
- `provider_name`: str
- `cli_command`: str
- `model_chorus_config`: dict

---

### `is_provider_enabled_in_config(provider_name, model_chorus_config) -> bool`

**Language:** python
**Defined in:** `model_chorus/tests/test_helpers.py:42`
**Complexity:** 2

**Description:**
> Check if a provider is enabled in model_chorus_config.yaml.

Args:
    provider_name: Name of the provider (e.g., 'claude', 'gemini', 'codex')
    model_chorus_config: The loaded model_chorus_config providers section

Returns:
    bool: True if enabled in config or if no config exists (default: enabled)

**Parameters:**
- `provider_name`: str
- `model_chorus_config`: dict

---

### `list_providers(check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1943`
**Complexity:** 8

**Decorators:** `@app.command()`

**Description:**
> List all available providers and their models.

Use --check to verify which providers are actually installed and working.

**Parameters:**
- `check`: bool

---

### `load_config(config_path) -> ModelChorusConfig`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/config.py:426`
**Complexity:** 1

**Description:**
> Load configuration (convenience function).

Args:
    config_path: Optional explicit path to config file

Returns:
    Loaded configuration

**Parameters:**
- `config_path`: Path | None

---

### `load_model_chorus_config() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_helpers.py:15`
**Complexity:** 4

**Description:**
> Load model_chorus_config.yaml to determine which providers are enabled.

Returns:
    dict: The 'providers' section from model_chorus_config.yaml, or empty dict if not found.

---

### `main() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:562`
**Complexity:** 2

**Description:**
> Main entry point.

---

### `async main() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/basic_workflow.py:58`
**Complexity:** 2

**Description:**
> Main entry point for the example.

---

### `async main() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/chat_example.py:190`
**Complexity:** 1

**Description:**
> Run all examples.

---

### `async main() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/provider_integration.py:99`
**Complexity:** 2

**Description:**
> Main entry point for the provider example.

---

### `async main() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/thinkdeep_example.py:458`
**Complexity:** 1

**Description:**
> Run all examples.

---

### `main() -> None`

**Language:** python
**Defined in:** `model_chorus/fix_thinkdeep_calls.py:110`
**Complexity:** 4

**Description:**
> Main function.

---

### `main() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:2039`
**Complexity:** 1

**Description:**
> Main entry point for CLI.

---

### `main() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:920`
⚠️ **Complexity:** 21 (High)

**Description:**
> CLI entry point for setup commands.

---

### `minimal_citation() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_citation_integration.py:42`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Citation with only required fields.

---

### `mock_brainstorming_result() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:129`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create mock brainstorming result for convergent analysis testing.

---

### `mock_claude_provider_full() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:293`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create fully mocked Claude provider for integration tests.

---

### `mock_claude_response() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:108`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock response from Claude CLI --output-format json.

---

### `mock_codex_provider_full() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:305`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create fully mocked Codex provider for integration tests.

---

### `mock_codex_response() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:133`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock JSONL response from Codex CLI --json.

---

### `mock_cursor_agent_provider_full() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:311`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create fully mocked Cursor Agent provider for integration tests.

---

### `mock_cursor_agent_response() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:143`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock JSON response from Cursor Agent CLI --output-format json.

---

### `mock_gemini_provider_full() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:299`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create fully mocked Gemini provider for integration tests.

---

### `mock_provider() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:322`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Generic mock ModelProvider for testing.

Provides a simple mock provider suitable for most unit tests.
For integration tests with smart context handling, use the
provider-specific fixtures above (mock_claude_provider_full, etc.).

---

### `mock_provider() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow.py:17`
**Complexity:** 8

**Decorators:** `@pytest.fixture`

**Description:**
> Mock ModelProvider for testing.

---

### `mock_provider() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_ideate_workflow_integration.py:18`
**Complexity:** 7

**Decorators:** `@pytest.fixture`

**Description:**
> Mock ModelProvider for testing.

---

### `mock_provider() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:38`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create a mock ModelProvider for testing middleware.

---

### `mock_subprocess_run() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:159`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock subprocess.run for CLI command execution.

---

### `mock_workflow_result() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:33`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock workflow result for CLI testing.

---

### `async multi_step_investigation_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/thinkdeep_example.py:61`
**Complexity:** 10

**Description:**
> Multi-step investigation showing hypothesis evolution and confidence progression.

This example demonstrates a systematic investigation of a performance issue,
showing how hypotheses are formed, tested, and confidence evolves across steps.

---

### `async multi_turn_conversation_example() -> None`

**Language:** python
**Defined in:** `model_chorus/examples/chat_example.py:45`
**Complexity:** 4

**Description:**
> Multi-turn conversation with continuation.

---

### `prepend_system_constraints(custom_prompt) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/prompts.py:49`
**Complexity:** 2

**Description:**
> Prepend read-only environment constraints to a custom system prompt.

Combines the read-only constraints with any workflow-specific
system prompt instructions. Constraints are prepended so they
establish context before task-specific guidance.

Args:
    custom_prompt: Optional workflow-specific system prompt.
                  If None, returns just the constraints.

Returns:
    Combined system prompt with constraints prepended

Example:
    >>> ideation_prompt = "You are a creative brainstorming expert..."
    >>> final_prompt = prepend_system_constraints(ideation_prompt)
    >>> # Returns constraints + ideation prompt

**Parameters:**
- `custom_prompt`: str | None

---

### `process_file(filepath) -> None`

**Language:** python
**Defined in:** `model_chorus/fix_thinkdeep_calls.py:81`
**Complexity:** 2

**Description:**
> Process a single test file.

**Parameters:**
- `filepath`: None

---

### `provider(provider_name, mock_claude_provider_full, mock_gemini_provider_full, mock_codex_provider_full, mock_cursor_agent_provider_full) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_chat_integration.py:57`
**Complexity:** 3

**Decorators:** `@pytest.fixture`

**Description:**
> Create provider instance based on provider name.

Uses mock providers if USE_MOCK_PROVIDERS=true, otherwise uses real CLI providers.
Automatically configures the fastest model for each provider to minimize test time and cost.

**Parameters:**
- `provider_name`: None
- `mock_claude_provider_full`: None
- `mock_gemini_provider_full`: None
- `mock_codex_provider_full`: None
- `mock_cursor_agent_provider_full`: None

---

### `provider_name(request) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:96`
**Complexity:** 1

**Decorators:** `@pytest.fixture(params=[pytest.param('claude', marks=pytest.mark.skipif(not CLAUDE_AVAILABLE, reason='Claude not available (config disabled or CLI not found)')), pytest.param('gemini', marks=pytest.mark.skipif(not GEMINI_AVAILABLE, reason='Gemini not available (config disabled or CLI not found)')), pytest.param('codex', marks=pytest.mark.skipif(not CODEX_AVAILABLE, reason='Codex not available (config disabled or CLI not found)')), pytest.param('cursor-agent', marks=pytest.mark.skipif(not CURSOR_AGENT_AVAILABLE, reason='Cursor Agent not available (config disabled or CLI not found)'))])`

**Description:**
> Parameterized fixture for provider names with auto-skipping.

Tests using this fixture will be automatically run once for each available provider.
Providers that are disabled in ai_config.yaml or don't have their CLI installed
will be automatically skipped.

**Parameters:**
- `request`: None

---

### `pytest_configure(config) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:35`
**Complexity:** 1

**Description:**
> Register custom markers for provider-specific tests.

**Parameters:**
- `config`: None

---

### `resolve_context_files(files) -> tuple[list[str], list[str], list[str]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:88`
⚠️ **Complexity:** 13 (High)

**Description:**
> Normalize and resolve context file paths with legacy mapping support.

Args:
    files: Raw string (comma-separated) or iterable of file paths.

Returns:
    Tuple containing:
        resolved_paths: Paths that exist after normalization/mapping.
        remapped_notices: Descriptions of legacy paths that were remapped.
        missing_files: Original entries that could not be resolved.

**Parameters:**
- `files`: str | Sequence[str] | None

---

### `review_json_data(review_json_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:28`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Load review JSON data.

**Parameters:**
- `review_json_path`: None

---

### `review_json_path() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:19`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Path to the review JSON file.

---

### `async run_all_examples() -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:498`
**Complexity:** 4

**Description:**
> Run all examples in sequence.

---

### `async run_specific_example(example_name) -> None`

**Language:** python
**Defined in:** `examples/workflow_examples.py:537`
**Complexity:** 2

**Description:**
> Run a specific example by name.

**Parameters:**
- `example_name`: str

---

### `sample_generation_request() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/conftest.py:169`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Sample GenerationRequest for testing.

---

### `sample_request() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:47`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create a sample GenerationRequest for testing.

---

### `sample_response() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_middleware.py:57`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create a sample GenerationResponse for testing.

---

### `score_cluster_coherence(cluster, model_name) -> float`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:803`
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
- `cluster`: list[CitationMap]
- `model_name`: str

---

### `score_cluster_separation(clusters, model_name) -> float`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:850`
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
- `clusters`: list[list[CitationMap]]
- `model_name`: str

---

### `score_clustering_quality(clusters, model_name) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:922`
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
- `clusters`: list[list[CitationMap]]
- `model_name`: str

---

### `set_progress_enabled(enabled) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/core/progress.py:19`
**Complexity:** 1

**Description:**
> Enable or disable progress output.

Args:
    enabled: True to enable progress output, False to disable

**Parameters:**
- `enabled`: bool

---

### `start(scenario, provider, continuation_id, files, personas, system, temperature, max_tokens, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:108`
⚠️ **Complexity:** 44 (High)

**Decorators:** `@study_app.command()`

**Description:**
> Conduct persona-based collaborative research.

The STUDY workflow provides multi-persona investigation with role-based
orchestration, enabling collaborative exploration of complex topics through
specialized personas with distinct expertise.

Example:
    # Start new investigation
    model-chorus study start --scenario "Explore authentication system patterns"

    # Continue investigation
    model-chorus study start --scenario "Deep dive into OAuth 2.0" --continue thread-id-123

    # Include files
    model-chorus study start --scenario "Analyze this codebase" -f src/auth.py -f tests/test_auth.py

    # Specify personas
    model-chorus study start --scenario "Security analysis" --persona SecurityExpert --persona Architect

**Parameters:**
- `scenario`: str
- `provider`: str | None
- `continuation_id`: str | None
- `files`: list[str] | None
- `personas`: list[str] | None
- `system`: str | None
- `temperature`: float | None
- `max_tokens`: int | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `storage(temp_db) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:33`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create and initialize a LongTermStorage instance.

**Parameters:**
- `temp_db`: None

---

### `study_next(investigation, provider, files, max_tokens, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:427`
⚠️ **Complexity:** 35 (High)

**Decorators:** `@study_app.command(name='next')`

**Description:**
> Continue an existing STUDY investigation.

This command automatically continues an investigation using the existing
thread context and personas. It's a convenience wrapper around the study
command with automatic continuation.

The command retrieves the investigation's conversation history and prompts
the next investigation step based on the current state.

Example:
    # Continue investigation with automatic next step
    model-chorus study-next --investigation thread-id-123

    # Continue with additional files
    model-chorus study-next --investigation thread-id-123 -f new_data.py

    # Continue with specific provider
    model-chorus study-next --investigation thread-id-123 -p gemini

**Parameters:**
- `investigation`: str
- `provider`: str | None
- `files`: list[str] | None
- `max_tokens`: int | None
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `study_view(investigation, persona, show_all, format_json, verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/study_commands.py:707`
⚠️ **Complexity:** 31 (High)

**Decorators:** `@study_app.command(name='view')`

**Description:**
> View memory and conversation history for a STUDY investigation.

This command displays the conversation memory for an investigation,
including all messages, persona contributions, and investigation metadata.
Useful for reviewing investigation history and debugging.

Example:
    # View investigation summary
    model-chorus study-view --investigation thread-id-123

    # View all messages
    model-chorus study-view --investigation thread-id-123 --show-all

    # Filter by persona
    model-chorus study-view --investigation thread-id-123 --persona Researcher

    # Output as JSON
    model-chorus study-view --investigation thread-id-123 --json

**Parameters:**
- `investigation`: str
- `persona`: str | None
- `show_all`: bool
- `format_json`: bool
- `verbose`: bool

---

### `summarize_cluster(cluster, model_name, max_length) -> str`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/workflows/argument/semantic.py:666`
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
- `cluster`: list[CitationMap]
- `model_name`: str
- `max_length`: int

---

### `temp_db() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:22`
**Complexity:** 2

**Decorators:** `@pytest.fixture`

**Description:**
> Create a temporary database file for testing.

---

### `temp_output_file(tmp_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:68`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create temporary output file path.

**Parameters:**
- `tmp_path`: None

---

### `temp_test_file(tmp_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_cli_integration.py:60`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Create temporary test file for file input tests.

**Parameters:**
- `tmp_path`: None

---

### `test_cache_clear(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:183`
**Complexity:** 2

**Description:**
> Test clearing all entries from cache.

**Parameters:**
- `cache`: None

---

### `test_cache_delete(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:129`
**Complexity:** 1

**Description:**
> Test deleting entries from cache.

**Parameters:**
- `cache`: None

---

### `test_cache_eviction_metric(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:84`
**Complexity:** 2

**Description:**
> Test that eviction count is tracked correctly.

**Parameters:**
- `cache`: None

---

### `test_cache_metadata(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:203`
**Complexity:** 2

**Description:**
> Test retrieving cache metadata.

**Parameters:**
- `cache`: None

---

### `test_cache_metrics_hits_and_misses(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:61`
**Complexity:** 1

**Description:**
> Test that cache hit/miss metrics are tracked correctly.

**Parameters:**
- `cache`: None

---

### `test_cache_query(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:151`
**Complexity:** 2

**Description:**
> Test querying cache with filters.

**Parameters:**
- `cache`: None

---

### `test_cache_update_existing(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:101`
**Complexity:** 1

**Description:**
> Test updating an existing cache entry.

**Parameters:**
- `cache`: None

---

### `async test_consensus_applies_provider_model_override_without_mutation() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_consensus_provider_models.py:29`
**Complexity:** 3

**Decorators:** `@pytest.mark.asyncio`

**Description:**
> Ensure provider-specific metadata adds model override and preserves shared metadata.

---

### `test_delete(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:140`
**Complexity:** 1

**Description:**
> Test deleting entries.

**Parameters:**
- `storage`: None

---

### `test_dimension_scores_structure(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:78`
**Complexity:** 2

**Description:**
> Test that dimension_scores has the expected structure.

**Parameters:**
- `review_json_data`: None

---

### `test_get_metadata(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:188`
**Complexity:** 2

**Description:**
> Test retrieving storage metadata.

**Parameters:**
- `storage`: None

---

### `test_issues_have_locations(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:147`
**Complexity:** 4

**Description:**
> Test that issues have location information when applicable.

**Parameters:**
- `review_json_data`: None

---

### `test_issues_structure(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:89`
**Complexity:** 3

**Description:**
> Test that issues list has the expected structure.

**Parameters:**
- `review_json_data`: None

---

### `test_lru_eviction(cache) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_cache.py:23`
**Complexity:** 2

**Description:**
> Test that LRU eviction works correctly when cache is full.

**Parameters:**
- `cache`: None

---

### `test_memory_references(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:166`
**Complexity:** 1

**Description:**
> Test that memory references are saved and retrieved.

**Parameters:**
- `storage`: None

---

### `test_overall_score_valid(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:61`
**Complexity:** 1

**Description:**
> Test that overall_score is a valid integer between 0 and 10.

**Parameters:**
- `review_json_data`: None

---

### `test_persistence_across_sessions(temp_db) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:70`
**Complexity:** 1

**Description:**
> Test that data persists across storage instances (sessions).

**Parameters:**
- `temp_db`: None

---

### `test_query_by_investigation(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:98`
**Complexity:** 2

**Description:**
> Test querying entries by investigation ID.

**Parameters:**
- `storage`: None

---

### `test_query_by_persona(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:119`
**Complexity:** 2

**Description:**
> Test querying entries by persona.

**Parameters:**
- `storage`: None

---

### `test_recommendation_valid(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:68`
**Complexity:** 1

**Description:**
> Test that recommendation is a valid value.

**Parameters:**
- `review_json_data`: None

---

### `test_review_json_exists(review_json_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:34`
**Complexity:** 1

**Description:**
> Test that review JSON file exists.

**Parameters:**
- `review_json_path`: None

---

### `test_review_json_has_required_fields(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:48`
**Complexity:** 2

**Description:**
> Test that review JSON has all required top-level fields.

**Parameters:**
- `review_json_data`: None

---

### `test_review_json_valid_json(review_json_path) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:41`
**Complexity:** 1

**Description:**
> Test that review JSON file contains valid JSON.

**Parameters:**
- `review_json_path`: None

---

### `test_review_response_completeness(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:164`
**Complexity:** 2

**Description:**
> Test that review response has meaningful content.

**Parameters:**
- `review_json_data`: None

---

### `test_review_response_consistency(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:130`
**Complexity:** 3

**Description:**
> Test consistency between overall_score and recommendation.

**Parameters:**
- `review_json_data`: None

---

### `test_save_and_retrieve(storage) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/workflows/study/memory/test_persistence.py:41`
**Complexity:** 1

**Description:**
> Test that entries can be saved and retrieved correctly.

**Parameters:**
- `storage`: None

---

### `async test_shared_request_model_overrides_provider_metadata() -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_consensus_provider_models.py:64`
**Complexity:** 1

**Decorators:** `@pytest.mark.asyncio`

**Description:**
> Shared request metadata should take precedence over provider-level overrides.

---

### `test_strengths_structure(review_json_data) -> None`

**Language:** python
**Defined in:** `model_chorus/tests/test_review_response.py:113`
**Complexity:** 2

**Description:**
> Test that strengths list has the expected structure.

**Parameters:**
- `review_json_data`: None

---

### `thinkdeep(step, step_number, total_steps, next_step_required, findings, provider, continuation_id, hypothesis, confidence, files_checked, relevant_files, thinking_mode, use_assistant_model, output, verbose, skip_provider_check) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1242`
⚠️ **Complexity:** 41 (High)

**Decorators:** `@app.command()`

**Description:**
> Start a ThinkDeep investigation for systematic problem analysis.

ThinkDeep provides multi-step investigation with explicit hypothesis tracking,
confidence progression, and state management across investigation steps.

Example:
    # Start new investigation
    model-chorus thinkdeep --step "Investigate why API latency increased" --step-number 1 --total-steps 3 --next-step-required --findings "Examining deployment logs" --confidence exploring

    # Continue investigation
    model-chorus thinkdeep --continuation-id "thread-123" --step "Check database query performance" --step-number 2 --total-steps 3 --next-step-required --findings "Found N+1 query pattern" --confidence medium --hypothesis "N+1 queries causing slowdown"

    # Final step (omit --next-step-required)
    model-chorus thinkdeep --continuation-id "thread-123" --step "Verify fix resolves issue" --step-number 3 --total-steps 3 --findings "Latency reduced to baseline" --confidence high --hypothesis "Confirmed: N+1 queries were root cause"

**Parameters:**
- `step`: str
- `step_number`: int
- `total_steps`: int
- `next_step_required`: bool
- `findings`: str
- `provider`: str | None
- `continuation_id`: str | None
- `hypothesis`: str | None
- `confidence`: str
- `files_checked`: str | None
- `relevant_files`: str | None
- `thinking_mode`: str | None
- `use_assistant_model`: bool
- `output`: Path | None
- `verbose`: bool
- `skip_provider_check`: bool

---

### `thinkdeep_status(thread_id, show_steps, show_files, verbose) -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:1585`
⚠️ **Complexity:** 18 (High)

**Decorators:** `@app.command(name='thinkdeep-status')`

**Description:**
> Inspect the state of an ongoing ThinkDeep investigation.

Shows current hypotheses, confidence level, investigation progress,
and optionally all steps and examined files.

Example:
    # View basic status
    model-chorus thinkdeep-status thread-id-123

    # View with all steps
    model-chorus thinkdeep-status thread-id-123 --steps

    # View with files
    model-chorus thinkdeep-status thread-id-123 --files --verbose

**Parameters:**
- `thread_id`: str
- `show_steps`: bool
- `show_files`: bool
- `verbose`: bool

---

### `validate_citation(citation) -> tuple[bool, list[str]]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/utils/citation_formatter.py:233`
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

### `validate_config(project_root) -> dict[str, Any]`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/setup.py:716`
**Complexity:** 5

**Description:**
> Validate .model-chorusrc configuration file.

Args:
    project_root: Project root directory (defaults to cwd)

Returns:
    Dict with validation result

**Parameters:**
- `project_root`: Path | None

---

### `version() -> None`

**Language:** python
**Defined in:** `model_chorus/src/model_chorus/cli/main.py:2028`
**Complexity:** 1

**Decorators:** `@app.command()`

**Description:**
> Show version information.

---


## 📦 Dependencies

### `examples/workflow_examples.py`

- `asyncio`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.ClaudeProvider`
- `model_chorus.providers.GeminiProvider`
- `model_chorus.workflows.ArgumentWorkflow`
- `model_chorus.workflows.IdeateWorkflow`
- `pathlib.Path`
- `sys`

### `model_chorus/examples/basic_workflow.py`

- `asyncio`
- `model_chorus.core.BaseWorkflow`
- `model_chorus.core.WorkflowRegistry`
- `model_chorus.core.WorkflowRequest`
- `model_chorus.core.WorkflowResult`

### `model_chorus/examples/chat_example.py`

- `asyncio`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.ClaudeProvider`
- `model_chorus.workflows.ChatWorkflow`
- `pathlib.Path`

### `model_chorus/examples/provider_integration.py`

- `asyncio`
- `model_chorus.providers.GenerationRequest`
- `model_chorus.providers.GenerationResponse`
- `model_chorus.providers.ModelCapability`
- `model_chorus.providers.ModelConfig`
- `model_chorus.providers.ModelProvider`

### `model_chorus/examples/thinkdeep_example.py`

- `asyncio`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.ClaudeProvider`
- `model_chorus.providers.GeminiProvider`
- `model_chorus.workflows.ConfidenceLevel`
- `model_chorus.workflows.ThinkDeepWorkflow`
- `pathlib.Path`

### `model_chorus/fix_thinkdeep_calls.py`

- `pathlib.Path`
- `re`

### `model_chorus/src/model_chorus/cli/__init__.py`

- `main.app`
- `main.main`

### `model_chorus/src/model_chorus/cli/main.py`

- `asyncio`
- `collections.abc.Sequence`
- `config.get_config_loader`
- `core.config.get_claude_config_loader`
- `core.context_ingestion.BinaryFileError`
- `core.context_ingestion.ContextIngestionService`
- `core.context_ingestion.FileTooLargeError`
- `core.conversation.ConversationMemory`
- `json`
- `model_chorus.__version__`
- `pathlib.Path`
- `primitives.OutputFormatter`
- `primitives.ProviderResolver`
- `primitives.WorkflowContext`
- `providers.ClaudeProvider`
- `providers.CodexProvider`
- `providers.CursorAgentProvider`
- `providers.GeminiProvider`
- `providers.GenerationRequest`
- `providers.base_provider.ModelProvider`
- `providers.cli_provider.ProviderUnavailableError`
- `rich.console.Console`
- `rich.table.Table`
- `study_commands.study_app`
- `typer`
- `typing.cast`
- `workflows.ArgumentWorkflow`
- `workflows.ChatWorkflow`
- `workflows.ConsensusStrategy`
- `workflows.ConsensusWorkflow`
- `workflows.IdeateWorkflow`
- `workflows.ThinkDeepWorkflow`

### `model_chorus/src/model_chorus/cli/primitives.py`

- `rich.console.Console`
- `typing.TYPE_CHECKING`

### `model_chorus/src/model_chorus/cli/setup.py`

- `asyncio`
- `json`
- `pathlib.Path`
- `subprocess`
- `sys`
- `typing.Any`

### `model_chorus/src/model_chorus/cli/study_commands.py`

- `asyncio`
- `core.config.get_config_loader`
- `core.conversation.ConversationMemory`
- `json`
- `pathlib.Path`
- `providers.ClaudeProvider`
- `providers.CodexProvider`
- `providers.CursorAgentProvider`
- `providers.GeminiProvider`
- `providers.base_provider.ModelProvider`
- `providers.cli_provider.ProviderUnavailableError`
- `rich.console.Console`
- `typer`
- `typing.Any`
- `workflows.study.StudyWorkflow`

### `model_chorus/src/model_chorus/config/__init__.py`

- `loader.ConfigLoader`
- `loader.get_config_loader`
- `models.GenerationDefaults`
- `models.ModelChorusConfig`
- `models.ProviderConfig`
- `models.WorkflowConfig`

### `model_chorus/src/model_chorus/config/loader.py`

- `json`
- `models.ModelChorusConfig`
- `models.ProviderConfig`
- `models.WorkflowConfig`
- `pathlib.Path`
- `typing.Any`

### `model_chorus/src/model_chorus/config/models.py`

- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.field_validator`
- `typing.Literal`

### `model_chorus/src/model_chorus/core/__init__.py`

- `base_workflow.BaseWorkflow`
- `base_workflow.WorkflowResult`
- `base_workflow.WorkflowStep`
- `contradiction.Contradiction`
- `contradiction.ContradictionSeverity`
- `contradiction.assess_contradiction_severity`
- `contradiction.detect_contradiction`
- `contradiction.detect_contradictions_batch`
- `contradiction.detect_polarity_opposition`
- `contradiction.generate_contradiction_explanation`
- `contradiction.generate_reconciliation_suggestion`
- `conversation.ConversationMemory`
- `gap_analysis.Gap`
- `gap_analysis.GapSeverity`
- `gap_analysis.GapType`
- `gap_analysis.assess_gap_severity`
- `gap_analysis.detect_gaps`
- `gap_analysis.detect_logical_gaps`
- `gap_analysis.detect_missing_evidence`
- `gap_analysis.detect_unsupported_claims`
- `gap_analysis.generate_gap_recommendation`
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
- `prompts.get_read_only_system_prompt`
- `prompts.prepend_system_constraints`
- `registry.WorkflowRegistry`

### `model_chorus/src/model_chorus/core/base_workflow.py`

- `abc.ABC`
- `abc.abstractmethod`
- `asyncio`
- `conversation.ConversationMemory`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `logging`
- `models.ConversationMessage`
- `models.ConversationThread`
- `typing.Any`
- `typing.Literal`
- `typing.TYPE_CHECKING`
- `workflow_runner.ExecutionMetrics`
- `workflow_runner.WorkflowRunner`

### `model_chorus/src/model_chorus/core/clustering.py`

- `dataclasses.dataclass`
- `dataclasses.field`
- `numpy`
- `typing.Any`

### `model_chorus/src/model_chorus/core/config.py`

- `json`
- `pathlib.Path`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.field_validator`
- `typing.Any`

### `model_chorus/src/model_chorus/core/context_ingestion.py`

- `chardet`
- `logging`
- `pathlib.Path`

### `model_chorus/src/model_chorus/core/contradiction.py`

- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.field_validator`
- `re`
- `typing.Any`

### `model_chorus/src/model_chorus/core/conversation.py`

- `datetime.UTC`
- `datetime.datetime`
- `datetime.timedelta`
- `filelock`
- `json`
- `logging`
- `models.ConversationMessage`
- `models.ConversationThread`
- `pathlib.Path`
- `typing.Any`
- `typing.Literal`
- `uuid`

### `model_chorus/src/model_chorus/core/gap_analysis.py`

- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.field_validator`
- `typing.Any`

### `model_chorus/src/model_chorus/core/models.py`

- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `typing.Any`
- `typing.Literal`

### `model_chorus/src/model_chorus/core/progress.py`

- `rich.console.Console`
- `sys`

### `model_chorus/src/model_chorus/core/registry.py`

- `base_workflow.BaseWorkflow`
- `collections.abc.Callable`
- `inspect`

### `model_chorus/src/model_chorus/core/role_orchestration.py`

- `asyncio`
- `dataclasses.dataclass`
- `dataclasses.field`
- `enum.Enum`
- `logging`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.field_validator`
- `typing.Any`

### `model_chorus/src/model_chorus/core/state.py`

- `datetime.UTC`
- `datetime.datetime`
- `logging`
- `models.ConversationState`
- `pathlib.Path`
- `threading`
- `typing.Any`

### `model_chorus/src/model_chorus/core/workflow_runner.py`

- `collections.abc.Callable`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `enum.Enum`
- `logging`
- `typing.Any`
- `typing.TYPE_CHECKING`

### `model_chorus/src/model_chorus/providers/__init__.py`

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

### `model_chorus/src/model_chorus/providers/base_provider.py`

- `abc.ABC`
- `abc.abstractmethod`
- `dataclasses.dataclass`
- `dataclasses.field`
- `enum.Enum`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/claude_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `base_provider.TokenUsage`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/cli_provider.py`

- `abc.abstractmethod`
- `asyncio`
- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelProvider`
- `json`
- `logging`
- `os`
- `pathlib.Path`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/codex_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `base_provider.TokenUsage`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/cursor_agent_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `base_provider.TokenUsage`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/gemini_provider.py`

- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelCapability`
- `base_provider.ModelConfig`
- `base_provider.TokenUsage`
- `cli_provider.CLIProvider`
- `json`
- `logging`
- `os`
- `pathlib.Path`
- `shutil`
- `typing.Any`

### `model_chorus/src/model_chorus/providers/middleware.py`

- `abc.ABC`
- `abc.abstractmethod`
- `asyncio`
- `base_provider.GenerationRequest`
- `base_provider.GenerationResponse`
- `base_provider.ModelProvider`
- `dataclasses.dataclass`
- `dataclasses.field`
- `enum.Enum`
- `logging`
- `time.time`
- `typing.Any`
- `typing.Callable`

### `model_chorus/src/model_chorus/utils/__init__.py`

- `model_chorus.utils.citation_formatter.CitationStyle`
- `model_chorus.utils.citation_formatter.calculate_citation_confidence`
- `model_chorus.utils.citation_formatter.calculate_citation_map_confidence`
- `model_chorus.utils.citation_formatter.format_citation`
- `model_chorus.utils.citation_formatter.format_citation_map`
- `model_chorus.utils.citation_formatter.validate_citation`

### `model_chorus/src/model_chorus/utils/citation_formatter.py`

- `enum.Enum`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/__init__.py`

- `argument.ArgumentWorkflow`
- `chat.ChatWorkflow`
- `consensus.ConsensusResult`
- `consensus.ConsensusStrategy`
- `consensus.ConsensusWorkflow`
- `consensus.ProviderConfig`
- `core.models.ConfidenceLevel`
- `core.models.Hypothesis`
- `core.models.InvestigationPhase`
- `core.models.InvestigationStep`
- `core.models.StudyState`
- `core.models.ThinkDeepState`
- `ideate.IdeateWorkflow`
- `study.StudyWorkflow`
- `thinkdeep.ThinkDeepWorkflow`

### `model_chorus/src/model_chorus/workflows/argument/__init__.py`

- `model_chorus.workflows.argument.argument_workflow.ArgumentWorkflow`
- `model_chorus.workflows.argument.semantic.cluster_claims_hierarchical`
- `model_chorus.workflows.argument.semantic.cluster_claims_kmeans`
- `model_chorus.workflows.argument.semantic.compute_claim_similarity`
- `model_chorus.workflows.argument.semantic.compute_cluster_statistics`
- `model_chorus.workflows.argument.semantic.compute_embedding`
- `model_chorus.workflows.argument.semantic.cosine_similarity`
- `model_chorus.workflows.argument.semantic.find_similar_claims`
- `model_chorus.workflows.argument.semantic.get_cluster_representative`

### `model_chorus/src/model_chorus/workflows/argument/argument_workflow.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.conversation.ConversationMemory`
- `core.models.ArgumentMap`
- `core.models.ArgumentPerspective`
- `core.progress.emit_stage`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `core.prompts.prepend_system_constraints`
- `core.registry.WorkflowRegistry`
- `core.role_orchestration.ModelRole`
- `core.role_orchestration.OrchestrationPattern`
- `core.role_orchestration.OrchestrationResult`
- `core.role_orchestration.RoleOrchestrator`
- `logging`
- `providers.ModelProvider`
- `typing.Any`
- `uuid`

### `model_chorus/src/model_chorus/workflows/argument/semantic.py`

- `functools.lru_cache`
- `hashlib`
- `model_chorus.core.models.Citation`
- `model_chorus.core.models.CitationMap`
- `numpy`
- `sentence_transformers.SentenceTransformer`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/chat.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.conversation.ConversationMemory`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `core.prompts.prepend_system_constraints`
- `logging`
- `providers.GenerationRequest`
- `providers.ModelProvider`
- `typing.Any`
- `uuid`

### `model_chorus/src/model_chorus/workflows/consensus.py`

- `asyncio`
- `core.progress.emit_provider_complete`
- `core.progress.emit_provider_start`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `dataclasses.dataclass`
- `dataclasses.field`
- `dataclasses.replace`
- `enum.Enum`
- `logging`
- `providers.GenerationRequest`
- `providers.GenerationResponse`
- `providers.ModelProvider`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/ideate/__init__.py`

- `ideate_workflow.IdeateWorkflow`

### `model_chorus/src/model_chorus/workflows/ideate/ideate_workflow.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.base_workflow.WorkflowStep`
- `core.conversation.ConversationMemory`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `core.registry.WorkflowRegistry`
- `core.role_orchestration.ModelRole`
- `core.role_orchestration.OrchestrationPattern`
- `core.role_orchestration.OrchestrationResult`
- `core.role_orchestration.RoleOrchestrator`
- `logging`
- `providers.GenerationRequest`
- `providers.ModelProvider`
- `typing.Any`
- `uuid`

### `model_chorus/src/model_chorus/workflows/study/__init__.py`

- `study_workflow.StudyWorkflow`

### `model_chorus/src/model_chorus/workflows/study/config.py`

- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/context_analysis.py`

- `core.models.ConfidenceLevel`
- `core.models.InvestigationPhase`
- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `json`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.field_validator`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/memory/__init__.py`

- `cache.ShortTermCache`
- `controller.MemoryController`
- `models.MemoryEntry`
- `models.MemoryMetadata`
- `models.MemoryQuery`
- `models.MemoryType`
- `persistence.LongTermStorage`

### `model_chorus/src/model_chorus/workflows/study/memory/cache.py`

- `collections.OrderedDict`
- `logging`
- `models.MemoryEntry`
- `models.MemoryMetadata`
- `models.MemoryQuery`
- `threading`

### `model_chorus/src/model_chorus/workflows/study/memory/controller.py`

- `cache.ShortTermCache`
- `logging`
- `models.MemoryEntry`
- `models.MemoryMetadata`
- `models.MemoryQuery`
- `persistence.LongTermStorage`
- `uuid`

### `model_chorus/src/model_chorus/workflows/study/memory/models.py`

- `datetime.UTC`
- `datetime.datetime`
- `enum.Enum`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/memory/persistence.py`

- `datetime.UTC`
- `datetime.datetime`
- `json`
- `logging`
- `models.MemoryEntry`
- `models.MemoryMetadata`
- `models.MemoryQuery`
- `models.MemoryType`
- `pathlib.Path`
- `sqlite3`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/persona_base.py`

- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/persona_router.py`

- `context_analysis.ContextAnalysisInput`
- `context_analysis.ContextAnalysisResult`
- `context_analysis.analyze_context`
- `core.models.StudyState`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `logging`
- `persona_base.Persona`
- `persona_base.PersonaRegistry`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/personas/__init__.py`

- `critic.CriticPersona`
- `critic.create_critic`
- `persona_base.PersonaRegistry`
- `planner.PlannerPersona`
- `planner.create_planner`
- `researcher.ResearcherPersona`
- `researcher.create_researcher`

### `model_chorus/src/model_chorus/workflows/study/personas/critic.py`

- `persona_base.Persona`
- `persona_base.PersonaResponse`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/personas/planner.py`

- `persona_base.Persona`
- `persona_base.PersonaResponse`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/personas/researcher.py`

- `persona_base.Persona`
- `persona_base.PersonaResponse`
- `typing.Any`

### `model_chorus/src/model_chorus/workflows/study/state_machine.py`

- `core.models.ConfidenceLevel`
- `core.models.InvestigationPhase`
- `core.models.StudyState`
- `logging`

### `model_chorus/src/model_chorus/workflows/study/study_workflow.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.base_workflow.WorkflowStep`
- `core.conversation.ConversationMemory`
- `core.models.ConversationMessage`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `core.registry.WorkflowRegistry`
- `datetime.UTC`
- `datetime.datetime`
- `logging`
- `persona_router.PersonaRouter`
- `personas.get_default_registry`
- `providers.ModelProvider`
- `typing.Any`
- `uuid`

### `model_chorus/src/model_chorus/workflows/thinkdeep.py`

- `core.base_workflow.BaseWorkflow`
- `core.base_workflow.WorkflowResult`
- `core.conversation.ConversationMemory`
- `core.models.ConfidenceLevel`
- `core.models.Hypothesis`
- `core.models.InvestigationStep`
- `core.models.ThinkDeepState`
- `core.progress.emit_progress`
- `core.progress.emit_workflow_complete`
- `core.progress.emit_workflow_start`
- `logging`
- `providers.GenerationRequest`
- `providers.GenerationResponse`
- `providers.ModelProvider`
- `typing.Any`
- `typing.Literal`
- `typing.cast`
- `uuid`

### `model_chorus/tests/conftest.py`

- `model_chorus.providers.base_provider.GenerationResponse`
- `os`
- `pytest`
- `test_helpers`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

### `model_chorus/tests/test_argument_workflow.py`

- `model_chorus.core.models.ArgumentMap`
- `model_chorus.core.role_orchestration.OrchestrationPattern`
- `model_chorus.core.role_orchestration.OrchestrationResult`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.argument.ArgumentWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`
- `uuid`

### `model_chorus/tests/test_chat_integration.py`

- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.ClaudeProvider`
- `model_chorus.providers.CodexProvider`
- `model_chorus.providers.CursorAgentProvider`
- `model_chorus.providers.GeminiProvider`
- `model_chorus.workflows.ChatWorkflow`
- `os`
- `pytest`
- `test_helpers.ANY_PROVIDER_AVAILABLE`

### `model_chorus/tests/test_chat_workflow.py`

- `model_chorus.workflows.ChatWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

### `model_chorus/tests/test_citation.py`

- `json`
- `model_chorus.core.models.Citation`
- `model_chorus.core.models.CitationMap`
- `pydantic.ValidationError`
- `pytest`

### `model_chorus/tests/test_citation_integration.py`

- `model_chorus.core.models.Citation`
- `model_chorus.core.models.CitationMap`
- `model_chorus.utils.citation_formatter.CitationStyle`
- `model_chorus.utils.citation_formatter.calculate_citation_confidence`
- `model_chorus.utils.citation_formatter.calculate_citation_map_confidence`
- `model_chorus.utils.citation_formatter.format_citation`
- `model_chorus.utils.citation_formatter.format_citation_map`
- `model_chorus.utils.citation_formatter.validate_citation`
- `pytest`

### `model_chorus/tests/test_claude_provider.py`

- `json`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.claude_provider.ClaudeProvider`
- `pytest`
- `unittest.mock.patch`

### `model_chorus/tests/test_cli_integration.py`

- `json`
- `model_chorus.cli`
- `model_chorus.cli.main.app`
- `model_chorus.core.base_workflow.WorkflowResult`
- `model_chorus.core.base_workflow.WorkflowStep`
- `pytest`
- `sys`
- `typer.testing.CliRunner`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`

### `model_chorus/tests/test_cli_primitives.py`

- `model_chorus.cli.primitives.OutputFormatter`
- `model_chorus.cli.primitives.ProviderResolver`
- `model_chorus.cli.primitives.WorkflowContext`
- `model_chorus.providers.cli_provider.ProviderDisabledError`
- `model_chorus.providers.cli_provider.ProviderUnavailableError`
- `pytest`
- `typer`
- `unittest.mock.MagicMock`
- `unittest.mock.Mock`
- `unittest.mock.patch`

### `model_chorus/tests/test_clustering.py`

- `model_chorus.core.clustering.ClusterResult`
- `model_chorus.core.clustering.SemanticClustering`
- `numpy`
- `pytest`
- `unittest.mock.Mock`
- `unittest.mock.patch`

### `model_chorus/tests/test_codex_provider.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.codex_provider.CodexProvider`
- `pytest`
- `unittest.mock.patch`

### `model_chorus/tests/test_concurrent_conversations.py`

- `asyncio`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.chat.ChatWorkflow`
- `model_chorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `time`
- `unittest.mock.AsyncMock`
- `uuid`

### `model_chorus/tests/test_config.py`

- `json`
- `model_chorus.core.config.ConfigLoader`
- `model_chorus.core.config.GenerationDefaults`
- `model_chorus.core.config.ModelChorusConfig`
- `model_chorus.core.config.WorkflowConfig`
- `model_chorus.core.config.get_config_loader`
- `pytest`

### `model_chorus/tests/test_consensus_provider_models.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.consensus.ConsensusWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_consensus_workflow.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.consensus.ConsensusStrategy`
- `model_chorus.workflows.consensus.ConsensusWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_context_ingestion.py`

- `model_chorus.core.context_ingestion.BinaryFileError`
- `model_chorus.core.context_ingestion.ContextIngestionService`
- `model_chorus.core.context_ingestion.DEFAULT_MAX_FILE_SIZE_KB`
- `model_chorus.core.context_ingestion.DEFAULT_WARN_FILE_SIZE_KB`
- `model_chorus.core.context_ingestion.FileTooLargeError`
- `pathlib.Path`
- `pytest`
- `tempfile`

### `model_chorus/tests/test_contradiction.py`

- `model_chorus.core.contradiction.Contradiction`
- `model_chorus.core.contradiction.ContradictionSeverity`
- `model_chorus.core.contradiction.assess_contradiction_severity`
- `model_chorus.core.contradiction.detect_contradiction`
- `model_chorus.core.contradiction.detect_contradictions_batch`
- `model_chorus.core.contradiction.detect_polarity_opposition`
- `model_chorus.core.contradiction.generate_contradiction_explanation`
- `model_chorus.core.contradiction.generate_reconciliation_suggestion`
- `pytest`

### `model_chorus/tests/test_conversation.py`

- `json`
- `model_chorus.core.conversation.ConversationMemory`
- `uuid`

### `model_chorus/tests/test_cursor_agent_provider.py`

- `json`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.cursor_agent_provider.CursorAgentProvider`
- `pytest`
- `unittest.mock.patch`

### `model_chorus/tests/test_gap_analysis.py`

- `model_chorus.core.gap_analysis.Gap`
- `model_chorus.core.gap_analysis.GapSeverity`
- `model_chorus.core.gap_analysis.GapType`
- `model_chorus.core.gap_analysis.assess_gap_severity`
- `model_chorus.core.gap_analysis.detect_gaps`
- `model_chorus.core.gap_analysis.detect_logical_gaps`
- `model_chorus.core.gap_analysis.detect_missing_evidence`
- `model_chorus.core.gap_analysis.detect_unsupported_claims`
- `model_chorus.core.gap_analysis.generate_gap_recommendation`
- `pytest`

### `model_chorus/tests/test_gemini_integration.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.gemini_provider.GeminiProvider`
- `pytest`
- `subprocess`
- `test_helpers.GEMINI_AVAILABLE`

### `model_chorus/tests/test_helpers.py`

- `pathlib.Path`
- `shutil`
- `subprocess`
- `yaml`

### `model_chorus/tests/test_ideate_workflow.py`

- `model_chorus.core.base_workflow.WorkflowResult`
- `model_chorus.core.base_workflow.WorkflowStep`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.ideate.ideate_workflow.IdeateWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

### `model_chorus/tests/test_ideate_workflow_integration.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.ideate.IdeateWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`
- `unittest.mock.patch`

### `model_chorus/tests/test_integration.py`

- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.claude_provider.ClaudeProvider`
- `model_chorus.providers.codex_provider.CodexProvider`
- `model_chorus.workflows.consensus.ConsensusStrategy`
- `model_chorus.workflows.consensus.ConsensusWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.patch`

### `model_chorus/tests/test_memory_management.py`

- `asyncio`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.chat.ChatWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_middleware.py`

- `asyncio`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.providers.base_provider.ModelProvider`
- `model_chorus.providers.middleware.CircuitBreakerConfig`
- `model_chorus.providers.middleware.CircuitBreakerMiddleware`
- `model_chorus.providers.middleware.CircuitOpenError`
- `model_chorus.providers.middleware.CircuitState`
- `model_chorus.providers.middleware.ConfigError`
- `model_chorus.providers.middleware.Middleware`
- `model_chorus.providers.middleware.ProviderError`
- `model_chorus.providers.middleware.RetryConfig`
- `model_chorus.providers.middleware.RetryExhaustedError`
- `model_chorus.providers.middleware.RetryMiddleware`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

### `model_chorus/tests/test_providers/test_cli_interface.py`

- `model_chorus.providers.CLIProvider`
- `model_chorus.providers.ClaudeProvider`
- `model_chorus.providers.CodexProvider`
- `model_chorus.providers.CursorAgentProvider`
- `model_chorus.providers.GeminiProvider`
- `model_chorus.providers.GenerationRequest`
- `model_chorus.providers.ModelProvider`
- `pytest`

### `model_chorus/tests/test_review_response.py`

- `json`
- `pathlib.Path`
- `pytest`

### `model_chorus/tests/test_role_orchestration.py`

- `dataclasses.dataclass`
- `model_chorus.core.role_orchestration.ModelRole`
- `model_chorus.core.role_orchestration.OrchestrationPattern`
- `model_chorus.core.role_orchestration.OrchestrationResult`
- `model_chorus.core.role_orchestration.RoleOrchestrator`
- `model_chorus.core.role_orchestration.SynthesisStrategy`
- `pytest`
- `unittest.mock.patch`

### `model_chorus/tests/test_semantic_similarity.py`

- `model_chorus.core.models.Citation`
- `model_chorus.core.models.CitationMap`
- `model_chorus.workflows.argument.semantic.add_similarity_to_citation`
- `model_chorus.workflows.argument.semantic.cluster_claims_hierarchical`
- `model_chorus.workflows.argument.semantic.cluster_claims_kmeans`
- `model_chorus.workflows.argument.semantic.compute_claim_similarity`
- `model_chorus.workflows.argument.semantic.compute_claim_similarity_batch`
- `model_chorus.workflows.argument.semantic.compute_cluster_statistics`
- `model_chorus.workflows.argument.semantic.compute_embedding`
- `model_chorus.workflows.argument.semantic.cosine_similarity`
- `model_chorus.workflows.argument.semantic.find_duplicate_claims`
- `model_chorus.workflows.argument.semantic.find_similar_claims`
- `model_chorus.workflows.argument.semantic.get_cluster_representative`
- `numpy`
- `pytest`

### `model_chorus/tests/test_standardization.py`

- `json`
- `model_chorus.providers.base_provider.TokenUsage`
- `model_chorus.providers.claude_provider.ClaudeProvider`
- `model_chorus.providers.gemini_provider.GeminiProvider`

### `model_chorus/tests/test_state.py`

- `json`
- `model_chorus.core.models.ConversationState`
- `model_chorus.core.state.StateManager`
- `model_chorus.core.state.get_default_state_manager`
- `pytest`
- `threading`
- `time`

### `model_chorus/tests/test_thinkdeep_complex.py`

- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.core.models.ConfidenceLevel`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_thinkdeep_expert_validation.py`

- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.core.models.ConfidenceLevel`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_thinkdeep_models.py`

- `json`
- `model_chorus.core.models.ConfidenceLevel`
- `model_chorus.core.models.Hypothesis`
- `model_chorus.core.models.InvestigationStep`
- `model_chorus.core.models.ThinkDeepState`
- `pydantic.ValidationError`
- `pytest`

### `model_chorus/tests/test_thinkdeep_workflow.py`

- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.core.models.ConfidenceLevel`
- `model_chorus.core.models.InvestigationStep`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/test_workflow_integration_chaining.py`

- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.providers.base_provider.GenerationRequest`
- `model_chorus.providers.base_provider.GenerationResponse`
- `model_chorus.workflows.chat.ChatWorkflow`
- `model_chorus.workflows.consensus.ConsensusStrategy`
- `model_chorus.workflows.consensus.ConsensusWorkflow`
- `model_chorus.workflows.thinkdeep.ThinkDeepWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`

### `model_chorus/tests/workflows/study/memory/test_cache.py`

- `model_chorus.workflows.study.memory.MemoryEntry`
- `model_chorus.workflows.study.memory.MemoryQuery`
- `model_chorus.workflows.study.memory.ShortTermCache`
- `pytest`

### `model_chorus/tests/workflows/study/memory/test_persistence.py`

- `model_chorus.workflows.study.memory.LongTermStorage`
- `model_chorus.workflows.study.memory.MemoryEntry`
- `model_chorus.workflows.study.memory.MemoryQuery`
- `model_chorus.workflows.study.memory.MemoryType`
- `os`
- `pytest`
- `tempfile`

### `model_chorus/tests/workflows/study/test_personas.py`

- `model_chorus.workflows.study.persona_base.Persona`
- `model_chorus.workflows.study.persona_base.PersonaRegistry`
- `model_chorus.workflows.study.persona_base.PersonaResponse`
- `model_chorus.workflows.study.personas.CriticPersona`
- `model_chorus.workflows.study.personas.PlannerPersona`
- `model_chorus.workflows.study.personas.ResearcherPersona`
- `model_chorus.workflows.study.personas.create_critic`
- `model_chorus.workflows.study.personas.create_default_personas`
- `model_chorus.workflows.study.personas.create_planner`
- `model_chorus.workflows.study.personas.create_researcher`
- `model_chorus.workflows.study.personas.get_default_registry`
- `pytest`

### `model_chorus/tests/workflows/study/test_routing.py`

- `model_chorus.core.models.StudyState`
- `model_chorus.workflows.study.persona_router.PersonaRouter`
- `model_chorus.workflows.study.persona_router.RoutingDecision`
- `model_chorus.workflows.study.personas.get_default_registry`
- `pytest`
- `unittest.mock.patch`

### `model_chorus/tests/workflows/study/test_state_machine.py`

- `model_chorus.core.models.ConfidenceLevel`
- `model_chorus.core.models.InvestigationPhase`
- `model_chorus.core.models.StudyState`
- `model_chorus.workflows.study.state_machine.InvestigationStateMachine`
- `pytest`

### `model_chorus/tests/workflows/study/test_study_workflow.py`

- `datetime.datetime`
- `model_chorus.core.base_workflow.WorkflowResult`
- `model_chorus.core.base_workflow.WorkflowStep`
- `model_chorus.core.conversation.ConversationMemory`
- `model_chorus.workflows.study.study_workflow.StudyWorkflow`
- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.Mock`
