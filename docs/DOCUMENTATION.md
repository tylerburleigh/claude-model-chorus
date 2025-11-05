# claude-model-chorus Documentation

**Version:** 1.0.0
**Generated:** 2025-11-05 12:47:47

---

## 📊 Project Statistics

- **Total Files:** 29
- **Total Lines:** 4204
- **Total Classes:** 32
- **Total Functions:** 11
- **Avg Complexity:** 2.82
- **Max Complexity:** 16
- **High Complexity Functions:**
  - consensus (16)



## 🏛️ Classes

### `BaseWorkflow`

**Language:** python
**Inherits from:** `ABC`
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:47`

**Description:**
> Abstract base class for all ModelChorus workflows.

All workflow implementations (thinkdeep, debug, consensus, etc.) must inherit
from this class and implement the run() method.

Attributes:
    name: Human-readable name of the workflow
    description: Brief description of what this workflow does
    config: Configuration dictionary for the workflow

**Methods:**
- `__init__()`
- `run()`
- `synthesize()`
- `get_result()`
- `validate_config()`
- `__repr__()`

---

### `CLIProvider`

**Language:** python
**Inherits from:** `ModelProvider`
**Defined in:** `modelchorus/src/modelchorus/providers/cli_provider.py:23`

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
- `build_command()`
- `parse_response()`
- `execute_command()`
- `generate()`
- `supports_vision()`
- `__repr__()`

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

### `ConsensusConfig`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:303`

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
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:50`

**Description:**
> Response from text generation.

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
**Defined in:** `modelchorus/src/modelchorus/providers/base_provider.py:60`

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
**Defined in:** `modelchorus/src/modelchorus/core/models.py:255`

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
**Defined in:** `modelchorus/src/modelchorus/core/models.py:153`

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
**Defined in:** `modelchorus/src/modelchorus/core/models.py:12`

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
**Defined in:** `modelchorus/src/modelchorus/core/models.py:86`

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
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:26`

**Description:**
> Result of a workflow execution.

**Methods:**
- `add_step()`

---

### `WorkflowStep`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/core/base_workflow.py:15`

**Description:**
> Represents a single step in a workflow execution.

---

### `WorkflowStep`

**Language:** python
**Inherits from:** `BaseModel`
**Defined in:** `modelchorus/src/modelchorus/core/models.py:193`

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

### `consensus(prompt, providers, strategy, system, temperature, max_tokens, timeout, output, verbose) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:54`
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

### `get_provider_by_name(name) -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:35`
**Complexity:** 2

**Description:**
> Get provider instance by name.

**Parameters:**
- `name`: str

---

### `list_providers() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:239`
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
**Defined in:** `modelchorus/examples/provider_integration.py:101`
**Complexity:** 2

**Description:**
> Main entry point for the provider example.

---

### `main() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:277`
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

### `mock_subprocess_run() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:45`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Mock subprocess.run for CLI command execution.

---

### `sample_generation_request() -> None`

**Language:** python
**Defined in:** `modelchorus/tests/conftest.py:55`
**Complexity:** 1

**Decorators:** `@pytest.fixture`

**Description:**
> Sample GenerationRequest for testing.

---

### `version() -> None`

**Language:** python
**Defined in:** `modelchorus/src/modelchorus/cli/main.py:266`
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

### `modelchorus/examples/provider_integration.py`

- `asyncio`
- `modelchorus.providers.GenerationRequest`
- `modelchorus.providers.GenerationResponse`
- `modelchorus.providers.ModelCapability`
- `modelchorus.providers.ModelConfig`
- `modelchorus.providers.ModelProvider`

### `modelchorus/src/modelchorus/cli/__init__.py`

- `main.app`
- `main.main`

### `modelchorus/src/modelchorus/cli/main.py`

- `asyncio`
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
- `workflows.ConsensusStrategy`
- `workflows.ConsensusWorkflow`

### `modelchorus/src/modelchorus/core/__init__.py`

- `base_workflow.BaseWorkflow`
- `base_workflow.WorkflowResult`
- `base_workflow.WorkflowStep`
- `models.ConsensusConfig`
- `models.ModelResponse`
- `models.ModelSelection`
- `models.WorkflowRequest`
- `models.WorkflowResponse`
- `models.WorkflowStep`
- `registry.WorkflowRegistry`

### `modelchorus/src/modelchorus/core/base_workflow.py`

- `abc.ABC`
- `abc.abstractmethod`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/core/models.py`

- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`

### `modelchorus/src/modelchorus/core/registry.py`

- `base_workflow.BaseWorkflow`
- `inspect`
- `typing.Callable`
- `typing.Dict`
- `typing.Optional`
- `typing.Type`

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

### `modelchorus/src/modelchorus/workflows/__init__.py`

- `consensus.ConsensusResult`
- `consensus.ConsensusStrategy`
- `consensus.ConsensusWorkflow`
- `consensus.ProviderConfig`

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

### `modelchorus/tests/conftest.py`

- `pytest`
- `unittest.mock.AsyncMock`
- `unittest.mock.MagicMock`

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
