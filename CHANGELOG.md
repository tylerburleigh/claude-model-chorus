# Changelog

All notable changes to ModelChorus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2025-11-10

### Added
- ThinkDeep workflow now accepts an optional `relevant_files` list, folding explicitly supplied paths into investigation state, prompts, and metadata (`relevant_files_this_step`, `relevant_files`).
- `model-chorus thinkdeep` surfaces a `--relevant-files` option that validates paths (including legacy remapping) and mirrors the new metadata fields in console output and JSON exports.

### Changed
- ThinkDeep CLI now remaps known legacy context file paths (e.g. `src/claude_skills/...`) to current locations and skips unresolved files with warnings instead of exiting.

## [0.5.0] - 2025-11-08

### Removed

#### RESEARCH Workflow
- **Complete removal of RESEARCH workflow** - No longer within the scope of ModelChorus
  - Removed `model-chorus research` CLI command
  - Deleted ResearchWorkflow implementation (~1,750 lines)
  - Removed ResearchEvidence, ResearchValidation, ResearchDossier, and ResearchState model classes
  - Deleted research skill (`skills/research/`)
  - Deleted research agent (`agents/research.md`)
  - Removed all research workflow tests (~1,200 lines)
  - Updated all cross-references in remaining workflows
  - Removed research-specific documentation

#### Impact
- **Workflows reduced from 6 to 5**: CHAT, CONSENSUS, THINKDEEP, ARGUMENT, IDEATE remain
- **~9,500 lines of code removed** from the codebase
- **Breaking change**: `model-chorus research` command no longer available
- **Breaking change**: ResearchWorkflow class no longer importable from `model_chorus.workflows`

### Changed
- Documentation updated to reflect 5 workflows instead of 6
- Router skill updated to route among 5 workflows
- All workflow selection guides updated to remove RESEARCH references
- Examples updated to remove research workflow demonstrations

## [0.4.0] - 2025-11-07

### Added

#### Provider Fallback and Resilience
- **Automatic Provider Fallback**: All workflows now support fallback provider chains
  - Primary provider fails → automatically tries fallback providers in order
  - Configurable via `fallback_providers` in `.model-chorusrc`
  - Works across CHAT, RESEARCH, ARGUMENT, IDEATE, and THINKDEEP workflows
  - Example: `fallback_providers: [gemini, codex, cursor-agent]`

- **Proactive Availability Checking**: Workflows check provider availability before execution
  - Fail-fast behavior: detects unavailable providers immediately
  - Clear error messages with installation instructions
  - Concurrent availability checks for minimal overhead
  - Optional `--skip-provider-check` flag for faster startup

- **Enhanced Error Messages**: Provider unavailability errors now include:
  - Specific reason (CLI not found, permission denied, etc.)
  - Installation command for the specific provider
  - Helpful suggestions (check installations, update config)
  - Reference to `model-chorus list-providers --check`

- **CLI Improvements**:
  - `list-providers --check` command to verify all provider installations
  - All workflow commands support `--skip-provider-check` flag
  - Verbose output shows provider initialization and fallback attempts
  - Installation commands added: claude, gemini, codex, cursor-agent

#### Configuration
- **Fallback Provider Configuration**: `.model-chorusrc.example` updated with comprehensive fallback examples
  - Each workflow includes fallback provider list
  - Example configurations for all 4 providers
  - Documentation on provider fallback behavior
  - Comments explaining fallback chain execution order

### Changed

#### Workflows
- **ResearchWorkflow**: Added `fallback_providers` parameter and availability checking
  - Constructor accepts `fallback_providers: Optional[List[ModelProvider]]`
  - `run()` method accepts `skip_provider_check: bool` parameter
  - All `provider.generate()` calls use `_execute_with_fallback()`

- **ArgumentWorkflow**: Added `fallback_providers` parameter and availability checking
  - Constructor accepts `fallback_providers: Optional[List[ModelProvider]]`
  - `run()` method accepts `skip_provider_check: bool` parameter
  - Provider failures handled gracefully with fallback

- **IdeateWorkflow**: Added `fallback_providers` parameter and availability checking
  - Constructor accepts `fallback_providers: Optional[List[ModelProvider]]`
  - `run()` method accepts `skip_provider_check: bool` parameter
  - All 5 provider.generate() calls use fallback mechanism

- **ThinkDeepWorkflow**: Added `fallback_providers` parameter and availability checking
  - Constructor accepts `fallback_providers: Optional[List[ModelProvider]]`
  - `run()` method accepts `skip_provider_check: bool` parameter
  - Main investigation uses fallback (expert provider intentionally excluded)

- **ChatWorkflow**: Added `fallback_providers` parameter and availability checking
  - Constructor accepts `fallback_providers: Optional[List[ModelProvider]]`
  - `run()` method accepts `skip_provider_check: bool` parameter
  - Conversation continuity maintained across fallback providers

#### CLI Commands
- All workflow CLI commands now:
  - Load `fallback_providers` from configuration
  - Initialize fallback provider instances
  - Pass fallback providers to workflow constructors
  - Support `--skip-provider-check` flag
  - Show verbose output for provider initialization

#### Infrastructure
- **BaseWorkflow**: New fallback infrastructure methods
  - `_execute_with_fallback()`: Tries primary then fallback providers sequentially
  - `check_provider_availability()`: Concurrent provider availability testing
  - Returns provider name, response, and failed providers list

- **CLIProvider**: Enhanced error handling
  - `ProviderUnavailableError` exception with suggestions
  - `check_availability()` method for testing CLI installation
  - Improved error classification (permanent vs transient)
  - Smart retry logic that skips permanent errors

### Documentation
- **TESTING_FALLBACK.md**: Comprehensive manual testing guide
  - 8 test scenarios covering all fallback cases
  - Step-by-step testing procedures
  - Expected results for each scenario
  - Debugging tips and troubleshooting
  - Success criteria checklist

- **Configuration Examples**: Updated `.model-chorusrc.example`
  - All workflows show fallback provider configuration
  - Comprehensive comments explaining fallback behavior
  - Provider fallback section with usage examples

### Technical Details
- Fallback mechanism uses sequential provider attempts with error tracking
- Availability checking uses concurrent async execution for performance
- Failed providers logged with warning messages
- Successful fallback provider reported in metadata
- Provider errors distinguished from model/content errors
- Fallback only triggers on provider-level failures (CLI missing, auth errors)

## [0.3.0] - 2025-11-06

### Added

#### ARGUMENT Workflow
- New `ArgumentWorkflow` for structured dialectical reasoning and argument analysis
- Three-role debate system: Creator (pro), Skeptic (con), Moderator (synthesis)
- Balanced multi-perspective analysis avoiding single-perspective bias
- Identification of trade-offs and common ground
- Conversation continuity for iterative argument refinement
- CLI command: `model-chorus argument` with continuation support
- Python API with conversation memory integration
- Comprehensive documentation in `docs/workflows/ARGUMENT.md`
- Use cases for policy analysis, technology evaluation, and balanced decision-making

#### IDEATE Workflow
- New `IdeateWorkflow` for collaborative multi-model brainstorming
- Multi-provider idea generation with parallel execution
- Automatic idea clustering and categorization
- Synthesis of complementary perspectives from different models
- Constructive handling of model disagreements
- CLI command: `model-chorus ideate` with multi-provider support
- Python API with idea clustering and synthesis
- Comprehensive documentation in `docs/workflows/IDEATE.md`
- Use cases for feature brainstorming, creative problem-solving, and innovation sessions

#### RESEARCH Workflow
- New `ResearchWorkflow` for multi-source research with citation tracking
- Automatic citation extraction and tracking across multiple models
- Semantic clustering of related ideas and concepts
- Multi-model knowledge synthesis with overlap detection
- Handling of contradictory information across sources
- CLI command: `model-chorus research` with multi-provider support
- Python API with citation engine and clustering algorithms
- Role-based orchestration framework for specialized provider roles
- Comprehensive documentation in `docs/workflows/RESEARCH.md`
- Use cases for topic research, literature review, and knowledge exploration

#### Documentation
- Complete workflow guides for ARGUMENT, IDEATE, and RESEARCH in `docs/workflows/`
- Workflow decision trees and selection guidance
- CLI and Python API examples for all new workflows
- Use case patterns and best practices
- Integration examples with existing workflows

### Changed
- Enhanced README.md to document all six workflows (CHAT, CONSENSUS, THINKDEEP, ARGUMENT, IDEATE, RESEARCH)
- Updated "Key Features" section to reflect six-workflow system
- Expanded CLI Commands section with new workflow commands
- Added CLI Options documentation for ARGUMENT, IDEATE, and RESEARCH
- Updated Quick Start sections with examples for all new workflows
- Enhanced architecture documentation to reflect complete workflow suite

### Technical Details
- `ArgumentWorkflow` with three-role debate orchestration
- `IdeateWorkflow` with multi-provider idea clustering
- `ResearchWorkflow` with citation engine and semantic clustering
- Role-based orchestration framework for specialized provider coordination
- Citation tracking and clustering algorithms
- Enhanced workflow abstractions for specialized use cases
- Maintained async architecture throughout new features

## [0.2.0] - 2025-11-06

### Added

#### CHAT Workflow
- New `ChatWorkflow` for simple single-model conversations with continuity
- Multi-turn conversation support with conversation memory
- Thread-based conversation persistence across sessions
- File context sharing in conversations
- CLI commands: `model-chorus chat` with continuation support
- Python API with `continuation_id` parameter for follow-up messages
- Comprehensive test coverage for chat workflow

#### THINKDEEP Workflow
- New `ThinkDeepWorkflow` for systematic investigation and extended reasoning
- Hypothesis tracking and evolution throughout investigation
- Confidence progression system (exploring → low → medium → high → very_high → almost_certain → certain)
- Optional expert validation from second AI model
- Investigation step execution with state persistence
- File examination tracking across investigation steps
- CLI commands: `model-chorus thinkdeep` and `model-chorus thinkdeep-status`
- Investigation state inspection with `--steps` and `--files` flags
- Python API with investigation state management
- Comprehensive test coverage for thinkdeep workflow

#### Conversation Infrastructure
- Core `ConversationMemory` class for managing multi-turn conversations
- Thread-based conversation state management
- Conversation persistence and retrieval
- Message history tracking with roles (user/assistant)
- Context continuity across workflow invocations

#### Documentation
- Complete workflow guide in `docs/WORKFLOWS.md`
- Decision trees for choosing between CHAT, THINKDEEP, and CONSENSUS
- CHAT workflow documentation with CLI and Python examples
- THINKDEEP workflow documentation with use cases and patterns
- Conversation API documentation and patterns
- Example code for all new workflows

### Changed
- Enhanced README.md with CHAT and THINKDEEP workflow sections
- Updated architecture documentation to reflect three-workflow system
- Improved CLI help text with workflow-specific options

### Technical Details
- `BaseWorkflow` extended with conversation memory support
- Pydantic models for investigation state (Hypothesis, InvestigationStep, ThinkDeepState)
- ConfidenceLevel enum for tracking investigation progress
- Workflow state persistence across invocations
- Async architecture maintained throughout new features

## [0.1.0] - 2025-11-05

### Added
- Initial release with CONSENSUS workflow
- Multi-provider support (Claude, Gemini, Codex, Cursor Agent)
- Five consensus strategies: all_responses, first_valid, majority, weighted, synthesize
- CLI interface with `model-chorus consensus` command
- Python API with `ConsensusWorkflow` class
- Provider abstraction layer with CLI-based provider implementations
- Type-safe request/response models with Pydantic
- Async execution for parallel provider calls
- Rich terminal output formatting
- Comprehensive test suite
- Claude Code plugin integration

[0.3.0]: https://github.com/tylerburleigh/claude-model-chorus/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tylerburleigh/claude-model-chorus/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tylerburleigh/claude-model-chorus/releases/tag/v0.1.0
