# Changelog

All notable changes to ModelChorus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-06

### Added

#### ARGUMENT Workflow
- New `ArgumentWorkflow` for structured dialectical reasoning and argument analysis
- Three-role debate system: Creator (pro), Skeptic (con), Moderator (synthesis)
- Balanced multi-perspective analysis avoiding single-perspective bias
- Identification of trade-offs and common ground
- Conversation continuity for iterative argument refinement
- CLI command: `modelchorus argument` with continuation support
- Python API with conversation memory integration
- Comprehensive documentation in `docs/workflows/ARGUMENT.md`
- Use cases for policy analysis, technology evaluation, and balanced decision-making

#### IDEATE Workflow
- New `IdeateWorkflow` for collaborative multi-model brainstorming
- Multi-provider idea generation with parallel execution
- Automatic idea clustering and categorization
- Synthesis of complementary perspectives from different models
- Constructive handling of model disagreements
- CLI command: `modelchorus ideate` with multi-provider support
- Python API with idea clustering and synthesis
- Comprehensive documentation in `docs/workflows/IDEATE.md`
- Use cases for feature brainstorming, creative problem-solving, and innovation sessions

#### RESEARCH Workflow
- New `ResearchWorkflow` for multi-source research with citation tracking
- Automatic citation extraction and tracking across multiple models
- Semantic clustering of related ideas and concepts
- Multi-model knowledge synthesis with overlap detection
- Handling of contradictory information across sources
- CLI command: `modelchorus research` with multi-provider support
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
- CLI commands: `modelchorus chat` with continuation support
- Python API with `continuation_id` parameter for follow-up messages
- Comprehensive test coverage for chat workflow

#### THINKDEEP Workflow
- New `ThinkDeepWorkflow` for systematic investigation and extended reasoning
- Hypothesis tracking and evolution throughout investigation
- Confidence progression system (exploring → low → medium → high → very_high → almost_certain → certain)
- Optional expert validation from second AI model
- Investigation step execution with state persistence
- File examination tracking across investigation steps
- CLI commands: `modelchorus thinkdeep` and `modelchorus thinkdeep-status`
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
- CLI interface with `modelchorus consensus` command
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
