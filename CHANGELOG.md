# Changelog

All notable changes to ModelChorus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/tylerburleigh/claude-model-chorus/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tylerburleigh/claude-model-chorus/releases/tag/v0.1.0
