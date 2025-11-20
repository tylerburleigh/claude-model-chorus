# High-Leverage Improvements

## CLI Architecture
- Break the monolithic `model_chorus/src/model_chorus/cli/main.py` into reusable command primitives so provider resolution, memory wiring, availability checks, and JSON output are implemented once and inherited by each workflow command.
- Build a shared context ingestion/streaming layer (chunking, size limits, consistent error handling) instead of each command calling `construct_prompt_with_files` directly.

## Workflow & Provider Platform
- Introduce a workflow runner abstraction above `BaseWorkflow._execute_with_fallback` to own provider selection, throttling, telemetry, and fan-out policies while individual workflows focus on domain logic.
- Add provider middleware for retries, backoff, structured errors, and circuit breakers so every CLI workflow gains resiliency without duplicating logic.

## Conversation State & Context
- Replace file-per-thread JSON persistence in `ConversationMemory` with a lightweight transactional store (e.g., SQLite/WAL) to support concurrent CLI sessions, indexing, and richer queries like "list recent investigations."
- Reuse `ConversationMemory.build_conversation_history` for all prompt construction and add token-budgeting so every workflow benefits from the same context compaction strategy.

## Testing & Tooling
- Consolidate duplicated test trees (`model_chorus/tests` vs. root `tests`) and provide fixtures against a modularized CLI to reduce brittle `sys.path` hacks and undefined mocks.
- Wire formatting/lint/type-check commands (black, ruff, mypy) plus a CI job that runs mock-provider unit tests so regressions in shared infrastructure are caught automatically.

## Developer Experience & Extensibility
- Promote a typed configuration object that both the CLI and library share, reducing the number of places new providers/workflows must touch.
- Add a workflow/extension registry (`model-chorus list-workflows`, plugin descriptors) so documentation and CLI capabilities stay in sync and new workflows can register themselves declaratively.
