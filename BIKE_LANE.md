### Task Context

**Goal:** Resolve inconsistencies and bugs in the `model-chorus` CLI, primarily discovered while running tests from `docs/TESTING_PLAYBOOK.md`.

**Summary of Changes Implemented:**

1.  **Standardized `--prompt` Flag:** The `chat`, `argument`, `ideate`, and `consensus` CLI commands were modified to use a required `--prompt` flag instead of a positional argument for consistency.
2.  **Added `--file` to Consensus:** The `consensus` command was updated to accept a `--file` argument, aligning its functionality with other workflows.
3.  **Centralized File Handling:** A `construct_prompt_with_files` helper function was created in `cli/main.py` to provide a standard way for all commands to prepend file content to the user's prompt, complete with a descriptive preamble. All relevant commands were refactored to use this function.
4.  **Fixed Provider CLI Interaction:** Debugged a systemic issue where multi-line prompts caused provider CLIs to hang. The fix involved modifying the base `CLIProvider` to correctly pass prompt data via `stdin` and updating the `claude` provider to use the standard `-` argument to read from `stdin`.
5.  **Fixed `ConversationMemory` Bug:** Resolved an issue where conversation threads were not being created on the first turn, causing "thread not found" errors in multi-turn workflows. The `add_message` method was updated to implicitly create a thread if one doesn't exist.
6.  **Verified JSON Output Standard:** Confirmed that all providers (`claude`, `gemini`, `codex`, `cursor-agent`) are standardized to request and parse JSON output.

---
### Deferred Ideas (Bike Lane)

*   **Standardize Prompt Flags in Providers:** Investigate and standardize the use of an explicit prompt flag (e.g., `-p` or `--prompt`) across all provider CLI wrappers (`claude`, `codex`, `cursor-agent`), similar to the fix applied to the `gemini` provider. This would make the command construction more robust and less reliant on positional arguments or implicit stdin.
*   **Improve Consensus CLI Output:** For the `all_responses` strategy, consider changing the "Consensus Response:" header to something more descriptive like "Individual Provider Responses:" to avoid confusion, or remove it entirely if the table and provider headers are deemed sufficient.
