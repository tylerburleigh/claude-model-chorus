# Resolver Follow-up Notes

Context: the ThinkDeep CLI now relies on `resolve_context_files` to translate legacy `src/claude_skills/...` paths, normalize context inputs, and warn (rather than abort) when files cannot be found. The helper lives in `model_chorus/src/model_chorus/cli/main.py` and currently backs only the ThinkDeep command.

## 1. Extend Resolver Coverage
- Integrate `resolve_context_files` into `chat`, `argument`, and `consensus` commands so all workflows share the same path normalization rules.
- Decide on strictness: ThinkDeep now skips missing files with a warning, but other commands presently exit. Consider a configuration flag (CLI option or `.model-chorusrc` field) to toggle between *warn-and-continue* and *fail-fast* behaviours.
- Introduce unit or integration specs for each command mirroring the new ThinkDeep tests to prevent regressions.

## 2. Make Legacy Mapping Configurable
- Externalize `LEGACY_PATH_MAPPINGS` into a config-driven structure (e.g., `config.get_legacy_path_map()` or a YAML block) so teams can add their own historical directory prefixes without code changes.
- Support environment-variable overrides for ad-hoc runs (CI migration scripts, local hotfixes).
- Document the extension process in the CLI guide once the configuration channel is ready.

## 3. Packaging & Test Environment Hygiene
- Remove the globally installed `model_chorus` package during local development or ensure the repo path is first on `PYTHONPATH` to avoid importing stale resolver logic. (The full `pytest` run currently conflicts with the packaged tests for this reason.)
- Update developer docs with a short snippet (`pip uninstall model-chorus` or `PYTHONPATH=model_chorus/src`) so contributors don’t hit the same import mismatch.
- When the next release is cut, confirm the wheel/sdist contains the new helper and that `python -m model_chorus.cli.main` behaves correctly without extra environment tweaks.

## 4. Diagnostics & Observability
- Consider emitting structured logs (or a verbose-mode table) summarizing remapped vs. skipped files to help investigators triage large runs.
- Track skipped paths in workflow metadata so downstream agents can surface missing context in reports.

## 5. Longer-Term Ideas
- Allow wildcard globs in `--files-checked` that resolve to multiple files via the helper.
- Expose the resolver via a utility module if non-CLI components (e.g., orchestration services) need the same remapping logic.
