---
name: model-chorus:precommit
description: Validates git changes and repository state before committing with systematic analysis for multi-repository validation, security review, change impact assessment, and completeness verification
---

# ModelChorus: PreCommit

Git change validation workflow that reviews staged/unstaged changes before committing, ensuring code quality, security, and completeness.

## When to Use

Use this skill when you need to:
- **Pre-commit validation** - Review changes before git commit
- **Change impact assessment** - Understand ripple effects of changes
- **Security review** - Check for exposed secrets or vulnerabilities
- **Completeness check** - Ensure all related changes included
- **Multi-repository validation** - Review changes across multiple repos
- **Test coverage verification** - Check if tests cover new code

## How It Works

PreCommit workflow:

1. **Change Detection** - Examine staged and/or unstaged changes via git diff
2. **Systematic Analysis** - Review changes for:
   - Security issues (secrets, vulnerabilities)
   - Missing tests
   - Code quality
   - Completeness (related files updated?)
   - Breaking changes
3. **Issue Documentation** - Track findings with severity
4. **Expert Validation** - Optional external model validates analysis

## Usage

Invoke PreCommit to review your changes:

```
Use PreCommit to validate changes in:
/absolute/path/to/repository

Review: [staged/unstaged/both]
Focus: [security/tests/completeness/all]
```

## Parameters

- **step** (required): Validation narrative for this step
- **step_number** (required): Current step (starts at 1)
- **total_steps** (required): Total validation steps
- **next_step_required** (required): True if more validation needed
- **findings** (required): Findings from git diff analysis
- **model** (required): AI model to use

**Required (Step 1):**
- **path**: Absolute path to repository root

**Optional:**
- **include_staged**: Include staged changes (default: true)
- **include_unstaged**: Include unstaged changes (default: true)
- **compare_to**: Git ref to diff against (branch/tag/commit)
- **precommit_type**: external (default) or internal
- **confidence**: exploring, low, medium, high, very_high, almost_certain, certain
- **files_checked**: Absolute paths of examined files
- **hypothesis**: Current theory about issues
- **relevant_files**: Files involved in changes (absolute paths)
- **relevant_context**: Methods/functions involved
- **issues_found**: List with severity levels
- **focus_on**: Specific areas to emphasize
- **severity_filter**: Minimum severity to report
- **continuation_id**: Resume previous validation
- **temperature**: 0=deterministic, 1=creative
- **thinking_mode**: minimal, low, medium, high, max

## Examples

### Example 1: Standard Pre-Commit Validation

```
PreCommit Validation:

Step 1: Validate changes before commit

Path: /home/user/project

Include staged: true
Include unstaged: true

Findings: "Reviewing git diff. Changed files:
- src/auth/middleware.py (70 lines added, 15 removed)
- tests/test_auth.py (30 lines added)
- requirements.txt (1 line added: pyjwt==2.8.0)

Will check: security, test coverage, breaking changes."

Relevant files:
- /home/user/project/src/auth/middleware.py
- /home/user/project/tests/test_auth.py
- /home/user/project/requirements.txt

Step: 1/3
Next: true

Model: gpt-5-codex
Precommit type: external
```

### Example 2: Security-Focused Pre-Commit

```
PreCommit Validation:

Step 1: Security check before commit

Path: /home/user/project

Focus on: "secrets, API keys, passwords, SQL injection"

Findings: "Scanning changes for security issues.
Found 2 potential concerns in payment_api.py."

Issues found:
- severity: critical
  description: "Line 45: API key hardcoded in source"
- severity: high
  description: "Line 78: SQL string concatenation (injection risk)"

Relevant files:
- /home/user/project/api/payment_api.py

Step: 1/3
Next: true

Model: gpt-5-pro
Thinking mode: max
```

### Example 3: Compare Against Branch

```
PreCommit Validation:

Step 1: Review feature branch before merge

Path: /home/user/project
Compare to: main

Findings: "Comparing feature/auth-refactor against main branch.
15 files changed, 450 additions, 200 deletions."

Focus on: "Breaking changes, test coverage, documentation updates"

Relevant files:
- /home/user/project/src/auth/* (8 files)
- /home/user/project/tests/* (5 files)
- /home/user/project/README.md
- /home/user/project/CHANGELOG.md

Step: 1/3
Next: true

Model: gemini-2.5-pro
```

### Example 4: Multi-Repository Validation

```
PreCommit Validation:

Step 1: Validate API and client changes together

Path: /home/user/api-server

Include staged: true
Precommit type: external

Findings: "API contract changed in endpoints.py.
Must verify client-side updates in separate repo."

Issues found:
- severity: high
  description: "Breaking change: removed /v1/users endpoint"
- severity: medium
  description: "No client update detected (check ../client repo)"

Relevant files:
- /home/user/api-server/api/endpoints.py
- /home/user/api-server/docs/api_spec.yaml

Step: 1/3
Next: true
```

## Validation Steps

**External Validation** (default): Up to 3 steps
1. Analysis - Review changes, identify issues
2. Follow-ups - Deep dive on concerns
3. Summary - Final report

**Internal Validation**: 1 step
- Quick self-review only
- Use for trivial changes

## What PreCommit Checks

**Security:**
- Exposed secrets (API keys, passwords, tokens)
- Hardcoded credentials
- SQL injection risks
- XSS vulnerabilities

**Completeness:**
- Missing tests for new code
- Missing documentation updates
- Incomplete feature (missing files?)
- Related files not updated

**Quality:**
- Code style issues
- Complex functions
- Missing error handling

**Impact:**
- Breaking changes
- API contract changes
- Database migrations
- Dependency updates

## Severity Levels

**critical** - Exposed secrets, severe security issues
**high** - Breaking changes, major bugs
**medium** - Missing tests, code quality
**low** - Style issues, suggestions

## Best Practices

**DO:**
- ✅ Run before every commit
- ✅ Review both staged and unstaged
- ✅ Check for exposed secrets
- ✅ Verify test coverage
- ✅ Use absolute repository path
- ✅ Fix critical/high issues before committing

**DON'T:**
- ❌ Commit with critical/high issues unresolved
- ❌ Skip security checks
- ❌ Ignore missing tests
- ❌ Forget to check related files
- ❌ Use relative paths

## Comparison Modes

**Staged + Unstaged** (default):
```
include_staged: true
include_unstaged: true
```

**Staged Only** (ready to commit):
```
include_staged: true
include_unstaged: false
```

**Branch Comparison** (before merge):
```
compare_to: "main"
```

When `compare_to` is set, `include_staged` and `include_unstaged` are ignored.

## Model Selection

**Recommended models:**
- `gpt-5-codex` - Best for code changes
- `gpt-5-pro` - Security and completeness
- `gemini-2.5-pro` - Large diffs (1M context)

## Expert Validation

Default: External model validates findings. Set `precommit_type: internal` for quick self-review only.

## Integration with ModelChorus

Uses `mcp__zen__precommit` tool from Zen MCP server for systematic git change validation.

## Workflow Integration

**Typical Flow:**
1. Make code changes
2. Run PreCommit validation
3. Fix critical/high issues
4. Run tests
5. Commit changes
6. Optional: Run again for unstaged changes

**With Git Hooks:**
Add to `.git/hooks/pre-commit` to automate validation.

## See Also

- **codereview** - Review code without git context
- **debug** - Debug issues found in validation
- **consensus** - Get opinions on breaking changes
- **chat** - Quick questions about changes
