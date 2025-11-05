---
name: model-chorus:codereview
description: Performs systematic, step-by-step code review with expert validation covering quality, security, performance, and architecture
---

# ModelChorus: CodeReview

Comprehensive code review workflow with systematic analysis across quality, security, performance, and architecture dimensions, followed by expert validation.

## When to Use

Use this skill when you need:
- **Pre-commit reviews** - Review code before committing
- **Pull request analysis** - Comprehensive PR review
- **Security audit** - Focus on security vulnerabilities
- **Performance review** - Identify performance issues
- **Architecture assessment** - Evaluate design patterns
- **Quick sanity check** - Fast review for simple changes

## How It Works

CodeReview uses a structured approach:

1. **Review Strategy** - Plan which aspects to review
2. **Systematic Analysis** - Review code across dimensions:
   - Code quality (readability, maintainability)
   - Security (vulnerabilities, best practices)
   - Performance (efficiency, bottlenecks)
   - Architecture (design patterns, structure)
3. **Issue Documentation** - Track findings with severity
4. **Expert Validation** - Optional external model validates findings

## Usage

Invoke CodeReview with files to review:

```
Use CodeReview to review:
[Description of what to review]

Files:
- /absolute/path/to/file1.py
- /absolute/path/to/file2.ts

Review type: [full/security/performance/quick]
Focus: [specific areas to emphasize]
```

## Parameters

- **step** (required): Review narrative for this step
- **step_number** (required): Current step (starts at 1)
- **total_steps** (required): Total review steps
- **next_step_required** (required): True if more review needed
- **findings** (required): Findings (positive and negative)
- **model** (required): AI model to use

**Required (Step 1):**
- **relevant_files**: All files/dirs under review (absolute paths)

**Optional:**
- **review_type**: full, security, performance, quick (default: full)
- **review_validation_type**: external (default) or internal
- **confidence**: exploring, low, medium, high, very_high, almost_certain, certain
- **files_checked**: Absolute paths of examined files
- **hypothesis**: Current theory about issues
- **relevant_context**: Methods/functions involved
- **issues_found**: List with severity (critical/high/medium/low)
- **focus_on**: Specific areas to emphasize
- **standards**: Coding standards to enforce
- **severity_filter**: Minimum severity to report (default: all)
- **continuation_id**: Resume previous review
- **temperature**: 0=deterministic, 1=creative
- **thinking_mode**: minimal, low, medium, high, max

## Review Types

**full** (default) - Complete review of all dimensions
- Code quality
- Security
- Performance
- Architecture

**security** - Focus on security issues only
- SQL injection, XSS, CSRF
- Authentication/authorization
- Data exposure
- Input validation

**performance** - Focus on performance issues
- Algorithmic complexity
- Database queries
- Caching opportunities
- Resource usage

**quick** - Fast sanity check
- Obvious issues only
- Critical bugs
- Basic best practices

## Examples

### Example 1: Full PR Review

```
CodeReview:

Step 1: Review authentication refactoring PR

Review type: full

Relevant files:
- /home/user/project/auth/middleware.py
- /home/user/project/auth/jwt_handler.py
- /home/user/project/tests/test_auth.py

Findings: "Reviewing 3 files for quality, security, performance,
and architecture. Will check JWT implementation, test coverage,
and error handling."

Step: 1/2
Next: true

Model: gpt-5-codex
Thinking mode: high
Review validation: external
```

### Example 2: Security-Focused Review

```
CodeReview:

Step 1: Security audit of payment processing

Review type: security

Relevant files:
- /home/user/project/payments/processor.py
- /home/user/project/api/payment_endpoints.py
- /home/user/project/models/transaction.py

Focus on: "SQL injection, authentication, data exposure,
input validation for payment amounts"

Findings: "Reviewing payment flow for security issues.
Critical: verify input sanitization, check for SQL injection,
review authentication requirements."

Issues found:
- severity: high
  description: "payment_endpoints.py line 45 - SQL injection risk"

Step: 1/2
Next: true

Model: gpt-5-pro
Thinking mode: max
Severity filter: high
```

### Example 3: Performance Review

```
CodeReview:

Step 1: Review slow API endpoint

Review type: performance

Relevant files:
- /home/user/project/api/analytics.py
- /home/user/project/db/queries.py

Focus on: "Database queries, N+1 problems, caching opportunities"

Findings: "Analyzing query patterns and load times. Analytics
endpoint makes 100+ DB queries per request."

Issues found:
- severity: high
  description: "N+1 query problem in get_user_stats()"
- severity: medium
  description: "Missing index on analytics_events.user_id"

Step: 1/2
Next: true

Model: gemini-2.5-pro
```

### Example 4: Quick Review

```
CodeReview:

Step 1: Quick check before commit

Review type: quick

Relevant files:
- /home/user/project/utils/validators.py

Findings: "Quick sanity check of validation utilities.
Code is clean, follows conventions, has tests."

Confidence: high
Step: 1/1
Next: false

Model: gpt-5-codex
Review validation: internal
```

## Severity Levels

**critical** - Security vulnerabilities, data loss risks
**high** - Bugs, major performance issues
**medium** - Code quality, minor issues
**low** - Style, suggestions

Filter results with `severity_filter` parameter.

## Review Workflow

**Step 1: Strategy**
- Outline review approach
- List files to examine
- Define focus areas
- Set severity thresholds

**Step 2: Analysis** (if external validation)
- Review each file systematically
- Document findings with severity
- Update files_checked list
- Track issues found

**Final Output:**
- Comprehensive findings report
- Issues categorized by severity
- Recommendations for improvement
- Positive aspects noted

## Validation Types

**external** (default):
- 2 steps: analysis + summary
- External model validates findings
- Use for important reviews

**internal**:
- 1 step: your analysis only
- No external validation
- Use for quick reviews

## Best Practices

**DO:**
- ✅ Use absolute file paths
- ✅ Cover all dimensions (quality, security, performance, architecture)
- ✅ Document both issues AND positives
- ✅ Assign appropriate severity levels
- ✅ Be specific about issues and locations
- ✅ Avoid pasting large code snippets (reference files instead)

**DON'T:**
- ❌ Review more than 10-15 files at once
- ❌ Skip positive findings
- ❌ Use relative paths
- ❌ Over-flag low-severity issues
- ❌ Dump code in review narrative

## Model Selection

**Recommended models:**
- `gpt-5-codex` - Best for code-specific review
- `gemini-2.5-pro` - Excellent analysis (1M context)
- `gpt-5-pro` - Strong reasoning for complex code

## Standards Enforcement

Specify coding standards to enforce:
```
standards: "PEP 8, type hints required, docstrings for public APIs,
max function length 50 lines, test coverage >80%"
```

## Integration with ModelChorus

Uses `mcp__zen__codereview` tool from Zen MCP server for structured code review workflows.

## See Also

- **precommit** - Review changes before committing
- **debug** - Debug specific issues found
- **consensus** - Get multiple opinions on design
- **thinkdeep** - Deep analysis of architectural patterns
