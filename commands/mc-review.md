# ModelChorus Review

Quick access to comprehensive code review workflow for quality, security, performance, and architecture analysis.

## Usage

```
/mc-review [files to review]
```

## What It Does

Launches the ModelChorus code review skill to:
- Review code comprehensively
- Find security vulnerabilities
- Identify performance issues
- Assess architecture quality
- Check best practices

## Examples

```
/mc-review src/auth/middleware.py
```

```
/mc-review api/ tests/ --focus security
```

```
/mc-review payment_processor.py --type security
```

## Review Types

- **full** - Complete review (quality, security, performance, architecture)
- **security** - Focus on security issues only
- **performance** - Focus on performance problems
- **quick** - Fast sanity check

## Features

- Systematic analysis across all dimensions
- Issue severity tracking (critical/high/medium/low)
- Expert validation
- Positive findings noted
- Coding standards enforcement

## What Gets Reviewed

**Code Quality:**
- Readability and maintainability
- Code organization
- Best practices
- Style consistency

**Security:**
- SQL injection, XSS, CSRF
- Authentication/authorization
- Data exposure
- Input validation

**Performance:**
- Algorithmic complexity
- Database query efficiency
- Caching opportunities
- Resource usage

**Architecture:**
- Design patterns
- Code structure
- Dependencies
- Testability

## See Also

- `/mc-chat` - For design discussions
- `/mc-debug` - For debugging issues found
- Skill: `model-chorus:codereview` for full review capabilities
- Skill: `model-chorus:precommit` for pre-commit validation
