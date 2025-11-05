# ModelChorus Debug

Quick access to systematic debugging workflow for investigating bugs, errors, and unexpected behavior.

## Usage

```
/mc-debug [describe the bug or issue]
```

## What It Does

Launches the ModelChorus debug skill to:
- Investigate complex bugs systematically
- Find root causes of errors
- Debug performance issues
- Analyze race conditions
- Solve mysterious intermittent problems

## Examples

```
/mc-debug API endpoint randomly returns 500 errors under load
```

```
/mc-debug Memory leak in Python service - grows from 200MB to 4GB over 24 hours
```

```
/mc-debug Order processing occasionally creates duplicate charges
```

## Features

- Systematic hypothesis testing
- Evidence-based investigation
- Root cause identification
- Expert validation
- Confidence tracking
- Can conclude "no bug found" with evidence

## Process

1. **Problem Documentation** - Clearly state issue and symptoms
2. **Hypothesis Formation** - Develop theories about root cause
3. **Evidence Collection** - Gather data systematically
4. **Root Cause Analysis** - Identify actual cause
5. **Validation** - Expert review of findings

## See Also

- `/mc-chat` - For quick debugging discussions
- `/mc-review` - For preventive code review
- Skill: `model-chorus:debug` for full debugging capabilities
