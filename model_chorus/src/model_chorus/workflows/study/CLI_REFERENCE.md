# STUDY Workflow - CLI Reference Guide

Complete command-line interface documentation for the STUDY workflow with detailed examples and usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Commands Overview](#commands-overview)
3. [study start](#study-start)
4. [study-next](#study-next)
5. [study view](#study-view)
6. [Common Patterns](#common-patterns)
7. [Output Formats](#output-formats)
8. [Error Handling](#error-handling)

---

## Quick Start

### Start Your First Investigation

```bash
model-chorus study start --scenario "What are the main design patterns in authentication systems?"
```

### Continue a Previous Investigation

```bash
model-chorus study start --scenario "Explore OAuth 2.0 deeper" --continue <thread-id>
```

### View Investigation History

```bash
model-chorus study view --investigation <thread-id>
```

---

## Commands Overview

| Command | Purpose | Status |
|---------|---------|--------|
| `study start` | Start new investigation or continue existing | ✓ Implemented |
| `study-next` | Convenience wrapper for continuation | ✓ Implemented |
| `study view` | View investigation history and results | ✓ Implemented |

---

## study start

Start a new investigation or continue an existing one.

### Syntax

```bash
model-chorus study start \
  --scenario "Investigation topic" \
  [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--scenario` | The research question or investigation topic (required) |

### Options

#### Provider Selection

```bash
--provider PROVIDER
-p PROVIDER
```

Select which AI model provider to use.

**Supported providers**: `claude`, `gemini`, `codex`, `cursor-agent`

**Default**: Uses value from config file or `claude`

**Examples**:
```bash
# Use Claude
model-chorus study start --scenario "..." --provider claude

# Use Gemini
model-chorus study start --scenario "..." --provider gemini

# Use abbreviation
model-chorus study start --scenario "..." -p codex
```

#### Persona Selection

```bash
--persona PERSONA_NAME
--persona PERSONA_NAME [--persona ANOTHER_PERSONA]
```

Specify which personas to use for investigation. Can specify multiple times.

**Default personas**: `Researcher`, `Critic`

**Examples**:
```bash
# Use specific personas
model-chorus study start \
  --scenario "Design a microservices architecture" \
  --persona Architect \
  --persona SecurityExpert \
  --persona PerformanceEngineer

# Use single custom persona
model-chorus study start \
  --scenario "..." \
  --persona DomainExpert

# Mix and match
model-chorus study start \
  --scenario "Analyze this code" \
  --persona CodeReviewer \
  --persona SecurityExpert
```

#### Context Files

```bash
--file FILE_PATH
-f FILE_PATH
```

Include file contents in investigation context. Can specify multiple times.

**Examples**:
```bash
# Single file
model-chorus study start \
  --scenario "Analyze this authentication module" \
  --file src/auth.py

# Multiple files
model-chorus study start \
  --scenario "Review this microservice" \
  --file src/main.ts \
  --file src/models.ts \
  --file tests/main.spec.ts

# Using abbreviation
model-chorus study start \
  --scenario "..." \
  -f src/service.py \
  -f src/middleware.py

# Absolute paths
model-chorus study start \
  --scenario "..." \
  --file /Users/username/project/src/auth.py
```

#### Generation Parameters

```bash
--temperature TEMP_VALUE
-t TEMP_VALUE
```

Control generation creativity (0.0 = deterministic, 1.0 = creative).

**Valid range**: 0.0 - 1.0

**Default**: 0.7

**Examples**:
```bash
# More deterministic (better for analysis)
model-chorus study start \
  --scenario "Analyze authentication patterns" \
  --temperature 0.3

# More creative (better for brainstorming)
model-chorus study start \
  --scenario "Generate new design patterns" \
  --temperature 0.9

# Using abbreviation
model-chorus study start --scenario "..." -t 0.5
```

```bash
--max-tokens TOKENS
```

Limit maximum response length.

**Examples**:
```bash
# Keep responses concise
model-chorus study start \
  --scenario "..." \
  --max-tokens 1000

# Allow longer responses
model-chorus study start \
  --scenario "..." \
  --max-tokens 4000
```

#### System Prompt

```bash
--system SYSTEM_PROMPT
```

Provide a custom system prompt for context.

**Examples**:
```bash
model-chorus study start \
  --scenario "Analyze security patterns" \
  --system "You are analyzing from the perspective of a security-first team"

# Multi-line system prompt
model-chorus study start \
  --scenario "..." \
  --system "You are evaluating code quality with focus on:
- Maintainability
- Performance
- Security"
```

#### Continuation

```bash
--continue THREAD_ID
-c THREAD_ID
```

Continue a previous investigation.

**Examples**:
```bash
# Continue with explicit thread ID
model-chorus study start \
  --scenario "Explore OAuth 2.0 deeper" \
  --continue 550e8400-e29b-41d4-a716-446655440000

# Continue with prompt suggestion
model-chorus study start \
  --scenario "..." \
  -c thread-id-123
```

#### Output

```bash
--output FILE_PATH
-o FILE_PATH
```

Save results to JSON file.

**Examples**:
```bash
# Save to file
model-chorus study start \
  --scenario "..." \
  --output results.json

# Using abbreviation
model-chorus study start --scenario "..." -o analysis.json

# With absolute path
model-chorus study start \
  --scenario "..." \
  --output /Users/username/Desktop/results.json
```

#### Verbose Output

```bash
--verbose
-v
```

Show detailed execution information including timestamps and metadata.

**Examples**:
```bash
# Verbose output
model-chorus study start --scenario "..." --verbose

# Using abbreviation
model-chorus study start --scenario "..." -v
```

#### Provider Check

```bash
--skip-provider-check
```

Skip provider availability check (faster startup but fails later if provider unavailable).

**Use when**: Running on slow networks or for quick testing

**Examples**:
```bash
model-chorus study start \
  --scenario "..." \
  --skip-provider-check
```

### Complete Example

```bash
model-chorus study start \
  --scenario "Analyze authentication patterns in OAuth 2.0 and SAML, focusing on security implications" \
  --provider claude \
  --persona SecurityExpert \
  --persona Architect \
  --persona PerformanceEngineer \
  --file src/auth_module.py \
  --file src/oauth_handler.ts \
  --temperature 0.5 \
  --max-tokens 3000 \
  --output auth_analysis.json \
  --verbose
```

### Output

#### Console Output

```
✓ STUDY investigation completed

Thread ID: 550e8400-e29b-41d4-a716-446655440000
Status: New investigation started
Personas: SecurityExpert, Architect, PerformanceEngineer

Investigation Steps:

[SecurityExpert]:
From a security perspective, OAuth 2.0 uses the Authorization Code flow...

[Architect]:
The architectural benefits of OAuth 2.0 include separation of concerns...

[PerformanceEngineer]:
Performance-wise, OAuth 2.0 can introduce latency in token exchange...

Research Synthesis:
OAuth 2.0 provides strong security guarantees through token-based auth,
architectural flexibility through delegation, and reasonable performance
when cached properly. Consider SAML for enterprise scenarios requiring
stronger federation...

To continue this investigation, use: --continue 550e8400-e29b-41d4-a716-446655440000
```

#### JSON Output (with --output flag)

```json
{
  "scenario": "Analyze authentication patterns in OAuth 2.0...",
  "provider": "claude",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_continuation": false,
  "personas_used": ["SecurityExpert", "Architect", "PerformanceEngineer"],
  "steps": [
    {
      "persona": "SecurityExpert",
      "content": "Security analysis of OAuth 2.0...",
      "metadata": {
        "step_number": 1,
        "model": "claude-3-opus"
      }
    },
    {
      "persona": "Architect",
      "content": "Architectural perspective on OAuth 2.0...",
      "metadata": {
        "step_number": 2,
        "model": "claude-3-opus"
      }
    },
    {
      "persona": "PerformanceEngineer",
      "content": "Performance analysis of OAuth 2.0...",
      "metadata": {
        "step_number": 3,
        "model": "claude-3-opus"
      }
    }
  ],
  "synthesis": "Combined analysis focusing on security, architecture, and performance...",
  "model": "claude-3-opus",
  "usage": {
    "total_tokens": 3250
  }
}
```

---

## study-next

Convenience wrapper for continuing investigations. Automatically continues using the existing thread context.

### Syntax

```bash
model-chorus study-next \
  --investigation THREAD_ID \
  [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--investigation` | Thread ID of investigation to continue (required) |

### Options

#### Provider Selection

```bash
--provider PROVIDER
-p PROVIDER
```

Switch providers for continuation.

**Examples**:
```bash
# Continue with same provider
model-chorus study-next --investigation thread-id-123

# Switch to different provider
model-chorus study-next \
  --investigation thread-id-123 \
  --provider gemini
```

#### Additional Context

```bash
--file FILE_PATH
-f FILE_PATH
```

Add new context files to existing investigation.

**Examples**:
```bash
model-chorus study-next \
  --investigation thread-id-123 \
  --file new_context.py
```

#### Output

```bash
--output FILE_PATH
-o FILE_PATH
```

Save continuation results.

**Examples**:
```bash
model-chorus study-next \
  --investigation thread-id-123 \
  --output continuation_results.json
```

#### Verbose Output

```bash
--verbose
-v
```

Show execution details.

### Complete Example

```bash
model-chorus study-next \
  --investigation 550e8400-e29b-41d4-a716-446655440000 \
  --provider claude \
  --file additional_context.py \
  --output continuation_results.json \
  --verbose
```

### Output

Similar to `study start`, showing continuation results with previous context preserved.

---

## study view

View investigation history and conversation memory.

### Syntax

```bash
model-chorus study view \
  --investigation THREAD_ID \
  [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--investigation` | Thread ID to view (required) |

### Options

#### Persona Filter

```bash
--persona PERSONA_NAME
```

Show only messages from specific persona.

**Examples**:
```bash
# View SecurityExpert contributions
model-chorus study view \
  --investigation thread-id-123 \
  --persona SecurityExpert

# View Architect perspective
model-chorus study view \
  --investigation thread-id-123 \
  --persona Architect
```

#### Full Content

```bash
--show-all
```

Don't truncate long messages.

**Examples**:
```bash
# Show complete messages
model-chorus study view \
  --investigation thread-id-123 \
  --show-all

# Show full messages for specific persona
model-chorus study view \
  --investigation thread-id-123 \
  --persona SecurityExpert \
  --show-all
```

#### JSON Output

```bash
--json
```

Output in JSON format for processing.

**Examples**:
```bash
# Get JSON output
model-chorus study view \
  --investigation thread-id-123 \
  --json

# Process with jq
model-chorus study view \
  --investigation thread-id-123 \
  --json | jq '.messages | length'
```

#### Verbose Output

```bash
--verbose
-v
```

Show timestamps and detailed metadata.

**Examples**:
```bash
model-chorus study view \
  --investigation thread-id-123 \
  --verbose
```

### Complete Example

```bash
model-chorus study view \
  --investigation 550e8400-e29b-41d4-a716-446655440000 \
  --persona SecurityExpert \
  --show-all \
  --verbose
```

### Output

#### Default Output

```
STUDY Investigation Memory

Investigation ID: 550e8400-e29b-41d4-a716-446655440000
Total Messages: 6

Conversation History:

[Message 1]
  Role: user
  Persona: User
  Content: Analyze authentication patterns in OAuth 2.0...

──────────────────────────────────────────────────────────

[Message 2]
  Role: assistant
  Persona: SecurityExpert
  Content: From a security perspective, OAuth 2.0...

──────────────────────────────────────────────────────────
```

#### JSON Output

```json
{
  "investigation_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_messages": 6,
  "filtered_messages": 6,
  "metadata": {},
  "messages": [
    {
      "role": "user",
      "content": "Analyze authentication patterns in OAuth 2.0...",
      "metadata": {"workflow": "study"},
      "timestamp": "2025-11-08T23:45:00Z"
    },
    {
      "role": "assistant",
      "content": "From a security perspective, OAuth 2.0...",
      "metadata": {"persona": "SecurityExpert"},
      "timestamp": "2025-11-08T23:46:15Z"
    }
  ]
}
```

---

## Common Patterns

### Pattern 1: Quick Analysis with Single Persona

```bash
model-chorus study start \
  --scenario "Quick authentication review" \
  --persona SecurityExpert \
  --temperature 0.5
```

### Pattern 2: Multi-Perspective Code Review

```bash
model-chorus study start \
  --scenario "Review and analyze this service" \
  --file src/service.ts \
  --file tests/service.spec.ts \
  --persona CodeReviewer \
  --persona SecurityExpert \
  --persona PerformanceEngineer \
  --output code_review.json
```

### Pattern 3: Iterative Research

```bash
# First investigation
THREAD=$(model-chorus study start \
  --scenario "OAuth 2.0 overview" \
  --output initial_analysis.json \
  | grep "Thread ID:" | awk '{print $NF}')

# Deepen the analysis
model-chorus study start \
  --scenario "Deep dive into OAuth 2.0 refresh tokens" \
  --continue $THREAD \
  --output deeper_analysis.json

# Another angle
model-chorus study start \
  --scenario "Security vulnerabilities in OAuth 2.0 implementations" \
  --continue $THREAD \
  --output security_analysis.json
```

### Pattern 4: File-Based Investigation

```bash
# Analyze entire codebase
model-chorus study start \
  --scenario "Analyze this authentication system for patterns and issues" \
  --file src/auth/oauth.ts \
  --file src/auth/jwt.ts \
  --file src/auth/middleware.ts \
  --file tests/auth.spec.ts \
  --persona SecurityExpert \
  --persona Architect \
  --output auth_system_analysis.json \
  --verbose
```

### Pattern 5: Configuration-Based Investigation

```bash
# Use different configurations for different aspects
# Security analysis
model-chorus study start \
  --scenario "Security review of authentication" \
  --temperature 0.3 \
  --persona SecurityExpert \
  --output security.json

# Design review
model-chorus study start \
  --scenario "Architecture review of authentication" \
  --temperature 0.5 \
  --persona Architect \
  --output architecture.json

# Performance review
model-chorus study start \
  --scenario "Performance analysis of authentication" \
  --temperature 0.4 \
  --persona PerformanceEngineer \
  --output performance.json
```

### Pattern 6: Batch Investigation with History

```bash
#!/bin/bash

# Start investigation
echo "Starting investigation..."
result=$(model-chorus study start \
  --scenario "OAuth 2.0 analysis" \
  --output initial.json \
  --verbose)

# Extract thread ID (adjust parsing as needed)
thread_id=$(echo "$result" | grep "Thread ID:" | awk '{print $NF}')
echo "Thread ID: $thread_id"

# View the investigation
echo ""
echo "Investigation Summary:"
model-chorus study view --investigation "$thread_id"

# Continue investigation
echo ""
echo "Continuing investigation..."
model-chorus study start \
  --scenario "Deeper analysis" \
  --continue "$thread_id" \
  --output continuation.json

# View updated investigation
echo ""
echo "Updated Investigation:"
model-chorus study view --investigation "$thread_id" --show-all
```

### Pattern 7: Filtering Results

```bash
# View only security expert perspective
model-chorus study view \
  --investigation thread-id-123 \
  --persona SecurityExpert \
  --json | jq '.messages[] | select(.metadata.persona=="SecurityExpert")'

# Count messages per persona
model-chorus study view \
  --investigation thread-id-123 \
  --json | jq '.messages | group_by(.metadata.persona) | map({persona: .[0].metadata.persona, count: length})'

# Extract just the content
model-chorus study view \
  --investigation thread-id-123 \
  --json | jq '.messages[].content'
```

---

## Output Formats

### Console Format

Human-readable output with:
- Thread ID display
- Status information
- Persona labels
- Investigation steps
- Synthesis summary
- Helpful hints for continuation

### JSON Format

Machine-readable JSON with:
- Full metadata
- All messages with timestamps
- Complete synthesis
- Token usage (if available)
- Investigation parameters

### Saved File Format

Same as JSON format when using `--output` flag.

---

## Error Handling

### Provider Not Available

**Error**:
```
Error: No providers available for study workflow:
  - claude: API key not configured

To fix this:
  Check installations: model-chorus list-providers --check
  Install missing providers or update .model-chorusrc
```

**Solutions**:
```bash
# Check available providers
model-chorus list-providers --check

# Configure API key
export ANTHROPIC_API_KEY="your-api-key"

# Retry command
model-chorus study start --scenario "..."
```

### Investigation Not Found

**Error**:
```
Error: Investigation not found: invalid-thread-id
Make sure you're using the correct thread ID from a previous investigation.

To start a new investigation, use: model-chorus study --scenario "..."
```

**Solutions**:
```bash
# View available investigations
ls ~/.model-chorus/conversations/

# Use correct thread ID
model-chorus study start --scenario "..." --continue correct-thread-id
```

### File Not Found

**Error**:
```
Error: File not found: src/missing-file.py
```

**Solutions**:
```bash
# Check file exists
ls src/missing-file.py

# Use correct path
model-chorus study start --scenario "..." --file src/correct-file.py

# Use absolute path
model-chorus study start --scenario "..." --file /Users/username/project/src/file.py
```

### Interrupted Investigation

**Error**:
```
Interrupted by user
```

**Note**: Press Ctrl+C to interrupt. Results up to interruption are preserved in conversation memory.

---

## Tips and Tricks

### Store Thread IDs

```bash
# Save thread ID to file for later use
model-chorus study start --scenario "..." | tee investigation.log | grep "Thread ID:" > thread_id.txt

# Use saved thread ID
thread=$(cat thread_id.txt | awk '{print $NF}')
model-chorus study view --investigation "$thread"
```

### Pipe Results

```bash
# Get just the synthesis
model-chorus study view --investigation thread-id-123 --json | jq '.synthesis'

# Count total messages
model-chorus study view --investigation thread-id-123 --json | jq '.messages | length'
```

### Compare Personas

```bash
# Create reports for each persona
for persona in SecurityExpert Architect Developer; do
  model-chorus study view \
    --investigation thread-id-123 \
    --persona "$persona" \
    --show-all > "${persona}_perspective.txt"
done
```

### Archive Investigations

```bash
# Export investigation to archive
model-chorus study view \
  --investigation thread-id-123 \
  --json > investigation_backup.json

# Later restore/view from backup
cat investigation_backup.json | jq '.synthesis'
```

---

## See Also

- [STUDY Workflow README](./README.md)
- [Model Chorus CLI Documentation](../../cli/main.py)
- [Configuration Guide](../../core/config.py)
