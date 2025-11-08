# ModelChorus Terminology Glossary

**Version:** 1.0
**Last Updated:** 2025-11-07

This glossary defines standardized terminology used across all ModelChorus workflows to ensure consistency and clarity for AI coding agents.

---

## Core Concepts

### Session Management

**session_id**
- Unique identifier for continuing multi-turn conversations
- **Standardized term** across all workflows (replaces `thread_id`, `continuation_id`)
- Format: `session-abc-123-def-456` (hyphen-separated alphanumeric)
- Used with `--continue` flag to resume previous conversations
- **State Preserved:**
  - **CHAT, ARGUMENT, IDEATE:** Conversation history only
  - **THINKDEEP:** Full investigation context (hypothesis, findings, confidence, files_checked)
  - **CONSENSUS:** Not supported (single-turn workflow)

**investigation_context**
- Extended state preservation for THINKDEEP workflow
- Includes: hypothesis, findings, confidence levels, files_checked, relevant_files
- More comprehensive than simple conversation history
- Allows resuming complex multi-step investigations

---

### Model & Provider Configuration

**provider**
- AI service provider powering the workflow
- **Standardized parameter:** `--provider [name]`
- Valid values: `claude`, `gemini`, `codex`, `cursor-agent`
- Default: `claude` (recommended for most use cases)
- Each provider has different strengths:
  - `claude`: General-purpose, strong reasoning
  - `gemini`: Large context windows, multi-modal
  - `codex`: Code-focused, technical tasks
  - `cursor-agent`: IDE integration, code editing

**temperature**
- Controls response creativity and randomness
- Range: 0.0 (deterministic) to 1.0 (creative)
- **Standardized default:** 0.7 for all workflows
- Lower values (0.2-0.5): Factual, consistent outputs
- Higher values (0.7-0.9): Creative, exploratory outputs
- Parameter: `--temperature [value]` or `-t [value]`

**thinking_mode** (THINKDEEP only)
- Controls reasoning depth and thoroughness
- Values: `minimal`, `low`, `medium`, `high`, `max`
- Default: `medium`
- Higher modes: More thorough analysis, longer response times
- Parameter: `--thinking-mode [mode]`

---

### File & Path Management

**file**
- Input file(s) providing context to the workflow
- **Standard parameter:** `--file [path]` or `-f [path]`
- **Path requirements:** Always use absolute paths
- Can be specified multiple times: `-f file1.py -f file2.py`
- Common use: Provide codebase context for analysis

**output**
- Destination file for saving workflow results
- **Standard parameter:** `--output [path]` or `-o [path]`
- Format: JSON (default) or text/markdown (workflow-specific)
- Includes: Result data, metadata, session_id for continuation

**absolute_path**
- Full file system path from root directory
- Format: `/home/user/project/file.py` (Unix) or `C:\Users\...` (Windows)
- **Requirement:** All file paths in ModelChorus must be absolute
- Prevents ambiguity about file locations
- Example: `/home/tyler/Documents/GitHub/project/src/main.py`

**relative_path**
- Path relative to current working directory
- **Not recommended** for ModelChorus workflows (use absolute paths)
- Can cause errors if working directory changes

---

### Workflow Execution

**step** (THINKDEEP only)
- Individual investigation phase in multi-step analysis
- Numbered sequentially: step 1, step 2, step 3, etc.
- Each step advances the hypothesis and accumulates findings
- Related parameters: `step_number`, `total_steps`, `next_step_required`

**hypothesis** (THINKDEEP only)
- Current working theory about the problem being investigated
- Evolves as evidence is gathered across steps
- Can be revised or replaced based on findings
- Represents agent's best understanding at current step

**confidence** (THINKDEEP only)
- Agent's certainty level in the current hypothesis
- Scale: `exploring` → `low` → `medium` → `high` → `very_high` → `almost_certain` → `certain`
- Increases as supporting evidence accumulates
- Can decrease if contradictory evidence found

**findings**
- Evidence, insights, or discoveries made during analysis
- **Context-dependent meaning:**
  - **THINKDEEP:** Evidence accumulated during investigation
- Cumulative: New findings build on previous ones
- Structured: Organized by relevance and importance

---

### Consensus & Multi-Model Workflows

**strategy** (CONSENSUS only)
- Method for combining multiple model responses
- Values:
  - `all_responses`: Show all model outputs separately
  - `synthesize`: Combine into unified response
  - `debate`: Models challenge each other's reasoning
  - `vote`: Simple majority voting
- Parameter: `--strategy [name]`
- Affects: Output structure and synthesis approach

**synthesis**
- Process of combining multiple outputs into cohesive result
- **Used differently across workflows:**
  - **CONSENSUS:** Combining multiple model responses
  - **ARGUMENT:** Synthesizing role perspectives (Creator/Skeptic/Moderator)
  - **IDEATE:** Merging generated ideas
- Common goal: Unified, comprehensive output from multiple sources

**role** (ARGUMENT only)
- Perspective taken in structured argument analysis
- Values: `Creator` (proposes idea), `Skeptic` (challenges), `Moderator` (synthesizes)
- Executed sequentially for balanced analysis
- No parallel in other workflows

---

### Ideation

**idea** (IDEATE only)
- Individual concept generated during ideation session
- Enumerated: Idea 1, Idea 2, Idea 3, etc.
- Parameter: `--num-ideas [count]` controls quantity generated
- Synthesis combines ideas into unified output

**num_ideas** (IDEATE only)
- Number of distinct ideas to generate
- Parameter: `--num-ideas [count]`
- Default: 3-5 (workflow-determined)
- Higher counts: More diversity, longer execution time

---

### Output & Results

**workflow_result**
- Generic term for any workflow's output
- Structure varies by workflow but includes common metadata
- Always contains: `session_id` (if applicable), timestamp, provider info
- Saved to file via `--output` parameter

**model_response**
- Direct, unprocessed output from AI model
- Distinguished from synthesized or processed results
- Used in CONSENSUS to show individual model outputs
- May include: Raw text, structured data, metadata

**synthesized_result**
- Output created by combining multiple inputs
- Examples:
  - CONSENSUS: Multiple model responses merged
  - ARGUMENT: Three role perspectives unified
  - IDEATE: Multiple ideas combined into coherent direction
- Contrasts with single model_response

**metadata**
- Execution information attached to workflow results
- Common fields:
  - Provider and model used
  - Execution timestamp
  - Token usage
  - Temperature and other parameters
  - Session ID for continuation
- Enables: Reproducibility, debugging, cost tracking

---

## Parameter Format Standards

### CLI Parameters
- **Format:** kebab-case with double dashes
- **Example:** `--session-id`, `--thinking-mode`, `--output-file`
- **Short flags:** Single dash with single letter: `-f`, `-o`, `-t`
- **Boolean flags:** Presence indicates true: `--verbose`, `--json`

### Internal Parameters
- **Format:** snake_case for code/config
- **Example:** `session_id`, `thinking_mode`, `output_file`
- **Reason:** Matches Python naming conventions
- **Conversion:** CLI kebab-case automatically converts to internal snake_case

### File Paths
- **Always absolute:** `/full/path/to/file`
- **Never relative:** `./file` or `../other/file`
- **Quote if spaces:** `"/path/with spaces/file.txt"`
- **Platform-aware:** Unix (`/home/user`) vs Windows (`C:\Users\`)

---

## Workflow-Specific Terminology

### CHAT
- Simple single-turn or multi-turn conversations
- No special terminology beyond core concepts
- Minimal state preservation in continuation

### CONSENSUS
- Multi-model consultation with strategy-based synthesis
- Key terms: `strategy`, `provider_response`, `synthesis`
- No session continuation support

### THINKDEEP
- Extended multi-step investigation workflow
- Key terms: `step`, `hypothesis`, `confidence`, `findings`, `investigation_context`
- Rich state preservation for complex analysis

### ARGUMENT
- Structured debate with role-based perspectives
- Key terms: `role` (Creator/Skeptic/Moderator), `synthesis`
- Sequential role execution pattern

### IDEATE
- Creative idea generation with synthesis
- Key terms: `idea`, `num_ideas`, `synthesis`
- Divergent thinking followed by convergence

---

## Common Patterns

### Multi-Turn Workflows
Workflows supporting session continuation:
- **CHAT:** Yes (conversation history)
- **CONSENSUS:** No (single-turn only)
- **THINKDEEP:** Yes (full investigation context)
- **ARGUMENT:** Yes (conversation history)
- **IDEATE:** Yes (conversation history)

### State Preservation Levels
- **Simple:** Conversation history only (CHAT, ARGUMENT, IDEATE)
- **Complex:** Full context with structured state (THINKDEEP)
- **None:** No continuation support (CONSENSUS)

### Multi-Model Workflows
Workflows that use multiple AI models:
- **CONSENSUS:** Yes (core feature)
- All others: Single model per execution (but can be chained)

---

## Deprecation Notice

### Replaced Terms

| Deprecated | Standard Replacement | Affected Workflows |
|------------|---------------------|-------------------|
| `thread_id` | `session_id` | CHAT, ARGUMENT, IDEATE |
| `continuation_id` | `session_id` | THINKDEEP |
| `--model` | `--provider` | THINKDEEP (legacy usage) |

**Migration:**
- Old parameters still supported for backward compatibility
- Will emit warnings recommending standard terms
- Planned removal: v2.0.0

---

## Usage Examples

### Starting a new session
```bash
modelchorus chat --provider claude --temperature 0.7 "Analyze this code structure"
```

### Continuing a session
```bash
modelchorus chat --continue session-abc-123 "What about error handling?"
```

### With file context
```bash
modelchorus thinkdeep --provider gemini -f /absolute/path/to/file.py "Debug this function"
```

### Saving output
```bash
modelchorus consensus --output /path/to/results.json "Analyze this topic"
```

### Complex multi-step investigation
```bash
modelchorus thinkdeep \
  --provider claude \
  --thinking-mode high \
  -f /project/src/main.py \
  -f /project/tests/test_main.py \
  --output /project/analysis.json \
  "Why is this test failing?"
```

---

## Related Documentation

- **Workflow Selection Guide:** `docs/WORKFLOW_SELECTION_GUIDE.md` - Choosing the right workflow
- **Skill Documentation:** `skills/*/SKILL.md` - Detailed workflow instructions
- **Parameter Reference:** See individual SKILL.md files for complete parameter lists

---

## Glossary Maintenance

**Ownership:** ModelChorus core team
**Review Cycle:** Quarterly or when new workflows added
**Change Process:**
1. Propose terminology change via issue/PR
2. Review for consistency across all workflows
3. Update glossary and affected SKILL.md files
4. Document in CHANGELOG with migration guidance

**Questions or suggestions?** Open an issue in the ModelChorus repository.

---

*This glossary is a living document. When in doubt, refer to this document for authoritative terminology definitions.*
