---
name: research
description: Systematic research with evidence extraction, source management, and citation formatting for comprehensive information gathering
---

# RESEARCH

## Overview

The RESEARCH workflow provides systematic research and information gathering through structured investigation with citation management. This workflow breaks research into multiple focused findings, manages sources, formats citations, and synthesizes comprehensive research dossiers.

**Key Capabilities:**
- Structured investigation broken into multiple focused findings
- Source tracking and reference management (supports local file ingestion)
- Configurable citation styles (informal, academic, technical)
- Adjustable research depth (shallow, moderate, thorough, comprehensive)
- Research dossier synthesis with organized themes and conclusions
- Conversation threading for iterative research refinement

**Use Cases:**
- Technical investigation requiring evidence-based analysis
- Literature reviews with proper source attribution
- Technology evaluation with comparative analysis
- Policy and standards research with compliance focus
- Product research and competitive analysis
- Documentation research for decision-making

## When to Use

Use the RESEARCH workflow when you need to:

- **Technical investigation** - Compare technologies with evidence, investigate best practices, analyze architectural patterns
- **Literature review** - Academic research, market research, technology trends analysis
- **Technology evaluation** - Framework comparisons, tool evaluation, architecture decisions
- **Policy & standards** - Compliance research (GDPR, HIPAA), security standards (OWASP), legal requirements
- **Product research** - Competitor analysis, user needs research, feature comparisons
- **Evidence-based decisions** - Decisions requiring multiple sources and proper citations

## When NOT to Use

Avoid the RESEARCH workflow when:

| Situation | Use Instead |
|-----------|-------------|
| You need to generate new ideas or brainstorm | **IDEATE** - Structured idea generation workflow |
| You need to evaluate arguments or make dialectical analysis | **ARGUMENT** - Three-role creator/skeptic/moderator analysis |
| You need hypothesis-driven debugging | **THINKDEEP** - Extended reasoning with hypothesis tracking |
| Simple factual question without citation needs | **CHAT** - Single-model conversation for straightforward queries |
| You need multiple model perspectives or consensus | **CONSENSUS** - Multi-model consultation with synthesis strategies |

## Research Depth Levels

RESEARCH provides four depth levels that control investigation thoroughness:

### shallow
**Quick overview with basic facts**
- 2-3 focused findings
- Surface-level coverage
- Fast execution
- Use for: Quick scans, basic understanding, time-constrained research

### moderate (Default/Recommended)
**Standard research depth with balanced coverage**
- 4-6 focused findings
- Balanced thoroughness vs speed
- Comprehensive without being excessive
- Use for: Most research tasks, standard investigations, balanced analysis

### thorough
**Comprehensive, detailed investigation**
- 7-9 focused findings
- Deep dive into topic
- Extensive source coverage
- Use for: Critical decisions, detailed analysis, publication-grade research

### comprehensive
**Exhaustive research with maximum detail**
- 10+ focused findings
- Maximum depth and breadth
- Longest execution time
- Use for: Major decisions, academic research, complete topic coverage

**Depth Selection Guide:**
```
Quick overview needed? � shallow (2-3 findings, fast)
Standard research task? � moderate (4-6 findings, recommended)
Critical decision? � thorough (7-9 findings, deep)
Exhaustive analysis? � comprehensive (10+ findings, maximum detail)
```

## Citation Styles

RESEARCH supports three citation formatting styles. Choose based on your output requirements:

### informal (Default)
**Conversational citation style for general use**

Format: Natural language attribution integrated into text

Example:
```
According to the GraphQL documentation, the type system provides
compile-time query validation. The GraphQL Foundation notes that
this feature reduces runtime errors significantly.
```

**Use for:**
- General documentation
- Internal reports
- Conversational contexts
- Quick reference materials

---

### academic
**APA-style citations for formal research**

Format: Author-date citations with formal reference list

Example:
```
The type system provides compile-time query validation (GraphQL
Foundation, 2024). Studies show this reduces runtime errors by
40-60% (Lee & Chen, 2023).

References:
GraphQL Foundation. (2024). GraphQL Specification v16.0.
Lee, A., & Chen, B. (2023). Runtime error reduction in typed APIs.
```

**Use for:**
- Academic papers
- Formal reports
- Research publications
- Citation-critical contexts

---

### technical
**Technical documentation citation style**

Format: Numbered references with technical specification format

Example:
```
Type system features:
- Compile-time validation [1]
- Schema introspection [2]
- Runtime type checking [3]

References:
[1] GraphQL Specification v16.0 (graphql.org)
[2] GraphQL Best Practices (graphql.org/learn)
[3] Type Safety in GraphQL (graphql.org/guides)
```

**Use for:**
- Technical documentation
- API documentation
- Engineering specs
- Standards documents

---

## Citation Style Selection Guide

```
General documentation? � informal (natural language)
Academic/formal report? � academic (APA-style)
Technical specs/docs? � technical (numbered references)
```

## Basic Usage

### Simple Example

```bash
modelchorus research "What are the key benefits and challenges of GraphQL?"
```

**Expected Output:**
The command conducts systematic research using the default provider (Claude) with moderate depth and informal citations. Returns multiple findings with sources and a synthesized research dossier.

```
Research Question: What are the key benefits and challenges of GraphQL?

--- Finding 1: Precise Data Fetching ---
[Evidence and explanation with sources]

--- Finding 2: Strong Type System ---
[Evidence and explanation with sources]

--- Finding 3: Developer Experience ---
[Evidence and explanation with sources]

--- Finding 4: Caching Challenges ---
[Evidence and explanation with sources]

--- Research Dossier ---
[Synthesized analysis with all findings, organized by themes]

Sources:
[1] GraphQL Documentation (graphql.org)
[2] REST vs GraphQL Study (example.com/study)

Session ID: research-thread-abc123
```

### Common Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `claude` | AI provider to use (`claude`, `gemini`, `codex`, `cursor-agent`) |
| `--depth` | `-d` | `thorough` | Research depth (`shallow`, `moderate`, `thorough`, `comprehensive`) |
| `--citation-style` | | `informal` | Citation format (`informal`, `academic`, `technical`) |
| `--continue` | `-c` | None | Thread ID to continue research session |
| `--file` | `-f` | None | Source file paths for context (repeatable) |
| `--system` | | None | Additional system prompt |
| `--temperature` | `-t` | `0.5` | Creativity level (0.0-1.0, lower for factual accuracy) |
| `--max-tokens` | | None | Maximum response length |
| `--output` | `-o` | None | Save result to JSON file |
| `--verbose` | `-v` | False | Show detailed execution info |

## Advanced Usage

### With Research Depth Control

```bash
# Quick overview (2-3 findings)
modelchorus research "What is GraphQL?" --depth shallow

# Standard research (4-6 findings, recommended)
modelchorus research "GraphQL benefits and challenges" --depth moderate

# Comprehensive research (7-9 findings)
modelchorus research "Complete analysis of GraphQL architecture" --depth thorough

# Exhaustive research (10+ findings)
modelchorus research "GraphQL ecosystem: complete technical analysis" --depth comprehensive
```

**Depth Selection Tips:**
- `shallow`: Quick scans, basic understanding, time-constrained
- `moderate`: Most research tasks (recommended default)
- `thorough`: Critical decisions, detailed analysis
- `comprehensive`: Academic research, complete topic coverage

### With Citation Style Selection

```bash
# Informal citations (default, natural language)
modelchorus research "API design patterns" --citation-style informal

# Academic citations (APA-style for formal reports)
modelchorus research "Microservices architecture patterns" --citation-style academic

# Technical citations (numbered references for specs)
modelchorus research "Kubernetes best practices" --citation-style technical
```

**Citation Style Tips:**
- `informal`: General documentation, internal reports
- `academic`: Formal papers, research publications
- `technical`: Technical docs, API documentation, engineering specs

### With Provider Selection

```bash
# Use Claude (excellent research synthesis)
modelchorus research "Container orchestration comparison" --provider claude

# Use Gemini (strong analytical research)
modelchorus research "Database scaling strategies" --provider gemini

# Use Codex (best for technical/code research)
modelchorus research "Python async patterns and best practices" --provider codex
```

**Provider Selection Tips:**
- `claude`: Best for general research, synthesis, nuanced analysis
- `gemini`: Strong for factual research, analytical comparisons
- `codex`: Optimized for technical and code-focused research
- `cursor-agent`: Ideal for development-focused research

### With Source Files

```bash
# Single source file
modelchorus research "Analyze our API design" --file docs/api_spec.yaml

# Multiple source files for comprehensive context
modelchorus research "Compare GraphQL and REST for our use case" --file docs/graphql_evaluation.pdf --file docs/rest_api_analysis.md --file docs/performance_requirements.md
```

**File Handling:**
- All file paths must exist before execution
- Files are read and included as sources in the research
- Multiple `-f` flags can be used to include multiple files
- Files are treated as authoritative sources in the research dossier
- Large files may be truncated based on provider token limits

### With Conversation Threading

```bash
# Initial research
modelchorus research "Microservices architecture patterns"
# Output includes: Session ID: research-thread-xyz789

# Continue with follow-up research
modelchorus research "Expand on service mesh patterns" --continue research-thread-xyz789

# Explore related topics in same session
modelchorus research "How do these patterns apply to our scale?" -c research-thread-xyz789
```

**Threading Notes:**
- Thread IDs persist across sessions
- Research history and sources are maintained
- Each follow-up builds on previous findings
- Thread IDs follow format: `research-thread-{uuid}`
- Use short flag `-c` or long flag `--continue`

### Adjusting Research Creativity

```bash
# Lower temperature for factual, precise research (recommended)
modelchorus research "GDPR compliance requirements" --temperature 0.3

# Default balanced setting for most research
modelchorus research "API design best practices" --temperature 0.5

# Higher temperature for exploratory research
modelchorus research "Future trends in cloud architecture" --temperature 0.7
```

**Temperature Guide:**
- `0.0-0.3`: Maximum factual accuracy (compliance, standards, specs)
- `0.4-0.6`: Balanced accuracy and insight (most research, recommended)
- `0.7-1.0`: Exploratory, trend analysis (future predictions, innovation research)

### Combining Options

```bash
# Comprehensive formal research with academic citations
modelchorus research "State of quantum computing 2024" --depth thorough --citation-style academic --provider claude --temperature 0.4 --output quantum_research.json

# Quick technical research with source files
modelchorus research "Evaluate GraphQL for our API" --depth shallow --citation-style technical --file current_api_spec.yaml --file requirements.md

# Iterative research session with threading
modelchorus research "Kubernetes deployment strategies" --depth moderate
# Returns: Session ID: research-thread-abc123

modelchorus research "Focus on blue-green deployment details" --continue research-thread-abc123 --citation-style technical
```

### Saving Results

```bash
# Save comprehensive research to JSON file
modelchorus research "GraphQL vs REST API comparison" --depth thorough --citation-style academic --output api_comparison_research.json
```

**Output file contains:**
- Original research question
- All findings with evidence
- Research dossier synthesis
- Complete source list
- Metadata (model name, token usage, timestamp, depth, citation style)
- Thread ID for continuation
- Provider information

## Best Practices

1. **Choose appropriate depth for the task** - Use `moderate` for most research, `shallow` for quick scans, `thorough` for critical decisions, and `comprehensive` for exhaustive analysis. Don't over-engineer with unnecessary depth.

2. **Match citation style to output format** - Use `informal` for general docs, `academic` for formal reports, and `technical` for engineering specs. Consistent citation style improves readability and credibility.

3. **Use threading for multi-stage research** - Save and reuse thread IDs when building on previous research. This maintains source tracking and provides coherent research progression.

4. **Include relevant source files** - When researching specific systems or documents, use `-f` flags to include local files as authoritative sources. This grounds research in actual context.

5. **Lower temperature for factual research** - Use temperatures 0.3-0.5 for fact-based research. Higher temperatures (0.7+) are only appropriate for trend analysis or exploratory research.

6. **Select provider based on research type** - Claude for synthesis and nuanced analysis, Gemini for factual comparison, Codex for technical/code research.

7. **Save important research to JSON** - Use `--output` to preserve research results, especially for thorough or comprehensive research that took significant time.

## Examples

### Example 1: Technology Comparison Research

**Scenario:** You need to decide between GraphQL and REST APIs for a new project.

**Command:**
```bash
modelchorus research "Compare GraphQL and REST APIs: benefits, challenges, and use cases" --depth thorough --citation-style technical --provider claude --output api_comparison.json
```

**Expected Outcome:** Comprehensive research with 7-9 findings covering benefits of each approach, implementation challenges, performance considerations, use case recommendations, and ecosystem maturity. Technical citations provide specific references for claims. Results saved to JSON for team review.

---

### Example 2: Compliance Research

**Scenario:** You need to understand GDPR requirements for user data handling.

**Command:**
```bash
modelchorus research "GDPR requirements for user data handling, storage, and deletion" --depth thorough --citation-style academic --temperature 0.3 --output gdpr_compliance.json
```

**Expected Outcome:** Detailed research with 7-9 findings covering legal requirements, technical implementation requirements, user rights (access, deletion, portability), consent management, data breach protocols, and penalties. Academic citations provide authoritative legal references. Low temperature ensures factual accuracy critical for compliance.

---

### Example 3: Iterative Research Session

**Scenario:** You're exploring microservices architecture and want to drill deeper progressively.

**Command:**
```bash
# Initial broad research
modelchorus research "Microservices architecture patterns and best practices" --depth moderate
# Returns: Session ID: research-thread-xyz789

# Drill into specific pattern
modelchorus research "Expand on service mesh patterns: implementation and tradeoffs" --continue research-thread-xyz789

# Explore application to specific context
modelchorus research "How do these patterns apply to our 10-person team with moderate traffic?" --continue research-thread-xyz789 --file team_context.md
```

**Expected Outcome:** Multi-turn research session where each iteration builds on previous findings. First call provides broad overview (4-6 findings). Second call drills into service mesh with more detail. Third call applies findings to specific team context using provided file. Thread maintains source tracking across all turns.

---

### Example 4: Source-Grounded Research

**Scenario:** You need to analyze existing documentation and provide research-backed recommendations.

**Command:**
```bash
modelchorus research "Evaluate our current API design against REST best practices" --depth moderate --citation-style technical --file current_api_spec.yaml --file api_design_docs.md --file performance_benchmarks.json --output api_evaluation.json
```

**Expected Outcome:** Research that treats provided files as authoritative sources, cross-references them with industry best practices, identifies gaps or improvements, and provides evidence-backed recommendations. Technical citations reference both provided files and external sources.

## Technical Contract

### Parameters

**Required:**
- `prompt` (string): The research topic or question to investigate with comprehensive evidence gathering

**Optional:**
- `--provider, -p` (string): AI provider to use for research investigation - Valid values: `claude`, `gemini`, `codex`, `cursor-agent` - Default: `claude`
- `--depth, -d` (string): Research thoroughness level - Valid values: `shallow`, `moderate`, `thorough`, `comprehensive` - Default: `moderate` - Controls number of focused findings (2-3 for shallow, 4-6 for moderate, 7-9 for thorough, 10+ for comprehensive)
- `--citation-style` (string): Citation formatting style - Valid values: `informal`, `academic`, `technical` - Default: `informal` - `academic` for formal papers, `technical` for technical docs
- `--continue, -c` (string): Session ID to continue previous research - Format: `research-thread-{uuid}` - Maintains full research context including sources and findings across iterations
- `--file, -f` (string, repeatable): File paths to include as authoritative sources - Can be specified multiple times - Files must exist before execution - Treated as primary sources and cross-referenced with external research
- `--system` (string): Additional system prompt to customize research focus - Useful for specifying constraints, priorities, or specific research angles
- `--temperature, -t` (float): Response creativity level - Range: 0.0-1.0 - Default: 0.7 - Recommended: 0.3-0.5 for factual research, 0.6-0.8 for trend analysis or exploratory research
- `--max-tokens` (integer): Maximum response length in tokens - Provider-specific limits apply - Affects detail level of research dossier
- `--output, -o` (string): Path to save JSON output file - Creates or overwrites file at specified path - Recommended for thorough/comprehensive research to preserve findings
- `--verbose, -v` (boolean): Enable detailed execution information - Default: false - Shows provider details and timing

### Return Format

The RESEARCH workflow returns a JSON object with the following structure:

```json
{
  "result": "=== Finding 1: [Title] ===\n[Evidence and analysis]\nCitations: [sources]\n\n=== Finding 2: [Title] ===\n[Evidence and analysis]\nCitations: [sources]\n\n... [depth-based count] ...\n\n=== Research Dossier ===\n[Synthesis, themes, conclusions]",
  "session_id": "research-thread-abc-123-def-456",
  "metadata": {
    "provider": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "depth": "moderate",
    "citation_style": "informal",
    "num_findings": 5,
    "sources_referenced": ["source1", "source2", "source3"],
    "files_provided": ["api_spec.yaml", "context.md"],
    "temperature": 0.7,
    "prompt_tokens": 500,
    "completion_tokens": 1200,
    "total_tokens": 1700,
    "timestamp": "2025-11-07T10:30:00Z"
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | Complete research dossier with numbered findings (each with evidence and citations), followed by synthesis section with themes and conclusions |
| `session_id` | string | Session ID for continuing this research (format: `research-thread-{uuid}`) |
| `metadata.provider` | string | The AI provider used for this research |
| `metadata.model` | string | Specific model version used by the provider |
| `metadata.depth` | string | Research depth level used (`shallow`, `moderate`, `thorough`, `comprehensive`) |
| `metadata.citation_style` | string | Citation formatting style used (`informal`, `academic`, `technical`) |
| `metadata.num_findings` | integer | Number of focused findings generated (varies by depth level) |
| `metadata.sources_referenced` | array[string] | List of sources cited in the research findings |
| `metadata.files_provided` | array[string] | List of local files used as authoritative sources (if any) |
| `metadata.temperature` | float | Temperature setting used for this research (0.0-1.0) |
| `metadata.prompt_tokens` | integer | Number of tokens in the input (prompt + context + history + files) |
| `metadata.completion_tokens` | integer | Number of tokens in the research dossier output |
| `metadata.total_tokens` | integer | Total tokens consumed (prompt_tokens + completion_tokens) |
| `metadata.timestamp` | string | ISO 8601 timestamp of when the research was completed |

**Usage Notes:**
- Save the `session_id` to continue research with follow-up questions or deeper investigation
- The `result` includes both individual findings (with citations) and a synthesized research dossier
- Token costs scale with `depth` - comprehensive research uses significantly more tokens than shallow
- Provided files via `--file` are treated as authoritative sources and appear in citations
- Citation format affects how sources are referenced throughout the findings
- Lower temperature (0.3-0.5) recommended for factual accuracy in research
- Use `--output` to save important research results, especially for thorough/comprehensive investigations

## Troubleshooting

### Issue: Research depth insufficient

**Symptoms:** Research feels shallow or missing key aspects of the topic

**Cause:** Using `shallow` depth for complex topics, or `moderate` depth for critical decisions

**Solution:**
```bash
# Increase depth for more comprehensive coverage
modelchorus research "Your topic" --depth thorough

# For exhaustive research
modelchorus research "Your topic" --depth comprehensive
```

---

### Issue: Citations not in expected format

**Symptoms:** Citations are informal when formal format needed, or vice versa

**Cause:** Using default `informal` citation style when `academic` or `technical` needed

**Solution:**
```bash
# For formal reports, use academic citations
modelchorus research "Your topic" --citation-style academic

# For technical documentation, use technical citations
modelchorus research "Your topic" --citation-style technical
```

---

### Issue: Research not building on previous findings

**Symptoms:** Follow-up research doesn't reference earlier findings, context lost

**Cause:** Not using `--continue` flag to link research sessions

**Solution:**
```bash
# Always save thread ID from initial research
modelchorus research "Initial topic"
# Note the thread ID in output

# Use --continue for follow-up research
modelchorus research "Related topic" --continue research-thread-abc123
```

---

### Issue: Research too generic or not grounded in context

**Symptoms:** Research doesn't reference specific systems, lacks local context

**Cause:** Not providing relevant source files with `--file` flags

**Solution:**
```bash
# Include relevant files as authoritative sources
modelchorus research "Your topic" --file relevant_doc1.md --file relevant_spec.yaml --file context_file.txt
```

---

### Issue: Provider initialization failed

**Symptoms:** Error about provider not being available or initialization failure

**Cause:** Selected provider is not configured or invalid provider name

**Solution:**
```bash
# Check available providers
modelchorus list-providers

# Use a valid, configured provider name
modelchorus research "Your topic" --provider claude
```

## Related Workflows

- **CHAT** - When you need simple conversational queries without structured findings or citation management
- **IDEATE** - When you need to generate new ideas rather than gather existing information
- **ARGUMENT** - When you need dialectical analysis with pros/cons rather than comprehensive information gathering
- **THINKDEEP** - When you need hypothesis-driven investigation rather than broad information collection
- **CONSENSUS** - When you need multiple model perspectives on research rather than single-model systematic gathering

---

**See Also:**
- ModelChorus Documentation: `/docs/WORKFLOWS.md`
- Provider Information: `modelchorus list-providers`
- General CLI Help: `modelchorus --help`
