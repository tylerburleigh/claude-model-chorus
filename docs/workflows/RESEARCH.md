# RESEARCH Workflow

Systematic research with evidence extraction, source management, and citation formatting for comprehensive information gathering.

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Type** | Single-model systematic research |
| **Models** | 1 provider (with source analysis) |
| **Best For** | Research questions, evidence gathering, technical investigation, literature review |
| **Conversation** | ✅ Yes (supports continuation) |
| **State** | ✅ Stateful (conversation threading + source tracking) |
| **Output** | Multiple findings + synthesized dossier with citations |

---

## What It Does

The RESEARCH workflow conducts systematic research and information gathering through structured investigation. Unlike simple Q&A, it provides comprehensive research with:

**Structured Investigation:**
- Breaks research into multiple focused findings
- Each finding addresses specific aspects of the question
- Systematic coverage of the topic

**Source Management:**
- Tracks sources and references
- Supports file ingestion for local documents
- Maintains source credibility ratings
- Organizes findings by source

**Citation Formatting:**
- Multiple citation styles (informal, academic, technical)
- Proper attribution and references
- Source-backed claims
- Verifiable information

**Research Depth Control:**
- Shallow: Quick overview, basic facts
- Moderate: Standard research depth (recommended)
- Thorough: Comprehensive, detailed investigation

**Research Dossier:**
- Synthesizes all findings into coherent report
- Organized by themes and categories
- Includes recommendations and conclusions
- Full citation list

**Result:** Comprehensive, well-researched analysis with proper citations, ready for use in documentation, reports, or decision-making.

---

## When to Use

Use RESEARCH when you need:

### Technical Investigation
```bash
# Compare technologies with evidence
modelchorus research "Benefits and challenges of GraphQL vs REST APIs" \
  --depth moderate --citations academic

# Investigate best practices
modelchorus research "API versioning best practices and industry standards"
```

### Literature Review
```bash
# Academic research
modelchorus research "Current state of quantum computing research" \
  --depth thorough --citations academic

# Market research
modelchorus research "Trends in SaaS pricing models 2024"
```

### Technology Evaluation
```bash
# Framework comparison
modelchorus research "Next.js vs Remix for React applications" \
  --depth moderate

# Tool evaluation
modelchorus research "CI/CD platforms comparison: GitHub Actions, GitLab CI, Jenkins"
```

### Policy & Standards Research
```bash
# Compliance research
modelchorus research "GDPR requirements for user data handling" \
  --citations technical

# Security standards
modelchorus research "OWASP Top 10 security vulnerabilities and mitigations"
```

### Product Research
```bash
# Competitor analysis
modelchorus research "Key features of top project management tools" \
  --depth thorough

# User needs research
modelchorus research "Developer productivity pain points and solutions"
```

---

## When NOT to Use

**Don't use RESEARCH when:**

| Situation | Use Instead | Reason |
|-----------|-------------|--------|
| Need to generate ideas | **IDEATE** | RESEARCH gathers info, doesn't create new ideas |
| Need to evaluate an argument | **ARGUMENT** | RESEARCH is descriptive, not dialectical |
| Need hypothesis-driven debugging | **THINKDEEP** | THINKDEEP is for investigation, RESEARCH for gathering |
| Simple factual question | **CHAT** | RESEARCH adds overhead for structured findings |
| Need multiple model perspectives | **CONSENSUS** | RESEARCH uses single model |

---

## CLI Usage

### Basic Invocation

**Simple research query:**
```bash
modelchorus research "What are the key benefits and challenges of GraphQL?"
```

**Expected output:**
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

Thread ID: research-thread-abc123
```

### Research Depth Levels

**Control investigation thoroughness:**
```bash
# Shallow research (quick overview)
modelchorus research "What is GraphQL?" --depth shallow

# Moderate research (standard, recommended)
modelchorus research "GraphQL benefits and challenges" --depth moderate

# Thorough research (comprehensive, detailed)
modelchorus research "Complete analysis of GraphQL architecture" --depth thorough
```

**Depth characteristics:**
- **Shallow:** 2-3 findings, basic coverage, fast
- **Moderate:** 4-6 findings, balanced depth, recommended
- **Thorough:** 7+ findings, comprehensive, slower

### Citation Styles

**Choose citation format:**
```bash
# Informal citations (default)
modelchorus research "API design patterns" --citations informal

# Academic citations (APA-style)
modelchorus research "Microservices architecture patterns" --citations academic

# Technical citations (technical docs style)
modelchorus research "Kubernetes best practices" --citations technical
```

**Citation style examples:**

**Informal:**
```
According to the GraphQL documentation, the type system provides...
```

**Academic:**
```
The type system provides compile-time query validation (GraphQL Foundation, 2024).
```

**Technical:**
```
Type system features:
- Compile-time validation [1]
- Schema introspection [2]

[1] GraphQL Specification v16.0 (graphql.org)
[2] GraphQL Best Practices (graphql.org/learn)
```

### With Provider Selection

**Specify AI provider:**
```bash
# Use Claude (excellent research synthesis)
modelchorus research "Container orchestration comparison" --provider claude

# Use Gemini (strong analytical research)
modelchorus research "Database scaling strategies" --provider gemini

# Use Codex (best for technical research)
modelchorus research "Python async patterns" --provider codex
```

### With Source Files

**Provide local documents for research:**
```bash
# Single source file
modelchorus research "Analyze our API design" \
  --file docs/api_spec.yaml

# Multiple sources
modelchorus research "Compare GraphQL and REST for our use case" \
  --file docs/graphql_evaluation.pdf \
  --file docs/rest_api_analysis.md \
  --file docs/performance_requirements.md
```

### With Continuation (Threading)

**Continue research session:**
```bash
# Initial research
modelchorus research "Microservices architecture patterns"
# Returns: Thread ID: research-thread-xyz789

# Drill deeper into specific finding
modelchorus research "Expand on service mesh patterns" \
  --continue research-thread-xyz789

# Explore related topic
modelchorus research "How do these patterns apply to our scale?" \
  --continue research-thread-xyz789
```

### With Custom Configuration

**Temperature and tokens:**
```bash
# Lower temperature for factual research
modelchorus research "GDPR compliance requirements" --temperature 0.3

# Limit response length
modelchorus research "Quick overview of GraphQL" \
  --depth shallow --max-tokens 1500

# Comprehensive research
modelchorus research "Complete guide to microservices" \
  --depth thorough --max-tokens 5000
```

---

## Python API Usage

### Basic Usage

```python
import asyncio
from modelchorus.workflows import ResearchWorkflow
from modelchorus.providers import ClaudeProvider
from modelchorus.core.conversation import ConversationMemory

async def conduct_research():
    # Initialize provider and memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create workflow
    workflow = ResearchWorkflow(
        provider=provider,
        conversation_memory=memory
    )

    # Conduct research
    result = await workflow.run(
        prompt="What are the key benefits and challenges of GraphQL compared to REST?",
        research_depth="moderate",
        citation_style="informal",
        temperature=0.5
    )

    if result.success:
        # Access individual findings
        for step in result.steps:
            finding_name = step.metadata.get('name', 'Finding')
            print(f"\n--- {finding_name} ---")
            print(step.content)

        # Access research dossier
        print("\n--- Research Dossier ---")
        print(result.synthesis)

        # Access metadata
        thread_id = result.metadata.get('thread_id')
        sources_count = result.metadata.get('sources_analyzed', 0)
        print(f"\nThread ID: {thread_id}")
        print(f"Sources analyzed: {sources_count}")
    else:
        print(f"Research failed: {result.error}")

# Run
asyncio.run(conduct_research())
```

### With Source Files

```python
async def research_with_sources():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ResearchWorkflow(provider=provider, conversation_memory=memory)

    # Ingest source documents
    workflow.ingest_source(
        title="GraphQL Specification v16.0",
        url="https://spec.graphql.org/",
        source_type="specification",
        credibility="high"
    )

    workflow.ingest_source(
        title="REST vs GraphQL Performance Study",
        url="docs/performance_study.pdf",
        source_type="research_paper",
        credibility="high"
    )

    # Conduct research with sources
    result = await workflow.run(
        prompt="Compare GraphQL and REST for data-heavy mobile applications",
        files=["docs/graphql_spec.md", "docs/rest_study.pdf"],
        research_depth="thorough",
        citation_style="academic",
        temperature=0.4
    )

    if result.success:
        print(result.synthesis)
        print(f"\nSources analyzed: {result.metadata.get('sources_analyzed', 0)}")
```

### With Continuation

```python
async def iterative_research():
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ResearchWorkflow(provider=provider, conversation_memory=memory)

    # Initial research
    result1 = await workflow.run(
        prompt="What is the current state of quantum computing?",
        research_depth="moderate"
    )

    thread_id = result1.metadata['thread_id']

    # Drill deeper into specific area
    result2 = await workflow.run(
        prompt="Focus specifically on quantum computing applications in cryptography",
        continuation_id=thread_id,
        research_depth="thorough"
    )

    print(result2.synthesis)
```

### Depth Comparison

```python
async def compare_depths(question: str):
    """Research same question at different depths."""
    provider = ClaudeProvider()
    memory = ConversationMemory()

    depths = ["shallow", "moderate", "thorough"]
    results = {}

    for depth in depths:
        workflow = ResearchWorkflow(provider=provider, conversation_memory=memory)

        result = await workflow.run(
            prompt=question,
            research_depth=depth,
            temperature=0.5
        )

        if result.success:
            results[depth] = {
                'findings': len(result.steps),
                'dossier_length': len(result.synthesis),
                'content': result.synthesis
            }

    return results
```

### Accessing Result Components

```python
result = await workflow.run(
    prompt="Research question",
    research_depth="moderate",
    citation_style="academic"
)

if result.success:
    # Individual findings
    findings = [
        {
            'title': step.metadata.get('name', f'Finding {i}'),
            'content': step.content,
            'metadata': step.metadata
        }
        for i, step in enumerate(result.steps, 1)
    ]

    # Research dossier (synthesis)
    dossier = result.synthesis

    # Metadata
    thread_id = result.metadata['thread_id']
    model_used = result.metadata.get('model', 'unknown')
    sources_analyzed = result.metadata.get('sources_analyzed', 0)
    findings_count = len(result.steps)

    print(f"Research completed:")
    print(f"  Model: {model_used}")
    print(f"  Findings: {findings_count}")
    print(f"  Sources: {sources_analyzed}")
    print(f"  Thread: {thread_id}")
```

### Error Handling

```python
async def robust_research():
    try:
        provider = ClaudeProvider()
        memory = ConversationMemory()
        workflow = ResearchWorkflow(provider=provider, conversation_memory=memory)

        result = await workflow.run(
            prompt="Research topic",
            research_depth="moderate",
            citation_style="academic"
        )

        if result.success:
            return result.steps, result.synthesis
        else:
            # Handle workflow failure
            print(f"Research failed: {result.error}")
            # Retry with different parameters, log error, etc.
            return None, None

    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        # Check API keys, provider availability, etc.
        return None, None
```

---

## Advanced Features

### Source Ingestion and Management

Track and reference sources systematically:

```python
workflow = ResearchWorkflow(provider, memory)

# Add various source types
workflow.ingest_source(
    title="GraphQL Specification",
    url="https://spec.graphql.org/",
    source_type="specification",
    credibility="high"
)

workflow.ingest_source(
    title="Performance Analysis Study",
    url="docs/study.pdf",
    source_type="research_paper",
    credibility="medium"  # Self-published, not peer-reviewed
)

workflow.ingest_source(
    title="Community Blog Post",
    url="https://blog.example.com/graphql",
    source_type="blog",
    credibility="low"  # Opinion piece
)

# Sources are tracked and referenced in findings
result = await workflow.run(
    prompt="GraphQL performance characteristics",
    files=["docs/study.pdf"],
    citation_style="academic"
)
```

**Source credibility levels:**
- **high:** Official docs, peer-reviewed research, specifications
- **medium:** Technical blogs, case studies, self-published research
- **low:** Opinion pieces, unverified claims, promotional content

### Research Depth Strategies

Optimize depth for your use case:

```python
# Quick overview: shallow depth
result = await workflow.run(
    prompt="What is Kubernetes?",
    research_depth="shallow",
    temperature=0.5
)
# Returns: 2-3 findings, basic explanation

# Standard research: moderate depth (recommended)
result = await workflow.run(
    prompt="Kubernetes deployment strategies",
    research_depth="moderate",
    temperature=0.5
)
# Returns: 4-6 findings, balanced coverage

# Comprehensive analysis: thorough depth
result = await workflow.run(
    prompt="Complete guide to Kubernetes architecture and operations",
    research_depth="thorough",
    temperature=0.4
)
# Returns: 7+ findings, comprehensive coverage
```

### Citation Style Selection

Match citation style to your audience:

```python
# Informal citations - blogs, documentation, general use
result = await workflow.run(
    prompt="GraphQL benefits",
    citation_style="informal"
)
# Output: "According to the GraphQL documentation..."

# Academic citations - papers, formal reports
result = await workflow.run(
    prompt="GraphQL performance analysis",
    citation_style="academic"
)
# Output: "Performance improvements were observed (Author, 2024)."

# Technical citations - technical docs, API references
result = await workflow.run(
    prompt="GraphQL API design patterns",
    citation_style="technical"
)
# Output: "[1] GraphQL Best Practices (graphql.org/learn)"
```

### Iterative Deep Dives

Build on previous research systematically:

```python
async def deep_research(topic: str):
    """Multi-turn research session."""
    workflow = ResearchWorkflow(provider, memory)

    # Step 1: Overview
    result1 = await workflow.run(
        prompt=f"Overview of {topic}",
        research_depth="shallow"
    )
    thread_id = result1.metadata['thread_id']

    # Step 2: Drill into key finding
    result2 = await workflow.run(
        prompt="Expand on the most important aspect from the overview",
        continuation_id=thread_id,
        research_depth="moderate"
    )

    # Step 3: Specific application
    result3 = await workflow.run(
        prompt="How does this apply to enterprise-scale deployments?",
        continuation_id=thread_id,
        research_depth="thorough"
    )

    return [result1, result2, result3]
```

### Temperature Tuning for Research

Balance factual accuracy with synthesis quality:

```python
# Very factual, conservative (0.2-0.4)
# - Stick closely to sources
# - Minimal inference
# - Best for compliance, specifications
result = await workflow.run(
    prompt="GDPR requirements",
    temperature=0.3,
    research_depth="thorough"
)

# Balanced research (0.4-0.6) [RECOMMENDED]
# - Good synthesis
# - Reasonable inference
# - Best for general research
result = await workflow.run(
    prompt="GraphQL vs REST comparison",
    temperature=0.5,
    research_depth="moderate"
)

# More interpretive (0.6-0.8)
# - Creative synthesis
# - More inference
# - Best for exploratory research
result = await workflow.run(
    prompt="Future trends in web development",
    temperature=0.7,
    research_depth="moderate"
)
```

---

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Research question or topic |
| `research_depth` | `str` | `"moderate"` | Investigation depth: shallow, moderate, thorough |
| `citation_style` | `str` | `"informal"` | Citation format: informal, academic, technical |
| `temperature` | `float` | `0.5` | Creativity level (0.0-1.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `system_prompt` | `str` | `None` | Additional context or constraints |
| `continuation_id` | `str` | `None` | Thread ID to continue session |
| `files` | `List[str]` | `[]` | File paths for source documents |

### Research Depth Guidelines

| Depth | Findings | Detail | Speed | Use Case |
|-------|----------|--------|-------|----------|
| **shallow** | 2-3 | Basic | Fast | Quick overview, definitions |
| **moderate** | 4-6 | Balanced | Medium | Standard research (recommended) |
| **thorough** | 7+ | Comprehensive | Slower | Deep analysis, complete understanding |

### Citation Style Comparison

| Style | Format | Best For | Example |
|-------|--------|----------|---------|
| **informal** | Natural language | Blogs, docs, general use | "According to X..." |
| **academic** | APA-style | Papers, formal reports | "(Author, 2024)" |
| **technical** | Numbered references | Technical docs, specs | "[1] Source Name" |

### Temperature Guidelines

| Range | Research Style | Best For |
|-------|---------------|----------|
| **0.2-0.4** | Strictly factual | Compliance, legal, specifications |
| **0.4-0.6** | Balanced synthesis | General research (recommended) |
| **0.6-0.8** | Interpretive analysis | Exploratory, trend analysis |

---

## Best Practices

### 1. Match Depth to Need

```python
# Don't over-research simple questions
result = await workflow.run(
    prompt="What is GraphQL?",
    research_depth="shallow"  # Not "thorough"
)

# Use thorough for complex topics
result = await workflow.run(
    prompt="Complete analysis of distributed systems patterns",
    research_depth="thorough"
)
```

### 2. Provide Sources When Available

```python
# Better with sources
workflow.ingest_source("Official Spec", "spec.graphql.org", "spec", "high")

result = await workflow.run(
    prompt="GraphQL type system",
    files=["docs/graphql_spec.md"],
    citation_style="academic"
)
```

### 3. Use Appropriate Citation Style

```python
# Blog post or documentation
result = await workflow.run(prompt="...", citation_style="informal")

# Academic paper or formal report
result = await workflow.run(prompt="...", citation_style="academic")

# Technical documentation
result = await workflow.run(prompt="...", citation_style="technical")
```

### 4. Iterate for Comprehensive Research

```python
# Start broad
result1 = await workflow.run("Overview of topic", research_depth="shallow")

# Drill down
result2 = await workflow.run(
    "Expand on aspect X",
    continuation_id=thread_id,
    research_depth="moderate"
)

# Go deep on specific point
result3 = await workflow.run(
    "Technical details of Y",
    continuation_id=thread_id,
    research_depth="thorough"
)
```

### 5. Lower Temperature for Factual Research

```python
# Factual, compliance-focused
result = await workflow.run(
    prompt="GDPR data retention requirements",
    temperature=0.3,  # Very factual
    research_depth="thorough",
    citation_style="technical"
)
```

### 6. Save Research Results

```python
import json
from pathlib import Path

result = await workflow.run(prompt="Research topic", research_depth="moderate")

if result.success:
    research_data = {
        'question': "Research topic",
        'depth': "moderate",
        'findings': [
            {
                'title': s.metadata.get('name'),
                'content': s.content
            }
            for s in result.steps
        ],
        'dossier': result.synthesis,
        'sources': result.metadata.get('sources_analyzed', 0)
    }

    Path("research_output.json").write_text(json.dumps(research_data, indent=2))
```

---

## Common Use Patterns

### Pattern 1: Technology Comparison

```python
async def compare_technologies(tech_a: str, tech_b: str, context: dict):
    """Research-based technology comparison."""
    workflow = ResearchWorkflow(provider, memory)

    prompt = f"Compare {tech_a} and {tech_b} for {context['use_case']}"

    system_prompt = f"""
    Context:
    - Use case: {context['use_case']}
    - Scale: {context['scale']}
    - Team size: {context['team_size']}
    - Constraints: {context['constraints']}

    Focus on: performance, scalability, developer experience, ecosystem, cost
    """

    result = await workflow.run(
        prompt=prompt,
        system_prompt=system_prompt,
        research_depth="thorough",
        citation_style="technical",
        temperature=0.5
    )

    return result.synthesis
```

### Pattern 2: Best Practices Research

```python
async def research_best_practices(domain: str, sources: List[str]):
    """Gather industry best practices with citations."""
    workflow = ResearchWorkflow(provider, memory)

    # Ingest industry sources
    for source in sources:
        workflow.ingest_source(
            title=source['title'],
            url=source['url'],
            source_type=source['type'],
            credibility=source['credibility']
        )

    result = await workflow.run(
        prompt=f"Best practices for {domain}",
        files=[s['path'] for s in sources if 'path' in s],
        research_depth="thorough",
        citation_style="technical",
        temperature=0.4
    )

    return {
        'findings': result.steps,
        'best_practices': result.synthesis,
        'sources': result.metadata.get('sources_analyzed', 0)
    }
```

### Pattern 3: Compliance Research

```python
async def compliance_research(regulation: str, requirements: List[str]):
    """Research compliance requirements with authoritative sources."""
    workflow = ResearchWorkflow(provider, memory)

    requirements_text = "\n".join(f"- {r}" for r in requirements)

    result = await workflow.run(
        prompt=f"{regulation} compliance requirements",
        system_prompt=f"Focus on these specific requirements:\n{requirements_text}",
        research_depth="thorough",
        citation_style="academic",
        temperature=0.3  # Very factual for compliance
    )

    return result
```

---

## Troubleshooting

### Issue: Findings Too Shallow

**Symptoms:** Brief findings without sufficient detail

**Solutions:**
```python
# Increase research depth
result = await workflow.run(prompt="...", research_depth="thorough")

# Increase max_tokens
result = await workflow.run(prompt="...", max_tokens=5000)

# Provide more specific prompt
result = await workflow.run(
    prompt="Comprehensive analysis of X including Y and Z"
)
```

### Issue: Missing Citations

**Symptoms:** Findings lack proper source references

**Solutions:**
```python
# Use academic or technical citation style
result = await workflow.run(prompt="...", citation_style="academic")

# Provide source files
workflow.ingest_source("Source Name", "url", "type", "high")
result = await workflow.run(
    prompt="...",
    files=["source1.pdf", "source2.md"],
    citation_style="technical"
)

# Request citations explicitly
result = await workflow.run(
    prompt="...",
    system_prompt="Include citations for all claims"
)
```

### Issue: Too Many Findings

**Symptoms:** Overwhelming number of findings, too verbose

**Solutions:**
```python
# Use shallow depth
result = await workflow.run(prompt="...", research_depth="shallow")

# Limit max_tokens
result = await workflow.run(prompt="...", max_tokens=2000)

# Make prompt more specific
result = await workflow.run(
    prompt="Focused question about specific aspect of X"
)
```

### Issue: Lost Thread Context

**Symptoms:** Continuation doesn't reference previous research

**Solutions:**
```python
# Verify thread ID
print(f"Thread ID: {result.metadata['thread_id']}")

# Check thread exists
threads = memory.list_threads()
print(threads)

# Check conversation history
history = memory.get_thread_history(thread_id)
print(f"Messages: {len(history)}")
```

### Issue: Outdated or Incorrect Information

**Symptoms:** Research contains outdated or incorrect facts

**Solutions:**
```python
# Provide current sources
workflow.ingest_source("Latest Spec 2024", "url", "spec", "high")

# Lower temperature for factual accuracy
result = await workflow.run(prompt="...", temperature=0.3)

# Use authoritative sources only
result = await workflow.run(
    prompt="...",
    system_prompt="Only cite authoritative, official sources"
)
```

### Issue: Provider Errors

**Symptoms:** "Provider failed" or API errors

**Solutions:**
```bash
# Check API keys
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY

# Test provider CLI
claude --version
gemini --version

# Try different provider
modelchorus research "test query" --provider gemini
```

---

## Comparison with Other Workflows

### RESEARCH vs CHAT

| Aspect | RESEARCH | CHAT |
|--------|----------|------|
| **Structure** | Systematic findings + dossier | Freeform conversation |
| **Citations** | Yes (multiple styles) | No formal citations |
| **Depth Control** | Shallow/Moderate/Thorough | No depth levels |
| **Best for** | Comprehensive research | Quick questions |
| **Use when** | Need evidence-backed analysis | Need simple answers |

**Example:**
```bash
# Use RESEARCH for comprehensive analysis
modelchorus research "GraphQL vs REST comparison" --depth thorough

# Use CHAT for quick question
modelchorus chat "What is GraphQL?" -p claude
```

### RESEARCH vs ARGUMENT

| Aspect | RESEARCH | ARGUMENT |
|--------|----------|----------|
| **Purpose** | Gather and synthesize information | Analyze arguments dialectically |
| **Output** | Findings + dossier | Pro/Con/Synthesis |
| **Process** | Investigation | Debate |
| **Best for** | Information gathering | Evaluating claims |
| **Use when** | Need evidence | Need balanced analysis of position |

**Example:**
```bash
# Use RESEARCH to gather facts
modelchorus research "GraphQL adoption rates and trends"

# Use ARGUMENT to evaluate claim
modelchorus argument "GraphQL should replace all REST APIs"
```

### RESEARCH vs THINKDEEP

| Aspect | RESEARCH | THINKDEEP |
|--------|----------|-----------|
| **Purpose** | Information gathering | Hypothesis-driven investigation |
| **Structure** | Findings + dossier | Hypothesis tracking + confidence |
| **Best for** | Literature review | Debugging, root cause analysis |
| **Focus** | Descriptive | Investigative |
| **Use when** | Need comprehensive info | Need to solve specific problem |

**Example:**
```bash
# Use RESEARCH for information
modelchorus research "Common causes of API latency"

# Use THINKDEEP for investigation
modelchorus thinkdeep "Why is our specific API slow?" -f src/api.py
```

---

## Real-World Examples

### Example 1: Technology Comparison

```bash
modelchorus research "Next.js vs Remix for React applications" \
  --depth thorough \
  --citations technical \
  --file docs/requirements.md
```

**Expected output:**
- **Finding 1:** Server-side rendering capabilities
- **Finding 2:** Developer experience and tooling
- **Finding 3:** Performance characteristics
- **Finding 4:** Ecosystem and community support
- **Finding 5:** Deployment and hosting options
- **Finding 6:** Learning curve and documentation
- **Dossier:** Comprehensive comparison with recommendations
- **Sources:** Official docs, benchmarks, case studies

### Example 2: Best Practices Research

```bash
modelchorus research "API versioning best practices and strategies" \
  --depth moderate \
  --citations academic \
  --temperature 0.5
```

**Expected output:**
- **Finding 1:** URL path versioning (pros/cons)
- **Finding 2:** Header-based versioning
- **Finding 3:** Content negotiation
- **Finding 4:** Semantic versioning for APIs
- **Finding 5:** Deprecation strategies
- **Dossier:** Industry standards and recommendations
- **Sources:** REST APIs books, industry blogs, specifications

### Example 3: Compliance Research

```bash
modelchorus research "GDPR requirements for user data retention and deletion" \
  --depth thorough \
  --citations technical \
  --temperature 0.3
```

**Expected output:**
- **Finding 1:** Data retention limits
- **Finding 2:** Right to erasure requirements
- **Finding 3:** Consent requirements
- **Finding 4:** Data portability obligations
- **Finding 5:** Documentation requirements
- **Finding 6:** Penalties for non-compliance
- **Finding 7:** Implementation best practices
- **Dossier:** Complete compliance guide
- **Sources:** GDPR official text, legal guidance, case studies

---

## See Also

**Related Documentation:**
- [WORKFLOWS.md](../WORKFLOWS.md) - Complete workflow comparison guide
- [ARGUMENT Workflow](ARGUMENT.md) - Dialectical reasoning for argument analysis
- [IDEATE Workflow](IDEATE.md) - Creative brainstorming and idea generation
- [CHAT Workflow](CHAT.md) - Simple conversation (coming soon)
- [THINKDEEP Workflow](THINKDEEP.md) - Hypothesis-driven investigation (coming soon)
- [CONVERSATION_INFRASTRUCTURE.md](../CONVERSATION_INFRASTRUCTURE.md) - Threading details

**Code Examples:**
- [examples/workflow_examples.py](../../examples/workflow_examples.py) - Complete Python examples
- [examples/research_basic.py](../../examples/) - Basic RESEARCH usage

**API Reference:**
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Complete API documentation
- [README.md](../../README.md) - Getting started guide

**Related Workflows:**
- Use **CHAT** for simple questions without structured research
- Use **ARGUMENT** to evaluate research findings dialectically
- Use **THINKDEEP** for hypothesis-driven debugging
- Combine **RESEARCH** + **ARGUMENT** to research then evaluate
- Combine **RESEARCH** + **IDEATE** to research then brainstorm solutions
