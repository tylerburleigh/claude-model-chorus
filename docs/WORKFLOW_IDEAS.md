# ModelChorus: New Workflow Ideas

**Document Version:** 1.0
**Date:** 2025-11-06
**Status:** Proposal

---

## Overview

This document details 6 new workflow ideas for ModelChorus, expanding the platform beyond its current capabilities (CONSENSUS, CHAT, THINKDEEP) into general-purpose knowledge work and document processing.

### Strategic Goals

1. **Broaden Audience** - Serve knowledge workers beyond developers (researchers, analysts, writers, strategists)
2. **Showcase Orchestration** - Demonstrate role-based multi-model coordination
3. **Build Reusable Infrastructure** - Create primitives (citation engine, document store) for future workflows
4. **Differentiate Platform** - Offer capabilities not easily replicated in single-model tools

### Workflow Categories

**General-Purpose Workflows (3)**
- ARGUMENT - Structured debate and argument analysis
- RESEARCH - Evidence-based research with validation
- IDEATE - Creative ideation with structured refinement

**Document Workflows (3)**
- DIGEST - Single-document summarization
- SYNTHESIZE - Multi-document synthesis and analysis
- REVIEW - Critical document analysis

---

## General-Purpose Workflows

---

### 1. ARGUMENT - Argument Cartography

**Tagline:** Deconstruct complex topics into structured argument maps with multi-perspective analysis

#### Overview

ARGUMENT transforms a topic or thesis into a comprehensive map of arguments, counter-arguments, and synthesis. It moves beyond single-model opinions by stress-testing ideas through structured debate between models playing distinct roles.

#### Orchestration Pattern

**Role-Based Orchestration:** Sequential debate with distinct model roles

```
User Topic → Thesis Generator → Skeptic/Critic → Moderator/Synthesizer → Argument Map
```

#### Step-by-Step Process

**Step 1: Thesis Generation**
- **Model Role:** Creator (e.g., Claude)
- **Input:** User's topic (e.g., "The future of remote work")
- **Task:** Generate core thesis with 3-5 primary supporting arguments
- **Output:** Structured thesis with supporting evidence

**Step 2: Critical Rebuttal**
- **Model Role:** Skeptic (e.g., Gemini)
- **Input:** Thesis from Step 1
- **Task:** Generate strong counter-arguments, identify weaknesses, challenge assumptions
- **Output:** Critical analysis with specific rebuttals

**Step 3: Synthesis & Adjudication**
- **Model Role:** Moderator (e.g., GPT-4)
- **Input:** Original thesis + Critical rebuttal
- **Task:** Synthesize debate, identify logical fallacies, highlight key contentions
- **Output:** Comprehensive argument map with:
  - Core thesis
  - Supporting arguments
  - Counter-arguments
  - Logical analysis
  - Areas for further research
  - Strength assessment of each position

#### Model Roles

| Role | Model Suggestion | Responsibility |
|------|------------------|----------------|
| **Creator** | Claude Opus | Generate well-reasoned thesis |
| **Skeptic** | Gemini 2.5 Pro | Challenge and find weaknesses |
| **Moderator** | GPT-4 | Synthesize and adjudicate |

#### Use Cases

- **Research Planning** - Explore a research question before diving into literature
- **Decision Analysis** - Evaluate strategic decisions from multiple angles
- **Debate Preparation** - Understand both sides of an argument
- **Critical Thinking** - Teach students to analyze complex topics
- **Policy Analysis** - Examine policy proposals comprehensively
- **Brainstorm Validation** - Stress-test ideas before investing resources

#### User Value

- **Bias Reduction** - Multiple perspectives prevent single-model bias
- **Comprehensive Analysis** - See all sides of an argument
- **Blind Spot Detection** - Skeptic reveals weaknesses you might miss
- **Structured Output** - Clear argument map vs. unstructured discussion
- **Time Savings** - Automated debate vs. manual research

#### Input/Output

**Input:**
```bash
modelchorus argument "Should companies mandate office return?" \
  --creator claude \
  --skeptic gemini \
  --moderator gpt-4
```

**Output:**
```markdown
# Argument Map: Should companies mandate office return?

## Core Thesis (by Creator)
[Structured thesis with supporting arguments...]

## Critical Rebuttal (by Skeptic)
[Counter-arguments and challenges...]

## Synthesis (by Moderator)
### Key Points of Contention
1. ...
2. ...

### Logical Fallacies Identified
- ...

### Strongest Arguments (Pro)
- ...

### Strongest Arguments (Con)
- ...

### Areas for Further Research
- ...
```

#### State Management

```python
class ArgumentState:
    topic: str
    thesis: ThesisDocument
    rebuttals: List[Rebuttal]
    synthesis: SynthesisDocument
    argument_map: ArgumentMap

class ArgumentMap:
    core_claims: List[Claim]
    supporting_evidence: List[Evidence]
    counter_arguments: List[CounterArgument]
    logical_analysis: LogicalAnalysis
    strength_scores: Dict[str, float]
```

#### Implementation Complexity

- **Effort:** 4-5 days
- **Complexity:** Medium
- **Dependencies:** Conversation infrastructure (existing)
- **New Components:** Argument mapping, logical analysis framework

#### Orchestration Value

**9/10** - Role-based orchestration at its finest. Cannot be replicated with single-model tools.

---

### 2. RESEARCH - Evidence Dossier Orchestrator

**Tagline:** Build validated research dossiers with cross-validated evidence and citation tracking

#### Overview

RESEARCH creates curated, validated research dossiers from user-provided sources. It goes beyond simple summaries by fact-checking claims, detecting contradictions, and maintaining rigorous citation tracking throughout the process.

#### Orchestration Pattern

**Parallel Validation → Sequential Synthesis**

```
Sources → [Parallel Extraction] → Fact-Checking → Contradiction Detection → Synthesis → Dossier
```

#### Step-by-Step Process

**Step 1: Source Ingestion**
- **Task:** Load and parse provided sources (documents, URLs, notes)
- **Output:** Structured source registry with metadata

**Step 2: Parallel Evidence Extraction**
- **Model Role:** Extractors (multiple models)
- **Input:** Individual sources
- **Task:** Extract claims, hypotheses, data points, methodologies
- **Output:** Structured evidence entries per source

**Step 3: Fact-Checking & Validation**
- **Model Role:** Validator
- **Input:** Extracted claims
- **Task:** Cross-reference claims, check for internal contradictions, assess confidence
- **Output:** Validated claims with confidence scores

**Step 4: Contradiction Detection**
- **Model Role:** Analyst
- **Input:** All validated claims
- **Task:** Identify contradictions, inconsistencies, and gaps
- **Output:** Contradiction map with severity levels

**Step 5: Hypothesis Logging**
- **Model Role:** Synthesizer
- **Input:** Validated evidence + contradictions
- **Task:** Generate research hypotheses, identify knowledge gaps
- **Output:** Hypothesis tree with supporting/contradicting evidence

**Step 6: Dossier Compilation**
- **Model Role:** Writer
- **Input:** All structured data
- **Task:** Compile coherent research dossier with citations
- **Output:** Final dossier with:
  - Executive summary
  - Key findings
  - Evidence tables
  - Contradiction analysis
  - Hypotheses and gaps
  - Full citations

#### Model Roles

| Role | Responsibility |
|------|----------------|
| **Extractors** | Pull structured data from sources |
| **Validator** | Cross-check claims for consistency |
| **Analyst** | Identify contradictions and patterns |
| **Synthesizer** | Generate hypotheses from evidence |
| **Writer** | Compile final dossier |

#### Use Cases

- **Academic Literature Review** - Systematically analyze research papers
- **Competitive Intelligence** - Analyze competitor information with validation
- **Market Research** - Synthesize market data with contradiction detection
- **Investigative Journalism** - Track evidence and contradictions
- **Grant Proposals** - Build evidence base for research proposals
- **Policy Research** - Analyze policy documents with fact-checking

#### User Value

- **Validated Evidence** - Claims are cross-checked, not blindly accepted
- **Contradiction Detection** - Automatically identify inconsistencies
- **Citation Integrity** - Every claim traced to source
- **Time Savings** - Automated evidence synthesis vs. manual compilation
- **Hypothesis Generation** - AI identifies research gaps

#### Input/Output

**Input:**
```bash
modelchorus research "Climate change mitigation strategies" \
  --sources paper1.pdf paper2.pdf article.txt \
  --validate true \
  --detect-contradictions true
```

**Output:**
```markdown
# Research Dossier: Climate change mitigation strategies

## Executive Summary
[Synthesized overview...]

## Key Findings
1. [Finding with citation]
2. [Finding with citation]

## Evidence Table
| Claim | Source | Confidence | Contradictions |
|-------|--------|------------|----------------|
| ... | Paper 1, p.5 | High | None |
| ... | Paper 2, p.12 | Medium | Contradicts Finding 3 |

## Contradictions Detected
### High Severity
- [Contradiction between Source A and Source B]

### Medium Severity
- [Minor inconsistency...]

## Research Hypotheses
1. [Hypothesis with supporting evidence]
2. [Hypothesis with knowledge gap]

## Citations
[Full citation list]
```

#### State Management

```python
class ResearchState:
    topic: str
    sources: List[Source]
    claims: List[Claim]
    contradictions: List[Contradiction]
    hypotheses: List[Hypothesis]
    dossier: ResearchDossier

class Claim:
    content: str
    source_id: str
    location: str  # page, paragraph
    confidence: float
    contradicts: List[str]  # other claim IDs
    supports: List[str]  # hypothesis IDs
```

#### Implementation Complexity

- **Effort:** 5-7 days
- **Complexity:** Medium-High
- **Dependencies:** THINKDEEP hypothesis tracking (existing)
- **New Components:** Fact-checking, contradiction detection, citation engine

#### Orchestration Value

**8/10** - Systematic validation and contradiction detection not possible with single-model approaches.

---

### 3. IDEATE - Creative Ideation Funnel

**Tagline:** Transform vague concepts into detailed outlines through structured creativity

#### Overview

IDEATE guides users from initial concepts to fleshed-out outlines by leveraging different models for divergent brainstorming and convergent analysis. It automates and enhances the creative process while maintaining structure.

#### Orchestration Pattern

**Divergent → Convergent → Elaboration**

```
Seed Idea → [Parallel Brainstorming] → Clustering & Scoring → User Selection → Detailed Outline
```

#### Step-by-Step Process

**Step 1: Divergent Brainstorming**
- **Model Role:** Ideators (multiple models in parallel)
- **Input:** Seed idea (e.g., "a productivity app for remote teams")
- **Task:** Generate diverse ideas for features, audiences, angles, approaches
- **Output:** Wide array of possibilities (30-50 ideas)

**Step 2: Convergent Analysis**
- **Model Role:** Analyst
- **Input:** All brainstormed ideas
- **Task:**
  - Cluster similar ideas into themes
  - Score against user criteria (novelty, feasibility, impact)
  - Identify recurring patterns
- **Output:** Organized idea clusters with scores

**Step 3: User Selection**
- **Interface:** Present clusters to user
- **Task:** User selects most promising cluster(s)
- **Output:** Selected direction

**Step 4: Elaboration**
- **Model Role:** Writer/Planner
- **Input:** Selected cluster + seed idea
- **Task:** Expand into detailed concept outline or business case
- **Output:** Comprehensive outline with:
  - Problem statement
  - Solution approach
  - Key features/components
  - Target audience
  - Success metrics
  - Implementation considerations
  - Next steps

#### Model Roles

| Role | Model | Responsibility |
|------|-------|----------------|
| **Ideator 1** | Claude | Creative feature brainstorming |
| **Ideator 2** | GPT-4 | Market angle brainstorming |
| **Ideator 3** | Gemini | Technical approach brainstorming |
| **Analyst** | Claude Opus | Clustering and scoring |
| **Writer** | GPT-4 | Detailed outline generation |

#### Use Cases

- **Product Development** - Brainstorm product features and positioning
- **Content Creation** - Generate content ideas and outlines
- **Business Strategy** - Explore strategic directions
- **Marketing Campaigns** - Develop campaign concepts
- **Research Projects** - Explore research approaches
- **Creative Writing** - Develop story concepts and outlines

#### User Value

- **Overcome Creative Blocks** - Automated diverse idea generation
- **Structured Creativity** - Organized vs. chaotic brainstorming
- **Multiple Perspectives** - Different models = different angles
- **Rapid Iteration** - Fast exploration of idea space
- **Quality Filtering** - Automatic scoring and clustering

#### Input/Output

**Input:**
```bash
modelchorus ideate "A tool to help developers learn new programming languages" \
  --criteria "novelty:high,feasibility:medium,impact:high" \
  --ideators claude,gpt-4,gemini
```

**Output:**
```markdown
# Ideation Results

## Seed Idea
A tool to help developers learn new programming languages

## Generated Ideas (42 total)

### Cluster 1: Interactive Learning (Score: 8.5/10)
- AI pair programming tutor
- Real-time code translation
- Progressive difficulty challenges
- Instant feedback on syntax
[12 ideas in cluster]

### Cluster 2: Community Learning (Score: 7.8/10)
- Language exchange matching
- Code review marketplace
- Learning buddy system
[8 ideas in cluster]

### Cluster 3: Gamification (Score: 7.2/10)
- Achievement system
- Language proficiency badges
- Competitive coding challenges
[10 ideas in cluster]

## User Selection Prompt
Select cluster(s) to elaborate: [1, 2, 3, or custom combination]

## Detailed Outline (After Selection)
[Comprehensive outline based on selected cluster...]
```

#### State Management

```python
class IdeationState:
    seed_idea: str
    criteria: Dict[str, str]
    generated_ideas: List[Idea]
    clusters: List[IdeaCluster]
    selected_cluster: IdeaCluster
    outline: DetailedOutline

class IdeaCluster:
    theme: str
    ideas: List[Idea]
    score: float
    criteria_breakdown: Dict[str, float]
```

#### Implementation Complexity

- **Effort:** 4-6 days
- **Complexity:** Medium
- **Dependencies:** Conversation infrastructure (existing)
- **New Components:** Clustering algorithm, scoring framework

#### Orchestration Value

**8/10** - Leverages different models for different cognitive modes (divergent vs. convergent thinking).

---

## Document Workflows

---

### 4. DIGEST - Single-Document Summary

**Tagline:** Fast, structured summaries of single documents with key insights extraction

#### Overview

DIGEST provides quick, high-quality summaries of individual documents through a sequential refinement pipeline. While single-document summarization is common, DIGEST adds structure, quality gates, and optional enhancements (action items, key quotes, sentiment).

#### Orchestration Pattern

**Sequential Refinement Pipeline**

```
Document → Preprocessing → Extractive Highlights → Abstractive Summary → Enhancement → Final Digest
```

#### Step-by-Step Process

**Step 1: Preprocessing & Section Detection**
- **Task:** Parse document, detect sections, extract metadata
- **Output:** Structured document with sections identified

**Step 2: Extractive Highlight Selection**
- **Model Role:** Fast extractor (e.g., Claude Haiku)
- **Input:** Preprocessed document
- **Task:** Identify key sentences, important data points, critical quotes
- **Output:** Extracted highlights with locations

**Step 3: Abstractive Rewrite**
- **Model Role:** Summarizer (e.g., Claude Opus, GPT-4)
- **Input:** Highlights + full context
- **Task:** Generate coherent summary with tone controls
- **Output:** Structured summary

**Step 4: Optional Enhancements**
- **Model Role:** Enhancer
- **Input:** Summary + original document
- **Task:** Extract action items, key quotes, generate questions
- **Output:** Enhanced digest

#### Model Roles

| Role | Model | Responsibility |
|------|-------|----------------|
| **Extractor** | Claude Haiku | Fast highlight identification |
| **Summarizer** | GPT-4 / Claude Opus | Coherent summary generation |
| **Enhancer** | Claude | Action items, insights |

#### Use Cases

- **Meeting Notes** - Summarize meeting recordings or notes
- **Business Reports** - Executive summaries of long reports
- **News Articles** - Quick digests of news
- **Research Papers** - Paper summaries for literature review
- **Email Threads** - Summarize long email chains
- **Technical Documentation** - Extract key points from docs

#### User Value

- **Time Savings** - Read 10-page doc in 1 minute
- **Structured Output** - Consistent format vs. raw summaries
- **Key Insights** - Highlights what matters most
- **Action Items** - Automatically extract next steps
- **Searchable** - Digest serves as document index

#### Input/Output

**Input:**
```bash
modelchorus digest meeting-notes.txt \
  --format executive \
  --extract-actions true \
  --extract-quotes true
```

**Output:**
```markdown
# Document Digest

**Source:** meeting-notes.txt
**Date:** 2025-11-06
**Length:** 12 pages → 1 page summary

## Executive Summary
[2-3 paragraph coherent summary...]

## Key Points
- [Main point 1]
- [Main point 2]
- [Main point 3]

## Action Items
- [ ] [Action item with owner]
- [ ] [Action item with deadline]

## Key Quotes
> "[Important quote from page 5]"

## Questions Raised
- [Open question 1]
- [Open question 2]

## Metadata
- Topics: Strategy, Product, Timeline
- Sentiment: Positive
- Urgency: Medium
```

#### State Management

```python
class DigestState:
    document: Document
    sections: List[Section]
    highlights: List[Highlight]
    summary: Summary
    action_items: List[ActionItem]
    key_quotes: List[Quote]
```

#### Implementation Complexity

- **Effort:** 2-3 days
- **Complexity:** Low-Medium
- **Dependencies:** Document ingestion (new)
- **New Components:** Document parser, section detector

#### Orchestration Value

**6/10** - Sequential pipeline with quality gates. Single-doc summarization is common, but structured enhancements add value.

---

### 5. SYNTHESIZE - Multi-Document Analysis

**Tagline:** Cross-document theme extraction, contradiction detection, and synthesis with rigorous citations

#### Overview

SYNTHESIZE is the flagship document workflow. It analyzes multiple documents in parallel to identify common themes, contradictions, and knowledge gaps, then synthesizes findings into a coherent report with verifiable citations. This is where orchestration provides maximum advantage over single-model approaches.

#### Orchestration Pattern

**Parallel Extraction → Sequential Synthesis & Refinement**

```
Documents → [Parallel Extraction] → Theme Clustering → Contradiction Detection → Synthesis → Report
```

#### Step-by-Step Process

**Step 1: Ingest & Chunk**
- **Task:** Upload multiple documents, parse, chunk for processing
- **Output:** Document registry with chunks and metadata

**Step 2: Parallel Extraction**
- **Model Role:** Extractors (one per document or chunk)
- **Input:** Individual documents
- **Task:** Extract:
  - Key claims and findings
  - Data points and statistics
  - Methodologies and approaches
  - Conclusions and recommendations
  - Structured summaries
- **Output:** Structured extraction per document with citations

**Step 3: Theme Clustering**
- **Model Role:** Analyst
- **Input:** All extracted claims
- **Task:** Identify recurring themes and patterns across documents
- **Output:** Theme clusters with supporting documents

**Step 4: Contradiction Detection**
- **Model Role:** Validator
- **Input:** Claims grouped by theme
- **Task:** Identify agreements, contradictions, and inconsistencies
- **Output:** Contradiction map with severity and evidence

**Step 5: Knowledge Gap Analysis**
- **Model Role:** Synthesizer
- **Input:** Themes + contradictions
- **Task:** Identify what's missing, unanswered questions, research gaps
- **Output:** Gap analysis

**Step 6: Report Compilation**
- **Model Role:** Writer
- **Input:** All structured data
- **Task:** Generate coherent synthesis report with embedded citations
- **Output:** Final synthesis report

#### Model Roles

| Role | Model | Responsibility |
|------|-------|----------------|
| **Extractors** | Claude Haiku (fast) | Parallel claim extraction |
| **Analyst** | Gemini 2.5 Pro | Theme clustering |
| **Validator** | Claude Opus | Contradiction detection |
| **Synthesizer** | GPT-4 | Gap analysis |
| **Writer** | Claude Opus | Report generation |

#### Use Cases

- **Literature Reviews** - Synthesize academic papers
- **Market Research** - Analyze market reports and data
- **Competitive Analysis** - Compare competitor information
- **News Analysis** - Cross-source news verification
- **Policy Research** - Synthesize policy documents
- **Legal Discovery** - Analyze case documents

#### User Value

- **Comprehensive Synthesis** - "What's the consensus across sources?"
- **Contradiction Detection** - Automatic inconsistency identification
- **Bias Reduction** - Multiple models check each other
- **Citation Integrity** - Every claim traceable to source
- **Time Savings** - Hours/days → minutes
- **Verifiable Output** - Citations allow verification

#### Input/Output

**Input:**
```bash
modelchorus synthesize "AI safety research trends" \
  --documents paper1.pdf paper2.pdf report1.docx article.txt \
  --detect-contradictions true \
  --citation-style apa
```

**Output:**
```markdown
# Synthesis Report: AI safety research trends

**Documents Analyzed:** 4
**Date:** 2025-11-06
**Citation Style:** APA

## Executive Summary
[Coherent synthesis of all documents...]

## Common Themes

### Theme 1: Alignment Challenges (4/4 documents)
This theme appears consistently across all analyzed sources:
- **Claim:** AI alignment remains unsolved [1, 2, 3, 4]
- **Claim:** Scalable oversight is critical [1, 3]
- **Claim:** Mechanistic interpretability shows promise [2, 4]

**Supporting Documents:**
- Document 1 (Smith et al., 2024): Comprehensive survey, p.12-15
- Document 2 (Jones, 2024): Empirical study, p.8
- Document 3 (Tech Report, 2024): Industry perspective, §4
- Document 4 (Conference Paper, 2024): Novel approach, p.22

### Theme 2: Governance Frameworks (3/4 documents)
[Similar structure...]

## Contradictions Detected

### High Severity
**Contradiction:** Timeline for AGI development
- **Source A** (Document 1, p.45): "AGI likely within 5-10 years"
- **Source B** (Document 3, p.12): "AGI unlikely before 2050"
- **Analysis:** Different definitions of AGI and development assumptions

### Medium Severity
[Additional contradictions...]

## Knowledge Gaps

The following areas lack coverage in analyzed documents:
1. **International coordination mechanisms** - Mentioned but not detailed
2. **Economic impacts of safety measures** - Not addressed
3. **Public communication strategies** - Single brief mention

## Agreements & Consensus

The following claims have strong agreement (3+ sources):
- AI safety research funding is increasing [1, 2, 3, 4]
- Technical alignment is harder than anticipated [1, 2, 4]

## Recommendations

Based on synthesis:
1. [Recommendation based on evidence]
2. [Recommendation addressing gaps]

## Full Citations

[1] Smith, J., et al. (2024). "AI Alignment Survey"...
[2] Jones, A. (2024). "Empirical Study"...
[3] Tech Company (2024). "Industry Report"...
[4] Conference (2024). "Novel Approaches"...

## Appendix: Document Index

| Doc ID | Title | Type | Date | Pages |
|--------|-------|------|------|-------|
| 1 | AI Alignment Survey | Academic | 2024 | 67 |
| 2 | Empirical Study | Academic | 2024 | 23 |
| 3 | Industry Report | Business | 2024 | 45 |
| 4 | Novel Approaches | Conference | 2024 | 12 |
```

#### State Management

```python
class SynthesisState:
    topic: str
    documents: List[Document]
    extractions: List[Extraction]
    themes: List[Theme]
    contradictions: List[Contradiction]
    gaps: List[KnowledgeGap]
    citation_map: CitationMap
    synthesis_report: Report

class Theme:
    name: str
    claims: List[Claim]
    supporting_docs: List[DocumentReference]
    confidence: float

class Contradiction:
    theme: str
    claim_a: Claim
    claim_b: Claim
    severity: str  # high, medium, low
    analysis: str

class Claim:
    content: str
    source_document: str
    location: str  # page, section
    confidence: float
```

#### Implementation Complexity

- **Effort:** 7-10 days
- **Complexity:** High
- **Dependencies:** Document ingestion, citation engine (new)
- **New Components:**
  - Robust citation tracking system
  - Theme clustering algorithm
  - Contradiction detection logic
  - Multi-document state management

#### Orchestration Value

**9/10** - This is where multi-model orchestration provides maximum advantage. Single models cannot match the systematic analysis and verification.

---

### 6. REVIEW - Critical Document Analysis

**Tagline:** 360-degree analysis of critical documents through multi-perspective committee review

#### Overview

REVIEW provides comprehensive analysis of single important documents (contracts, business plans, technical papers) by running parallel analyses from different expert perspectives, then synthesizing into a unified multi-faceted review.

#### Orchestration Pattern

**Role-Based Committee → Synthesis**

```
Document → [Parallel Committee] → Consolidation → Multi-Faceted Review
           [Summarizer]
           [Critic]
           [Amplifier]
```

#### Step-by-Step Process

**Step 1: Document Ingestion**
- **Task:** Load and parse critical document
- **Output:** Structured document ready for analysis

**Step 2: Parallel Committee Analysis**
Three models analyze simultaneously with different roles:

**Agent A: Summarizer**
- **Role:** Neutral observer
- **Task:** Provide objective, executive summary
- **Output:** Balanced summary without judgment

**Agent B: Critic**
- **Role:** Skeptic/Devil's Advocate
- **Task:** Identify:
  - Logical fallacies
  - Unstated assumptions
  - Potential weaknesses
  - Missing information
  - Risks and concerns
- **Output:** Critical analysis

**Agent C: Amplifier**
- **Role:** Advocate
- **Task:** Highlight:
  - Strongest arguments
  - Most innovative points
  - Unique contributions
  - Strategic opportunities
- **Output:** Positive analysis

**Step 3: Consolidation**
- **Model Role:** Moderator (using CONSENSUS pattern)
- **Input:** All three analyses
- **Task:** Integrate perspectives into coherent review
- **Output:** Multi-faceted review document

#### Model Roles

| Role | Model | Perspective |
|------|-------|-------------|
| **Summarizer** | Claude Opus | Neutral, objective |
| **Critic** | Gemini 2.5 Pro | Skeptical, challenging |
| **Amplifier** | GPT-4 | Optimistic, opportunity-focused |
| **Moderator** | Claude Opus | Synthesis and integration |

#### Use Cases

- **Business Plan Review** - Comprehensive vetting before investment
- **Legal Contract Analysis** - Multi-perspective contract review
- **Technical Paper Evaluation** - Peer review assistance
- **Strategic Document Assessment** - Policy/strategy evaluation
- **Proposal Review** - Grant or project proposal analysis
- **Manuscript Evaluation** - Book or article review

#### User Value

- **360-Degree Analysis** - See all angles, not just one perspective
- **Risk Identification** - Critic reveals what you might miss
- **Opportunity Discovery** - Amplifier highlights hidden value
- **Balanced Assessment** - Not overly optimistic or pessimistic
- **Decision Confidence** - Comprehensive review enables better decisions

#### Input/Output

**Input:**
```bash
modelchorus review business-plan.pdf \
  --summarizer claude \
  --critic gemini \
  --amplifier gpt-4 \
  --depth comprehensive
```

**Output:**
```markdown
# Critical Review: Business Plan for TechCo

**Document:** business-plan.pdf
**Date:** 2025-11-06
**Review Type:** Comprehensive

## Executive Summary (Neutral Perspective)

[Objective summary of the business plan without bias...]

**Key Components:**
- Market opportunity: $X billion
- Team: Y members with Z experience
- Product: AI-powered platform
- Financial projections: Revenue of $A by year 3

## Critical Analysis (Skeptic Perspective)

### Logical Fallacies Identified
1. **Survivorship Bias** (p.12): Comparison only to successful competitors
2. **Hasty Generalization** (p.23): Small pilot extrapolated to full market

### Unstated Assumptions
- Assumes regulatory environment remains stable
- Requires customer acquisition cost below $X (not validated)
- Depends on technology that hasn't been proven at scale

### Potential Weaknesses
1. **Market Entry** - Crowded market with entrenched competitors (p.8)
2. **Team Gaps** - No CTO listed, technical leadership unclear (p.15)
3. **Financial Model** - Aggressive growth assumptions without justification (p.30)

### Risks & Concerns
- **High:** Technical feasibility unproven
- **Medium:** Customer acquisition strategy vague
- **Low:** International expansion timeline optimistic

### Missing Information
- Competitive response strategy
- Detailed technical architecture
- Customer validation data
- Contingency plans

## Opportunity Analysis (Amplifier Perspective)

### Strongest Arguments
1. **Market Timing** - First-mover in emerging category (p.5)
2. **Team Experience** - Founders have 20+ years combined experience (p.14)
3. **Technology Moat** - Proprietary algorithm with patent pending (p.22)

### Most Innovative Points
- Novel approach to customer onboarding (p.18)
- Unique pricing model aligned with customer success (p.25)
- Strategic partnerships with industry leaders (p.28)

### Strategic Opportunities
- Potential acquisition target for larger players
- Platform could expand beyond initial market
- Technology applicable to adjacent industries

### Unique Contributions
This plan addresses a genuine market gap that competitors have overlooked.

## Integrated Assessment (Moderator Synthesis)

### Overall Viability: **Medium-High**

**Strengths:**
- Clear market opportunity with growing demand
- Experienced founding team
- Innovative technology approach

**Weaknesses:**
- Unproven technology at scale
- Aggressive financial projections
- Competitive landscape underestimated

**Recommendations:**
1. **Critical:** Validate technical feasibility with proof-of-concept
2. **Important:** Revise financial model with conservative scenarios
3. **Important:** Develop detailed competitive response strategy
4. **Advisable:** Add technical co-founder or CTO

### Decision Framework

**Proceed if:**
- Technical proof-of-concept successful
- Team strengthened with technical leadership
- Customer validation data obtained

**Reconsider if:**
- Technology assumptions prove unfounded
- Customer acquisition costs exceed projections
- Regulatory barriers emerge

## Confidence Scores

| Aspect | Score | Rationale |
|--------|-------|-----------|
| Market Opportunity | 8/10 | Well-researched, growing market |
| Team Capability | 6/10 | Strong but missing technical lead |
| Technology Feasibility | 5/10 | Unproven at scale |
| Financial Projections | 4/10 | Overly optimistic |
| Competitive Position | 6/10 | Differentiated but challenged |
| **Overall** | **6.5/10** | Promising but needs validation |

## Next Steps

1. Request technical proof-of-concept results
2. Interview team about CTO search
3. Validate customer acquisition assumptions
4. Revise financial model with sensitivity analysis
5. Decision checkpoint in 60 days
```

#### State Management

```python
class ReviewState:
    document: Document
    summary: SummaryAnalysis
    critique: CriticalAnalysis
    amplification: OpportunityAnalysis
    synthesis: IntegratedReview
    confidence_scores: Dict[str, float]

class CriticalAnalysis:
    fallacies: List[LogicalFallacy]
    assumptions: List[Assumption]
    weaknesses: List[Weakness]
    risks: List[Risk]
    missing_info: List[str]

class OpportunityAnalysis:
    strengths: List[Strength]
    innovations: List[Innovation]
    opportunities: List[Opportunity]
```

#### Implementation Complexity

- **Effort:** 4-6 days
- **Complexity:** Medium
- **Dependencies:** CONSENSUS synthesis (existing), DIGEST (for structure)
- **New Components:** Committee coordination, multi-perspective integration

#### Orchestration Value

**8/10** - Role-based committee provides richer analysis than single-model review. Unique multi-perspective value.

---

## Implementation Priority Matrix

### Recommended Phasing

**Phase 2A: High-Impact General Workflows**
1. **ARGUMENT** (4-5 days) - Highest differentiation, role-based orchestration showcase
2. **RESEARCH** (5-7 days) - Builds citation infrastructure for document workflows
3. **IDEATE** (4-6 days) - Appeals to creators, complements analytical workflows

**Total Phase 2A: 13-18 days**

**Phase 2B: Core Document Workflows**
4. **DIGEST** (2-3 days) - Fast foundation, tests document ingestion
5. **SYNTHESIZE** (7-10 days) - Flagship document capability, builds on RESEARCH
6. **REVIEW** (4-6 days) - High-value variant using CONSENSUS + DIGEST patterns

**Total Phase 2B: 13-19 days**

**Combined Total: 26-37 days (5-7 weeks)**

### Dependency Graph

```
Existing Workflows
├─ CONSENSUS
│  └─ REVIEW (uses consensus pattern)
├─ CHAT
│  └─ [All workflows use conversation infrastructure]
└─ THINKDEEP
   ├─ RESEARCH (uses hypothesis tracking)
   └─ SYNTHESIZE (uses investigation patterns)

New Infrastructure
├─ Document Ingestion
│  ├─ DIGEST
│  ├─ SYNTHESIZE
│  └─ REVIEW
└─ Citation Engine
   ├─ RESEARCH
   └─ SYNTHESIZE
```

### Complexity vs. Value Matrix

```
High Value
│
│  SYNTHESIZE ●        ● ARGUMENT
│              RESEARCH ●
│                    ● REVIEW
│          IDEATE ●
│                  ● DIGEST
│
└─────────────────────────────── High Complexity
   Low                           High
```

---

## Common Infrastructure Needs

### 1. Citation & Provenance Engine

**Required by:** RESEARCH, SYNTHESIZE

**Components:**
- Document ID generation
- Chunk/page/paragraph location tracking
- Claim → Source mapping
- Citation formatting (APA, MLA, Chicago)
- Verification confidence scoring

**Schema:**
```python
class Citation:
    claim_id: str
    source_document_id: str
    location: Location  # page, para, line
    quote: Optional[str]
    confidence: float
    verified: bool

class Location:
    document: str
    page: Optional[int]
    section: Optional[str]
    paragraph: Optional[int]
    line_start: Optional[int]
```

### 2. Document Ingestion Pipeline

**Required by:** DIGEST, SYNTHESIZE, REVIEW

**Components:**
- Format parsing (PDF, DOCX, HTML, TXT, MD)
- Section detection
- Metadata extraction
- Intelligent chunking
- OCR for scanned documents (future)

**Schema:**
```python
class Document:
    id: str
    title: str
    source: str
    format: str
    metadata: DocumentMetadata
    sections: List[Section]
    chunks: List[Chunk]

class DocumentMetadata:
    author: Optional[str]
    date: Optional[str]
    publication: Optional[str]
    doi: Optional[str]
    url: Optional[str]
```

### 3. Theme Clustering Algorithm

**Required by:** SYNTHESIZE, IDEATE

**Components:**
- Semantic similarity computation
- Clustering algorithm (K-means, hierarchical)
- Theme naming/summarization
- Cluster scoring

### 4. Contradiction Detection

**Required by:** RESEARCH, SYNTHESIZE

**Components:**
- Claim comparison logic
- Contradiction severity scoring
- Explanation generation
- Reconciliation suggestions

### 5. Role-Based Orchestration Framework

**Required by:** ARGUMENT, REVIEW, IDEATE

**Components:**
- Role definition system
- Sequential/parallel execution coordinator
- Role-specific prompt management
- Synthesis coordinator

---

## Success Metrics

### Workflow Adoption
- **Target:** 50% of users try at least 1 new workflow in first month
- **Measure:** Workflow invocation counts

### User Value
- **Target:** 4.5+ average rating across all workflows
- **Measure:** User satisfaction surveys

### Orchestration Differentiation
- **Target:** Users report workflows provide value beyond single-model tools
- **Measure:** Qualitative feedback, comparison studies

### Citation Accuracy
- **Target:** 95%+ citation accuracy (claim correctly linked to source)
- **Measure:** Automated validation + spot checks

### Time Savings
- **Target:** 10x time reduction for multi-document tasks
- **Measure:** User-reported time comparisons

---

## Risk Assessment

### High Priority Risks

**Risk 1: Citation Hallucination**
- **Problem:** Models generate plausible but incorrect citations
- **Impact:** Critical - undermines trust
- **Mitigation:** Strict provenance tracking, verification loops, confidence scoring

**Risk 2: State Management Complexity**
- **Problem:** Tracking claims, contradictions, citations across documents is complex
- **Impact:** High - system unreliable if state corrupted
- **Mitigation:** Robust state schema, validation, automated testing

**Risk 3: Document Format Variability**
- **Problem:** PDFs, scans, complex layouts
- **Impact:** Medium - reduces usability
- **Mitigation:** Start with clean formats, expand gradually

### Medium Priority Risks

**Risk 4: Orchestration Latency**
- **Problem:** Multi-step workflows may be slow
- **Impact:** Medium - user frustration
- **Mitigation:** Parallel execution, fast models for extraction, progress indicators

**Risk 5: Cost Scaling**
- **Problem:** Multi-model workflows increase API costs
- **Impact:** Medium - limits adoption
- **Mitigation:** Optimize model selection, offer single-model variants, cost transparency

---

## Future Extensions

### Advanced Workflows (Phase 3+)

**SCENARIO** - Strategic Scenario Planning Board
- Business/strategic scenario analysis
- Risk evaluation matrices
- Effort: 7-9 days

**ROOTCAUSE** - Root Cause Analysis Investigator
- 5 Whys investigation
- Causal tree mapping
- Effort: 4-5 days

**REFINE** - Multi-Format Content Refinery
- Multi-channel content creation
- Brand voice consistency
- Effort: 5-6 days

**TEACH** - Guided Learning Studio
- Adaptive curriculum planning
- Socratic tutoring
- Effort: 8-10 days

**COMPLIANCE** - Legal/Compliance Document Analysis
- Regulatory analysis
- Risk assessment
- Effort: 8-10 days

**COMPARE** - Document Comparison Matrix
- Side-by-side comparison
- Decision matrices
- Effort: 5-7 days

### Infrastructure Extensions

- **Vector Storage** - For large document corpora
- **Graph Visualization** - For argument maps, causal trees
- **Real-time Collaboration** - Multi-user workflows
- **Custom Role Definitions** - User-defined model roles
- **Workflow Composition** - Chain workflows together
- **Quality Dashboards** - Monitoring citation accuracy, user satisfaction

---

## References

### Multi-Model Analysis Sources

**General-Purpose Workflows:**
- Gemini 2.5 Pro analysis (Confidence: 9/10)
- GPT-5 Codex analysis (Confidence: 7/10)

**Document Workflows:**
- Gemini 2.5 Pro analysis (Confidence: 8/10)
- GPT-5 Codex analysis (Confidence: 7/10)

### Industry Examples

**Research/Document Synthesis:**
- Perplexity (web search)
- SciSpace (academic papers)
- Consensus.app (research synthesis)
- Glean (enterprise knowledge)
- Notion AI (document processing)
- Thomson Reuters (legal)
- Connected Papers (academic graphs)

**Collaboration & Ideation:**
- Miro (brainstorming)
- Figma (design collaboration)
- Linear (project planning)

---

## Appendix: CLI Examples

### ARGUMENT
```bash
modelchorus argument "Universal basic income: viable policy?" \
  --creator claude \
  --skeptic gemini \
  --moderator gpt-4 \
  --output argument-map.md
```

### RESEARCH
```bash
modelchorus research "Quantum computing applications" \
  --sources paper1.pdf paper2.pdf article.txt \
  --validate true \
  --detect-contradictions true \
  --citation-style apa \
  --output research-dossier.md
```

### IDEATE
```bash
modelchorus ideate "Sustainable urban transportation" \
  --criteria "novelty:high,feasibility:medium,impact:high" \
  --ideators claude,gpt-4,gemini \
  --interactive true
```

### DIGEST
```bash
modelchorus digest quarterly-report.pdf \
  --format executive \
  --extract-actions true \
  --extract-quotes true \
  --output digest.md
```

### SYNTHESIZE
```bash
modelchorus synthesize "AI regulation approaches" \
  --documents eu-policy.pdf us-policy.pdf china-policy.pdf \
  --detect-contradictions true \
  --identify-gaps true \
  --citation-style chicago \
  --output synthesis-report.md
```

### REVIEW
```bash
modelchorus review startup-pitch.pdf \
  --summarizer claude \
  --critic gemini \
  --amplifier gpt-4 \
  --depth comprehensive \
  --output critical-review.md
```

---

**Document Status:** Draft proposal for review
**Next Steps:** Review → Prioritize → Prototype → Implement

