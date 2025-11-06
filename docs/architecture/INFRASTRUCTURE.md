# ModelChorus Infrastructure Components

This document describes the core infrastructure components that support ModelChorus workflows.

---

## Citation Engine

### Overview

The Citation Engine is a comprehensive system for tracking, formatting, and validating source citations in ModelChorus workflows. It provides structured citation management with academic-quality formatting, confidence scoring, and evidence tracking capabilities.

**Key Capabilities:**
- **Citation Tracking:** Track sources with location, confidence, and metadata
- **Multiple Format Styles:** APA, MLA, and Chicago citation formats
- **Validation:** Automated citation quality and completeness checks
- **Confidence Scoring:** Multi-factor confidence assessment for citations and claims
- **Evidence Mapping:** Map claims to supporting citations for verification

**Supported Workflows:**
- **RESEARCH:** Source tracking, evidence gathering, and research dossier generation
- **ARGUMENT:** Claim-to-citation mapping, evidence verification, and dialectical analysis

**Location:** `modelchorus/src/modelchorus/utils/citation_formatter.py`

---

### Architecture

#### Data Models

The citation engine uses two core Pydantic models defined in `modelchorus/src/modelchorus/core/models.py`:

##### Citation Model

Represents a single source citation with metadata.

**Fields:**
- `source` (str, required): Source identifier (URL, file path, document ID, DOI, etc.)
  - Minimum length: 1 character
  - Examples: URLs, file paths, ISBNs, DOIs
- `location` (str, optional): Specific location within source
  - Examples: "Section 3.2", "Page 45", "Figure 4", "00:15:30 (timestamp)"
- `confidence` (float, required): Confidence level in citation accuracy
  - Range: 0.0-1.0
  - Validation: Must be between 0.0 and 1.0
- `snippet` (str, optional): Text snippet from the source
  - Useful for evidence verification
- `metadata` (dict, optional): Additional citation metadata
  - Flexible key-value pairs
  - Common fields: `author`, `year`, `publication_date`, `title`, `citation_type`

**Example:**
```python
from modelchorus.core.models import Citation

citation = Citation(
    source="https://arxiv.org/abs/2401.12345",
    location="Section 3.2, Figure 4",
    confidence=0.95,
    snippet="Our experiments show a 23% improvement in accuracy...",
    metadata={
        "author": "Smith et al.",
        "publication_date": "2024-01-15",
        "title": "Machine Learning Advances",
        "citation_type": "academic_paper"
    }
)
```

##### CitationMap Model

Maps claims to their supporting citations for evidence tracking and verification.

**Fields:**
- `claim_id` (str, required): Unique identifier for the claim
  - Minimum length: 1 character
- `claim_text` (str, required): The actual claim or argument text
  - Minimum length: 1 character
- `citations` (List[Citation], optional): List of Citation objects supporting this claim
  - Defaults to empty list
- `strength` (float, required): Overall strength of citation support
  - Range: 0.0-1.0
  - Typically calculated from citation confidences
- `metadata` (dict, optional): Additional mapping metadata
  - Common fields: `argument_type`, `verification_status`, `citation_count`

**Example:**
```python
from modelchorus.core.models import Citation, CitationMap

citations = [
    Citation(
        source="https://arxiv.org/abs/2401.12345",
        location="Section 3.2",
        confidence=0.95,
        snippet="Our experiments show a 23% improvement..."
    ),
    Citation(
        source="paper2.pdf",
        location="Figure 4",
        confidence=0.85,
        snippet="Results demonstrate significant gains..."
    )
]

citation_map = CitationMap(
    claim_id="claim-001",
    claim_text="Machine learning models improve accuracy by 23%",
    citations=citations,
    strength=0.9,
    metadata={
        "argument_type": "empirical",
        "verification_status": "verified",
        "citation_count": 2
    }
)
```

---

#### Citation Formatting

The citation engine supports three academic citation styles, each with distinct formatting rules.

##### Supported Styles

**1. APA (American Psychological Association)**
- Format: `Author(s). (Year). Title. Retrieved from Source.`
- Best for: Psychology, education, social sciences
- Example: `Smith, J. (2024). Machine Learning. Retrieved from https://arxiv.org/abs/2401.12345`

**2. MLA (Modern Language Association)**
- Format: `Author(s). "Title." Source, Year. Location.`
- Best for: Humanities, literature, arts
- Example: `Smith, J. "Machine Learning." https://arxiv.org/abs/2401.12345, 2024.`

**3. Chicago (Chicago Manual of Style)**
- Format: `Author(s). "Title." Source (Year): Location.`
- Best for: History, business, fine arts
- Example: `Smith, J. "Machine Learning." https://arxiv.org/abs/2401.12345 (2024): Section 3.2`

##### Style Selection

```python
from modelchorus.utils.citation_formatter import CitationStyle

# Available styles
CitationStyle.APA       # "apa"
CitationStyle.MLA       # "mla"
CitationStyle.CHICAGO   # "chicago"
```

---

#### Validation & Confidence Scoring

The citation engine includes comprehensive validation and confidence scoring to assess citation quality.

##### Citation Validation

Validates citations for completeness and quality.

**Validation Checks:**
1. **Required Fields:** Source present and non-empty
2. **Confidence Bounds:** Confidence value between 0.0 and 1.0
3. **Metadata Completeness:** Checks for recommended fields (author, year, title)
4. **Source Format:** Recognizes URLs, file paths, DOIs

**Example:**
```python
from modelchorus.utils.citation_formatter import validate_citation

is_valid, issues = validate_citation(citation)
if not is_valid:
    print(f"Validation issues: {', '.join(issues)}")
```

**Returns:**
- `is_valid` (bool): True if citation meets minimum requirements
- `issues` (List[str]): List of validation issue messages

##### Confidence Scoring

Multi-factor confidence assessment evaluates citation reliability.

**Scoring Factors:**
1. **Base Confidence (40%):** Original confidence value from citation
2. **Metadata Completeness (30%):** Presence of author, year, title, snippet
3. **Source Quality (20%):** Academic sources rated higher (arXiv, DOI, academic domains)
4. **Location Specificity (10%):** Presence and detail of location information

**Example:**
```python
from modelchorus.utils.citation_formatter import calculate_citation_confidence

scores = calculate_citation_confidence(citation)

print(f"Overall confidence: {scores['overall_confidence']:.2f}")
print(f"Metadata completeness: {scores['metadata_score']:.2f}")
print(f"Source quality: {scores['source_quality_score']:.2f}")
print(f"Location specificity: {scores['location_score']:.2f}")
```

**Returns Dictionary:**
- `overall_confidence` (float): Final weighted confidence score (0.0-1.0)
- `base_confidence` (float): Original confidence value
- `metadata_score` (float): Completeness score (0.0-1.0)
- `source_quality_score` (float): Quality score (0.0-1.0)
- `location_score` (float): Specificity score (0.0-1.0)
- `factors` (dict): Detailed breakdown of scoring factors

##### Citation Map Confidence

Aggregate confidence scoring for claims supported by multiple citations.

**Scoring Formula:**
- Average citation confidence (50%)
- CitationMap strength value (30%)
- Citation count factor (20%) - plateaus at 5 citations

**Example:**
```python
from modelchorus.utils.citation_formatter import calculate_citation_map_confidence

scores = calculate_citation_map_confidence(citation_map)

print(f"Overall confidence: {scores['overall_confidence']:.2f}")
print(f"Citation count: {scores['citation_count']}")
print(f"Average citation confidence: {scores['average_citation_confidence']:.2f}")
print(f"Min/Max confidence: {scores['min_confidence']:.2f} / {scores['max_confidence']:.2f}")
```

---

### Usage Examples

#### Basic Citation Creation

```python
from modelchorus.core.models import Citation

# Minimal citation
citation = Citation(
    source="https://example.com/research",
    confidence=0.8
)

# Complete citation with all fields
citation = Citation(
    source="https://arxiv.org/abs/2401.12345",
    location="Section 3.2, Figure 4",
    confidence=0.95,
    snippet="Our experiments show a 23% improvement in accuracy...",
    metadata={
        "author": "Smith, J.",
        "year": "2024",
        "title": "Machine Learning Advances",
        "citation_type": "academic_paper",
        "peer_reviewed": True
    }
)
```

#### Citation Formatting

**Format Single Citation:**
```python
from modelchorus.utils.citation_formatter import format_citation, CitationStyle

# APA format
apa_citation = format_citation(citation, CitationStyle.APA)
# Output: "Smith, J. (2024). Machine Learning Advances. Retrieved from https://arxiv.org/abs/2401.12345 (Section 3.2, Figure 4)"

# MLA format
mla_citation = format_citation(citation, CitationStyle.MLA)
# Output: "Smith, J. "Machine Learning Advances." https://arxiv.org/abs/2401.12345, 2024. Section 3.2, Figure 4"

# Chicago format
chicago_citation = format_citation(citation, CitationStyle.CHICAGO)
# Output: "Smith, J. "Machine Learning Advances." https://arxiv.org/abs/2401.12345 (2024): Section 3.2, Figure 4"
```

**Format Citation Map:**
```python
from modelchorus.utils.citation_formatter import format_citation_map, CitationStyle

# Format with claim text and all citations
formatted = format_citation_map(citation_map, CitationStyle.APA, include_claim=True)
print(formatted)

# Output:
# Claim: Machine learning models improve accuracy by 23%
#
# Citations:
# 1. Smith, J. (2024). ML Research. Retrieved from https://arxiv.org/abs/2401.12345
# 2. Doe, A. (2024). AI Studies. Retrieved from paper2.pdf
```

#### Citation Validation

```python
from modelchorus.utils.citation_formatter import validate_citation, calculate_citation_confidence

# Validate citation
is_valid, issues = validate_citation(citation)

if is_valid:
    print("✓ Citation is valid")
else:
    print("✗ Citation has issues:")
    for issue in issues:
        print(f"  - {issue}")

# Calculate confidence score
scores = calculate_citation_confidence(citation)

print(f"Overall confidence: {scores['overall_confidence']:.2f}")
print(f"  Base confidence: {scores['base_confidence']:.2f}")
print(f"  Metadata complete: {scores['metadata_score']:.2f}")
print(f"  Source quality: {scores['source_quality_score']:.2f}")
print(f"  Location specific: {scores['location_score']:.2f}")
```

#### Integration in Workflows

**RESEARCH Workflow - Source Tracking:**
```python
from modelchorus.workflows import ResearchWorkflow
from modelchorus.providers import ClaudeProvider
from modelchorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ResearchWorkflow(provider=provider, conversation_memory=memory)

# Ingest sources with credibility ratings
workflow.ingest_source(
    title="GraphQL Specification v16.0",
    url="https://spec.graphql.org/",
    source_type="specification",
    credibility="high"
)

workflow.ingest_source(
    title="REST vs GraphQL Study",
    url="docs/performance_study.pdf",
    source_type="research_paper",
    credibility="high"
)

# Conduct research with citations
result = await workflow.run(
    prompt="Compare GraphQL and REST performance",
    citation_style="academic"
)

# Generate research dossier with citations
dossier = workflow.generate_dossier(
    evidence_items=evidence,
    include_citations=True,
    citation_style="academic"
)
```

**ARGUMENT Workflow - Evidence Verification:**
```python
from modelchorus.core.models import Citation, CitationMap

# Create citations for proponent argument
pro_citation = Citation(
    source="https://study.com/typescript-errors",
    location="Results section",
    confidence=0.90,
    snippet="TypeScript reduces runtime errors by 15%",
    metadata={"author": "Research Team", "year": "2024"}
)

# Map citation to claim
pro_claim = CitationMap(
    claim_id="pro-typescript-001",
    claim_text="TypeScript significantly reduces runtime errors",
    citations=[pro_citation],
    strength=0.90,
    metadata={
        "argument_type": "empirical",
        "stance": "proponent",
        "verification_status": "verified"
    }
)
```

---

### API Reference

#### format_citation()

Format a Citation object according to the specified style.

```python
def format_citation(
    citation: Citation,
    style: CitationStyle = CitationStyle.APA
) -> str
```

**Parameters:**
- `citation` (Citation): The Citation object to format
- `style` (CitationStyle): The citation style (APA, MLA, or Chicago)

**Returns:**
- `str`: Formatted citation string according to the specified style

**Example:**
```python
formatted = format_citation(citation, CitationStyle.APA)
```

---

#### format_citation_map()

Format a CitationMap object with all its citations.

```python
def format_citation_map(
    citation_map: CitationMap,
    style: CitationStyle = CitationStyle.APA,
    include_claim: bool = True
) -> str
```

**Parameters:**
- `citation_map` (CitationMap): The CitationMap object to format
- `style` (CitationStyle): The citation style to use
- `include_claim` (bool): Whether to include the claim text in the output

**Returns:**
- `str`: Formatted string with claim and all citations

**Example:**
```python
formatted = format_citation_map(citation_map, CitationStyle.APA, include_claim=True)
```

---

#### validate_citation()

Validate a Citation object for completeness and quality.

```python
def validate_citation(citation: Citation) -> Tuple[bool, List[str]]
```

**Parameters:**
- `citation` (Citation): The Citation object to validate

**Returns:**
- `Tuple[bool, List[str]]`: Tuple of (is_valid, issues)
  - `is_valid`: True if citation meets minimum requirements
  - `issues`: List of validation issue messages

**Example:**
```python
is_valid, issues = validate_citation(citation)
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

---

#### calculate_citation_confidence()

Calculate a detailed confidence score for a citation's reliability.

```python
def calculate_citation_confidence(citation: Citation) -> Dict[str, Any]
```

**Parameters:**
- `citation` (Citation): The Citation object to score

**Returns:**
- `Dict[str, Any]`: Dictionary with:
  - `overall_confidence` (float): Final confidence score (0.0-1.0)
  - `base_confidence` (float): Original confidence value
  - `metadata_score` (float): Completeness score (0.0-1.0)
  - `source_quality_score` (float): Quality score (0.0-1.0)
  - `location_score` (float): Specificity score (0.0-1.0)
  - `factors` (dict): Detailed breakdown of scoring factors

**Example:**
```python
scores = calculate_citation_confidence(citation)
print(f"Overall: {scores['overall_confidence']:.2f}")
```

---

#### calculate_citation_map_confidence()

Calculate aggregate confidence scores for a CitationMap.

```python
def calculate_citation_map_confidence(citation_map: CitationMap) -> Dict[str, Any]
```

**Parameters:**
- `citation_map` (CitationMap): The CitationMap object to score

**Returns:**
- `Dict[str, Any]`: Dictionary with:
  - `overall_confidence` (float): Aggregate confidence for the claim (0.0-1.0)
  - `citation_count` (int): Number of citations
  - `average_citation_confidence` (float): Mean confidence across citations
  - `min_confidence` (float): Lowest confidence citation
  - `max_confidence` (float): Highest confidence citation
  - `strength` (float): Original strength value from CitationMap
  - `individual_scores` (List[Dict]): List of confidence scores per citation

**Example:**
```python
scores = calculate_citation_map_confidence(citation_map)
print(f"Claim supported by {scores['citation_count']} citations")
print(f"Overall confidence: {scores['overall_confidence']:.2f}")
```

---

### Design Decisions

#### Why Three Citation Styles?

Different academic disciplines and use cases require different citation formats:
- **APA:** Widely used in social sciences, emphasizes author and date
- **MLA:** Standard in humanities, focuses on page numbers and print sources
- **Chicago:** Flexible style used in history and business, supports footnotes

Supporting all three ensures ModelChorus can generate citations appropriate for any domain.

#### Confidence Scoring Rationale

The multi-factor confidence scoring system weights different aspects based on their reliability indicators:
- **Base confidence (40%):** User or model's assessment of citation accuracy
- **Metadata (30%):** Complete metadata indicates thorough citation work
- **Source quality (20%):** Academic sources are more reliable than blogs
- **Location (10%):** Specific locations enable verification

This weighting prioritizes direct confidence assessment while accounting for objective quality indicators.

#### Metadata Flexibility

Citation metadata uses flexible dictionaries rather than rigid schemas because:
1. Different source types require different metadata fields
2. Academic standards evolve over time
3. Users may need custom fields for specific workflows
4. Forward compatibility for future enhancements

Common metadata fields are documented but not enforced, allowing adaptation to diverse use cases.

#### Integration Approach

The citation engine is designed as a standalone utility that workflows can integrate as needed:
- **Loose coupling:** Workflows import citation utilities, not vice versa
- **Optional usage:** Workflows can track citations or not
- **Flexible integration:** Each workflow determines how to use citations
- **Shared models:** Common Citation/CitationMap models ensure interoperability

This approach allows workflows to use citations in workflow-specific ways while maintaining consistency.

---

### Best Practices

#### When to Add Citations

**Always cite:**
- Empirical claims with specific numbers or statistics
- Direct quotes from sources
- Paraphrased ideas from specific sources
- Technical specifications or standards

**Consider citing:**
- General concepts with established sources
- Background information from authoritative sources
- Comparative claims about technologies or methods

**Don't need to cite:**
- Common knowledge in the domain
- Your own original analysis or reasoning
- Hypothetical scenarios or thought experiments

#### Choosing Confidence Scores

**High Confidence (0.8-1.0):**
- Direct quotes from primary sources
- Official documentation or specifications
- Peer-reviewed academic research
- First-hand data or experiments

**Medium Confidence (0.5-0.7):**
- Secondary sources (textbooks, review papers)
- Well-regarded blog posts from experts
- Industry reports from reputable firms
- Conference presentations

**Low Confidence (0.0-0.4):**
- Opinion pieces or editorials
- Unverified claims from unknown sources
- Paraphrased information without verification
- Outdated sources (depending on topic)

#### Metadata Recommendations

**Essential metadata for academic citations:**
```python
metadata = {
    "author": "Smith, J.",           # Author name(s)
    "year": "2024",                  # Publication year
    "title": "Paper Title",          # Work title
}
```

**Recommended additional metadata:**
```python
metadata = {
    # Essential
    "author": "Smith, J.",
    "year": "2024",
    "title": "Paper Title",

    # Helpful for verification
    "publication_date": "2024-01-15",
    "citation_type": "academic_paper",

    # Quality indicators
    "peer_reviewed": True,
    "doi": "10.1000/xyz123",

    # Context
    "accessed_date": "2024-11-06",
    "keywords": ["machine learning", "optimization"]
}
```

#### Citation Quality Guidelines

**High-quality citations include:**
1. Specific source URL or identifier
2. Precise location within source (section, page, timestamp)
3. Direct snippet showing the relevant content
4. Complete metadata (author, date, title)
5. Appropriate confidence score

**Example of a high-quality citation:**
```python
citation = Citation(
    source="https://arxiv.org/abs/2401.12345",
    location="Section 3.2, Figure 4",
    confidence=0.95,
    snippet="Our experiments demonstrate a 23% ± 2% improvement in accuracy across all tested datasets",
    metadata={
        "author": "Smith, J. and Doe, A.",
        "year": "2024",
        "title": "Advanced Machine Learning Techniques",
        "publication_date": "2024-01-15",
        "citation_type": "academic_paper",
        "peer_reviewed": True,
        "doi": "10.1234/arxiv.2401.12345"
    }
)
```

---

### Testing

#### Test Coverage

The citation engine has comprehensive test coverage in `modelchorus/tests/test_citation.py`:

**Citation Model Tests:**
- Citation creation with all fields
- Minimal citation with required fields only
- Field validation (empty source, confidence bounds)
- Various source types (URLs, file paths, DOIs, ISBNs)
- Location format variations
- Metadata flexibility
- Serialization and deserialization
- JSON roundtrip

**CitationMap Model Tests:**
- CitationMap creation with multiple citations
- Minimal citation map
- Field validation
- Single and multiple citation handling
- Metadata flexibility
- Nested citation validation
- Serialization and JSON roundtrip

**Integration Tests:**
- Claim-to-evidence mapping
- Multiple claims from same source
- Citation strength calculation
- Confidence-based filtering
- ARGUMENT workflow citation tracking

#### Running Tests

```bash
# Run all citation tests
pytest modelchorus/tests/test_citation.py -v

# Run specific test class
pytest modelchorus/tests/test_citation.py::TestCitation -v
pytest modelchorus/tests/test_citation.py::TestCitationMap -v

# Run with coverage
pytest modelchorus/tests/test_citation.py --cov=modelchorus.core.models --cov=modelchorus.utils.citation_formatter
```

#### Test Examples

See `modelchorus/tests/test_citation.py` for extensive examples of:
- Citation model usage patterns
- CitationMap construction
- Validation edge cases
- Serialization scenarios
- Integration patterns for workflows

---

### Future Enhancements

**Planned improvements to the citation engine:**

1. **Additional Citation Styles:**
   - Harvard style (common in UK and Australia)
   - IEEE style (common in engineering)
   - Vancouver style (common in medicine)
   - Custom style templates

2. **Enhanced Validation:**
   - DOI resolution and verification
   - URL accessibility checks
   - Author name format validation
   - Publication date validation
   - Citation duplication detection

3. **Citation Deduplication:**
   - Detect duplicate citations from same source
   - Merge citations with different locations
   - Handle different URL formats for same source
   - Smart source matching

4. **Cross-Reference Tracking:**
   - Track citation relationships between claims
   - Detect contradictory citations
   - Build citation graphs
   - Identify citation clusters

5. **Export Formats:**
   - BibTeX export
   - RIS export
   - EndNote XML
   - Zotero/Mendeley integration

6. **Automatic Metadata Extraction:**
   - Extract metadata from DOIs
   - Parse PDF metadata
   - Web scraping for online sources
   - Integration with citation databases

---

## See Also

**Related Documentation:**
- [RESEARCH Workflow](../workflows/RESEARCH.md) - Research workflow using citation engine
- [ARGUMENT Workflow](../workflows/ARGUMENT.md) - Argument workflow using citations
- [WORKFLOWS.md](../WORKFLOWS.md) - Complete workflow overview

**Code Files:**
- `modelchorus/src/modelchorus/utils/citation_formatter.py` - Citation formatting implementation
- `modelchorus/src/modelchorus/core/models.py` - Citation and CitationMap models
- `modelchorus/tests/test_citation.py` - Comprehensive test suite

**API Reference:**
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Complete API documentation
- [README.md](../../README.md) - Getting started guide
