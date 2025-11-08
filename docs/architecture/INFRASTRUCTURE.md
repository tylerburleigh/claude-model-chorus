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

**Location:** `model_chorus/src/model_chorus/utils/citation_formatter.py`

---

### Architecture

#### Data Models

The citation engine uses two core Pydantic models defined in `model_chorus/src/model_chorus/core/models.py`:

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
from model_chorus.core.models import Citation

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
from model_chorus.core.models import Citation, CitationMap

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
from model_chorus.utils.citation_formatter import CitationStyle

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
from model_chorus.utils.citation_formatter import validate_citation

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
from model_chorus.utils.citation_formatter import calculate_citation_confidence

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
from model_chorus.utils.citation_formatter import calculate_citation_map_confidence

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
from model_chorus.core.models import Citation

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
from model_chorus.utils.citation_formatter import format_citation, CitationStyle

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
from model_chorus.utils.citation_formatter import format_citation_map, CitationStyle

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
from model_chorus.utils.citation_formatter import validate_citation, calculate_citation_confidence

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
from model_chorus.workflows import ResearchWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

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
from model_chorus.core.models import Citation, CitationMap

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

## Semantic Clustering Engine

### Overview

The Semantic Clustering Engine provides theme-based organization of textual content through machine learning-powered semantic similarity analysis. It groups similar ideas, claims, or findings into coherent clusters, enabling workflows to organize large collections of information automatically.

**Key Capabilities:**
- **Semantic Similarity:** Compute meaning-based similarity using sentence transformers
- **Multiple Algorithms:** K-means and hierarchical clustering methods
- **Automatic Naming:** Generate descriptive labels for clusters
- **Quality Scoring:** Assess cluster coherence and quality
- **Flexible Integration:** Works with any text-based content

**Supported Workflows:**
- **IDEATE:** Organize brainstormed ideas into thematic groups for convergent analysis
- **ARGUMENT:** Cluster claims by topic for better organization
- **RESEARCH:** Group findings by theme (future enhancement)

**Location:** `model_chorus/src/model_chorus/core/clustering.py`

---

### Architecture

#### Core Components

The clustering engine consists of three primary components:

1. **Embedding Computation:** Converts text into semantic vectors using sentence transformers
2. **Clustering Algorithms:** Groups embeddings using K-means or hierarchical methods
3. **Post-Processing:** Names, summarizes, and scores clusters

#### Data Models

##### ClusterResult

Represents the result of a clustering operation with all metadata.

**Fields:**
- `cluster_id` (int, required): Unique identifier for this cluster
- `items` (List[int], required): List of item indices belonging to this cluster
- `centroid` (np.ndarray, required): Cluster centroid in embedding space
- `name` (str, optional): Human-readable cluster name/label
- `summary` (str, optional): Brief summary of cluster theme
- `quality_score` (float, optional): Cluster coherence score (0.0-1.0)
- `metadata` (dict, optional): Additional cluster information
  - `size` (int): Number of items in cluster
  - `method` (str): Clustering method used ("kmeans" or "hierarchical")

**Example:**
```python
from model_chorus.core.clustering import ClusterResult
import numpy as np

cluster = ClusterResult(
    cluster_id=0,
    items=[0, 2, 5],
    centroid=np.array([0.1, 0.2, ..., 0.9]),
    name="Python programming ideas",
    summary="Ideas related to Python development, best practices, and tooling",
    quality_score=0.85,
    metadata={
        "size": 3,
        "method": "kmeans"
    }
)
```

---

#### Semantic Similarity

The clustering engine uses **sentence transformers** to compute semantic embeddings. By default, it uses the `all-MiniLM-L6-v2` model, which balances speed and accuracy.

##### How It Works

1. **Text → Embeddings:** Each text is converted to a 384-dimensional vector that captures its semantic meaning
2. **Similarity Computation:** Vectors are compared using cosine similarity (angle between vectors)
3. **Similar texts** have embeddings that point in similar directions, resulting in high similarity scores

##### Supported Similarity Metrics

**Cosine Similarity (Default):**
- Measures angle between vectors
- Range: -1.0 to 1.0 (typically 0.0 to 1.0 for normalized vectors)
- Best for: Text similarity, semantic meaning
- Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`

**Euclidean Distance:**
- Measures geometric distance between vectors
- Converted to similarity: `1 / (1 + distance)`
- Best for: Spatial clustering

**Dot Product:**
- Simple vector multiplication
- Best for: Raw embedding comparisons

**Example:**
```python
from model_chorus.core.clustering import SemanticClustering

clustering = SemanticClustering()

texts = [
    "Python is great for data science",
    "I love Python for machine learning",
    "Java is verbose but powerful"
]

# Compute embeddings
embeddings = clustering.compute_embeddings(texts)
print(embeddings.shape)  # (3, 384)

# Compute similarity matrix
similarity = clustering.compute_similarity(embeddings, metric="cosine")
print(similarity.shape)  # (3, 3)

# Check similarity between first two texts (both about Python)
print(f"Python texts similarity: {similarity[0, 1]:.3f}")  # High score

# Check similarity between Python and Java texts
print(f"Python vs Java similarity: {similarity[0, 2]:.3f}")  # Lower score
```

---

#### Clustering Algorithms

The engine supports two clustering algorithms, each with different characteristics:

##### K-Means Clustering

**How it works:**
1. Randomly initialize K cluster centers
2. Assign each item to nearest center
3. Update centers to mean of assigned items
4. Repeat until convergence

**Characteristics:**
- **Fast:** O(n × k × i) where n=items, k=clusters, i=iterations
- **Deterministic:** Same random seed produces same results
- **Assumes spherical clusters:** Works best when clusters are roughly equal size
- **Requires K:** Must specify number of clusters in advance

**Best for:**
- Large datasets (>100 items)
- When you know the approximate number of themes
- When speed is important

**Example:**
```python
clustering = SemanticClustering()

texts = ["Python is great", "I love Python", "Java is verbose", "C++ is fast"]
clusters = clustering.cluster(texts, n_clusters=2, method="kmeans", random_state=42)

for cluster in clusters:
    print(f"Cluster {cluster.cluster_id}: {cluster.name}")
    print(f"  Items: {cluster.items}")
    print(f"  Quality: {cluster.quality_score:.2f}")
```

##### Hierarchical Clustering

**How it works:**
1. Start with each item as its own cluster
2. Merge the two most similar clusters
3. Repeat until K clusters remain

**Characteristics:**
- **Slower:** O(n³) complexity
- **Deterministic:** No random initialization
- **Flexible shapes:** Can handle irregular cluster shapes
- **Creates hierarchy:** Can visualize as dendrogram

**Linkage methods:**
- **ward** (default): Minimizes within-cluster variance
- **complete**: Maximum distance between clusters
- **average**: Average distance between all pairs
- **single**: Minimum distance between clusters

**Best for:**
- Smaller datasets (<100 items)
- When cluster hierarchy is meaningful
- When you want dendrogram visualization

**Example:**
```python
clustering = SemanticClustering()

texts = ["Python is great", "I love Python", "Java is verbose", "C++ is fast"]
clusters = clustering.cluster(texts, n_clusters=2, method="hierarchical")

for cluster in clusters:
    print(f"Cluster {cluster.cluster_id}: {cluster.name}")
```

---

#### Cluster Naming and Summarization

The engine automatically generates names and summaries for clusters:

##### Cluster Naming

**Current implementation:**
- Uses the shortest text in the cluster as the name
- Truncates to 50 characters if needed
- Simple but effective for most cases

**Future enhancement:**
- Use LLM to generate thematic names based on cluster content
- Example: "Python Development" instead of "Python is great"

**Example:**
```python
texts = [
    "Python is great for data science",
    "Python programming",
    "I love Python for machine learning"
]

# Cluster name would be "Python programming" (shortest)
name = clustering.name_cluster(texts)
```

##### Cluster Summarization

**Current implementation:**
- Concatenates all texts with semicolons
- Truncates to 200 characters if needed
- Provides quick overview of cluster content

**Future enhancement:**
- Use LLM to generate coherent summaries
- Example: "This cluster focuses on Python's strengths in data science and machine learning applications."

---

#### Quality Scoring

Each cluster receives a quality score measuring its coherence:

**Scoring Formula:**
```
Quality = Average cosine similarity of items to centroid
```

**Score Interpretation:**
- **0.8-1.0:** Excellent coherence (highly similar items)
- **0.6-0.8:** Good coherence (related items)
- **0.4-0.6:** Moderate coherence (somewhat related)
- **0.0-0.4:** Poor coherence (diverse or unrelated items)

**Why it matters:**
- High-quality clusters indicate clear thematic groupings
- Low-quality clusters may indicate:
  - Too many clusters (over-segmentation)
  - Items don't naturally group
  - Need to adjust clustering parameters

**Example:**
```python
embeddings = clustering.compute_embeddings(cluster_texts)
centroid = embeddings.mean(axis=0)
quality = clustering.score_cluster(embeddings, centroid)

if quality > 0.8:
    print("Excellent cluster coherence!")
elif quality > 0.6:
    print("Good cluster coherence")
else:
    print("Consider adjusting number of clusters")
```

---

### Usage Examples

#### Basic Clustering

**Simple text clustering:**
```python
from model_chorus.core.clustering import SemanticClustering

# Initialize engine
clustering = SemanticClustering(
    model_name="all-MiniLM-L6-v2",  # Default model
    cache_embeddings=True  # Cache for faster repeated calls
)

# Cluster texts
texts = [
    "Python is great for data science",
    "I love Python for ML",
    "Java is verbose but powerful",
    "C++ offers fine-grained control",
    "JavaScript is ubiquitous on the web"
]

clusters = clustering.cluster(
    texts=texts,
    n_clusters=3,
    method="kmeans",
    random_state=42
)

# Inspect results
for cluster in clusters:
    print(f"\n{cluster.name}")
    print(f"Quality: {cluster.quality_score:.2f}")
    print(f"Items ({len(cluster.items)}):")
    for idx in cluster.items:
        print(f"  - {texts[idx]}")
```

#### IDEATE Workflow Integration

**Organize brainstormed ideas:**
```python
from model_chorus.workflows import IdeateWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

provider = ClaudeProvider()
memory = ConversationMemory()
workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

# Brainstorm ideas
brainstorm_result = await workflow.brainstorm(
    prompt="Ways to improve developer productivity",
    perspectives=["efficiency", "tooling", "collaboration"],
    ideas_per_perspective=5
)

# Convergent analysis with clustering
analysis_result = await workflow.convergent_analysis(
    brainstorming_result=brainstorm_result,
    num_clusters=4,  # Group into 4 themes
    scoring_criteria=["feasibility", "impact", "effort"]
)

# The clustering engine automatically groups similar ideas
print(analysis_result.synthesis)
```

**How it works:**
1. Extract ideas from brainstorming result (15 ideas from 3 perspectives)
2. Cluster similar ideas using semantic clustering (4 themes)
3. Score each cluster based on feasibility, impact, effort
4. Synthesize analysis with clusters and recommendations

#### ARGUMENT Workflow Integration

**Cluster argument claims by topic:**
```python
from model_chorus.workflows.argument.semantic import cluster_claims_kmeans
from model_chorus.core.models import CitationMap, Citation

# Create claims about TypeScript
claims = [
    CitationMap(
        claim_id="claim-1",
        claim_text="TypeScript reduces runtime errors by 15%",
        citations=[Citation(source="study.com", confidence=0.9)],
        strength=0.9
    ),
    CitationMap(
        claim_id="claim-2",
        claim_text="TypeScript improves developer productivity",
        citations=[Citation(source="blog.com", confidence=0.7)],
        strength=0.7
    ),
    CitationMap(
        claim_id="claim-3",
        claim_text="Learning TypeScript adds overhead for new developers",
        citations=[Citation(source="survey.com", confidence=0.8)],
        strength=0.8
    ),
]

# Cluster by topic
clusters = cluster_claims_kmeans(
    citation_maps=claims,
    n_clusters=2,
    random_state=42
)

for i, cluster in enumerate(clusters):
    print(f"\nCluster {i}:")
    for claim_map in cluster:
        print(f"  - {claim_map.claim_text}")
```

**Result:**
- **Cluster 0:** Benefits claims (errors, productivity)
- **Cluster 1:** Challenges claims (learning curve)

#### Advanced: Embedding Cache

**Speed up repeated clustering:**
```python
clustering = SemanticClustering(cache_embeddings=True)

texts = ["Python is great", "I love Python"]

# First call: computes embeddings
clusters1 = clustering.cluster(texts, n_clusters=2)  # Slower

# Second call: uses cached embeddings
clusters2 = clustering.cluster(texts, n_clusters=2)  # Much faster

# Cache persists across calls with same texts
print(f"Cache size: {len(clustering._embedding_cache)}")  # 2
```

#### Custom Similarity Metrics

**Try different metrics:**
```python
clustering = SemanticClustering()

texts = ["Python is great", "I love Python", "Java is verbose"]
embeddings = clustering.compute_embeddings(texts)

# Cosine similarity (default)
cosine_sim = clustering.compute_similarity(embeddings, metric="cosine")
print(f"Cosine similarity (Python texts): {cosine_sim[0, 1]:.3f}")

# Euclidean distance
euclidean_sim = clustering.compute_similarity(embeddings, metric="euclidean")
print(f"Euclidean similarity (Python texts): {euclidean_sim[0, 1]:.3f}")

# Dot product
dot_sim = clustering.compute_similarity(embeddings, metric="dot")
print(f"Dot product (Python texts): {dot_sim[0, 1]:.3f}")
```

---

### API Reference

#### SemanticClustering.__init__()

Initialize the semantic clustering engine.

```python
def __init__(
    model_name: str = "all-MiniLM-L6-v2",
    cache_embeddings: bool = True
)
```

**Parameters:**
- `model_name` (str): Name of sentence transformer model to use
  - Default: `"all-MiniLM-L6-v2"` (384-dim, balanced speed/quality)
  - Alternatives: `"all-mpnet-base-v2"` (768-dim, higher quality, slower)
- `cache_embeddings` (bool): Whether to cache computed embeddings for faster repeated calls

---

#### compute_embeddings()

Convert texts to semantic vector embeddings.

```python
def compute_embeddings(texts: List[str]) -> np.ndarray
```

**Parameters:**
- `texts` (List[str]): List of text strings to embed

**Returns:**
- `np.ndarray`: Array of shape (n_texts, embedding_dim) with embeddings

**Example:**
```python
embeddings = clustering.compute_embeddings(["Hello", "World"])
print(embeddings.shape)  # (2, 384)
```

---

#### compute_similarity()

Compute pairwise similarity matrix between embeddings.

```python
def compute_similarity(
    embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray
```

**Parameters:**
- `embeddings` (np.ndarray): Array of shape (n_items, embedding_dim)
- `metric` (str): Similarity metric - "cosine", "euclidean", or "dot"

**Returns:**
- `np.ndarray`: Similarity matrix of shape (n_items, n_items)

---

#### cluster()

Main entry point: cluster texts into semantic groups.

```python
def cluster(
    texts: List[str],
    n_clusters: int,
    method: str = "kmeans",
    random_state: Optional[int] = None
) -> List[ClusterResult]
```

**Parameters:**
- `texts` (List[str]): List of texts to cluster
- `n_clusters` (int): Number of clusters to create
- `method` (str): Clustering method - "kmeans" or "hierarchical"
- `random_state` (int, optional): Random seed for reproducibility (kmeans only)

**Returns:**
- `List[ClusterResult]`: List of ClusterResult objects with names, summaries, and scores

**Example:**
```python
clusters = clustering.cluster(
    texts=["Python rocks", "I love Python", "Java is verbose"],
    n_clusters=2,
    method="kmeans",
    random_state=42
)
```

---

#### cluster_kmeans()

Cluster embeddings using K-means algorithm (internal method).

```python
def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]
```

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (cluster_labels, centroids)

---

#### cluster_hierarchical()

Cluster embeddings using hierarchical clustering (internal method).

```python
def cluster_hierarchical(
    embeddings: np.ndarray,
    n_clusters: int,
    linkage: str = "ward"
) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Cluster labels

---

#### score_cluster()

Compute quality score for a cluster based on cohesion.

```python
def score_cluster(
    embeddings: np.ndarray,
    centroid: np.ndarray
) -> float
```

**Returns:**
- `float`: Quality score between 0.0 and 1.0

---

### Design Decisions

#### Why Sentence Transformers?

**Advantages:**
- **Pre-trained:** No training required, works out-of-box
- **Fast:** Efficient inference on CPU
- **Accurate:** State-of-the-art semantic similarity
- **Flexible:** Multiple models for different speed/quality trade-offs

**Alternatives considered:**
- Word2Vec: Less accurate for semantic meaning
- BERT embeddings: Slower, requires more resources
- OpenAI embeddings: API cost, network latency

#### Why K-Means as Default?

K-means balances speed, simplicity, and effectiveness:
- **Speed:** Handles 1000+ items efficiently
- **Predictable:** Clear spherical clusters
- **Deterministic:** Reproducible with random seed

Hierarchical clustering offered as alternative for:
- Smaller datasets where speed less critical
- When dendrogram visualization desired
- Irregular cluster shapes

#### Embedding Caching Strategy

Caching embeddings improves performance for repeated clustering:
- **Speed improvement:** 10-100x faster for repeated texts
- **Memory trade-off:** ~1.5KB per cached text (384 dims × 4 bytes)
- **Use case:** Re-clustering with different K values

Cache disabled by default in memory-constrained environments.

#### Quality Scoring Rationale

Using average cosine similarity to centroid because:
- **Simple:** Easy to understand and interpret
- **Effective:** Correlates well with human perception of coherence
- **Fast:** O(n) computation
- **Standard:** Used in scikit-learn and research literature

---

### Best Practices

#### Choosing Number of Clusters

**Start with sqrt(n/2):**
```python
import math

n_items = len(texts)
n_clusters = max(2, math.floor(math.sqrt(n_items / 2)))
```

**Experiment with different values:**
```python
# Try multiple cluster counts
for k in range(2, 6):
    clusters = clustering.cluster(texts, n_clusters=k)
    avg_quality = sum(c.quality_score for c in clusters) / len(clusters)
    print(f"K={k}: Average quality = {avg_quality:.2f}")
```

**Guidelines by dataset size:**
- **5-10 items:** 2-3 clusters
- **10-20 items:** 3-5 clusters
- **20-50 items:** 4-8 clusters
- **50+ items:** sqrt(n/2) or quality-based selection

#### When to Use Which Method

**Use K-means when:**
- Dataset has >50 items
- Clusters are roughly equal size
- Speed is important
- You have a target number of themes

**Use hierarchical when:**
- Dataset has <50 items
- Cluster sizes vary significantly
- You want to explore different cluster counts
- Dendrogram visualization would be useful

#### Improving Cluster Quality

**Low quality scores? Try:**
1. **Adjust K:** Increase/decrease number of clusters
2. **Filter noise:** Remove very dissimilar items first
3. **Use better model:** Switch to "all-mpnet-base-v2" for higher quality
4. **Hierarchical method:** May handle irregular shapes better

**Example:**
```python
# Try different K values and pick best
best_k = None
best_quality = 0.0

for k in range(2, 8):
    clusters = clustering.cluster(texts, n_clusters=k)
    avg_quality = sum(c.quality_score for c in clusters) / len(clusters)

    if avg_quality > best_quality:
        best_quality = avg_quality
        best_k = k

print(f"Best K: {best_k} with quality {best_quality:.2f}")
```

#### Memory Management

**For large datasets:**
```python
# Disable caching if memory is limited
clustering = SemanticClustering(cache_embeddings=False)

# Process in batches
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    clusters = clustering.cluster(batch, n_clusters=k)
```

---

### Use Cases

#### Organizing Brainstormed Ideas (IDEATE)

**Problem:** 15 ideas from 3 perspectives, need thematic grouping

**Solution:**
```python
# After brainstorming, cluster ideas
analysis = await workflow.convergent_analysis(
    brainstorming_result=result,
    num_clusters=4
)
```

**Output:** 4 themed clusters (e.g., "Automation", "Collaboration", "Tooling", "Process")

**Benefit:** Easier to evaluate ideas when organized by theme

---

#### Organizing Argument Claims (ARGUMENT)

**Problem:** Complex debates have many claims, hard to track themes

**Solution:**
```python
# Cluster claims by topic
clusters = cluster_claims_kmeans(claims, n_clusters=3)

# Pro/Con clusters automatically separate
for cluster in clusters:
    print(f"Theme: {cluster[0].claim_text[:50]}")
```

**Output:** Claims grouped by topic (benefits, challenges, performance)

**Benefit:** See debate structure at a glance, identify which topics have most evidence

---

#### Research Finding Organization (Future)

**Problem:** RESEARCH workflow generates 5-10 findings, need thematic synthesis

**Solution:**
```python
# Cluster findings by theme
finding_texts = [step.content for step in result.steps]
clusters = clustering.cluster(finding_texts, n_clusters=3)

# Generate themed dossier
for cluster in clusters:
    print(f"\n## {cluster.name}")
    for idx in cluster.items:
        print(f"- Finding {idx+1}: {finding_texts[idx][:100]}...")
```

**Output:** Research dossier organized by theme

**Benefit:** More coherent synthesis, easier to understand overall picture

---

### Testing

#### Test Coverage

Comprehensive tests in `model_chorus/tests/test_clustering.py`:

**Core Functionality Tests:**
- Embedding computation with various model sizes
- Similarity computation (cosine, euclidean, dot)
- K-means clustering with different K values
- Hierarchical clustering with different linkage methods
- Cluster naming and summarization
- Quality scoring

**Edge Cases:**
- Empty input lists
- Single item clustering
- K > number of items
- Identical texts
- Cache behavior

**Integration Tests:**
- IDEATE workflow clustering integration
- ARGUMENT workflow clustering integration
- End-to-end clustering pipeline

#### Running Tests

```bash
# Run all clustering tests
pytest model_chorus/tests/test_clustering.py -v

# Run specific test
pytest model_chorus/tests/test_clustering.py::TestSemanticClustering::test_cluster_kmeans -v

# Run with coverage
pytest model_chorus/tests/test_clustering.py --cov=model_chorus.core.clustering
```

---

### Future Enhancements

**Planned improvements:**

1. **LLM-Based Naming:**
   - Use provider LLMs to generate meaningful cluster names
   - Example: "Machine Learning Applications" vs "Python is great"

2. **Dynamic K Selection:**
   - Automatically determine optimal number of clusters
   - Use silhouette score or elbow method
   - Reduce need for manual K tuning

3. **Multi-Language Support:**
   - Use multilingual sentence transformers
   - Enable clustering across languages
   - Useful for international workflows

4. **Incremental Clustering:**
   - Add new items to existing clusters without re-clustering
   - Useful for streaming workflows
   - Maintain cluster stability

5. **Hierarchical Dendrogram Visualization:**
   - Generate visual dendrograms for hierarchical clustering
   - Export to PNG/SVG
   - Interactive exploration of cluster hierarchy

6. **Cluster Comparison:**
   - Compare clustering results with different K or methods
   - Measure cluster stability across runs
   - Help users choose best clustering

---

## Role-Based Orchestration Framework

### Overview

The Role-Based Orchestration Framework provides infrastructure for coordinating multiple AI models in specific roles, enabling sophisticated multi-model workflows. Each model can be assigned a distinct role (e.g., proponent, critic, analyst) with customized prompts and execution patterns, allowing workflows to leverage diverse perspectives and specialized behaviors.

**Key Capabilities:**
- **Role Assignment:** Assign specific roles to different models with customized prompts
- **Stance Configuration:** Define stances (for/against/neutral) for debate-style workflows
- **Execution Patterns:** Sequential, parallel, or hybrid orchestration
- **Synthesis Strategies:** Combine role outputs using concatenation, structured formatting, or AI synthesis
- **Provider Flexibility:** Map any model to any role with per-role parameter overrides

**Supported Workflows:**
- **ARGUMENT:** Sequential debate with Creator (for), Skeptic (against), and Moderator (synthesis)
- **IDEATE:** Parallel brainstorming from multiple creative perspectives
- **RESEARCH:** Multi-angle investigation with specialized research roles (planned)

**Location:** `model_chorus/src/model_chorus/core/role_orchestration.py`

---

### Architecture

#### Core Components

The orchestration framework consists of four primary components:

1. **ModelRole:** Defines a model's role, stance, and prompt customization
2. **RoleOrchestrator:** Coordinates execution of models in assigned roles
3. **OrchestrationPattern:** Defines execution strategy (sequential/parallel/hybrid)
4. **SynthesisStrategy:** Determines how to combine role outputs

#### Data Models

##### ModelRole

Represents a specific role assignment for an AI model in a multi-model workflow.

**Fields:**
- `role` (str, required): Descriptive name for this role
  - Examples: "proponent", "critic", "synthesizer", "analyst"
  - Min length: 1, Max length: 100
- `model` (str, required): Model identifier to assign to this role
  - Examples: "gpt-5", "gemini-2.5-pro", "claude-sonnet-4"
- `stance` (str, optional): Stance for debate-style workflows
  - Allowed values: "for", "against", "neutral"
  - Automatically validated and normalized to lowercase
- `stance_prompt` (str, optional): Additional prompt text to reinforce the stance
  - Appended to base prompt to guide model behavior
- `system_prompt` (str, optional): System-level prompt for this role
  - Sets overall behavior and expertise context
- `temperature` (float, optional): Temperature override for this role
  - Range: 0.0 to 1.0
  - Overrides workflow default for role-specific creativity
- `max_tokens` (int, optional): Max tokens override for this role
  - Overrides workflow default for role-specific response length
- `metadata` (dict, optional): Additional metadata
  - Common fields: `priority`, `tags`, `constraints`

**Example:**
```python
from model_chorus.core.role_orchestration import ModelRole

proponent = ModelRole(
    role="proponent",
    model="gpt-5",
    stance="for",
    stance_prompt="You are advocating FOR the proposal. Present strong supporting arguments.",
    system_prompt="You are an expert debater focused on building compelling cases.",
    temperature=0.8,
    max_tokens=4000,
    metadata={
        "priority": 1,
        "tags": ["debate", "advocacy"]
    }
)

critic = ModelRole(
    role="critic",
    model="gemini-2.5-pro",
    stance="against",
    stance_prompt="You are critically analyzing AGAINST the proposal. Identify weaknesses and risks.",
    system_prompt="You are a skeptical analyst who challenges assumptions.",
    temperature=0.7,
    max_tokens=3500
)
```

##### OrchestrationResult

Result from orchestrating multiple models with assigned roles.

**Fields:**
- `role_responses` (List[tuple], optional): List of (role_name, response) tuples in execution order
- `all_responses` (List, optional): List of all GenerationResponse objects
- `failed_roles` (List[str], optional): List of role names that failed to execute
- `pattern_used` (OrchestrationPattern, required): Orchestration pattern that was executed
- `execution_order` (List[str], optional): List of role names in the order they were executed
- `synthesized_output` (Any, optional): Synthesized/combined output (if synthesis enabled)
- `synthesis_strategy` (SynthesisStrategy, optional): Strategy used for synthesis
- `metadata` (dict, optional): Additional execution metadata
  - Common fields: `total_roles`, `successful_roles`, `failed_roles`, `timing`, `synthesis_metadata`

**Example:**
```python
from model_chorus.core.role_orchestration import OrchestrationResult, OrchestrationPattern

result = OrchestrationResult(
    role_responses=[
        ("proponent", GenerationResponse(content="Argument FOR...", model="gpt-5")),
        ("critic", GenerationResponse(content="Argument AGAINST...", model="gemini-2.5-pro")),
    ],
    pattern_used=OrchestrationPattern.SEQUENTIAL,
    execution_order=["proponent", "critic"],
    synthesized_output="After considering both perspectives...",
    synthesis_strategy=SynthesisStrategy.AI_SYNTHESIZE,
    metadata={
        "total_roles": 2,
        "successful_roles": 2,
        "failed_roles": 0
    }
)
```

---

#### Orchestration Patterns

The framework supports three execution patterns, each optimized for different workflow characteristics:

##### Sequential Pattern

**How it works:**
1. Execute first role
2. Wait for completion
3. Execute second role
4. Repeat until all roles complete
5. Return results in execution order

**Characteristics:**
- **Sequential dependency:** Each role can build on previous outputs
- **Ordered execution:** Deterministic, predictable order
- **Context building:** Later roles can reference earlier responses
- **Slower:** Total time = sum of all role execution times

**Best for:**
- Debate workflows where roles respond to each other
- Multi-stage analysis (research → analysis → synthesis)
- When later roles need context from earlier roles

**Example:**
```python
from model_chorus.core.role_orchestration import RoleOrchestrator, OrchestrationPattern

# Sequential debate: Creator → Skeptic → Moderator
orchestrator = RoleOrchestrator(
    roles=[creator_role, skeptic_role, moderator_role],
    provider_map=providers,
    pattern=OrchestrationPattern.SEQUENTIAL
)

result = await orchestrator.execute("Should we adopt TypeScript?")
# Creator generates thesis first
# Skeptic responds to creator's thesis
# Moderator synthesizes both perspectives
```

##### Parallel Pattern

**How it works:**
1. Launch all roles simultaneously using asyncio.gather
2. Wait for all to complete
3. Sort results by original role order
4. Return all responses

**Characteristics:**
- **Concurrent execution:** All roles execute at once
- **Independent perspectives:** Roles don't see each other's outputs
- **Faster:** Total time ≈ slowest role execution time
- **Unordered internally:** Results sorted by index after completion

**Best for:**
- Multiple independent perspectives (brainstorming, research)
- When roles don't need each other's context
- Maximizing throughput for large workflows
- Diverse expert opinions

**Example:**
```python
# Parallel brainstorming: Multiple perspectives simultaneously
orchestrator = RoleOrchestrator(
    roles=[perspective1, perspective2, perspective3],
    provider_map=providers,
    pattern=OrchestrationPattern.PARALLEL
)

result = await orchestrator.execute("Ways to improve developer productivity")
# All three perspectives generate ideas simultaneously
# Results combined after all complete
```

##### Hybrid Pattern (Planned)

**How it works:**
1. Mix sequential and parallel phases
2. Example: Parallel research → Sequential debate → Parallel voting

**Status:** Not yet implemented

**Characteristics:**
- **Flexible orchestration:** Combine sequential and parallel benefits
- **Complex workflows:** Multi-phase workflows with different patterns per phase
- **Performance optimization:** Parallel where possible, sequential where needed

**Planned for:** Future releases

---

#### Synthesis Strategies

After roles execute, their outputs can be synthesized using different strategies:

##### NONE

**Description:** No synthesis - return raw responses as-is

**Use when:**
- You want to process each response individually
- Custom synthesis logic needed
- Displaying responses separately

**Output:** Original `role_responses` list unchanged

**Example:**
```python
result = await orchestrator.execute(
    prompt="Should we adopt TypeScript?",
)
# No synthesis performed - default
for role_name, response in result.role_responses:
    print(f"{role_name}: {response.content}")
```

##### CONCATENATE

**Description:** Simple concatenation with role labels and separators

**Use when:**
- You want a readable combined text
- Clear attribution to each role needed
- Simple text output sufficient

**Output:** Single string with role-labeled sections

**Format:**
```
## PROPONENT

[Proponent's full response]

---

## CRITIC

[Critic's full response]
```

**Example:**
```python
result = await orchestrator.execute(
    prompt="Should we adopt TypeScript?",
)
synthesized = await orchestrator.synthesize(result, SynthesisStrategy.CONCATENATE)
print(synthesized.synthesized_output)
```

##### STRUCTURED

**Description:** Combine into structured dictionary with role keys

**Use when:**
- Programmatic access to individual responses needed
- Preserving response metadata (model, usage)
- Building APIs or data pipelines

**Output:** Dictionary with role names as keys

**Format:**
```python
{
    "proponent": {
        "content": "TypeScript provides...",
        "model": "gpt-5",
        "usage": {...}
    },
    "critic": {
        "content": "TypeScript adds complexity...",
        "model": "gemini-2.5-pro",
        "usage": {...}
    }
}
```

**Example:**
```python
synthesized = await orchestrator.synthesize(result, SynthesisStrategy.STRUCTURED)
proponent_content = synthesized.synthesized_output["proponent"]["content"]
```

##### AI_SYNTHESIZE

**Description:** Use AI model to intelligently combine responses

**Use when:**
- You want a coherent synthesis that resolves conflicts
- Integrating multiple perspectives into unified recommendation
- Producing publication-quality output

**Output:** AI-generated synthesis incorporating all perspectives

**Process:**
1. Build synthesis prompt with all role responses
2. Send to synthesis provider (default: first role's provider)
3. AI model integrates perspectives, identifies agreements/disagreements
4. Returns coherent, balanced final response

**Example:**
```python
synthesized = await orchestrator.synthesize(
    result,
    strategy=SynthesisStrategy.AI_SYNTHESIZE,
    synthesis_provider=claude_provider,  # Optional: specify synthesis model
    synthesis_prompt="Custom synthesis instructions..."  # Optional: override default
)
print(synthesized.synthesized_output)
# "After considering both perspectives, TypeScript offers strong benefits..."
```

---

### Usage Examples

#### Basic Sequential Debate (ARGUMENT)

**Two-role debate:**
```python
from model_chorus.core.role_orchestration import RoleOrchestrator, ModelRole, OrchestrationPattern
from model_chorus.providers import ClaudeProvider, GeminiProvider

# Define roles
proponent = ModelRole(
    role="proponent",
    model="claude",
    stance="for",
    stance_prompt="You are advocating FOR the proposal. Present strong supporting arguments.",
    temperature=0.8
)

critic = ModelRole(
    role="critic",
    model="gemini",
    stance="against",
    stance_prompt="You are critically analyzing AGAINST the proposal. Identify weaknesses and risks.",
    temperature=0.7
)

# Create provider map
providers = {
    "claude": ClaudeProvider(),
    "gemini": GeminiProvider()
}

# Create orchestrator
orchestrator = RoleOrchestrator(
    roles=[proponent, critic],
    provider_map=providers,
    pattern=OrchestrationPattern.SEQUENTIAL
)

# Execute debate
result = await orchestrator.execute("Should we adopt TypeScript for our project?")

# Display responses
for role_name, response in result.role_responses:
    print(f"\n## {role_name.upper()}")
    print(response.content)
```

#### Three-Role Sequential Analysis

**Analysis → Critique → Synthesis pattern:**
```python
# Define three roles
analyst = ModelRole(
    role="analyst",
    model="gpt-5",
    stance="neutral",
    stance_prompt="Analyze objectively, present facts and data.",
    temperature=0.5
)

skeptic = ModelRole(
    role="skeptic",
    model="gemini-2.5-pro",
    stance="against",
    stance_prompt="Challenge assumptions, identify risks and weaknesses.",
    temperature=0.6
)

synthesizer = ModelRole(
    role="synthesizer",
    model="claude-sonnet-4",
    stance="neutral",
    stance_prompt="Integrate both perspectives into balanced recommendation.",
    temperature=0.7
)

orchestrator = RoleOrchestrator(
    roles=[analyst, skeptic, synthesizer],
    provider_map=providers,
    pattern=OrchestrationPattern.SEQUENTIAL
)

# Execute with AI synthesis
result = await orchestrator.execute("Evaluate moving to microservices architecture")
synthesized = await orchestrator.synthesize(result, SynthesisStrategy.AI_SYNTHESIZE)

print(synthesized.synthesized_output)
```

#### Parallel Brainstorming (IDEATE)

**Multiple independent perspectives:**
```python
# Define multiple perspective roles
perspectives = [
    ModelRole(
        role="efficiency_expert",
        model="gpt-5",
        stance="neutral",
        stance_prompt="Focus on efficiency and optimization opportunities.",
        temperature=0.8
    ),
    ModelRole(
        role="ux_designer",
        model="gemini-2.5-pro",
        stance="neutral",
        stance_prompt="Focus on user experience and usability improvements.",
        temperature=0.9
    ),
    ModelRole(
        role="technical_architect",
        model="claude-sonnet-4",
        stance="neutral",
        stance_prompt="Focus on technical feasibility and architecture.",
        temperature=0.7
    )
]

# Parallel execution for speed
orchestrator = RoleOrchestrator(
    roles=perspectives,
    provider_map=providers,
    pattern=OrchestrationPattern.PARALLEL
)

result = await orchestrator.execute("Brainstorm ways to improve developer productivity")

# Structured output for programmatic access
synthesized = await orchestrator.synthesize(result, SynthesisStrategy.STRUCTURED)

for role_name, data in synthesized.synthesized_output.items():
    print(f"\n{role_name}: {data['content'][:200]}...")
```

#### ARGUMENT Workflow Integration

**How ARGUMENT uses RoleOrchestrator:**
```python
from model_chorus.workflows import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider

# ArgumentWorkflow internally creates three roles:
# 1. Creator (stance: "for") - Generates thesis
# 2. Skeptic (stance: "against") - Provides rebuttal
# 3. Moderator (stance: "neutral") - Synthesizes perspectives

workflow = ArgumentWorkflow(provider=ClaudeProvider())

result = await workflow.run(
    prompt="TypeScript should replace JavaScript for all new projects",
    temperature=0.7
)

# Workflow uses RoleOrchestrator internally:
# - Sequential pattern (roles build on each other)
# - Custom stance prompts per role
# - Final synthesis via Moderator role

print(result.synthesis)  # Balanced dialectical analysis
```

#### Custom Role Configuration

**Advanced role customization:**
```python
custom_role = ModelRole(
    role="devil's_advocate",
    model="gpt-5",
    stance="against",
    stance_prompt=(
        "You are playing devil's advocate. Your goal is to find flaws, "
        "challenge assumptions, and present worst-case scenarios. Be relentlessly "
        "critical and skeptical."
    ),
    system_prompt="You are a contrarian thinker who questions conventional wisdom.",
    temperature=0.9,  # Higher creativity for diverse criticism
    max_tokens=5000,  # Allow detailed critiques
    metadata={
        "priority": "high",
        "tags": ["critical_thinking", "risk_analysis"],
        "constraints": ["focus_on_risks", "identify_blind_spots"]
    }
)

# Role can access full prompt via get_full_prompt()
base_prompt = "Evaluate this proposal: ..."
full_prompt = custom_role.get_full_prompt(base_prompt)

# Output combines: system_prompt + stance_prompt + base_prompt
print(full_prompt)
```

---

### API Reference

#### ModelRole.__init__()

Create a new role assignment for an AI model.

```python
def __init__(
    role: str,
    model: str,
    stance: Optional[str] = None,
    stance_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    metadata: Dict[str, Any] = {}
)
```

**Parameters:**
- `role` (str): Descriptive name for this role (1-100 chars)
- `model` (str): Model identifier ("gpt-5", "gemini-2.5-pro", etc.)
- `stance` (str, optional): "for", "against", or "neutral"
- `stance_prompt` (str, optional): Additional prompt to reinforce stance
- `system_prompt` (str, optional): System-level behavior prompt
- `temperature` (float, optional): Temperature override (0.0-1.0)
- `max_tokens` (int, optional): Max tokens override
- `metadata` (dict, optional): Additional metadata

**Raises:**
- `ValueError`: If stance not in {"for", "against", "neutral"}
- `ValueError`: If temperature outside [0.0, 1.0]

---

#### ModelRole.get_full_prompt()

Construct full prompt by combining base prompt with role customizations.

```python
def get_full_prompt(base_prompt: str) -> str
```

**Parameters:**
- `base_prompt` (str): The base prompt from the workflow

**Returns:**
- `str`: Complete prompt with system_prompt + stance_prompt + base_prompt

**Example:**
```python
full = role.get_full_prompt("Analyze this proposal")
# Returns: "{system_prompt}\n\n{stance_prompt}\n\n{base_prompt}"
```

---

#### RoleOrchestrator.__init__()

Initialize the role orchestrator.

```python
def __init__(
    roles: List[ModelRole],
    provider_map: Dict[str, Any],
    pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
    default_timeout: float = 120.0
)
```

**Parameters:**
- `roles` (List[ModelRole]): List of ModelRole instances defining the workflow
- `provider_map` (Dict[str, Any]): Mapping from model identifiers to provider instances
- `pattern` (OrchestrationPattern): Execution pattern (default: SEQUENTIAL)
- `default_timeout` (float): Default timeout per model in seconds (default: 120.0)

**Raises:**
- `ValueError`: If roles list is empty
- `ValueError`: If pattern is HYBRID (not yet implemented)

---

#### RoleOrchestrator.execute()

Execute the orchestrated workflow with all roles.

```python
async def execute(
    base_prompt: str,
    context: Optional[str] = None
) -> OrchestrationResult
```

**Parameters:**
- `base_prompt` (str): The base prompt to send to all models
- `context` (Optional[str]): Additional context to include (e.g., from previous execution)

**Returns:**
- `OrchestrationResult`: Structured result with all responses and metadata

**Execution behavior:**
- Sequential: Executes roles one at a time in order
- Parallel: Executes all roles concurrently with asyncio.gather

**Example:**
```python
result = await orchestrator.execute(
    base_prompt="Should we adopt microservices?",
    context="Previous discussion established our current monolith has scaling issues."
)
```

---

#### RoleOrchestrator.synthesize()

Synthesize multiple role outputs into a unified result.

```python
async def synthesize(
    result: OrchestrationResult,
    strategy: SynthesisStrategy = SynthesisStrategy.CONCATENATE,
    synthesis_provider: Optional[Any] = None,
    synthesis_prompt: Optional[str] = None
) -> OrchestrationResult
```

**Parameters:**
- `result` (OrchestrationResult): Result from execute()
- `strategy` (SynthesisStrategy): How to combine outputs
- `synthesis_provider` (Optional[Any]): Provider for AI synthesis (defaults to first role's provider)
- `synthesis_prompt` (Optional[str]): Custom synthesis prompt (overrides default)

**Returns:**
- `OrchestrationResult`: Updated result with synthesized_output field populated

**Synthesis behaviors:**
- NONE: Returns original result unchanged
- CONCATENATE: Simple text concatenation with role labels
- STRUCTURED: Dictionary with role keys
- AI_SYNTHESIZE: AI-generated integration (falls back to CONCATENATE on error)

---

### Design Decisions

#### Why Role-Based Abstraction?

**Benefits:**
- **Flexibility:** Any model can play any role
- **Clarity:** Role names communicate purpose (proponent vs critic)
- **Customization:** Per-role prompts, temperatures, and constraints
- **Reusability:** Same roles work across different workflows

**Alternatives considered:**
- Hard-coded model assignments: Less flexible, tight coupling
- Workflow-specific implementations: Code duplication across workflows
- Simple multi-model calls: No role semantics, harder to understand

#### Why Sequential and Parallel Patterns?

**Sequential pattern enables:**
- Context building (later roles reference earlier outputs)
- Dialectical reasoning (thesis → antithesis → synthesis)
- Multi-stage analysis (research → analysis → recommendation)

**Parallel pattern enables:**
- Independent diverse perspectives
- Faster execution (concurrent API calls)
- Brainstorming and ideation workflows

**Hybrid pattern (future) enables:**
- Best of both: parallel where possible, sequential where needed
- Complex multi-phase workflows
- Performance optimization

#### Synthesis Strategy Flexibility

Different workflows need different synthesis approaches:
- **ARGUMENT:** AI synthesis for balanced dialectical conclusion
- **IDEATE:** Structured output for programmatic idea processing
- **RESEARCH:** Concatenation for comprehensive evidence review

Providing multiple strategies allows workflows to choose the best approach for their use case.

#### Provider Map Design

**Why explicit provider map?**
- **Flexibility:** Multiple models from same provider (gpt-5-mini, gpt-5-pro)
- **Testing:** Easy to mock providers for testing
- **Configuration:** Centralized provider management
- **Fallbacks:** Can implement fallback logic in provider map

**Alternative considered:**
- Auto-resolve providers from model names: Less explicit, harder to test

---

### Best Practices

#### Choosing Orchestration Pattern

**Use Sequential when:**
- Roles need to respond to each other
- Building context across multiple stages
- Debate or dialectical workflows
- Order matters for workflow logic

**Use Parallel when:**
- Roles are independent
- Speed is important
- Brainstorming or diverse perspectives
- Order doesn't matter for workflow logic

**Example decision:**
```python
# Debate: Use sequential (Skeptic needs to respond to Proponent)
pattern = OrchestrationPattern.SEQUENTIAL

# Brainstorming: Use parallel (all perspectives independent)
pattern = OrchestrationPattern.PARALLEL
```

#### Crafting Effective Role Prompts

**Good stance prompts:**
- Clear role definition
- Specific behavioral guidance
- Examples of desired output
- Appropriate tone/style

**Example:**
```python
# ✅ Good: Clear, specific, actionable
stance_prompt = (
    "You are a critical security analyst. Review the proposal for security "
    "vulnerabilities, threat vectors, and compliance risks. For each risk identified, "
    "assess severity (critical/high/medium/low) and provide concrete mitigation strategies."
)

# ❌ Bad: Vague, generic
stance_prompt = "You are a critic. Find problems."
```

#### Temperature Tuning by Role

Different roles benefit from different creativity levels:

**Conservative (0.3-0.5):**
- Analyst roles requiring factual accuracy
- Technical evaluation
- Compliance reviews

**Balanced (0.6-0.8):**
- General debate and discussion
- Balanced analysis
- Standard workflows (default)

**Creative (0.8-1.0):**
- Brainstorming and ideation
- Creative problem solving
- Exploring unconventional approaches

**Example:**
```python
roles = [
    ModelRole(role="analyst", model="gpt-5", temperature=0.4),      # Factual
    ModelRole(role="creative", model="gemini", temperature=0.9),    # Innovative
    ModelRole(role="synthesizer", model="claude", temperature=0.7)  # Balanced
]
```

#### Error Handling

**Orchestrator continues on role failures:**
```python
result = await orchestrator.execute("Analyze proposal")

# Check for failures
if result.failed_roles:
    print(f"Failed roles: {', '.join(result.failed_roles)}")
    print(f"Successful: {len(result.role_responses)}/{len(orchestrator.roles)}")

# Access successful responses
for role_name, response in result.role_responses:
    # Only successful roles appear here
    print(f"{role_name}: {response.content}")
```

#### Provider Resolution

**Map variations automatically:**
```python
provider_map = {
    "gpt5": openai_provider,
    "gpt-5": openai_provider,    # Handles both formats
    "gemini": gemini_provider,
    "gemini-2.5-pro": gemini_provider
}

# These all resolve to the same provider
ModelRole(role="role1", model="gpt5")
ModelRole(role="role2", model="gpt-5")
ModelRole(role="role3", model="gpt_5")  # Underscores also handled
```

---

### Use Cases

#### Structured Debate (ARGUMENT)

**Problem:** Need balanced analysis of controversial proposal

**Solution:**
```python
# Three-role debate
orchestrator = RoleOrchestrator(
    roles=[
        ModelRole(role="proponent", model="gpt-5", stance="for"),
        ModelRole(role="critic", model="gemini", stance="against"),
        ModelRole(role="moderator", model="claude", stance="neutral")
    ],
    provider_map=providers,
    pattern=OrchestrationPattern.SEQUENTIAL
)

result = await orchestrator.execute("Should we adopt microservices?")
synthesized = await orchestrator.synthesize(result, SynthesisStrategy.AI_SYNTHESIZE)
```

**Output:** Balanced dialectical analysis considering both perspectives

**Benefit:** Structured reasoning prevents confirmation bias

---

#### Multi-Perspective Brainstorming (IDEATE)

**Problem:** Need diverse ideas from different viewpoints

**Solution:**
```python
# Parallel perspectives
orchestrator = RoleOrchestrator(
    roles=[
        ModelRole(role="technical", model="gpt-5", temperature=0.8),
        ModelRole(role="business", model="gemini", temperature=0.9),
        ModelRole(role="user_experience", model="claude", temperature=0.85)
    ],
    provider_map=providers,
    pattern=OrchestrationPattern.PARALLEL
)

result = await orchestrator.execute("Ways to improve app performance")
```

**Output:** Diverse ideas from technical, business, and UX perspectives simultaneously

**Benefit:** Faster execution, diverse perspectives, no anchoring bias

---

#### Multi-Stage Analysis

**Problem:** Need comprehensive analysis with multiple stages

**Solution:**
```python
# Research → Analysis → Recommendation
orchestrator = RoleOrchestrator(
    roles=[
        ModelRole(role="researcher", model="gpt-5", temperature=0.5),
        ModelRole(role="analyst", model="gemini", temperature=0.6),
        ModelRole(role="strategist", model="claude", temperature=0.7)
    ],
    provider_map=providers,
    pattern=OrchestrationPattern.SEQUENTIAL
)

result = await orchestrator.execute("Evaluate cloud migration options")
```

**Output:** Research findings → Analysis of findings → Strategic recommendation

**Benefit:** Structured, multi-stage reasoning with context building

---

### Testing

#### Test Coverage

Comprehensive tests in `model_chorus/tests/test_role_orchestration.py`:

**ModelRole Tests:**
- Role creation with all fields
- Stance validation (for/against/neutral)
- Temperature validation (0.0-1.0)
- Prompt construction via get_full_prompt()
- Metadata handling

**RoleOrchestrator Tests:**
- Sequential execution
- Parallel execution
- Provider resolution with variations
- Error handling and partial failures
- Execution order preservation

**Synthesis Tests:**
- NONE strategy
- CONCATENATE strategy
- STRUCTURED strategy
- AI_SYNTHESIZE strategy
- Fallback behavior on AI synthesis failure

**Integration Tests:**
- ARGUMENT workflow integration
- IDEATE workflow integration
- End-to-end orchestration with real providers

#### Running Tests

```bash
# Run all orchestration tests
pytest model_chorus/tests/test_role_orchestration.py -v

# Run specific test
pytest model_chorus/tests/test_role_orchestration.py::TestRoleOrchestrator::test_sequential_execution -v

# Run with coverage
pytest model_chorus/tests/test_role_orchestration.py --cov=model_chorus.core.role_orchestration
```

---

### Future Enhancements

**Planned improvements:**

1. **Hybrid Pattern Implementation:**
   - Mix sequential and parallel phases
   - Example: Parallel research → Sequential debate → Parallel voting
   - Performance optimization for complex workflows

2. **Dynamic Role Assignment:**
   - Assign roles based on model capabilities
   - Auto-select best model for each role
   - Capability-based routing

3. **Role Dependencies:**
   - Explicit dependencies between roles (DAG)
   - Conditional execution based on previous results
   - Branching logic

4. **Enhanced Synthesis:**
   - Multiple synthesis models for comparison
   - Synthesis quality scoring
   - Consensus detection across roles

5. **Execution Monitoring:**
   - Real-time progress updates
   - Per-role timing and metrics
   - Execution visualization

6. **Role Templates:**
   - Pre-configured roles for common use cases
   - Role libraries (analyst, critic, synthesizer, etc.)
   - Easy workflow composition

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

The citation engine has comprehensive test coverage in `model_chorus/tests/test_citation.py`:

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
pytest model_chorus/tests/test_citation.py -v

# Run specific test class
pytest model_chorus/tests/test_citation.py::TestCitation -v
pytest model_chorus/tests/test_citation.py::TestCitationMap -v

# Run with coverage
pytest model_chorus/tests/test_citation.py --cov=model_chorus.core.models --cov=model_chorus.utils.citation_formatter
```

#### Test Examples

See `model_chorus/tests/test_citation.py` for extensive examples of:
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
- `model_chorus/src/model_chorus/utils/citation_formatter.py` - Citation formatting implementation
- `model_chorus/src/model_chorus/core/models.py` - Citation and CitationMap models
- `model_chorus/tests/test_citation.py` - Comprehensive test suite

**API Reference:**
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Complete API documentation
- [README.md](../../README.md) - Getting started guide
