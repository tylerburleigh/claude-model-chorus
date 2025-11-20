"""
Citation formatting utilities for ModelChorus ARGUMENT workflow.

Provides formatting functions to convert Citation objects into standard
citation formats (APA, MLA, Chicago) for academic and professional use.

Also includes validation and confidence scoring utilities for citation quality
assessment and verification.
"""

from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class CitationStyle(str, Enum):
    """Supported citation formatting styles."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"


def format_citation(
    citation: "Citation",  # type: ignore # Forward reference to avoid circular import
    style: CitationStyle = CitationStyle.APA,
) -> str:
    """
    Format a Citation object according to the specified style.

    Args:
        citation: The Citation object to format
        style: The citation style to use (APA, MLA, or Chicago)

    Returns:
        Formatted citation string according to the specified style

    Example:
        >>> from model_chorus.core.models import Citation
        >>> from model_chorus.utils.citation_formatter import format_citation, CitationStyle
        >>> c = Citation(
        ...     source="https://arxiv.org/abs/2401.12345",
        ...     confidence=0.95,
        ...     metadata={"author": "Smith, J.", "year": "2024", "title": "Machine Learning"}
        ... )
        >>> format_citation(c, CitationStyle.APA)
        'Smith, J. (2024). Machine Learning. Retrieved from https://arxiv.org/abs/2401.12345'
    """
    if style == CitationStyle.APA:
        return _format_apa(citation)
    elif style == CitationStyle.MLA:
        return _format_mla(citation)
    elif style == CitationStyle.CHICAGO:
        return _format_chicago(citation)
    else:
        raise ValueError(f"Unsupported citation style: {style}")


def _format_apa(citation: "Citation") -> str:  # type: ignore
    """
    Format citation in APA style.

    APA format: Author(s). (Year). Title. Source.

    Args:
        citation: The Citation object to format

    Returns:
        APA-formatted citation string
    """
    parts = []

    # Author
    author = citation.metadata.get("author")
    if author:
        parts.append(f"{author}")

    # Year
    year = citation.metadata.get("year") or citation.metadata.get("publication_date", "")
    if year:
        # Extract year if full date provided
        year_str = str(year)[:4] if len(str(year)) > 4 else str(year)
        parts.append(f"({year_str})")

    # Title
    title = citation.metadata.get("title")
    if title:
        parts.append(f"{title}.")

    # Source/Location
    source_part = f"Retrieved from {citation.source}"
    if citation.location:
        source_part += f" ({citation.location})"
    parts.append(source_part)

    return ". ".join(parts) if parts else citation.source


def _format_mla(citation: "Citation") -> str:  # type: ignore
    """
    Format citation in MLA style.

    MLA format: Author(s). "Title." Source, Year. Location.

    Args:
        citation: The Citation object to format

    Returns:
        MLA-formatted citation string
    """
    parts = []

    # Author
    author = citation.metadata.get("author")
    if author:
        parts.append(f"{author}.")

    # Title (in quotes)
    title = citation.metadata.get("title")
    if title:
        parts.append(f'"{title}."')

    # Source
    parts.append(f"{citation.source},")

    # Year
    year = citation.metadata.get("year") or citation.metadata.get("publication_date", "")
    if year:
        year_str = str(year)[:4] if len(str(year)) > 4 else str(year)
        parts.append(f"{year_str}.")

    # Location
    if citation.location:
        parts.append(citation.location)

    return " ".join(parts) if parts else citation.source


def _format_chicago(citation: "Citation") -> str:  # type: ignore
    """
    Format citation in Chicago style.

    Chicago format: Author(s). "Title." Source (Year): Location.

    Args:
        citation: The Citation object to format

    Returns:
        Chicago-formatted citation string
    """
    parts = []

    # Author
    author = citation.metadata.get("author")
    if author:
        parts.append(f"{author}.")

    # Title (in quotes)
    title = citation.metadata.get("title")
    if title:
        parts.append(f'"{title}."')

    # Source
    source_part = citation.source

    # Year (in parentheses after source)
    year = citation.metadata.get("year") or citation.metadata.get("publication_date", "")
    if year:
        year_str = str(year)[:4] if len(str(year)) > 4 else str(year)
        source_part += f" ({year_str})"

    # Location (after colon)
    if citation.location:
        source_part += f": {citation.location}"

    parts.append(source_part)

    return " ".join(parts) if parts else citation.source


def format_citation_map(
    citation_map: "CitationMap",  # type: ignore
    style: CitationStyle = CitationStyle.APA,
    include_claim: bool = True,
) -> str:
    """
    Format a CitationMap object with all its citations.

    Args:
        citation_map: The CitationMap object to format
        style: The citation style to use (APA, MLA, or Chicago)
        include_claim: Whether to include the claim text in the output

    Returns:
        Formatted string with claim and all citations

    Example:
        >>> formatted = format_citation_map(cm, CitationStyle.APA, include_claim=True)
        >>> print(formatted)
        Claim: Machine learning improves accuracy by 23%

        Citations:
        1. Smith, J. (2024). ML Research. Retrieved from https://arxiv.org/abs/2401.12345
        2. Doe, A. (2024). AI Studies. Retrieved from paper2.pdf
    """
    lines = []

    if include_claim:
        lines.append(f"Claim: {citation_map.claim_text}")
        lines.append("")  # Blank line

    if citation_map.citations:
        lines.append("Citations:")
        for idx, citation in enumerate(citation_map.citations, start=1):
            formatted = format_citation(citation, style)
            lines.append(f"{idx}. {formatted}")
    else:
        lines.append("No citations available")

    return "\n".join(lines)


# ============================================================================
# Citation Validation and Confidence Scoring
# ============================================================================


def validate_citation(citation: "Citation") -> Tuple[bool, List[str]]:  # type: ignore
    """
    Validate a Citation object for completeness and quality.

    Args:
        citation: The Citation object to validate

    Returns:
        Tuple of (is_valid, issues) where:
        - is_valid: True if citation meets minimum requirements
        - issues: List of validation issue messages

    Example:
        >>> is_valid, issues = validate_citation(citation)
        >>> if not is_valid:
        ...     print(f"Validation issues: {', '.join(issues)}")
    """
    issues = []

    # Check required fields
    if not citation.source or not citation.source.strip():
        issues.append("Missing or empty source")

    if citation.confidence < 0.0 or citation.confidence > 1.0:
        issues.append(f"Confidence {citation.confidence} out of valid range [0.0, 1.0]")

    # Check recommended metadata for quality
    metadata = citation.metadata or {}

    if not metadata.get("author"):
        issues.append("Missing author metadata (recommended)")

    if not metadata.get("year") and not metadata.get("publication_date"):
        issues.append("Missing year/publication_date metadata (recommended)")

    if not metadata.get("title"):
        issues.append("Missing title metadata (recommended)")

    # Check source format
    source_lower = citation.source.lower()
    if not (
        source_lower.startswith("http://")
        or source_lower.startswith("https://")
        or source_lower.endswith((".pdf", ".doc", ".docx", ".txt"))
        or "doi:" in source_lower
    ):
        issues.append("Source format not recognized (should be URL, file path, or DOI)")

    is_valid = len(issues) == 0
    return is_valid, issues


def calculate_citation_confidence(citation: "Citation") -> Dict[str, Any]:  # type: ignore
    """
    Calculate a detailed confidence score for a citation's reliability.

    Evaluates multiple factors:
    - Base confidence score (from citation.confidence)
    - Metadata completeness (author, year, title presence)
    - Source quality (URL vs file, academic domains)
    - Location specificity (page numbers, sections)

    Args:
        citation: The Citation object to score

    Returns:
        Dictionary with:
        - overall_confidence: Final confidence score (0.0-1.0)
        - base_confidence: Original confidence value
        - metadata_score: Completeness score for metadata (0.0-1.0)
        - source_quality_score: Quality score for source type (0.0-1.0)
        - location_score: Specificity score for location (0.0-1.0)
        - factors: Detailed breakdown of scoring factors

    Example:
        >>> scores = calculate_citation_confidence(citation)
        >>> print(f"Overall confidence: {scores['overall_confidence']:.2f}")
        >>> print(f"Metadata completeness: {scores['metadata_score']:.2f}")
    """
    metadata = citation.metadata or {}

    # Base confidence from citation
    base_confidence = citation.confidence

    # Metadata completeness score
    metadata_factors = {
        "has_author": bool(metadata.get("author")),
        "has_year": bool(metadata.get("year") or metadata.get("publication_date")),
        "has_title": bool(metadata.get("title")),
        "has_snippet": bool(citation.snippet),
    }
    metadata_score = sum(metadata_factors.values()) / len(metadata_factors)

    # Source quality score
    source_lower = citation.source.lower()
    source_quality = 0.5  # Default for unknown sources

    if "arxiv.org" in source_lower or "doi:" in source_lower:
        source_quality = 1.0  # Academic/peer-reviewed
    elif source_lower.startswith("https://"):
        source_quality = 0.8  # Secure web source
    elif source_lower.startswith("http://"):
        source_quality = 0.6  # Unsecure web source
    elif source_lower.endswith((".pdf", ".doc", ".docx")):
        source_quality = 0.7  # Document file

    # Location specificity score
    location_score = 0.0
    if citation.location:
        location_lower = citation.location.lower()
        location_score = 0.5  # Base for having location

        # Bonus for specific references
        if any(term in location_lower for term in ["page", "p.", "pp."]):
            location_score += 0.25
        if any(term in location_lower for term in ["section", "chapter", "§"]):
            location_score += 0.25

    # Calculate overall confidence (weighted average)
    overall_confidence = (
        base_confidence * 0.4  # 40% from original confidence
        + metadata_score * 0.3  # 30% from metadata completeness
        + source_quality * 0.2  # 20% from source quality
        + location_score * 0.1  # 10% from location specificity
    )

    return {
        "overall_confidence": round(overall_confidence, 3),
        "base_confidence": base_confidence,
        "metadata_score": round(metadata_score, 3),
        "source_quality_score": round(source_quality, 3),
        "location_score": round(location_score, 3),
        "factors": {
            "metadata_completeness": metadata_factors,
            "source_type": (
                "academic" if source_quality >= 0.9 else "web" if "http" in source_lower else "file"
            ),
            "has_specific_location": location_score > 0.5,
        },
    }


def calculate_citation_map_confidence(citation_map: "CitationMap") -> Dict[str, Any]:  # type: ignore
    """
    Calculate aggregate confidence scores for a CitationMap.

    Evaluates the overall quality of citations supporting a claim.

    Args:
        citation_map: The CitationMap object to score

    Returns:
        Dictionary with:
        - overall_confidence: Aggregate confidence for the claim (0.0-1.0)
        - citation_count: Number of citations
        - average_citation_confidence: Mean confidence across citations
        - min_confidence: Lowest confidence citation
        - max_confidence: Highest confidence citation
        - strength: Original strength value from CitationMap
        - individual_scores: List of confidence scores per citation

    Example:
        >>> scores = calculate_citation_map_confidence(citation_map)
        >>> print(f"Claim supported by {scores['citation_count']} citations")
        >>> print(f"Overall confidence: {scores['overall_confidence']:.2f}")
    """
    if not citation_map.citations:
        return {
            "overall_confidence": 0.0,
            "citation_count": 0,
            "average_citation_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "strength": citation_map.strength,
            "individual_scores": [],
        }

    # Calculate confidence for each citation
    individual_scores = [
        calculate_citation_confidence(citation) for citation in citation_map.citations
    ]

    overall_confidences = [score["overall_confidence"] for score in individual_scores]

    # Calculate aggregate metrics
    average_confidence = sum(overall_confidences) / len(overall_confidences)
    min_confidence = min(overall_confidences)
    max_confidence = max(overall_confidences)

    # Overall confidence combines:
    # - Average citation confidence (50%)
    # - CitationMap strength (30%)
    # - Citation count factor (20%) - more citations = higher confidence, plateaus at 5
    citation_count_factor = min(len(citation_map.citations) / 5.0, 1.0)

    overall_confidence = (
        average_confidence * 0.5 + citation_map.strength * 0.3 + citation_count_factor * 0.2
    )

    return {
        "overall_confidence": round(overall_confidence, 3),
        "citation_count": len(citation_map.citations),
        "average_citation_confidence": round(average_confidence, 3),
        "min_confidence": round(min_confidence, 3),
        "max_confidence": round(max_confidence, 3),
        "strength": citation_map.strength,
        "individual_scores": individual_scores,
    }
