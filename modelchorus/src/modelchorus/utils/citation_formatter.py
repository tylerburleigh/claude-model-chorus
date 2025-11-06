"""
Citation formatting utilities for ModelChorus ARGUMENT workflow.

Provides formatting functions to convert Citation objects into standard
citation formats (APA, MLA, Chicago) for academic and professional use.
"""

from typing import Optional
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
        >>> from modelchorus.core.models import Citation
        >>> from modelchorus.utils.citation_formatter import format_citation, CitationStyle
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
