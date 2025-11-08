"""
Pydantic models for STUDY workflow memory system.

This module defines the data models used for memory entries, metadata,
and memory operations in the persona-based investigation workflow.

The models support a two-tier memory architecture:
- Short-term cache: MemoryEntry instances in memory
- Long-term persistence: Serialized MemoryEntry stored in SQLite
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class MemoryType(str, Enum):
    """
    Type of memory entry for categorization and retrieval.

    Values:
        FINDING: Research findings or discovered facts
        HYPOTHESIS: Proposed hypotheses or theories
        EVIDENCE: Supporting evidence for hypotheses
        CONCLUSION: Final conclusions from investigation
        QUESTION: Open questions requiring investigation
        REFERENCE: References to external sources or context
        PERSONA_NOTE: Persona-specific observations or notes
    """

    FINDING = "finding"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    REFERENCE = "reference"
    PERSONA_NOTE = "persona_note"


class MemoryEntry(BaseModel):
    """
    Single memory entry in the investigation memory system.

    A memory entry captures a discrete piece of information from the investigation,
    including what was learned, who learned it (persona), when it was learned,
    and how confident they are in the finding.

    Memory entries are created during investigation steps and can reference other
    memory entries to build a knowledge graph of the investigation.

    Attributes:
        investigation_id: Unique identifier for the investigation
        session_id: Unique identifier for the current investigation session
        timestamp: ISO format timestamp of when entry was created
        persona: Persona identifier (e.g., 'researcher', 'critic', 'planner')
        findings: The main content/finding of this memory entry
        evidence: Supporting evidence or context for the finding
        confidence_before: Confidence level before this investigation step
        confidence_after: Confidence level after this investigation step
        memory_references: List of memory entry IDs referenced/related to this one
        memory_type: Type of memory entry for categorization
        metadata: Additional metadata (sources, tags, importance score, etc.)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "investigation_id": "inv-abc-123",
                "session_id": "sess-xyz-789",
                "timestamp": "2025-11-08T20:30:00Z",
                "persona": "researcher",
                "findings": "Found that memory architecture should use two-tier design",
                "evidence": "Analysis of existing ConversationMemory shows message-level tracking, need investigation-level tracking",
                "confidence_before": "low",
                "confidence_after": "high",
                "memory_references": ["mem-001", "mem-005"],
                "memory_type": "finding",
                "metadata": {
                    "importance": 0.9,
                    "tags": ["architecture", "memory-design"],
                    "sources": ["core/conversation.py"],
                },
            }
        }
    )

    investigation_id: str = Field(
        ...,
        description="Unique identifier for the investigation",
        min_length=1,
    )

    session_id: str = Field(
        ...,
        description="Unique identifier for the current investigation session",
        min_length=1,
    )

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO format timestamp of when entry was created",
    )

    persona: str = Field(
        ...,
        description="Persona identifier who created this memory entry",
        min_length=1,
    )

    findings: str = Field(
        ...,
        description="The main content or finding of this memory entry",
        min_length=1,
    )

    evidence: str = Field(
        default="",
        description="Supporting evidence or context for the finding",
    )

    confidence_before: str = Field(
        default="exploring",
        description="Confidence level before this investigation step",
    )

    confidence_after: str = Field(
        default="exploring",
        description="Confidence level after this investigation step",
    )

    memory_references: List[str] = Field(
        default_factory=list,
        description="List of memory entry IDs referenced or related to this one",
    )

    memory_type: MemoryType = Field(
        default=MemoryType.FINDING,
        description="Type of memory entry for categorization",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (sources, tags, importance, etc.)",
    )


class MemoryMetadata(BaseModel):
    """
    Metadata for memory operations and statistics.

    Tracks information about memory system usage, performance,
    and health metrics for monitoring and optimization.

    Attributes:
        total_entries: Total number of memory entries stored
        cache_entries: Number of entries currently in cache
        persisted_entries: Number of entries persisted to long-term storage
        investigation_count: Number of distinct investigations tracked
        last_cleanup: ISO timestamp of last cleanup/pruning operation
        cache_hit_rate: Percentage of cache hits vs misses
        avg_retrieval_time_ms: Average retrieval time in milliseconds
        storage_size_bytes: Total storage size in bytes (for persistence layer)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_entries": 150,
                "cache_entries": 25,
                "persisted_entries": 125,
                "investigation_count": 5,
                "last_cleanup": "2025-11-08T20:00:00Z",
                "cache_hit_rate": 0.85,
                "avg_retrieval_time_ms": 2.5,
                "storage_size_bytes": 1048576,
            }
        }
    )

    total_entries: int = Field(
        default=0,
        ge=0,
        description="Total number of memory entries stored",
    )

    cache_entries: int = Field(
        default=0,
        ge=0,
        description="Number of entries currently in cache",
    )

    persisted_entries: int = Field(
        default=0,
        ge=0,
        description="Number of entries persisted to long-term storage",
    )

    investigation_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct investigations tracked",
    )

    last_cleanup: Optional[str] = Field(
        default=None,
        description="ISO timestamp of last cleanup/pruning operation",
    )

    cache_hit_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of cache hits vs misses (0.0-1.0)",
    )

    avg_retrieval_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average retrieval time in milliseconds",
    )

    storage_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Total storage size in bytes (for persistence layer)",
    )


class MemoryQuery(BaseModel):
    """
    Query model for searching and filtering memory entries.

    Supports flexible filtering across multiple dimensions to retrieve
    relevant memory entries from cache or persistence storage.

    Attributes:
        investigation_id: Filter by specific investigation ID
        persona: Filter by persona identifier
        confidence_level: Filter by minimum confidence level
        time_range_start: Filter by start time (ISO format)
        time_range_end: Filter by end time (ISO format)
        memory_type: Filter by memory entry type
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        sort_by: Field to sort by (timestamp, confidence_after, etc.)
        sort_order: Sort order (asc or desc)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "investigation_id": "inv-abc-123",
                "persona": "researcher",
                "confidence_level": "medium",
                "time_range_start": "2025-11-08T00:00:00Z",
                "time_range_end": "2025-11-08T23:59:59Z",
                "memory_type": "finding",
                "limit": 50,
                "offset": 0,
                "sort_by": "timestamp",
                "sort_order": "desc",
            }
        }
    )

    investigation_id: Optional[str] = Field(
        default=None,
        description="Filter by specific investigation ID",
    )

    persona: Optional[str] = Field(
        default=None,
        description="Filter by persona identifier",
    )

    confidence_level: Optional[str] = Field(
        default=None,
        description="Filter by minimum confidence level",
    )

    time_range_start: Optional[str] = Field(
        default=None,
        description="Filter by start time (ISO format timestamp)",
    )

    time_range_end: Optional[str] = Field(
        default=None,
        description="Filter by end time (ISO format timestamp)",
    )

    memory_type: Optional[MemoryType] = Field(
        default=None,
        description="Filter by memory entry type",
    )

    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results to return",
    )

    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip (for pagination)",
    )

    sort_by: str = Field(
        default="timestamp",
        description="Field to sort by (timestamp, confidence_after, etc.)",
    )

    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort order: 'asc' (ascending) or 'desc' (descending)",
    )
