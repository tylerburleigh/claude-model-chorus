"""
Memory system for STUDY workflow persona-based investigations.

This package implements a two-tier memory architecture for persona-based collaborative
research, enabling context retention and knowledge accumulation across investigation steps.

Architecture:
-----------

Two-Tier Memory System:
    1. Short-term Cache (In-Memory):
       - Fast access for active investigation data
       - Persona-specific context for current investigation
       - Hypothesis tracking and intermediate findings
       - Session-based lifecycle

    2. Long-term Persistence (SQLite):
       - Durable storage for completed investigations
       - Cross-investigation knowledge retrieval
       - Historical context and patterns
       - Searchable investigation archives

Integration with Core:
    - Complements core.conversation.ConversationMemory with persona-aware capabilities
    - ConversationMemory: Message-level threading and turn management
    - PersonaMemory: Investigation-level context and knowledge retention
    - Both work together to provide complete memory coverage

Memory Lifecycle:
    1. Investigation Start: Create new memory context in cache
    2. Active Investigation: Store findings, hypotheses, persona interactions in cache
    3. Investigation Complete: Persist important findings to long-term storage
    4. Future Investigations: Retrieve relevant historical context from persistence

Key Features:
    - Persona-specific memory contexts (what each persona knows/remembers)
    - Automatic memory summarization and pruning
    - Context-aware retrieval (find relevant past investigations)
    - Memory consolidation (cache → persistence)
    - Investigation threading (link related investigations)

Components:
----------
    models.py: Data models for memory entries and metadata
    cache.py: In-memory short-term cache implementation
    persistence.py: SQLite-based long-term persistence layer
    controller.py: Memory controller coordinating cache and persistence

Usage:
-----
    >>> from model_chorus.workflows.study.memory import MemoryController
    >>>
    >>> # Create memory system
    >>> memory = MemoryController()
    >>>
    >>> # Store investigation finding
    >>> memory.store(
    ...     persona_id="researcher",
    ...     content="Found pattern X in dataset Y",
    ...     context_type="finding",
    ...     investigation_id="inv-123"
    ... )
    >>>
    >>> # Retrieve persona-specific context
    >>> context = memory.get_context(
    ...     persona_id="researcher",
    ...     investigation_id="inv-123"
    ... )

Future Extensions:
    - Vector embeddings for semantic similarity search
    - Cross-persona memory sharing and consensus
    - Memory importance scoring and automatic summarization
    - Investigation dependency tracking
"""

# Component imports (all implemented):
from .cache import ShortTermCache
from .controller import MemoryController
from .models import MemoryEntry, MemoryMetadata, MemoryQuery, MemoryType
from .persistence import LongTermStorage

__all__ = [
    # Data models (task 4-2 complete):
    "MemoryEntry",
    "MemoryMetadata",
    "MemoryType",
    "MemoryQuery",
    # Cache implementation (task 4-3 complete):
    "ShortTermCache",
    # Persistence implementation (task 4-4 complete):
    "LongTermStorage",
    # Controller implementation (task 4-5-1 complete):
    "MemoryController",
]

__version__ = "0.1.0"
