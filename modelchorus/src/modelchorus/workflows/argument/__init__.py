"""
ARGUMENT workflow module for evidence-based argumentation.

Provides semantic similarity computation, claim comparison, and citation analysis
for research and argumentation workflows.
"""

from modelchorus.workflows.argument.semantic import (
    compute_claim_similarity,
    find_similar_claims,
    compute_embedding,
    cosine_similarity,
)

__all__ = [
    "compute_claim_similarity",
    "find_similar_claims",
    "compute_embedding",
    "cosine_similarity",
]
