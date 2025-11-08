"""
ARGUMENT workflow module for evidence-based argumentation.

Provides semantic similarity computation, claim comparison, clustering,
and citation analysis for research and argumentation workflows.
"""

from model_chorus.workflows.argument.argument_workflow import ArgumentWorkflow
from model_chorus.workflows.argument.semantic import (
    compute_claim_similarity,
    find_similar_claims,
    compute_embedding,
    cosine_similarity,
    cluster_claims_kmeans,
    cluster_claims_hierarchical,
    get_cluster_representative,
    compute_cluster_statistics,
)

__all__ = [
    "ArgumentWorkflow",
    "compute_claim_similarity",
    "find_similar_claims",
    "compute_embedding",
    "cosine_similarity",
    "cluster_claims_kmeans",
    "cluster_claims_hierarchical",
    "get_cluster_representative",
    "compute_cluster_statistics",
]
