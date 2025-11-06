"""
Semantic similarity computation for claim comparison in ARGUMENT workflow.

Provides embedding-based similarity computation using sentence transformers
to identify similar claims, detect redundancy, and enable semantic search
across citation maps.
"""

import hashlib
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from modelchorus.core.models import Citation, CitationMap


# Global model instance (lazy-loaded)
_model: Optional[SentenceTransformer] = None
_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight model (384 dimensions)


def _get_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """
    Get or initialize the sentence transformer model.

    Uses lazy loading to avoid loading the model until needed.
    Model is cached globally to avoid repeated loading.

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformer instance
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def _normalize_text(text: str) -> str:
    """
    Normalize text for consistent embedding computation.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text (lowercase, stripped whitespace)
    """
    return text.strip().lower()


@lru_cache(maxsize=1000)
def _compute_embedding_cached(text_hash: str, text: str, model_name: str) -> np.ndarray:
    """
    Internal cached embedding computation.

    Uses text hash as cache key to enable LRU caching while avoiding
    unhashable numpy arrays as cache keys.

    Args:
        text_hash: SHA256 hash of the normalized text (for cache key)
        text: Actual text to compute embedding for
        model_name: Model to use for embedding

    Returns:
        Embedding vector as numpy array
    """
    model = _get_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def compute_embedding(
    text: str,
    model_name: str = _DEFAULT_MODEL,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute semantic embedding for text using sentence transformers.

    Embeddings are cached using LRU cache to avoid recomputing for
    duplicate text. Text is normalized before embedding computation
    for consistency.

    Args:
        text: Text to compute embedding for
        model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
        normalize: Whether to normalize text before embedding (default: True)

    Returns:
        Normalized embedding vector as numpy array (unit length if model normalizes)

    Example:
        >>> emb1 = compute_embedding("Machine learning improves accuracy")
        >>> emb2 = compute_embedding("ML enhances precision")
        >>> similarity = cosine_similarity(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.847
    """
    if normalize:
        text = _normalize_text(text)

    # Create cache key using hash of text
    text_hash = hashlib.sha256(text.encode()).hexdigest()

    return _compute_embedding_cached(text_hash, text, model_name)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Assumes embeddings are already normalized (unit length).
    If normalized, this reduces to simple dot product.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
        - 1.0 = identical/very similar
        - 0.0 = completely dissimilar
        - 0.5 = moderate similarity

    Example:
        >>> emb1 = compute_embedding("hello world")
        >>> emb2 = compute_embedding("hello world")
        >>> cosine_similarity(emb1, emb2)
        1.0
    """
    # For normalized embeddings, cosine similarity = dot product
    similarity = np.dot(embedding1, embedding2)
    # Clip to [0, 1] range in case of floating point errors
    return float(np.clip(similarity, 0.0, 1.0))


def compute_claim_similarity(
    claim1: str,
    claim2: str,
    model_name: str = _DEFAULT_MODEL,
) -> float:
    """
    Compute semantic similarity between two claim texts.

    Args:
        claim1: First claim text
        claim2: Second claim text
        model_name: Sentence transformer model to use

    Returns:
        Similarity score (0.0 to 1.0)
        - >= 0.9: Very similar (likely duplicates)
        - 0.7-0.9: Similar (related claims)
        - 0.5-0.7: Moderate similarity (overlapping topics)
        - < 0.5: Different claims

    Example:
        >>> score = compute_claim_similarity(
        ...     "AI improves software quality",
        ...     "Artificial intelligence enhances code quality"
        ... )
        >>> print(f"Similarity: {score:.3f}")
        Similarity: 0.875
    """
    emb1 = compute_embedding(claim1, model_name=model_name)
    emb2 = compute_embedding(claim2, model_name=model_name)
    return cosine_similarity(emb1, emb2)


def find_similar_claims(
    query_claim: str,
    citation_maps: List[CitationMap],
    threshold: float = 0.7,
    top_k: Optional[int] = None,
    model_name: str = _DEFAULT_MODEL,
) -> List[Tuple[CitationMap, float]]:
    """
    Find citation maps with claims similar to the query claim.

    Uses semantic similarity to identify related claims across a
    collection of citation maps. Useful for:
    - Detecting duplicate claims
    - Finding supporting evidence for new claims
    - Clustering related arguments

    Args:
        query_claim: Claim text to search for
        citation_maps: List of CitationMap objects to search within
        threshold: Minimum similarity score to include (default: 0.7)
        top_k: Optional limit on number of results (returns top-k most similar)
        model_name: Sentence transformer model to use

    Returns:
        List of (CitationMap, similarity_score) tuples, sorted by similarity (descending)

    Example:
        >>> maps = [citation_map1, citation_map2, citation_map3]
        >>> results = find_similar_claims(
        ...     "Machine learning improves accuracy",
        ...     maps,
        ...     threshold=0.7,
        ...     top_k=5
        ... )
        >>> for cm, score in results:
        ...     print(f"{score:.3f}: {cm.claim_text}")
        0.892: ML models enhance prediction accuracy
        0.745: AI systems improve results
    """
    query_embedding = compute_embedding(query_claim, model_name=model_name)

    # Compute similarities for all citation maps
    similarities: List[Tuple[CitationMap, float]] = []
    for cm in citation_maps:
        claim_embedding = compute_embedding(cm.claim_text, model_name=model_name)
        score = cosine_similarity(query_embedding, claim_embedding)

        if score >= threshold:
            similarities.append((cm, score))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Apply top_k limit if specified
    if top_k is not None:
        similarities = similarities[:top_k]

    return similarities


def compute_claim_similarity_batch(
    claims: List[str],
    model_name: str = _DEFAULT_MODEL,
) -> np.ndarray:
    """
    Compute pairwise similarities for a batch of claims.

    More efficient than calling compute_claim_similarity repeatedly,
    as it batches the embedding computation.

    Args:
        claims: List of claim texts
        model_name: Sentence transformer model to use

    Returns:
        NxN similarity matrix where result[i][j] = similarity(claims[i], claims[j])

    Example:
        >>> claims = [
        ...     "AI improves quality",
        ...     "ML enhances accuracy",
        ...     "Weather is sunny"
        ... ]
        >>> sim_matrix = compute_claim_similarity_batch(claims)
        >>> print(sim_matrix.shape)
        (3, 3)
        >>> print(f"Similarity[0,1]: {sim_matrix[0,1]:.3f}")
        Similarity[0,1]: 0.847
    """
    # Compute embeddings for all claims
    embeddings = [compute_embedding(claim, model_name=model_name) for claim in claims]
    embeddings_array = np.array(embeddings)

    # Compute pairwise similarities (matrix multiplication)
    # For normalized embeddings: similarity_matrix = embeddings @ embeddings.T
    similarity_matrix = np.dot(embeddings_array, embeddings_array.T)

    # Clip to [0, 1] range
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

    return similarity_matrix


def add_similarity_to_citation(
    citation: Citation,
    reference_claim: str,
    model_name: str = _DEFAULT_MODEL,
) -> Citation:
    """
    Add semantic similarity score to a citation's metadata.

    Computes similarity between the citation's snippet (if available)
    and a reference claim, storing the result in citation.metadata.

    Args:
        citation: Citation object to enhance
        reference_claim: Claim text to compare against
        model_name: Sentence transformer model to use

    Returns:
        Enhanced Citation object with similarity_score in metadata

    Example:
        >>> citation = Citation(
        ...     source="paper.pdf",
        ...     confidence=0.9,
        ...     snippet="ML improves accuracy by 23%"
        ... )
        >>> enhanced = add_similarity_to_citation(
        ...     citation,
        ...     "Machine learning enhances precision"
        ... )
        >>> print(enhanced.metadata["similarity_score"])
        0.847
    """
    if not citation.snippet:
        # No snippet to compare - set similarity to 0
        citation.metadata["similarity_score"] = 0.0
        citation.metadata["similarity_note"] = "No snippet available"
        return citation

    # Compute similarity between snippet and reference claim
    similarity = compute_claim_similarity(
        citation.snippet,
        reference_claim,
        model_name=model_name,
    )

    citation.metadata["similarity_score"] = round(similarity, 3)
    citation.metadata["similarity_model"] = model_name

    return citation


def find_duplicate_claims(
    citation_maps: List[CitationMap],
    threshold: float = 0.9,
    model_name: str = _DEFAULT_MODEL,
) -> List[List[CitationMap]]:
    """
    Detect groups of duplicate or near-duplicate claims.

    Uses high similarity threshold to identify claims that are
    essentially the same despite different wording.

    Args:
        citation_maps: List of CitationMap objects to check for duplicates
        threshold: Minimum similarity to consider duplicates (default: 0.9)
        model_name: Sentence transformer model to use

    Returns:
        List of duplicate groups, where each group is a list of similar CitationMaps

    Example:
        >>> maps = [citation_map1, citation_map2, citation_map3]
        >>> duplicates = find_duplicate_claims(maps, threshold=0.9)
        >>> for group in duplicates:
        ...     print(f"Found {len(group)} duplicates:")
        ...     for cm in group:
        ...         print(f"  - {cm.claim_text}")
        Found 2 duplicates:
          - AI improves accuracy
          - Artificial intelligence enhances precision
    """
    if len(citation_maps) < 2:
        return []

    # Compute all pairwise similarities
    claims = [cm.claim_text for cm in citation_maps]
    similarity_matrix = compute_claim_similarity_batch(claims, model_name=model_name)

    # Find duplicate groups using connected components
    n = len(citation_maps)
    visited = [False] * n
    duplicate_groups: List[List[CitationMap]] = []

    for i in range(n):
        if visited[i]:
            continue

        # Start new group with current claim
        group = [citation_maps[i]]
        visited[i] = True

        # Find all similar claims
        for j in range(i + 1, n):
            if not visited[j] and similarity_matrix[i, j] >= threshold:
                group.append(citation_maps[j])
                visited[j] = True

        # Only include groups with 2+ members
        if len(group) >= 2:
            duplicate_groups.append(group)

    return duplicate_groups
