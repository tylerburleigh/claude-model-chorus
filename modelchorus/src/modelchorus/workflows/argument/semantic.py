"""
Semantic similarity computation for claim comparison in ARGUMENT workflow.

Provides embedding-based similarity computation using sentence transformers
to identify similar claims, detect redundancy, and enable semantic search
across citation maps.
"""

import hashlib
from typing import List, Dict, Tuple, Optional, Any
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


# ============================================================================
# Clustering Algorithms
# ============================================================================


def cluster_claims_kmeans(
    citation_maps: List[CitationMap],
    n_clusters: int = 3,
    model_name: str = _DEFAULT_MODEL,
    random_state: int = 42,
) -> List[List[CitationMap]]:
    """
    Cluster claims using K-means algorithm on semantic embeddings.

    Groups claims into k clusters based on semantic similarity,
    useful for organizing large collections of claims by topic.

    Args:
        citation_maps: List of CitationMap objects to cluster
        n_clusters: Number of clusters to create (default: 3)
        model_name: Sentence transformer model to use
        random_state: Random seed for reproducibility

    Returns:
        List of clusters, where each cluster is a list of CitationMaps

    Raises:
        ValueError: If n_clusters > len(citation_maps)

    Example:
        >>> maps = [cm1, cm2, cm3, cm4, cm5]
        >>> clusters = cluster_claims_kmeans(maps, n_clusters=2)
        >>> print(f"Found {len(clusters)} clusters")
        Found 2 clusters
        >>> for i, cluster in enumerate(clusters):
        ...     print(f"Cluster {i}: {len(cluster)} claims")
        Cluster 0: 3 claims
        Cluster 1: 2 claims
    """
    from sklearn.cluster import KMeans

    if len(citation_maps) == 0:
        return []

    if n_clusters > len(citation_maps):
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot be greater than "
            f"number of citation maps ({len(citation_maps)})"
        )

    # Compute embeddings for all claims
    claims = [cm.claim_text for cm in citation_maps]
    embeddings = np.array([compute_embedding(claim, model_name=model_name) for claim in claims])

    # Run K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group citation maps by cluster
    clusters: List[List[CitationMap]] = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(citation_maps[idx])

    return clusters


def cluster_claims_hierarchical(
    citation_maps: List[CitationMap],
    n_clusters: int = 3,
    model_name: str = _DEFAULT_MODEL,
    linkage_method: str = "ward",
) -> List[List[CitationMap]]:
    """
    Cluster claims using hierarchical clustering on semantic embeddings.

    Uses agglomerative hierarchical clustering to group claims,
    building a tree-based hierarchy and cutting at specified level.

    Args:
        citation_maps: List of CitationMap objects to cluster
        n_clusters: Number of clusters to create (default: 3)
        model_name: Sentence transformer model to use
        linkage_method: Linkage method ('ward', 'complete', 'average', 'single')

    Returns:
        List of clusters, where each cluster is a list of CitationMaps

    Raises:
        ValueError: If n_clusters > len(citation_maps)

    Example:
        >>> maps = [cm1, cm2, cm3, cm4, cm5]
        >>> clusters = cluster_claims_hierarchical(
        ...     maps,
        ...     n_clusters=2,
        ...     linkage_method="ward"
        ... )
        >>> print(f"Found {len(clusters)} clusters")
        Found 2 clusters
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(citation_maps) == 0:
        return []

    if n_clusters > len(citation_maps):
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot be greater than "
            f"number of citation maps ({len(citation_maps)})"
        )

    # Compute embeddings for all claims
    claims = [cm.claim_text for cm in citation_maps]
    embeddings = np.array([compute_embedding(claim, model_name=model_name) for claim in claims])

    # Run hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    labels = hierarchical.fit_predict(embeddings)

    # Group citation maps by cluster
    clusters: List[List[CitationMap]] = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(citation_maps[idx])

    return clusters


def get_cluster_representative(
    cluster: List[CitationMap],
    model_name: str = _DEFAULT_MODEL,
) -> CitationMap:
    """
    Find the most representative claim in a cluster (centroid).

    Computes the claim closest to the cluster centroid in embedding space,
    useful for summarizing or labeling clusters.

    Args:
        cluster: List of CitationMaps in the cluster
        model_name: Sentence transformer model to use

    Returns:
        CitationMap closest to cluster centroid

    Example:
        >>> cluster = [cm1, cm2, cm3]
        >>> representative = get_cluster_representative(cluster)
        >>> print(f"Representative claim: {representative.claim_text}")
    """
    if len(cluster) == 0:
        raise ValueError("Cluster cannot be empty")

    if len(cluster) == 1:
        return cluster[0]

    # Compute embeddings for all claims in cluster
    claims = [cm.claim_text for cm in cluster]
    embeddings = np.array([compute_embedding(claim, model_name=model_name) for claim in claims])

    # Compute centroid
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # Normalize

    # Find claim closest to centroid
    similarities = [cosine_similarity(emb, centroid) for emb in embeddings]
    most_central_idx = np.argmax(similarities)

    return cluster[most_central_idx]


def generate_cluster_name(
    cluster: List[CitationMap],
    model_name: str = _DEFAULT_MODEL,
    max_words: int = 5,
) -> str:
    """
    Generate a concise name for a cluster based on representative claims.

    Uses extractive approach: identifies key terms from the cluster
    representative claim and constructs a short descriptive name.

    Args:
        cluster: List of CitationMaps in the cluster
        model_name: Sentence transformer model to use
        max_words: Maximum words in generated name (default: 5)

    Returns:
        Concise cluster name (e.g., "AI Quality Improvement")

    Example:
        >>> cluster = [cm1, cm2, cm3]  # Claims about "AI improves quality"
        >>> name = generate_cluster_name(cluster, max_words=4)
        >>> print(name)
        AI Quality Improvement
    """
    if len(cluster) == 0:
        return "Empty Cluster"

    if len(cluster) == 1:
        # For single-item clusters, use truncated claim text
        claim_text = cluster[0].claim_text
        words = claim_text.split()
        return " ".join(words[:max_words])

    # Get representative claim
    representative = get_cluster_representative(cluster, model_name=model_name)
    claim_text = representative.claim_text

    # Simple extractive approach: take first N words
    # Future enhancement: use NLP for key phrase extraction
    words = claim_text.split()

    # Capitalize first letter of each word for title case
    name_words = words[:max_words]
    name = " ".join(word.capitalize() for word in name_words)

    return name


def summarize_cluster(
    cluster: List[CitationMap],
    model_name: str = _DEFAULT_MODEL,
    max_length: int = 150,
) -> str:
    """
    Generate a detailed summary of cluster themes.

    Analyzes claims in cluster to identify common patterns
    and generates a descriptive summary.

    Args:
        cluster: List of CitationMaps in the cluster
        model_name: Sentence transformer model to use
        max_length: Maximum characters in summary (default: 150)

    Returns:
        Cluster summary (1-2 sentences)

    Example:
        >>> cluster = [cm1, cm2, cm3]
        >>> summary = summarize_cluster(cluster)
        >>> print(summary)
        This cluster focuses on AI quality improvement claims.
        All claims discuss machine learning enhancing accuracy.
    """
    if len(cluster) == 0:
        return "Empty cluster with no claims."

    if len(cluster) == 1:
        # For single-item clusters, return the claim itself
        claim_text = cluster[0].claim_text
        if len(claim_text) > max_length:
            return claim_text[:max_length - 3] + "..."
        return claim_text

    # Get representative claim as central theme
    representative = get_cluster_representative(cluster, model_name=model_name)

    # Create summary based on cluster size and representative
    num_claims = len(cluster)

    # Build summary sentence
    summary = (
        f"This cluster contains {num_claims} related claims. "
        f"Representative theme: {representative.claim_text}"
    )

    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary


def compute_cluster_statistics(
    clusters: List[List[CitationMap]],
    model_name: str = _DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Compute statistics and quality metrics for clusters.

    Provides insights into cluster quality, including size distribution,
    intra-cluster similarity, and representative claims.

    Args:
        clusters: List of clusters (each cluster is a list of CitationMaps)
        model_name: Sentence transformer model to use

    Returns:
        Dictionary with cluster statistics:
        - num_clusters: Number of clusters
        - cluster_sizes: List of cluster sizes
        - avg_cluster_size: Average cluster size
        - representatives: List of representative claims (one per cluster)
        - intra_cluster_similarities: Average similarity within each cluster

    Example:
        >>> stats = compute_cluster_statistics(clusters)
        >>> print(f"Number of clusters: {stats['num_clusters']}")
        >>> print(f"Average cluster size: {stats['avg_cluster_size']:.1f}")
        >>> print(f"Representatives: {stats['representatives']}")
    """
    if len(clusters) == 0:
        return {
            "num_clusters": 0,
            "cluster_sizes": [],
            "avg_cluster_size": 0.0,
            "representatives": [],
            "intra_cluster_similarities": [],
        }

    # Cluster sizes
    cluster_sizes = [len(cluster) for cluster in clusters]

    # Representative claims
    representatives = []
    for cluster in clusters:
        if len(cluster) > 0:
            rep = get_cluster_representative(cluster, model_name=model_name)
            representatives.append(rep.claim_text)
        else:
            representatives.append(None)

    # Intra-cluster similarities
    intra_cluster_sims = []
    for cluster in clusters:
        if len(cluster) <= 1:
            intra_cluster_sims.append(1.0)  # Perfect similarity for single-item clusters
            continue

        # Compute pairwise similarities within cluster
        claims = [cm.claim_text for cm in cluster]
        sim_matrix = compute_claim_similarity_batch(claims, model_name=model_name)

        # Get upper triangle (exclude diagonal) and compute average
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        avg_sim = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 1.0

        intra_cluster_sims.append(avg_sim)

    return {
        "num_clusters": len(clusters),
        "cluster_sizes": cluster_sizes,
        "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "representatives": representatives,
        "intra_cluster_similarities": intra_cluster_sims,
        "avg_intra_cluster_similarity": float(np.mean(intra_cluster_sims)) if intra_cluster_sims else 0.0,
    }


def score_cluster_coherence(
    cluster: List[CitationMap],
    model_name: str = _DEFAULT_MODEL,
) -> float:
    """
    Measure how tightly grouped claims are within a cluster.

    Computes average pairwise similarity between claims in the cluster.
    Higher scores indicate more coherent/similar claims.

    Args:
        cluster: List of CitationMaps in the cluster
        model_name: Sentence transformer model to use

    Returns:
        Coherence score (0.0 to 1.0)
        - 1.0 = perfect coherence (all claims identical)
        - 0.8-1.0 = high coherence (very similar claims)
        - 0.5-0.8 = moderate coherence
        - < 0.5 = low coherence (diverse claims)

    Example:
        >>> cluster = [cm1, cm2, cm3]
        >>> coherence = score_cluster_coherence(cluster)
        >>> print(f"Coherence: {coherence:.3f}")
        Coherence: 0.847
    """
    if len(cluster) == 0:
        return 0.0

    if len(cluster) == 1:
        return 1.0  # Single claim is perfectly coherent

    # Compute pairwise similarities
    claims = [cm.claim_text for cm in cluster]
    sim_matrix = compute_claim_similarity_batch(claims, model_name=model_name)

    # Get upper triangle (exclude diagonal) and compute average
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    if len(upper_triangle) == 0:
        return 1.0

    avg_similarity = float(np.mean(upper_triangle))
    return avg_similarity


def score_cluster_separation(
    clusters: List[List[CitationMap]],
    model_name: str = _DEFAULT_MODEL,
) -> float:
    """
    Measure how distinct clusters are from each other.

    Computes average distance between cluster centroids.
    Higher scores indicate better separation between clusters.

    Args:
        clusters: List of clusters (each cluster is a list of CitationMaps)
        model_name: Sentence transformer model to use

    Returns:
        Separation score (0.0 to 1.0)
        - 1.0 = perfect separation (clusters completely distinct)
        - 0.7-1.0 = high separation (well-separated clusters)
        - 0.5-0.7 = moderate separation
        - < 0.5 = low separation (overlapping clusters)

    Example:
        >>> clusters = [[cm1, cm2], [cm3, cm4]]
        >>> separation = score_cluster_separation(clusters)
        >>> print(f"Separation: {separation:.3f}")
        Separation: 0.723
    """
    if len(clusters) <= 1:
        return 1.0  # Single cluster or empty - perfect separation

    # Remove empty clusters
    non_empty_clusters = [c for c in clusters if len(c) > 0]

    if len(non_empty_clusters) <= 1:
        return 1.0

    # Compute centroid for each cluster
    centroids = []
    for cluster in non_empty_clusters:
        claims = [cm.claim_text for cm in cluster]
        embeddings = np.array([compute_embedding(claim, model_name=model_name) for claim in claims])

        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        centroids.append(centroid)

    centroids_array = np.array(centroids)

    # Compute pairwise distances between centroids
    # Distance = 1 - similarity (for cosine similarity)
    similarity_matrix = np.dot(centroids_array, centroids_array.T)
    distance_matrix = 1.0 - similarity_matrix

    # Get upper triangle (exclude diagonal)
    upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    if len(upper_triangle) == 0:
        return 1.0

    # Average inter-cluster distance
    avg_distance = float(np.mean(upper_triangle))

    # Convert to separation score (higher is better)
    # Distance is already 0-1, so we can use it directly
    separation = avg_distance

    return separation


def score_clustering_quality(
    clusters: List[List[CitationMap]],
    model_name: str = _DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Compute comprehensive quality metrics for clustering results.

    Combines multiple metrics to provide overall assessment of
    clustering quality, including coherence, separation, and
    interpretability measures.

    Args:
        clusters: List of clusters (each cluster is a list of CitationMaps)
        model_name: Sentence transformer model to use

    Returns:
        Dictionary with quality metrics:
        - coherence_scores: List of coherence scores (one per cluster)
        - avg_coherence: Average coherence across all clusters
        - separation: Inter-cluster separation score
        - silhouette_score: Sklearn silhouette coefficient (-1 to 1)
        - quality_score: Overall quality (0.0 to 1.0)
        - num_clusters: Number of clusters
        - cluster_sizes: List of cluster sizes
        - interpretability: Named clusters and summaries

    Example:
        >>> clusters = [[cm1, cm2], [cm3, cm4]]
        >>> quality = score_clustering_quality(clusters)
        >>> print(f"Quality: {quality['quality_score']:.3f}")
        >>> print(f"Coherence: {quality['avg_coherence']:.3f}")
        >>> print(f"Separation: {quality['separation']:.3f}")
        Quality: 0.812
        Coherence: 0.847
        Separation: 0.723
    """
    if len(clusters) == 0:
        return {
            "coherence_scores": [],
            "avg_coherence": 0.0,
            "separation": 0.0,
            "silhouette_score": 0.0,
            "quality_score": 0.0,
            "num_clusters": 0,
            "cluster_sizes": [],
            "interpretability": {
                "cluster_names": [],
                "cluster_summaries": [],
            },
        }

    # Remove empty clusters
    non_empty_clusters = [c for c in clusters if len(c) > 0]

    if len(non_empty_clusters) == 0:
        return {
            "coherence_scores": [],
            "avg_coherence": 0.0,
            "separation": 0.0,
            "silhouette_score": 0.0,
            "quality_score": 0.0,
            "num_clusters": 0,
            "cluster_sizes": [],
            "interpretability": {
                "cluster_names": [],
                "cluster_summaries": [],
            },
        }

    # Compute coherence for each cluster
    coherence_scores = [
        score_cluster_coherence(cluster, model_name=model_name)
        for cluster in non_empty_clusters
    ]
    avg_coherence = float(np.mean(coherence_scores))

    # Compute separation
    separation = score_cluster_separation(non_empty_clusters, model_name=model_name)

    # Compute silhouette score (requires sklearn)
    silhouette = 0.0
    if len(non_empty_clusters) > 1:
        try:
            from sklearn.metrics import silhouette_score as sklearn_silhouette

            # Flatten all claims and create labels
            all_claims = []
            labels = []
            for cluster_idx, cluster in enumerate(non_empty_clusters):
                for cm in cluster:
                    all_claims.append(cm.claim_text)
                    labels.append(cluster_idx)

            # Compute embeddings
            embeddings = np.array([
                compute_embedding(claim, model_name=model_name)
                for claim in all_claims
            ])

            # Compute silhouette score
            if len(set(labels)) > 1:  # Need at least 2 clusters
                silhouette = float(sklearn_silhouette(embeddings, labels, metric='cosine'))
                # Convert from [-1, 1] to [0, 1] range
                silhouette = (silhouette + 1.0) / 2.0
        except ImportError:
            silhouette = 0.0  # sklearn not available

    # Compute overall quality score
    # Weight: 40% coherence + 40% separation + 20% silhouette
    quality_score = (0.4 * avg_coherence) + (0.4 * separation) + (0.2 * silhouette)

    # Generate cluster names and summaries for interpretability
    cluster_names = [
        generate_cluster_name(cluster, model_name=model_name)
        for cluster in non_empty_clusters
    ]
    cluster_summaries = [
        summarize_cluster(cluster, model_name=model_name)
        for cluster in non_empty_clusters
    ]

    return {
        "coherence_scores": coherence_scores,
        "avg_coherence": avg_coherence,
        "separation": separation,
        "silhouette_score": silhouette,
        "quality_score": quality_score,
        "num_clusters": len(non_empty_clusters),
        "cluster_sizes": [len(c) for c in non_empty_clusters],
        "interpretability": {
            "cluster_names": cluster_names,
            "cluster_summaries": cluster_summaries,
        },
    }
