"""
Theme clustering algorithm for grouping similar ideas, claims, or hypotheses.

This module provides clustering capabilities for semantic grouping of textual content,
particularly for workflows like ARGUMENT, IDEATE, and RESEARCH where multiple claims
or ideas need to be organized into coherent themes.

Key Features:
- Semantic similarity computation using sentence transformers
- K-means and hierarchical clustering algorithms
- Automatic cluster naming and summarization
- Cluster quality scoring and metrics
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ClusterResult:
    """
    Result of a clustering operation.

    Attributes:
        cluster_id: Unique identifier for this cluster
        items: List of item indices belonging to this cluster
        centroid: Cluster centroid in embedding space
        name: Human-readable cluster name/label
        summary: Brief summary of cluster theme
        quality_score: Quality/coherence score (0.0 = poor, 1.0 = excellent)
        metadata: Additional cluster metadata
    """
    cluster_id: int
    items: List[int]
    centroid: np.ndarray
    name: str = ""
    summary: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation of cluster result."""
        return (
            f"ClusterResult(id={self.cluster_id}, "
            f"items={len(self.items)}, "
            f"name='{self.name}', "
            f"quality={self.quality_score:.2f})"
        )


class SemanticClustering:
    """
    Semantic clustering engine for grouping textual content by theme.

    This class provides methods for:
    1. Computing semantic similarity between texts using embeddings
    2. Clustering texts using K-means or hierarchical methods
    3. Naming and summarizing clusters
    4. Scoring cluster quality

    Example:
        >>> clustering = SemanticClustering(model_name="all-MiniLM-L6-v2")
        >>> texts = ["Python is great", "I love Python", "Java is verbose"]
        >>> clusters = clustering.cluster(texts, n_clusters=2)
        >>> for cluster in clusters:
        ...     print(f"{cluster.name}: {cluster.items}")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True
    ):
        """
        Initialize the semantic clustering engine.

        Args:
            model_name: Name of the sentence transformer model to use
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._model = None  # Lazy load

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute semantic embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (n_texts, embedding_dim)

        Example:
            >>> clustering = SemanticClustering()
            >>> texts = ["Hello world", "Goodbye world"]
            >>> embeddings = clustering.compute_embeddings(texts)
            >>> embeddings.shape
            (2, 384)  # 384-dimensional embeddings
        """
        self._load_model()

        # Check cache first
        if self.cache_embeddings:
            uncached_texts = []
            uncached_indices = []
            embeddings_list = [None] * len(texts)

            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    embeddings_list[i] = self._embedding_cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Compute uncached embeddings
            if uncached_texts:
                new_embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)
                for i, text in zip(uncached_indices, uncached_texts):
                    embedding = new_embeddings[uncached_indices.index(i)]
                    embeddings_list[i] = embedding
                    self._embedding_cache[text] = embedding

            return np.array(embeddings_list)
        else:
            return self._model.encode(texts, convert_to_numpy=True)

    def compute_similarity(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between embeddings.

        Args:
            embeddings: numpy array of shape (n_items, embedding_dim)
            metric: Similarity metric ("cosine", "euclidean", "dot")

        Returns:
            Similarity matrix of shape (n_items, n_items)

        Example:
            >>> clustering = SemanticClustering()
            >>> embeddings = np.random.rand(5, 384)
            >>> sim_matrix = clustering.compute_similarity(embeddings)
            >>> sim_matrix.shape
            (5, 5)
        """
        if metric == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            # Compute cosine similarity
            return np.dot(normalized, normalized.T)
        elif metric == "euclidean":
            # Compute pairwise Euclidean distances, convert to similarity
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings, embeddings, metric='euclidean')
            # Convert distance to similarity (closer = more similar)
            return 1 / (1 + distances)
        elif metric == "dot":
            # Simple dot product
            return np.dot(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def cluster_kmeans(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using K-means algorithm.

        Args:
            embeddings: numpy array of shape (n_items, embedding_dim)
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (cluster_labels, centroids) where:
            - cluster_labels: array of shape (n_items,) with cluster assignments
            - centroids: array of shape (n_clusters, embedding_dim)

        Example:
            >>> clustering = SemanticClustering()
            >>> embeddings = np.random.rand(10, 384)
            >>> labels, centroids = clustering.cluster_kmeans(embeddings, n_clusters=3)
            >>> labels.shape, centroids.shape
            ((10,), (3, 384))
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        return labels, centroids

    def cluster_hierarchical(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        linkage: str = "ward"
    ) -> np.ndarray:
        """
        Cluster embeddings using hierarchical clustering.

        Args:
            embeddings: numpy array of shape (n_items, embedding_dim)
            n_clusters: Number of clusters to create
            linkage: Linkage criterion ("ward", "complete", "average", "single")

        Returns:
            Cluster labels array of shape (n_items,)

        Example:
            >>> clustering = SemanticClustering()
            >>> embeddings = np.random.rand(10, 384)
            >>> labels = clustering.cluster_hierarchical(embeddings, n_clusters=3)
            >>> labels.shape
            (10,)
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = clustering.fit_predict(embeddings)

        return labels

    def name_cluster(
        self,
        texts: List[str],
        max_length: int = 50
    ) -> str:
        """
        Generate a name/label for a cluster based on member texts.

        This is a simple implementation that finds the most common words
        in the cluster texts. For production use, consider using an LLM
        to generate more meaningful names.

        Args:
            texts: List of texts in the cluster
            max_length: Maximum length of the generated name

        Returns:
            Cluster name string

        Example:
            >>> clustering = SemanticClustering()
            >>> texts = ["Python is great", "I love Python", "Python programming"]
            >>> clustering.name_cluster(texts)
            'Python programming'
        """
        if not texts:
            return "Empty Cluster"

        # Simple heuristic: use the shortest text as the cluster name
        # In production, you'd want to use an LLM to generate a summary
        name = min(texts, key=len)

        if len(name) > max_length:
            name = name[:max_length - 3] + "..."

        return name

    def summarize_cluster(
        self,
        texts: List[str],
        max_length: int = 200
    ) -> str:
        """
        Generate a summary for a cluster based on member texts.

        This is a simple implementation that concatenates texts.
        For production use, consider using an LLM for better summarization.

        Args:
            texts: List of texts in the cluster
            max_length: Maximum length of the summary

        Returns:
            Cluster summary string

        Example:
            >>> clustering = SemanticClustering()
            >>> texts = ["Python is great", "I love Python"]
            >>> clustering.summarize_cluster(texts)
            'Python is great; I love Python'
        """
        if not texts:
            return "No texts in cluster"

        summary = "; ".join(texts)

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        return summary

    def score_cluster(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray
    ) -> float:
        """
        Compute quality score for a cluster based on cohesion.

        Score is based on average similarity of items to centroid.
        Higher score = more cohesive cluster.

        Args:
            embeddings: numpy array of shape (n_items, embedding_dim) for cluster items
            centroid: numpy array of shape (embedding_dim,) for cluster center

        Returns:
            Quality score between 0.0 and 1.0

        Example:
            >>> clustering = SemanticClustering()
            >>> embeddings = np.random.rand(5, 384)
            >>> centroid = embeddings.mean(axis=0)
            >>> score = clustering.score_cluster(embeddings, centroid)
            >>> 0.0 <= score <= 1.0
            True
        """
        if len(embeddings) == 0:
            return 0.0

        # Normalize
        norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Compute cosine similarity to centroid
        similarities = np.dot(norm_embeddings, norm_centroid)

        # Average similarity as quality score
        return float(similarities.mean())

    def cluster(
        self,
        texts: List[str],
        n_clusters: int,
        method: str = "kmeans",
        random_state: Optional[int] = None
    ) -> List[ClusterResult]:
        """
        Cluster texts into semantic groups.

        This is the main entry point for clustering. It performs:
        1. Embedding computation
        2. Clustering using specified method
        3. Cluster naming and summarization
        4. Quality scoring

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters to create
            method: Clustering method ("kmeans" or "hierarchical")
            random_state: Random seed for reproducibility (kmeans only)

        Returns:
            List of ClusterResult objects

        Example:
            >>> clustering = SemanticClustering()
            >>> texts = ["Python is great", "I love Python", "Java is verbose", "C++ is fast"]
            >>> clusters = clustering.cluster(texts, n_clusters=2)
            >>> len(clusters)
            2
            >>> all(isinstance(c, ClusterResult) for c in clusters)
            True
        """
        if len(texts) == 0:
            return []

        if n_clusters > len(texts):
            n_clusters = len(texts)

        # Compute embeddings
        embeddings = self.compute_embeddings(texts)

        # Perform clustering
        if method == "kmeans":
            labels, centroids = self.cluster_kmeans(embeddings, n_clusters, random_state)
        elif method == "hierarchical":
            labels = self.cluster_hierarchical(embeddings, n_clusters)
            # Compute centroids for hierarchical clustering
            centroids = np.array([
                embeddings[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build cluster results
        results = []
        for cluster_id in range(n_clusters):
            # Get items in this cluster
            item_indices = np.where(labels == cluster_id)[0].tolist()
            cluster_texts = [texts[i] for i in item_indices]

            # Get centroid
            centroid = centroids[cluster_id]

            # Name and summarize
            name = self.name_cluster(cluster_texts)
            summary = self.summarize_cluster(cluster_texts)

            # Score quality
            cluster_embeddings = embeddings[item_indices]
            quality_score = self.score_cluster(cluster_embeddings, centroid)

            result = ClusterResult(
                cluster_id=cluster_id,
                items=item_indices,
                centroid=centroid,
                name=name,
                summary=summary,
                quality_score=quality_score,
                metadata={
                    "size": len(item_indices),
                    "method": method,
                }
            )
            results.append(result)

        return results
