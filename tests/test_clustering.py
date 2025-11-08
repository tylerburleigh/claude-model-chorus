"""
Unit tests for the semantic clustering algorithm.

Tests verify that:
- Embeddings are computed correctly and cached properly
- Similarity metrics work as expected
- K-means clustering produces valid results
- Hierarchical clustering produces valid results
- Cluster naming and summarization work correctly
- Cluster quality scoring is accurate
- Edge cases are handled properly
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from model_chorus.core.clustering import (
    SemanticClustering,
    ClusterResult,
)


def _check_sentence_transformers_available():
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


class TestClusterResult:
    """Test suite for ClusterResult dataclass."""

    def test_cluster_result_creation(self):
        """Test creating a ClusterResult with basic attributes."""
        centroid = np.array([0.1, 0.2, 0.3])
        result = ClusterResult(
            cluster_id=0,
            items=[0, 1, 2],
            centroid=centroid,
            name="Test Cluster",
            summary="A test cluster",
            quality_score=0.85,
        )

        assert result.cluster_id == 0
        assert result.items == [0, 1, 2]
        assert np.array_equal(result.centroid, centroid)
        assert result.name == "Test Cluster"
        assert result.summary == "A test cluster"
        assert result.quality_score == 0.85

    def test_cluster_result_repr(self):
        """Test string representation of ClusterResult."""
        centroid = np.array([0.1, 0.2, 0.3])
        result = ClusterResult(
            cluster_id=1,
            items=[0, 1],
            centroid=centroid,
            name="Python",
            quality_score=0.92,
        )

        repr_str = repr(result)
        assert "id=1" in repr_str
        assert "items=2" in repr_str
        assert "name='Python'" in repr_str
        assert "quality=0.92" in repr_str


class TestSemanticClustering:
    """Test suite for SemanticClustering class."""

    @pytest.fixture
    def clustering(self):
        """Create a SemanticClustering instance for testing."""
        return SemanticClustering(cache_embeddings=True)

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = Mock()
        # Return 384-dimensional embeddings (standard size for MiniLM)
        model.encode = Mock(side_effect=lambda texts, **kwargs: np.random.rand(len(texts), 384))
        return model

    def test_initialization(self, clustering):
        """Test SemanticClustering initialization."""
        assert clustering.model_name == "all-MiniLM-L6-v2"
        assert clustering.cache_embeddings is True
        assert clustering._model is None  # Lazy load
        assert clustering._embedding_cache == {}

    def test_lazy_model_loading(self, clustering, mock_model):
        """Test that model is loaded lazily on first use."""
        assert clustering._model is None

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            clustering._load_model()
            assert clustering._model is not None

    def test_compute_embeddings_basic(self, clustering, mock_model):
        """Test basic embedding computation."""
        texts = ["Hello world", "Goodbye world"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            embeddings = clustering.compute_embeddings(texts)

            assert embeddings.shape == (2, 384)
            assert mock_model.encode.called

    def test_compute_embeddings_caching(self, clustering, mock_model):
        """Test that embeddings are cached properly."""
        texts = ["Python is great", "I love Python"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # First call should compute
            embeddings1 = clustering.compute_embeddings(texts)
            call_count_1 = mock_model.encode.call_count

            # Second call with same texts should use cache
            embeddings2 = clustering.compute_embeddings(texts)
            call_count_2 = mock_model.encode.call_count

            # Cache hit - no additional encode calls
            assert call_count_2 == call_count_1
            assert np.array_equal(embeddings1, embeddings2)

    def test_compute_embeddings_no_cache(self, mock_model):
        """Test embedding computation without caching."""
        clustering = SemanticClustering(cache_embeddings=False)
        texts = ["Test 1", "Test 2"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            embeddings1 = clustering.compute_embeddings(texts)
            call_count_1 = mock_model.encode.call_count

            embeddings2 = clustering.compute_embeddings(texts)
            call_count_2 = mock_model.encode.call_count

            # No cache - should call encode both times
            assert call_count_2 == call_count_1 * 2

    def test_compute_similarity_cosine(self, clustering):
        """Test cosine similarity computation."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],  # Same as first
        ])

        sim_matrix = clustering.compute_similarity(embeddings, metric="cosine")

        assert sim_matrix.shape == (3, 3)
        # Check diagonal (self-similarity should be 1.0)
        assert np.allclose(np.diag(sim_matrix), 1.0)
        # Check that identical vectors have similarity 1.0
        assert np.allclose(sim_matrix[0, 2], 1.0)
        # Check orthogonal vectors have similarity 0.0
        assert np.allclose(sim_matrix[0, 1], 0.0)

    def test_compute_similarity_euclidean(self, clustering):
        """Test Euclidean similarity computation."""
        embeddings = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        sim_matrix = clustering.compute_similarity(embeddings, metric="euclidean")

        assert sim_matrix.shape == (3, 3)
        # Closer points should have higher similarity
        assert sim_matrix[0, 1] < sim_matrix[0, 0]  # (0,0) to (1,0) vs self

    def test_compute_similarity_dot(self, clustering):
        """Test dot product similarity."""
        embeddings = np.array([
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 1.0],
        ])

        sim_matrix = clustering.compute_similarity(embeddings, metric="dot")

        assert sim_matrix.shape == (3, 3)
        # Check symmetry
        assert np.allclose(sim_matrix, sim_matrix.T)

    def test_compute_similarity_invalid_metric(self, clustering):
        """Test that invalid similarity metric raises error."""
        embeddings = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Unknown similarity metric"):
            clustering.compute_similarity(embeddings, metric="invalid")

    def test_cluster_kmeans_basic(self, clustering):
        """Test basic K-means clustering."""
        # Create simple 2D embeddings with clear clusters
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
        ])

        labels, centroids = clustering.cluster_kmeans(embeddings, n_clusters=2, random_state=42)

        assert labels.shape == (4,)
        assert centroids.shape == (2, 2)
        # Verify that similar points are in the same cluster
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_cluster_hierarchical_basic(self, clustering):
        """Test basic hierarchical clustering."""
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
        ])

        labels = clustering.cluster_hierarchical(embeddings, n_clusters=2, linkage="ward")

        assert labels.shape == (4,)
        # Verify that similar points are in the same cluster
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]

    def test_cluster_hierarchical_linkages(self, clustering):
        """Test different linkage methods for hierarchical clustering."""
        embeddings = np.random.rand(10, 5)

        for linkage in ["ward", "complete", "average", "single"]:
            labels = clustering.cluster_hierarchical(embeddings, n_clusters=3, linkage=linkage)
            assert labels.shape == (10,)
            assert len(np.unique(labels)) == 3

    def test_name_cluster_basic(self, clustering):
        """Test basic cluster naming."""
        texts = ["Python programming is great", "I love Python", "Python is awesome"]
        name = clustering.name_cluster(texts)

        assert len(name) > 0
        assert len(name) <= 50

    def test_name_cluster_empty(self, clustering):
        """Test naming an empty cluster."""
        name = clustering.name_cluster([])
        assert name == "Empty Cluster"

    def test_name_cluster_long_text(self, clustering):
        """Test cluster naming with very long texts."""
        long_text = "A" * 100
        texts = [long_text]

        name = clustering.name_cluster(texts, max_length=50)

        assert len(name) <= 50
        assert name.endswith("...")

    def test_summarize_cluster_basic(self, clustering):
        """Test basic cluster summarization."""
        texts = ["Python is great", "I love Python"]
        summary = clustering.summarize_cluster(texts)

        assert "Python" in summary
        assert len(summary) > 0

    def test_summarize_cluster_empty(self, clustering):
        """Test summarizing an empty cluster."""
        summary = clustering.summarize_cluster([])
        assert summary == "No texts in cluster"

    def test_summarize_cluster_truncation(self, clustering):
        """Test that long summaries are truncated."""
        texts = ["A" * 100, "B" * 100, "C" * 100]
        summary = clustering.summarize_cluster(texts, max_length=50)

        assert len(summary) <= 50
        assert summary.endswith("...")

    def test_score_cluster_basic(self, clustering):
        """Test basic cluster quality scoring."""
        # Create embeddings that are close together (good cluster)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ])
        centroid = embeddings.mean(axis=0)

        score = clustering.score_cluster(embeddings, centroid)

        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high for cohesive cluster

    def test_score_cluster_empty(self, clustering):
        """Test scoring an empty cluster."""
        embeddings = np.array([]).reshape(0, 3)
        centroid = np.array([0.0, 0.0, 0.0])

        score = clustering.score_cluster(embeddings, centroid)

        assert score == 0.0

    def test_cluster_end_to_end(self, clustering, mock_model):
        """Test end-to-end clustering workflow."""
        texts = [
            "Python is great",
            "I love Python",
            "Java is verbose",
            "C++ is fast",
        ]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            results = clustering.cluster(texts, n_clusters=2, method="kmeans", random_state=42)

            assert len(results) == 2
            assert all(isinstance(r, ClusterResult) for r in results)

            # Check that all items are assigned
            all_items = []
            for result in results:
                all_items.extend(result.items)
            assert sorted(all_items) == [0, 1, 2, 3]

            # Check that each cluster has required fields
            for result in results:
                assert result.cluster_id >= 0
                assert len(result.items) > 0
                assert result.centroid is not None
                assert len(result.name) > 0
                assert len(result.summary) > 0
                assert 0.0 <= result.quality_score <= 1.0
                assert "size" in result.metadata
                assert result.metadata["method"] == "kmeans"

    def test_cluster_with_hierarchical(self, clustering, mock_model):
        """Test clustering with hierarchical method."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            results = clustering.cluster(texts, n_clusters=2, method="hierarchical")

            assert len(results) == 2
            assert all(r.metadata["method"] == "hierarchical" for r in results)

    def test_cluster_empty_texts(self, clustering):
        """Test clustering with empty text list."""
        results = clustering.cluster([], n_clusters=2)

        assert results == []

    def test_cluster_more_clusters_than_texts(self, clustering, mock_model):
        """Test clustering when n_clusters > number of texts."""
        texts = ["Text 1", "Text 2"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # Should auto-adjust to n_clusters=2
            results = clustering.cluster(texts, n_clusters=5, method="kmeans")

            assert len(results) == 2

    def test_cluster_invalid_method(self, clustering, mock_model):
        """Test clustering with invalid method."""
        texts = ["Text 1", "Text 2"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with pytest.raises(ValueError, match="Unknown clustering method"):
                clustering.cluster(texts, n_clusters=2, method="invalid")

    def test_cluster_single_text(self, clustering, mock_model):
        """Test clustering with a single text."""
        texts = ["Only one text"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            results = clustering.cluster(texts, n_clusters=1, method="kmeans")

            assert len(results) == 1
            assert results[0].items == [0]

    def test_cluster_reproducibility(self, clustering, mock_model):
        """Test that clustering with same random_state produces same results."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # Use fixed random seed to ensure mock returns consistent embeddings
            np.random.seed(42)
            results1 = clustering.cluster(texts, n_clusters=2, method="kmeans", random_state=42)

            # Reset mock to return same embeddings
            np.random.seed(42)
            results2 = clustering.cluster(texts, n_clusters=2, method="kmeans", random_state=42)

            # Results should be identical with same random state
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1.cluster_id == r2.cluster_id
                assert set(r1.items) == set(r2.items)


class TestClusteringIntegration:
    """Integration tests that verify clustering with real sentence-transformers (if available)."""

    @pytest.mark.skipif(
        not _check_sentence_transformers_available(),
        reason="sentence-transformers not installed"
    )
    def test_real_clustering_with_semantics(self):
        """Test real clustering to verify semantic similarity works."""
        clustering = SemanticClustering()

        # Texts with clear semantic groups
        texts = [
            "Python is a programming language",
            "Java is used for enterprise software",
            "Dogs are loyal pets",
            "Cats are independent animals",
            "Ruby is a dynamic language",
        ]

        results = clustering.cluster(texts, n_clusters=2, random_state=42)

        # Should create 2 clusters: programming languages vs animals
        # This is a basic smoke test - exact clustering may vary
        assert len(results) == 2
        assert all(len(r.items) > 0 for r in results)
