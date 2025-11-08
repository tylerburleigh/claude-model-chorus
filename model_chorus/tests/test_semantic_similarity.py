"""
Unit tests for semantic similarity computation in ARGUMENT workflow.

Tests embedding computation, similarity scoring, claim comparison,
and integration with Citation/CitationMap models.
"""

import pytest
import numpy as np

from model_chorus.core.models import Citation, CitationMap
from model_chorus.workflows.argument.semantic import (
    compute_embedding,
    cosine_similarity,
    compute_claim_similarity,
    find_similar_claims,
    compute_claim_similarity_batch,
    add_similarity_to_citation,
    find_duplicate_claims,
    cluster_claims_kmeans,
    cluster_claims_hierarchical,
    get_cluster_representative,
    compute_cluster_statistics,
)


class TestEmbeddingComputation:
    """Test embedding computation and caching."""

    def test_compute_embedding_returns_array(self):
        """Test that compute_embedding returns numpy array."""
        text = "Machine learning improves accuracy"
        embedding = compute_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1  # Should be 1D vector
        assert len(embedding) > 0  # Should have dimensions

    def test_compute_embedding_normalization(self):
        """Test that embeddings are normalized (unit length)."""
        text = "AI enhances precision"
        embedding = compute_embedding(text)

        # Compute L2 norm (should be 1.0 for normalized embeddings)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5  # Allow for floating point errors

    def test_compute_embedding_caching(self):
        """Test that embedding computation is cached."""
        text = "Test caching behavior"

        # First call computes embedding
        emb1 = compute_embedding(text)

        # Second call should return cached result (same object)
        emb2 = compute_embedding(text)

        # Should be identical arrays
        assert np.array_equal(emb1, emb2)

    def test_compute_embedding_case_insensitive(self):
        """Test that embedding computation is case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"

        emb1 = compute_embedding(text1)
        emb2 = compute_embedding(text2)

        # Should be identical (normalized to lowercase)
        assert np.array_equal(emb1, emb2)

    def test_compute_embedding_empty_string(self):
        """Test embedding computation for empty string."""
        # Empty or whitespace-only strings should still produce embeddings
        embedding = compute_embedding("")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_cosine_similarity_identical(self):
        """Test similarity between identical embeddings."""
        text = "Identical text"
        emb = compute_embedding(text)

        similarity = cosine_similarity(emb, emb)

        assert abs(similarity - 1.0) < 1e-5  # Should be 1.0

    def test_cosine_similarity_range(self):
        """Test that similarity is in [0, 1] range."""
        emb1 = compute_embedding("Machine learning")
        emb2 = compute_embedding("Weather forecast")

        similarity = cosine_similarity(emb1, emb2)

        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_similar_text(self):
        """Test similarity for similar texts."""
        emb1 = compute_embedding("Artificial intelligence improves accuracy")
        emb2 = compute_embedding("AI enhances precision")

        similarity = cosine_similarity(emb1, emb2)

        # Similar texts should have high similarity
        assert similarity > 0.6

    def test_cosine_similarity_different_text(self):
        """Test similarity for different texts."""
        emb1 = compute_embedding("Machine learning algorithms")
        emb2 = compute_embedding("Weather is sunny today")

        similarity = cosine_similarity(emb1, emb2)

        # Different texts should have low similarity
        assert similarity < 0.5


class TestClaimSimilarity:
    """Test claim-to-claim similarity computation."""

    def test_compute_claim_similarity_identical(self):
        """Test similarity for identical claims."""
        claim = "Machine learning improves accuracy by 23%"

        similarity = compute_claim_similarity(claim, claim)

        assert abs(similarity - 1.0) < 1e-5

    def test_compute_claim_similarity_similar(self):
        """Test similarity for similar claims."""
        claim1 = "AI improves software quality"
        claim2 = "Artificial intelligence enhances code quality"

        similarity = compute_claim_similarity(claim1, claim2)

        assert similarity > 0.7  # Should be quite similar

    def test_compute_claim_similarity_different(self):
        """Test similarity for different claims."""
        claim1 = "AI improves accuracy"
        claim2 = "Weather is nice today"

        similarity = compute_claim_similarity(claim1, claim2)

        assert similarity < 0.3  # Should be very different

    def test_compute_claim_similarity_symmetric(self):
        """Test that similarity is symmetric."""
        claim1 = "Machine learning models"
        claim2 = "Deep learning networks"

        sim12 = compute_claim_similarity(claim1, claim2)
        sim21 = compute_claim_similarity(claim2, claim1)

        assert abs(sim12 - sim21) < 1e-5


class TestFindSimilarClaims:
    """Test finding similar claims in citation maps."""

    @pytest.fixture
    def sample_citation_maps(self):
        """Create sample citation maps for testing."""
        return [
            CitationMap(
                claim_id="claim-1",
                claim_text="Machine learning improves accuracy",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="claim-2",
                claim_text="AI enhances precision and recall",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="claim-3",
                claim_text="Weather forecasting uses numerical models",
                citations=[],
                strength=0.8,
            ),
            CitationMap(
                claim_id="claim-4",
                claim_text="ML models achieve better results",
                citations=[],
                strength=0.88,
            ),
        ]

    def test_find_similar_claims_basic(self, sample_citation_maps):
        """Test basic similar claim finding."""
        query = "Machine learning achieves high accuracy"

        results = find_similar_claims(
            query,
            sample_citation_maps,
            threshold=0.7,
        )

        # Should find at least the very similar claims
        assert len(results) >= 1
        assert all(isinstance(cm, CitationMap) and isinstance(score, float) for cm, score in results)

    def test_find_similar_claims_threshold(self, sample_citation_maps):
        """Test threshold filtering."""
        query = "AI improves results"

        # Lower threshold should return more results
        results_low = find_similar_claims(query, sample_citation_maps, threshold=0.5)
        results_high = find_similar_claims(query, sample_citation_maps, threshold=0.8)

        assert len(results_low) >= len(results_high)

    def test_find_similar_claims_top_k(self, sample_citation_maps):
        """Test top_k limiting."""
        query = "ML enhances performance"

        results = find_similar_claims(
            query,
            sample_citation_maps,
            threshold=0.3,
            top_k=2,
        )

        assert len(results) <= 2

    def test_find_similar_claims_sorted(self, sample_citation_maps):
        """Test that results are sorted by similarity."""
        query = "Machine learning models"

        results = find_similar_claims(query, sample_citation_maps, threshold=0.5)

        # Check that scores are in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_claims_empty_list(self):
        """Test with empty citation map list."""
        query = "Test query"

        results = find_similar_claims(query, [])

        assert results == []


class TestBatchSimilarity:
    """Test batch similarity computation."""

    def test_compute_claim_similarity_batch_shape(self):
        """Test that batch computation returns correct shape."""
        claims = [
            "AI improves quality",
            "ML enhances accuracy",
            "Weather is sunny",
        ]

        sim_matrix = compute_claim_similarity_batch(claims)

        assert sim_matrix.shape == (3, 3)

    def test_compute_claim_similarity_batch_diagonal(self):
        """Test that diagonal is all 1.0 (self-similarity)."""
        claims = [
            "Claim one",
            "Claim two",
            "Claim three",
        ]

        sim_matrix = compute_claim_similarity_batch(claims)

        # Diagonal should be all 1.0
        diagonal = np.diag(sim_matrix)
        assert np.allclose(diagonal, 1.0, atol=1e-5)

    def test_compute_claim_similarity_batch_symmetric(self):
        """Test that similarity matrix is symmetric."""
        claims = [
            "AI models",
            "ML algorithms",
            "Data science",
        ]

        sim_matrix = compute_claim_similarity_batch(claims)

        # Matrix should be symmetric
        assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-5)

    def test_compute_claim_similarity_batch_range(self):
        """Test that all values are in [0, 1] range."""
        claims = [
            "Machine learning",
            "Deep learning",
            "Weather forecast",
        ]

        sim_matrix = compute_claim_similarity_batch(claims)

        assert np.all(sim_matrix >= 0.0)
        assert np.all(sim_matrix <= 1.0)


class TestCitationIntegration:
    """Test integration with Citation model."""

    def test_add_similarity_to_citation_with_snippet(self):
        """Test adding similarity to citation with snippet."""
        citation = Citation(
            source="https://arxiv.org/abs/2401.12345",
            confidence=0.95,
            snippet="Machine learning models improve accuracy by 23%",
            metadata={},
        )
        reference_claim = "ML enhances prediction precision"

        enhanced = add_similarity_to_citation(citation, reference_claim)

        assert "similarity_score" in enhanced.metadata
        assert "similarity_model" in enhanced.metadata
        assert 0.0 <= enhanced.metadata["similarity_score"] <= 1.0

    def test_add_similarity_to_citation_without_snippet(self):
        """Test adding similarity to citation without snippet."""
        citation = Citation(
            source="paper.pdf",
            confidence=0.8,
            snippet=None,
            metadata={},
        )
        reference_claim = "AI improves results"

        enhanced = add_similarity_to_citation(citation, reference_claim)

        assert enhanced.metadata["similarity_score"] == 0.0
        assert "similarity_note" in enhanced.metadata

    def test_add_similarity_preserves_existing_metadata(self):
        """Test that existing metadata is preserved."""
        citation = Citation(
            source="source.pdf",
            confidence=0.9,
            snippet="Test snippet",
            metadata={"author": "Smith", "year": 2024},
        )

        enhanced = add_similarity_to_citation(citation, "Test claim")

        assert enhanced.metadata["author"] == "Smith"
        assert enhanced.metadata["year"] == 2024
        assert "similarity_score" in enhanced.metadata


class TestDuplicateDetection:
    """Test duplicate claim detection."""

    def test_find_duplicate_claims_basic(self):
        """Test basic duplicate detection."""
        maps = [
            CitationMap(
                claim_id="claim-1",
                claim_text="AI improves accuracy",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="claim-2",
                claim_text="Artificial intelligence enhances precision",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="claim-3",
                claim_text="Weather is sunny today",
                citations=[],
                strength=0.8,
            ),
        ]

        duplicates = find_duplicate_claims(maps, threshold=0.7)

        # Should find at least one group of duplicates
        assert isinstance(duplicates, list)
        for group in duplicates:
            assert len(group) >= 2

    def test_find_duplicate_claims_high_threshold(self):
        """Test with high threshold (very similar only)."""
        maps = [
            CitationMap(
                claim_id="claim-1",
                claim_text="Machine learning",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="claim-2",
                claim_text="Machine learning",  # Exact duplicate
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="claim-3",
                claim_text="Deep learning networks",
                citations=[],
                strength=0.8,
            ),
        ]

        duplicates = find_duplicate_claims(maps, threshold=0.95)

        # Should find the exact duplicates
        assert len(duplicates) >= 1

    def test_find_duplicate_claims_no_duplicates(self):
        """Test with no duplicates."""
        maps = [
            CitationMap(
                claim_id="claim-1",
                claim_text="Weather is sunny",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="claim-2",
                claim_text="Machine learning models",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="claim-3",
                claim_text="Database optimization techniques",
                citations=[],
                strength=0.8,
            ),
        ]

        duplicates = find_duplicate_claims(maps, threshold=0.9)

        # Should find no duplicates
        assert duplicates == []

    def test_find_duplicate_claims_empty_list(self):
        """Test with empty list."""
        duplicates = find_duplicate_claims([], threshold=0.9)

        assert duplicates == []

    def test_find_duplicate_claims_single_item(self):
        """Test with single citation map."""
        maps = [
            CitationMap(
                claim_id="claim-1",
                claim_text="Single claim",
                citations=[],
                strength=0.9,
            ),
        ]

        duplicates = find_duplicate_claims(maps, threshold=0.9)

        assert duplicates == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_claim_text(self):
        """Test handling of empty claim text."""
        # Should not crash
        embedding = compute_embedding("")
        assert isinstance(embedding, np.ndarray)

    def test_very_long_claim(self):
        """Test handling of very long claim text."""
        long_claim = "Machine learning " * 1000  # Very long text

        embedding = compute_embedding(long_claim)
        assert isinstance(embedding, np.ndarray)

    def test_special_characters(self):
        """Test handling of special characters."""
        claim = "AI & ML improve accuracy (23% ± 2%) @ p<0.05 !"

        embedding = compute_embedding(claim)
        assert isinstance(embedding, np.ndarray)

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        claim = "机器学习提高准确性"  # Chinese text

        embedding = compute_embedding(claim)
        assert isinstance(embedding, np.ndarray)


class TestKMeansClustering:
    """Test K-means clustering functionality."""

    @pytest.fixture
    def diverse_citation_maps(self):
        """Create citation maps with diverse topics for clustering."""
        return [
            # ML/AI cluster
            CitationMap(
                claim_id="ml-1",
                claim_text="Machine learning improves accuracy",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="ml-2",
                claim_text="AI enhances prediction quality",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="ml-3",
                claim_text="Neural networks achieve better results",
                citations=[],
                strength=0.88,
            ),
            # Weather cluster
            CitationMap(
                claim_id="weather-1",
                claim_text="Weather forecasting uses numerical models",
                citations=[],
                strength=0.8,
            ),
            CitationMap(
                claim_id="weather-2",
                claim_text="Climate patterns affect temperature",
                citations=[],
                strength=0.82,
            ),
            # Database cluster
            CitationMap(
                claim_id="db-1",
                claim_text="Database optimization improves query speed",
                citations=[],
                strength=0.87,
            ),
            CitationMap(
                claim_id="db-2",
                claim_text="Indexing enhances database performance",
                citations=[],
                strength=0.86,
            ),
        ]

    def test_cluster_claims_kmeans_basic(self, diverse_citation_maps):
        """Test basic K-means clustering."""
        clusters = cluster_claims_kmeans(diverse_citation_maps, n_clusters=3)

        assert len(clusters) == 3
        assert all(isinstance(cluster, list) for cluster in clusters)
        # All citation maps should be assigned to exactly one cluster
        total_assigned = sum(len(cluster) for cluster in clusters)
        assert total_assigned == len(diverse_citation_maps)

    def test_cluster_claims_kmeans_empty_list(self):
        """Test K-means with empty list."""
        clusters = cluster_claims_kmeans([])

        assert clusters == []

    def test_cluster_claims_kmeans_single_cluster(self, diverse_citation_maps):
        """Test K-means with n_clusters=1."""
        clusters = cluster_claims_kmeans(diverse_citation_maps, n_clusters=1)

        assert len(clusters) == 1
        assert len(clusters[0]) == len(diverse_citation_maps)

    def test_cluster_claims_kmeans_too_many_clusters(self):
        """Test error when n_clusters > number of items."""
        maps = [
            CitationMap(claim_id="1", claim_text="Test", citations=[], strength=0.9),
            CitationMap(claim_id="2", claim_text="Test2", citations=[], strength=0.9),
        ]

        with pytest.raises(ValueError, match="n_clusters .* cannot be greater"):
            cluster_claims_kmeans(maps, n_clusters=5)

    def test_cluster_claims_kmeans_reproducibility(self, diverse_citation_maps):
        """Test that clustering is reproducible with same random_state."""
        clusters1 = cluster_claims_kmeans(diverse_citation_maps, n_clusters=3, random_state=42)
        clusters2 = cluster_claims_kmeans(diverse_citation_maps, n_clusters=3, random_state=42)

        # Should produce same clusters
        for c1, c2 in zip(clusters1, clusters2):
            assert len(c1) == len(c2)


class TestHierarchicalClustering:
    """Test hierarchical clustering functionality."""

    @pytest.fixture
    def sample_maps(self):
        """Create sample citation maps."""
        return [
            CitationMap(
                claim_id="1",
                claim_text="Machine learning models",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="2",
                claim_text="AI algorithms",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="3",
                claim_text="Weather forecasting",
                citations=[],
                strength=0.8,
            ),
            CitationMap(
                claim_id="4",
                claim_text="Climate prediction",
                citations=[],
                strength=0.82,
            ),
        ]

    def test_cluster_claims_hierarchical_basic(self, sample_maps):
        """Test basic hierarchical clustering."""
        clusters = cluster_claims_hierarchical(sample_maps, n_clusters=2)

        assert len(clusters) == 2
        total_assigned = sum(len(cluster) for cluster in clusters)
        assert total_assigned == len(sample_maps)

    def test_cluster_claims_hierarchical_empty_list(self):
        """Test hierarchical clustering with empty list."""
        clusters = cluster_claims_hierarchical([])

        assert clusters == []

    def test_cluster_claims_hierarchical_linkage_methods(self, sample_maps):
        """Test different linkage methods."""
        for linkage in ["ward", "complete", "average", "single"]:
            clusters = cluster_claims_hierarchical(
                sample_maps,
                n_clusters=2,
                linkage_method=linkage
            )

            assert len(clusters) == 2

    def test_cluster_claims_hierarchical_too_many_clusters(self):
        """Test error when n_clusters > number of items."""
        maps = [
            CitationMap(claim_id="1", claim_text="Test", citations=[], strength=0.9),
        ]

        with pytest.raises(ValueError, match="n_clusters .* cannot be greater"):
            cluster_claims_hierarchical(maps, n_clusters=2)


class TestClusterRepresentative:
    """Test cluster representative selection."""

    def test_get_cluster_representative_basic(self):
        """Test getting representative from cluster."""
        cluster = [
            CitationMap(
                claim_id="1",
                claim_text="Machine learning models",
                citations=[],
                strength=0.9,
            ),
            CitationMap(
                claim_id="2",
                claim_text="ML algorithms and techniques",
                citations=[],
                strength=0.85,
            ),
            CitationMap(
                claim_id="3",
                claim_text="Deep learning networks",
                citations=[],
                strength=0.88,
            ),
        ]

        representative = get_cluster_representative(cluster)

        assert representative in cluster
        assert isinstance(representative, CitationMap)

    def test_get_cluster_representative_single_item(self):
        """Test representative when cluster has single item."""
        cluster = [
            CitationMap(
                claim_id="1",
                claim_text="Single claim",
                citations=[],
                strength=0.9,
            ),
        ]

        representative = get_cluster_representative(cluster)

        assert representative == cluster[0]

    def test_get_cluster_representative_empty_cluster(self):
        """Test error when cluster is empty."""
        with pytest.raises(ValueError, match="Cluster cannot be empty"):
            get_cluster_representative([])


class TestClusterStatistics:
    """Test cluster statistics computation."""

    def test_compute_cluster_statistics_basic(self):
        """Test basic statistics computation."""
        clusters = [
            # Cluster 1: 3 similar claims
            [
                CitationMap(claim_id="1", claim_text="AI models", citations=[], strength=0.9),
                CitationMap(claim_id="2", claim_text="ML algorithms", citations=[], strength=0.85),
                CitationMap(claim_id="3", claim_text="Deep learning", citations=[], strength=0.88),
            ],
            # Cluster 2: 2 similar claims
            [
                CitationMap(claim_id="4", claim_text="Weather forecast", citations=[], strength=0.8),
                CitationMap(claim_id="5", claim_text="Climate prediction", citations=[], strength=0.82),
            ],
        ]

        stats = compute_cluster_statistics(clusters)

        assert stats["num_clusters"] == 2
        assert stats["cluster_sizes"] == [3, 2]
        assert stats["avg_cluster_size"] == 2.5
        assert stats["min_cluster_size"] == 2
        assert stats["max_cluster_size"] == 3
        assert len(stats["representatives"]) == 2
        assert len(stats["intra_cluster_similarities"]) == 2
        assert 0.0 <= stats["avg_intra_cluster_similarity"] <= 1.0

    def test_compute_cluster_statistics_empty(self):
        """Test statistics for empty cluster list."""
        stats = compute_cluster_statistics([])

        assert stats["num_clusters"] == 0
        assert stats["cluster_sizes"] == []
        assert stats["avg_cluster_size"] == 0.0

    def test_compute_cluster_statistics_single_item_clusters(self):
        """Test statistics when clusters have single items."""
        clusters = [
            [CitationMap(claim_id="1", claim_text="Test 1", citations=[], strength=0.9)],
            [CitationMap(claim_id="2", claim_text="Test 2", citations=[], strength=0.8)],
        ]

        stats = compute_cluster_statistics(clusters)

        # Single-item clusters should have similarity of 1.0
        assert all(sim == 1.0 for sim in stats["intra_cluster_similarities"])


class TestClusteringIntegration:
    """Test end-to-end clustering workflows."""

    def test_kmeans_to_statistics_pipeline(self):
        """Test complete pipeline: cluster -> compute statistics."""
        maps = [
            CitationMap(claim_id=str(i), claim_text=f"Claim about topic {i % 3}", citations=[], strength=0.9)
            for i in range(9)
        ]

        # Cluster
        clusters = cluster_claims_kmeans(maps, n_clusters=3)

        # Compute statistics
        stats = compute_cluster_statistics(clusters)

        assert stats["num_clusters"] == 3
        assert sum(stats["cluster_sizes"]) == 9
        assert len(stats["representatives"]) == 3

    def test_hierarchical_to_statistics_pipeline(self):
        """Test complete pipeline with hierarchical clustering."""
        maps = [
            CitationMap(claim_id="ml-1", claim_text="Machine learning", citations=[], strength=0.9),
            CitationMap(claim_id="ml-2", claim_text="AI algorithms", citations=[], strength=0.85),
            CitationMap(claim_id="db-1", claim_text="Database optimization", citations=[], strength=0.8),
            CitationMap(claim_id="db-2", claim_text="Query performance", citations=[], strength=0.82),
        ]

        # Cluster
        clusters = cluster_claims_hierarchical(maps, n_clusters=2)

        # Get representatives
        representatives = [get_cluster_representative(cluster) for cluster in clusters]

        assert len(representatives) == 2
        assert all(rep in maps for rep in representatives)
