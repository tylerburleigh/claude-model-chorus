"""
Tests for review response parsing and validation.
"""

import json
from pathlib import Path

import pytest

# Skip all tests in this module if the review JSON file doesn't exist
REVIEW_JSON_PATH = Path(__file__).parent.parent / "review_provider_standardization.json"
pytestmark = pytest.mark.skipif(
    not REVIEW_JSON_PATH.exists(),
    reason="Review JSON file not found - these tests validate a specific review output file",
)


@pytest.fixture
def review_json_path():
    """Path to the review JSON file."""
    # File is in the workspace root (model_chorus directory)
    # __file__ is tests/test_review_response.py
    # .parent.parent gets us to workspace root
    return Path(__file__).parent.parent / "review_provider_standardization.json"


@pytest.fixture
def review_json_data(review_json_path):
    """Load review JSON data."""
    with open(review_json_path) as f:
        return json.load(f)


def test_review_json_exists(review_json_path):
    """Test that review JSON file exists."""
    assert (
        review_json_path.exists()
    ), f"Review JSON file not found at {review_json_path}"


def test_review_json_valid_json(review_json_path):
    """Test that review JSON file contains valid JSON."""
    with open(review_json_path) as f:
        data = json.load(f)
    assert isinstance(data, dict), "Review JSON should be a dictionary"


def test_review_json_has_required_fields(review_json_data):
    """Test that review JSON has all required top-level fields."""
    required_fields = [
        "overall_score",
        "recommendation",
        "dimension_scores",
        "issues",
        "strengths",
    ]
    for field in required_fields:
        assert field in review_json_data, f"Missing required field: {field}"


def test_overall_score_valid(review_json_data):
    """Test that overall_score is a valid integer between 0 and 10."""
    score = review_json_data["overall_score"]
    assert isinstance(score, int), "overall_score should be an integer"
    assert 0 <= score <= 10, "overall_score should be between 0 and 10"


def test_recommendation_valid(review_json_data):
    """Test that recommendation is a valid value."""
    recommendation = review_json_data["recommendation"]
    assert isinstance(recommendation, str), "recommendation should be a string"
    valid_recommendations = ["APPROVE", "REVISE", "REJECT"]
    assert (
        recommendation in valid_recommendations
    ), f"recommendation should be one of {valid_recommendations}"


def test_dimension_scores_structure(review_json_data):
    """Test that dimension_scores has the expected structure."""
    dimension_scores = review_json_data["dimension_scores"]
    assert isinstance(dimension_scores, dict), "dimension_scores should be a dictionary"

    # Check that all values are integers between 0 and 10
    for dimension, score in dimension_scores.items():
        assert isinstance(score, int), f"Score for {dimension} should be an integer"
        assert 0 <= score <= 10, f"Score for {dimension} should be between 0 and 10"


def test_issues_structure(review_json_data):
    """Test that issues list has the expected structure."""
    issues = review_json_data["issues"]
    assert isinstance(issues, list), "issues should be a list"

    required_issue_fields = ["severity", "category", "description"]
    for issue in issues:
        assert isinstance(issue, dict), "Each issue should be a dictionary"
        for field in required_issue_fields:
            assert field in issue, f"Issue missing required field: {field}"

        # Validate severity
        valid_severities = ["critical", "high", "medium", "low"]
        assert (
            issue["severity"].lower() in valid_severities
        ), f"Invalid severity: {issue['severity']}. Should be one of {valid_severities}"

        # Validate that description is a string
        assert isinstance(
            issue["description"], str
        ), "Issue description should be a string"
        assert len(issue["description"]) > 0, "Issue description should not be empty"


def test_strengths_structure(review_json_data):
    """Test that strengths list has the expected structure."""
    strengths = review_json_data["strengths"]
    assert isinstance(strengths, list), "strengths should be a list"

    for strength in strengths:
        assert isinstance(strength, dict), "Each strength should be a dictionary"
        assert "category" in strength, "Strength missing required field: category"
        assert "description" in strength, "Strength missing required field: description"
        assert isinstance(
            strength["description"], str
        ), "Strength description should be a string"
        assert (
            len(strength["description"]) > 0
        ), "Strength description should not be empty"


def test_review_response_consistency(review_json_data):
    """Test consistency between overall_score and recommendation."""
    score = review_json_data["overall_score"]
    recommendation = review_json_data["recommendation"]

    # Generally, high scores should be APPROVE, low scores should be REVISE/REJECT
    if score >= 8:
        assert recommendation in [
            "APPROVE"
        ], f"High score ({score}) should typically result in APPROVE, got {recommendation}"
    elif score <= 5:
        assert recommendation in [
            "REVISE",
            "REJECT",
        ], f"Low score ({score}) should typically result in REVISE or REJECT, got {recommendation}"


def test_issues_have_locations(review_json_data):
    """Test that issues have location information when applicable."""
    issues = review_json_data["issues"]
    for issue in issues:
        # Location is optional but if present should be a string
        if "location" in issue:
            assert isinstance(
                issue["location"], str
            ), "Issue location should be a string"

        # Suggestion is optional but if present should be a string
        if "suggestion" in issue:
            assert isinstance(
                issue["suggestion"], str
            ), "Issue suggestion should be a string"


def test_review_response_completeness(review_json_data):
    """Test that review response has meaningful content."""
    # Should have at least some issues or strengths
    issues = review_json_data["issues"]
    strengths = review_json_data["strengths"]

    assert (
        len(issues) > 0 or len(strengths) > 0
    ), "Review should have at least some issues or strengths"

    # Should have at least one dimension score
    dimension_scores = review_json_data["dimension_scores"]
    assert len(dimension_scores) > 0, "Review should have at least one dimension score"
