"""
Tests for IdeateWorkflow functionality.

Tests divergent brainstorming, convergent analysis, and complete ideation workflow.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_chorus.core.base_workflow import WorkflowResult, WorkflowStep
from model_chorus.providers.base_provider import GenerationRequest, GenerationResponse
from model_chorus.workflows.ideate.ideate_workflow import IdeateWorkflow


@pytest.fixture
def mock_provider():
    """Mock ModelProvider for testing."""
    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.validate_api_key = MagicMock(return_value=True)

    # Mock async generate method with realistic brainstorming response
    async def mock_generate(request: GenerationRequest) -> GenerationResponse:
        # Check prompt type to return appropriate response
        prompt_content = ""
        if request.prompt:
            prompt_content = request.prompt.lower()

        # Extraction prompt
        if "extract individual" in prompt_content or "discrete ideas" in prompt_content:
            content = """**[IDEA-1] Gamification System** (from practical)
Add points and badges to user actions to increase engagement through game mechanics.

**[IDEA-2] Interactive Tutorials** (from user-focused)
Create guided walkthroughs that teach users features hands-on instead of documentation.

**[IDEA-3] AI-Powered Onboarding** (from innovative)
Use machine learning to personalize the onboarding experience based on user behavior patterns.

**[IDEA-4] Progress Tracking** (from practical)
Show users their completion status with visual progress bars and milestones.

**[IDEA-5] Community Features** (from user-focused)
Enable peer-to-peer support through forums and chat for better user connection."""

        # Clustering prompt
        elif "cluster" in prompt_content or "thematic" in prompt_content:
            content = """**[CLUSTER-1] Engagement Mechanics**
Ideas focused on keeping users actively involved through game-like features and rewards.

Ideas in this cluster:
- IDEA-1: Core gamification system provides engagement foundation
- IDEA-4: Progress tracking complements gamification with visual feedback

**[CLUSTER-2] Learning & Guidance**
Ideas centered on helping users learn and navigate the system effectively.

Ideas in this cluster:
- IDEA-2: Interactive tutorials provide hands-on learning experience
- IDEA-3: AI personalization adapts guidance to individual needs

**[CLUSTER-3] Social Connection**
Ideas promoting user interaction and community building.

Ideas in this cluster:
- IDEA-5: Community features enable peer support and connection"""

        # Scoring prompt
        elif "score" in prompt_content or "evaluat" in prompt_content:
            content = """**[SCORE-cluster-1] Engagement Mechanics**

Scores:
- Feasibility: 4/5 - Well-established patterns, clear implementation path
- Impact: 4/5 - Proven to significantly increase user engagement metrics
- Novelty: 2/5 - Common approach but effective

Overall Score: 3.3/5
Recommendation: Medium Priority

**[SCORE-cluster-2] Learning & Guidance**

Scores:
- Feasibility: 3/5 - AI personalization requires ML infrastructure and data
- Impact: 5/5 - Directly addresses user onboarding pain points
- Novelty: 4/5 - AI-driven onboarding is cutting edge

Overall Score: 4.0/5
Recommendation: High Priority

**[SCORE-cluster-3] Social Connection**

Scores:
- Feasibility: 4/5 - Standard community platform features
- Impact: 3/5 - Beneficial but not core to primary user goals
- Novelty: 2/5 - Common feature in modern applications

Overall Score: 3.0/5
Recommendation: Medium Priority"""

        # Default brainstorming response
        else:
            content = f"Brainstorming response for: {prompt_content[:50]}..."

        return GenerationResponse(
            content=content,
            model="test-model-1",
            usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            stop_reason="end_turn",
        )

    provider.generate = AsyncMock(side_effect=mock_generate)

    # Mock async check_availability method
    async def mock_check_availability():
        return (True, None)  # is_available=True, error=None

    provider.check_availability = AsyncMock(side_effect=mock_check_availability)
    return provider


@pytest.fixture
def ideate_workflow(mock_provider):
    """Create IdeateWorkflow instance for testing."""
    return IdeateWorkflow(provider=mock_provider)


@pytest.fixture
def mock_brainstorming_result():
    """Create mock brainstorming result for convergent analysis testing."""
    steps = [
        WorkflowStep(
            step_number=1,
            content="Practical ideas: Gamification system with points and badges. Progress tracking dashboard.",
            model="test-model",
            metadata={"perspective": "practical", "tokens": 100},
        ),
        WorkflowStep(
            step_number=2,
            content="Innovative ideas: AI-powered personalized onboarding. Machine learning recommendation engine.",
            model="test-model",
            metadata={"perspective": "innovative", "tokens": 120},
        ),
        WorkflowStep(
            step_number=3,
            content="User-focused ideas: Interactive tutorials. Community forums for peer support.",
            model="test-model",
            metadata={"perspective": "user-focused", "tokens": 90},
        ),
    ]

    return WorkflowResult(
        success=True,
        synthesis="Combined brainstorming from all perspectives",
        steps=steps,
        metadata={
            "workflow": "ideate-parallel",
            "perspectives": ["practical", "innovative", "user-focused"],
            "models_used": ["test-model", "test-model", "test-model"],
            "pattern": "parallel",
            "total_ideas": 3,
        },
    )


class TestIdeateWorkflowInitialization:
    """Test IdeateWorkflow initialization."""

    def test_initialization_with_provider(self, mock_provider):
        """Test basic initialization with provider."""
        workflow = IdeateWorkflow(provider=mock_provider)

        assert workflow.name == "Ideate"
        assert workflow.description == "Creative ideation and brainstorming workflow"
        assert workflow.provider == mock_provider

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            IdeateWorkflow(provider=None)

    def test_validate_config(self, ideate_workflow):
        """Test config validation."""
        assert ideate_workflow.validate_config({}) is True

    def test_get_provider(self, ideate_workflow, mock_provider):
        """Test get_provider method."""
        assert ideate_workflow.get_provider() == mock_provider


class TestConvergentAnalysis:
    """Test convergent analysis functionality."""

    @pytest.mark.asyncio
    async def test_convergent_analysis_with_brainstorming_result(
        self, ideate_workflow, mock_brainstorming_result
    ):
        """Test convergent analysis with mock brainstorming result."""
        result = await ideate_workflow.run_convergent_analysis(
            brainstorming_result=mock_brainstorming_result
        )

        assert result.success is True
        assert len(result.steps) == 3  # extraction, clustering, scoring
        # Verify steps exist with content
        assert result.steps[0].content is not None
        assert result.steps[1].content is not None
        assert result.steps[2].content is not None

    @pytest.mark.asyncio
    async def test_convergent_analysis_metadata(
        self, ideate_workflow, mock_brainstorming_result
    ):
        """Test that convergent analysis includes proper metadata."""
        result = await ideate_workflow.run_convergent_analysis(
            brainstorming_result=mock_brainstorming_result,
            scoring_criteria=["feasibility", "impact", "novelty"],
        )

        assert "workflow" in result.metadata
        assert result.metadata["workflow"] == "ideate-convergent"
        assert "scoring_criteria" in result.metadata
        assert result.metadata["scoring_criteria"] == [
            "feasibility",
            "impact",
            "novelty",
        ]

    @pytest.mark.asyncio
    async def test_convergent_analysis_raises_error_on_empty_result(
        self, ideate_workflow
    ):
        """Test that convergent analysis raises error with no steps."""
        empty_result = WorkflowResult(
            success=True, synthesis="Empty", steps=[], metadata={}
        )

        with pytest.raises(ValueError, match="Brainstorming result must have steps"):
            await ideate_workflow.run_convergent_analysis(
                brainstorming_result=empty_result
            )

    @pytest.mark.asyncio
    async def test_convergent_analysis_custom_criteria(
        self, ideate_workflow, mock_brainstorming_result
    ):
        """Test convergent analysis with custom scoring criteria."""
        custom_criteria = ["feasibility", "impact", "user_value"]

        result = await ideate_workflow.run_convergent_analysis(
            brainstorming_result=mock_brainstorming_result,
            scoring_criteria=custom_criteria,
        )

        assert result.metadata["scoring_criteria"] == custom_criteria


class TestIdeaExtraction:
    """Test idea extraction from brainstorming results."""

    @pytest.mark.asyncio
    async def test_extract_ideas_from_brainstorming(
        self, ideate_workflow, mock_brainstorming_result
    ):
        """Test idea extraction returns WorkflowStep with ideas."""
        extraction_step = await ideate_workflow._extract_ideas(
            brainstorming_result=mock_brainstorming_result
        )

        assert extraction_step.content is not None
        assert "extracted_ideas" in extraction_step.metadata
        assert "num_ideas" in extraction_step.metadata
        ideas = extraction_step.metadata["extracted_ideas"]
        assert len(ideas) > 0
        assert all("id" in idea for idea in ideas)
        assert all("label" in idea for idea in ideas)
        assert all("description" in idea for idea in ideas)

    @pytest.mark.asyncio
    async def test_extract_ideas_preserves_perspectives(
        self, ideate_workflow, mock_brainstorming_result
    ):
        """Test that idea extraction preserves perspective information."""
        extraction_step = await ideate_workflow._extract_ideas(
            brainstorming_result=mock_brainstorming_result
        )

        ideas = extraction_step.metadata["extracted_ideas"]
        # Check that ideas have perspective information
        assert all("perspective" in idea for idea in ideas)


class TestIdeaClustering:
    """Test idea clustering functionality."""

    @pytest.mark.asyncio
    async def test_cluster_ideas(self, ideate_workflow, mock_brainstorming_result):
        """Test idea clustering returns WorkflowStep with clusters."""
        # First extract ideas
        extraction_step = await ideate_workflow._extract_ideas(
            brainstorming_result=mock_brainstorming_result
        )

        # Then cluster them
        clustering_step = await ideate_workflow._cluster_ideas(
            extraction_step=extraction_step
        )

        assert clustering_step.content is not None
        assert "clusters" in clustering_step.metadata
        assert "num_clusters" in clustering_step.metadata

        clusters = clustering_step.metadata["clusters"]
        assert len(clusters) > 0
        assert all("id" in cluster for cluster in clusters)
        assert all("theme" in cluster for cluster in clusters)
        assert all("ideas" in cluster for cluster in clusters)


class TestIdeaScoring:
    """Test idea scoring functionality."""

    @pytest.mark.asyncio
    async def test_score_ideas(self, ideate_workflow, mock_brainstorming_result):
        """Test idea scoring returns WorkflowStep with scores."""
        # Extract and cluster ideas first
        extraction_step = await ideate_workflow._extract_ideas(
            brainstorming_result=mock_brainstorming_result
        )
        clustering_step = await ideate_workflow._cluster_ideas(
            extraction_step=extraction_step
        )

        # Score the clusters
        scoring_step = await ideate_workflow._score_ideas(
            clustering_step=clustering_step,
            scoring_criteria=["feasibility", "impact", "novelty"],
        )

        assert scoring_step.content is not None
        assert "scored_clusters" in scoring_step.metadata
        assert "scoring_criteria" in scoring_step.metadata

        scored_clusters = scoring_step.metadata["scored_clusters"]
        # Check structure of scored clusters if present
        assert isinstance(scored_clusters, list)
        if len(scored_clusters) > 0:
            assert all("overall_score" in cluster for cluster in scored_clusters)
            assert all("recommendation" in cluster for cluster in scored_clusters)


class TestCompleteIdeation:
    """Test complete ideation workflow (divergent + convergent)."""

    @pytest.mark.asyncio
    async def test_run_complete_ideation(self, mock_provider):
        """Test complete ideation workflow execution."""
        workflow = IdeateWorkflow(provider=mock_provider)

        # Create provider map with multiple providers for parallel brainstorming
        provider_map = {
            "model1": mock_provider,
            "model2": mock_provider,
            "model3": mock_provider,
        }

        result = await workflow.run_complete_ideation(
            prompt="How can we improve user onboarding?", provider_map=provider_map
        )

        assert result.success is True
        assert "workflow" in result.metadata
        assert result.metadata["workflow"] == "ideate-complete"
        assert "divergent_phase" in result.metadata
        assert "convergent_phase" in result.metadata

        # Check that steps include both divergent and convergent phases
        # Divergent: 3 perspectives
        # Convergent: extraction, clustering, scoring
        assert len(result.steps) >= 6

    @pytest.mark.asyncio
    async def test_complete_ideation_with_custom_parameters(self, mock_provider):
        """Test complete ideation with custom perspectives and criteria."""
        workflow = IdeateWorkflow(provider=mock_provider)

        provider_map = {"model1": mock_provider, "model2": mock_provider}

        custom_perspectives = ["technical", "business"]
        custom_criteria = ["feasibility", "impact", "effort"]

        result = await workflow.run_complete_ideation(
            prompt="How should we architect our microservices?",
            provider_map=provider_map,
            perspectives=custom_perspectives,
            scoring_criteria=custom_criteria,
        )

        assert result.success is True
        assert result.metadata["perspectives"] == custom_perspectives
        assert result.metadata["scoring_criteria"] == custom_criteria
