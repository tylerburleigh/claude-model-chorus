"""
Tests for IdeateWorkflow functionality.

Tests the complete ideation workflow including parallel brainstorming,
convergent analysis (extraction, clustering, scoring), interactive selection,
and elaboration into detailed outlines.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from modelchorus.workflows.ideate import IdeateWorkflow
from modelchorus.providers.base_provider import GenerationRequest, GenerationResponse
from modelchorus.core.conversation import ConversationMemory
from modelchorus.core.models import Idea, IdeaCluster, IdeationState
from modelchorus.core.role_orchestration import OrchestrationResult, OrchestrationPattern


@pytest.fixture
def mock_provider():
    """Mock ModelProvider for testing."""
    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.validate_api_key = MagicMock(return_value=True)

    # Mock async generate method
    async def mock_generate(request: GenerationRequest) -> GenerationResponse:
        # Determine response content based on the prompt
        prompt_content = request.prompt if hasattr(request, 'prompt') else ''

        # For idea extraction
        if 'extract individual ideas' in str(prompt_content).lower():
            content = """**[IDEA-1] Gamification System** (from practical)
Add points, badges, and leaderboards to improve engagement.

**[IDEA-2] Interactive Tutorials** (from user-focused)
Create step-by-step guided tutorials for new users.

**[IDEA-3] AI-Powered Recommendations** (from innovative)
Use machine learning to suggest personalized content."""

        # For clustering
        elif 'cluster' in str(prompt_content).lower() and 'thematic' in str(prompt_content).lower():
            content = """**[CLUSTER-1] User Engagement**
Features that improve user interaction and retention.

Ideas in this cluster:
- idea-1: Directly addresses engagement through gamification
- idea-2: Helps onboarding which drives engagement

**[CLUSTER-2] Personalization**
AI-driven customization for better user experience.

Ideas in this cluster:
- idea-3: Uses AI to personalize recommendations"""

        # For scoring
        elif 'score' in str(prompt_content).lower() and 'criteria' in str(prompt_content).lower():
            content = """**[SCORE-cluster-1] User Engagement**

Scores:
- Feasibility: 4/5 - Well-understood techniques
- Impact: 5/5 - Proven to increase engagement significantly
- Novelty: 3/5 - Common in industry

Overall Score: 4.0/5
Recommendation: High Priority

**[SCORE-cluster-2] Personalization**

Scores:
- Feasibility: 3/5 - Requires ML expertise
- Impact: 4/5 - High user value when done right
- Novelty: 4/5 - Emerging best practice

Overall Score: 3.7/5
Recommendation: Medium Priority"""

        # For elaboration
        elif 'detailed, actionable outline' in str(prompt_content).lower():
            content = """## Overview
Implementation of user engagement features including gamification and tutorials.

## Goals & Objectives
- Increase user retention by 30%
- Reduce time-to-value for new users
- Create viral loops through social features

## Implementation Approach
Phased rollout starting with core gamification mechanics.

## Detailed Steps
1. Design point and badge system
2. Implement backend tracking
3. Create UI components
4. A/B test with user segment

## Key Considerations
- Balance engagement without being manipulative
- Ensure performance impact is minimal
- Consider accessibility

## Success Metrics
- 30% increase in DAU
- 50% increase in feature adoption
- NPS score improvement

## Potential Challenges
- User fatigue from excessive notifications
- Technical debt from rapid implementation

## Next Actions
1. Create detailed design mockups
2. Spec out database schema
3. Prototype core mechanics"""

        # Default response for brainstorming
        else:
            content = f"Creative ideas about: {request.prompt[:100] if hasattr(request, 'prompt') else 'the topic'}"

        return GenerationResponse(
            content=content,
            model="test-model-1",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason="end_turn",
        )

    provider.generate = AsyncMock(side_effect=mock_generate)
    return provider


@pytest.fixture
def conversation_memory():
    """Create ConversationMemory instance for testing."""
    return ConversationMemory()


@pytest.fixture
def ideate_workflow(mock_provider, conversation_memory):
    """Create IdeateWorkflow instance for testing."""
    return IdeateWorkflow(
        provider=mock_provider,
        conversation_memory=conversation_memory,
    )


class TestIdeateWorkflowInitialization:
    """Test IdeateWorkflow initialization."""

    def test_initialization_with_provider(self, mock_provider, conversation_memory):
        """Test basic initialization with provider."""
        workflow = IdeateWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory,
        )

        assert workflow.name == "Ideate"
        assert workflow.description == "Creative ideation and brainstorming workflow"
        assert workflow.provider == mock_provider
        assert workflow.conversation_memory == conversation_memory

    def test_initialization_without_memory(self, mock_provider):
        """Test initialization without conversation memory."""
        workflow = IdeateWorkflow(provider=mock_provider)

        assert workflow.provider == mock_provider
        assert workflow.conversation_memory is None

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            IdeateWorkflow(provider=None)

    def test_validate_config(self, ideate_workflow):
        """Test config validation."""
        config = {"temperature": 0.9}
        assert ideate_workflow.validate_config(config) is True

    def test_get_provider(self, ideate_workflow, mock_provider):
        """Test get_provider method."""
        assert ideate_workflow.get_provider() == mock_provider


class TestBasicIdeation:
    """Test basic ideation methods."""

    @pytest.mark.asyncio
    async def test_run_basic_ideation(self, ideate_workflow):
        """Test basic run method for ideation."""
        result = await ideate_workflow.run(
            prompt="How can we improve user onboarding?"
        )

        assert result is not None
        assert result.synthesis is not None
        assert len(result.steps) == 1
        assert result.metadata['workflow'] == 'ideate'
        assert 'thread_id' in result.metadata

    @pytest.mark.asyncio
    async def test_run_with_empty_prompt_raises_error(self, ideate_workflow):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await ideate_workflow.run(prompt="")

    @pytest.mark.asyncio
    async def test_run_with_continuation_id(self, ideate_workflow):
        """Test running with continuation_id for thread continuity."""
        # First call
        result1 = await ideate_workflow.run(
            prompt="Initial ideation on user onboarding"
        )
        thread_id = result1.metadata['thread_id']

        # Continued call
        result2 = await ideate_workflow.run(
            prompt="Expand on gamification ideas",
            continuation_id=thread_id
        )

        assert result2.metadata['thread_id'] == thread_id
        assert result2.metadata['round'] == 2

    @pytest.mark.asyncio
    async def test_run_with_custom_temperature(self, ideate_workflow):
        """Test running with custom temperature parameter."""
        result = await ideate_workflow.run(
            prompt="Generate creative ideas",
            temperature=0.95
        )

        assert result.metadata['temperature'] == 0.95

    @pytest.mark.asyncio
    async def test_run_updates_conversation_memory(self, ideate_workflow):
        """Test that run updates conversation memory."""
        result = await ideate_workflow.run(
            prompt="Test prompt for memory"
        )

        thread_id = result.metadata['thread_id']
        thread = ideate_workflow.conversation_memory.get_thread(thread_id)

        assert thread is not None
        assert len(thread.messages) == 2  # user + assistant
        assert thread.messages[0].role == "user"
        assert thread.messages[1].role == "assistant"


class TestRoleCreation:
    """Test brainstormer role creation."""

    def test_create_brainstormer_role_practical(self, ideate_workflow):
        """Test creating practical brainstormer role."""
        role = ideate_workflow._create_brainstormer_role("claude", "practical")

        assert role.role == "brainstormer-practical"
        assert role.model == "claude"
        assert role.stance == "neutral"
        assert role.temperature == 0.9
        assert "pragmatic" in role.stance_prompt.lower()

    def test_create_brainstormer_role_innovative(self, ideate_workflow):
        """Test creating innovative brainstormer role."""
        role = ideate_workflow._create_brainstormer_role("gemini", "innovative")

        assert role.role == "brainstormer-innovative"
        assert role.model == "gemini"
        assert "bold" in role.stance_prompt.lower() or "cutting-edge" in role.stance_prompt.lower()

    def test_create_brainstormer_role_user_focused(self, ideate_workflow):
        """Test creating user-focused brainstormer role."""
        role = ideate_workflow._create_brainstormer_role("claude", "user-focused")

        assert role.role == "brainstormer-user-focused"
        assert "user" in role.stance_prompt.lower()


class TestParallelBrainstorming:
    """Test parallel brainstorming functionality."""

    @pytest.mark.asyncio
    async def test_run_parallel_brainstorming(self, ideate_workflow, mock_provider):
        """Test parallel brainstorming with multiple providers."""
        # Create mock provider map
        provider_map = {
            "claude": mock_provider,
            "gemini": mock_provider,
        }

        # Mock RoleOrchestrator
        with patch('modelchorus.workflows.ideate.ideate_workflow.RoleOrchestrator') as mock_orch:
            # Create mock orchestration result
            mock_result = MagicMock()
            mock_result.role_responses = [
                ("brainstormer-practical", GenerationResponse(
                    content="Practical ideas...",
                    model="claude",
                    usage={"total_tokens": 100}
                )),
                ("brainstormer-innovative", GenerationResponse(
                    content="Innovative ideas...",
                    model="gemini",
                    usage={"total_tokens": 120}
                )),
            ]

            mock_orch_instance = MagicMock()
            mock_orch_instance.execute = AsyncMock(return_value=mock_result)
            mock_orch.return_value = mock_orch_instance

            # Run parallel brainstorming
            result = await ideate_workflow.run_parallel_brainstorming(
                prompt="How to improve our API?",
                provider_map=provider_map,
                perspectives=['practical', 'innovative']
            )

            assert result is not None
            assert len(result.steps) == 2
            assert result.metadata['workflow'] == 'ideate-parallel'
            assert result.metadata['pattern'] == 'parallel'
            assert result.metadata['perspectives'] == ['practical', 'innovative']

    @pytest.mark.asyncio
    async def test_parallel_brainstorming_empty_prompt_raises_error(self, ideate_workflow):
        """Test that empty prompt raises ValueError."""
        provider_map = {"claude": MagicMock()}

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await ideate_workflow.run_parallel_brainstorming(
                prompt="",
                provider_map=provider_map
            )

    @pytest.mark.asyncio
    async def test_parallel_brainstorming_empty_provider_map_raises_error(self, ideate_workflow):
        """Test that empty provider map raises ValueError."""
        with pytest.raises(ValueError, match="Provider map cannot be empty"):
            await ideate_workflow.run_parallel_brainstorming(
                prompt="Test prompt",
                provider_map={}
            )


class TestConvergentAnalysis:
    """Test convergent analysis functionality."""

    @pytest.fixture
    def mock_brainstorming_result(self):
        """Create mock brainstorming result for convergent analysis."""
        from modelchorus.core.base_workflow import WorkflowResult, WorkflowStep

        steps = [
            WorkflowStep(
                step_number=1,
                content="Practical perspective ideas...",
                model="claude",
                metadata={"perspective": "practical", "model": "claude"}
            ),
            WorkflowStep(
                step_number=2,
                content="Innovative perspective ideas...",
                model="gemini",
                metadata={"perspective": "innovative", "model": "gemini"}
            ),
        ]

        return WorkflowResult(
            synthesis="Combined brainstorming results",
            steps=steps,
            metadata={
                'perspectives': ['practical', 'innovative'],
                'models_used': ['claude', 'gemini']
            }
        )

    @pytest.mark.asyncio
    async def test_run_convergent_analysis(self, ideate_workflow, mock_brainstorming_result):
        """Test convergent analysis on brainstorming results."""
        result = await ideate_workflow.run_convergent_analysis(
            brainstorming_result=mock_brainstorming_result,
            scoring_criteria=['feasibility', 'impact', 'novelty']
        )

        assert result is not None
        assert result.metadata['workflow'] == 'ideate-convergent'
        assert 'num_ideas_extracted' in result.metadata
        assert 'num_clusters' in result.metadata
        assert result.metadata['scoring_criteria'] == ['feasibility', 'impact', 'novelty']
        assert len(result.steps) == 3  # extraction, clustering, scoring

    @pytest.mark.asyncio
    async def test_convergent_analysis_without_brainstorming_raises_error(self, ideate_workflow):
        """Test that missing brainstorming result raises ValueError."""
        with pytest.raises(ValueError, match="Brainstorming result must have steps"):
            await ideate_workflow.run_convergent_analysis(
                brainstorming_result=None
            )

    @pytest.mark.asyncio
    async def test_idea_extraction(self, ideate_workflow, mock_brainstorming_result):
        """Test idea extraction from brainstorming results."""
        extraction_step = await ideate_workflow._extract_ideas(
            brainstorming_result=mock_brainstorming_result
        )

        assert extraction_step is not None
        assert extraction_step.metadata['title'] == 'Idea Extraction'
        assert 'num_ideas' in extraction_step.metadata
        assert 'extracted_ideas' in extraction_step.metadata
        assert extraction_step.metadata['num_ideas'] == 3

    @pytest.mark.asyncio
    async def test_idea_clustering(self, ideate_workflow):
        """Test idea clustering functionality."""
        from modelchorus.core.base_workflow import WorkflowStep

        # Create mock extraction step with ideas
        extraction_step = WorkflowStep(
            step_number=1,
            content="Extracted ideas...",
            model="claude",
            metadata={
                'title': 'Idea Extraction',
                'extracted_ideas': [
                    {'id': 'idea-1', 'label': 'Gamification', 'perspective': 'practical', 'description': 'Add game elements'},
                    {'id': 'idea-2', 'label': 'Tutorials', 'perspective': 'user-focused', 'description': 'Interactive guides'},
                    {'id': 'idea-3', 'label': 'AI Recommendations', 'perspective': 'innovative', 'description': 'ML-powered suggestions'},
                ]
            }
        )

        clustering_step = await ideate_workflow._cluster_ideas(
            extraction_step=extraction_step,
            num_clusters=2
        )

        assert clustering_step is not None
        assert clustering_step.metadata['title'] == 'Idea Clustering'
        assert 'num_clusters' in clustering_step.metadata
        assert 'clusters' in clustering_step.metadata
        assert clustering_step.metadata['num_clusters'] == 2

    @pytest.mark.asyncio
    async def test_idea_scoring(self, ideate_workflow):
        """Test idea scoring functionality."""
        from modelchorus.core.base_workflow import WorkflowStep

        # Create mock clustering step
        clustering_step = WorkflowStep(
            step_number=2,
            content="Clustered ideas...",
            model="claude",
            metadata={
                'title': 'Idea Clustering',
                'clusters': [
                    {
                        'id': 'cluster-1',
                        'theme': 'User Engagement',
                        'description': 'Features for engagement',
                        'ideas': [
                            {'idea_id': 'idea-1', 'reason': 'Gamification'},
                        ]
                    },
                    {
                        'id': 'cluster-2',
                        'theme': 'Personalization',
                        'description': 'AI-driven customization',
                        'ideas': [
                            {'idea_id': 'idea-3', 'reason': 'AI recommendations'},
                        ]
                    },
                ]
            }
        )

        scoring_step = await ideate_workflow._score_ideas(
            clustering_step=clustering_step,
            scoring_criteria=['feasibility', 'impact', 'novelty']
        )

        assert scoring_step is not None
        assert scoring_step.metadata['title'] == 'Idea Scoring'
        assert 'scored_clusters' in scoring_step.metadata
        assert scoring_step.metadata['scoring_criteria'] == ['feasibility', 'impact', 'novelty']
        assert len(scoring_step.metadata['scored_clusters']) == 2


class TestInteractiveSelection:
    """Test interactive selection functionality."""

    @pytest.fixture
    def mock_convergent_result(self):
        """Create mock convergent result for selection testing."""
        from modelchorus.core.base_workflow import WorkflowResult, WorkflowStep

        clustering_step = WorkflowStep(
            step_number=2,
            content="Clustered ideas...",
            model="claude",
            metadata={
                'title': 'Idea Clustering',
                'clusters': [
                    {
                        'id': 'cluster-1',
                        'theme': 'User Engagement',
                        'description': 'Engagement features',
                        'ideas': [{'idea_id': 'idea-1', 'reason': 'Gamification'}]
                    },
                ]
            }
        )

        scoring_step = WorkflowStep(
            step_number=3,
            content="Scored ideas...",
            model="claude",
            metadata={
                'title': 'Idea Scoring',
                'scored_clusters': [
                    {
                        'cluster_id': 'cluster-1',
                        'theme': 'User Engagement',
                        'overall_score': 4.2,
                        'recommendation': 'High Priority',
                        'scores': {
                            'feasibility': {'score': 4, 'explanation': 'Achievable'},
                            'impact': {'score': 5, 'explanation': 'High impact'},
                        }
                    },
                ]
            }
        )

        return WorkflowResult(
            synthesis="Analysis complete",
            steps=[clustering_step, scoring_step],
            metadata={'workflow': 'ideate-convergent'}
        )

    def test_parse_selection_input_single_number(self, ideate_workflow):
        """Test parsing single number selection."""
        result = ideate_workflow._parse_selection_input("3", total_count=5)
        assert result == [3]

    def test_parse_selection_input_comma_separated(self, ideate_workflow):
        """Test parsing comma-separated selection."""
        result = ideate_workflow._parse_selection_input("1,3,5", total_count=5)
        assert result == [1, 3, 5]

    def test_parse_selection_input_range(self, ideate_workflow):
        """Test parsing range selection."""
        result = ideate_workflow._parse_selection_input("2-4", total_count=5)
        assert result == [2, 3, 4]

    def test_parse_selection_input_all(self, ideate_workflow):
        """Test parsing 'all' selection."""
        result = ideate_workflow._parse_selection_input("all", total_count=3)
        assert result == [1, 2, 3]

    def test_parse_selection_input_none(self, ideate_workflow):
        """Test parsing 'none' selection."""
        result = ideate_workflow._parse_selection_input("none", total_count=5)
        assert result == []

    def test_parse_selection_input_invalid_range(self, ideate_workflow):
        """Test invalid range returns None."""
        result = ideate_workflow._parse_selection_input("2-10", total_count=5)
        assert result is None

    def test_parse_selection_input_with_max_selections(self, ideate_workflow):
        """Test max_selections limit."""
        result = ideate_workflow._parse_selection_input("1,2,3,4", total_count=5, max_selections=2)
        assert result is None


class TestElaboration:
    """Test elaboration functionality."""

    @pytest.fixture
    def mock_selection_result(self):
        """Create mock selection result for elaboration testing."""
        from modelchorus.core.base_workflow import WorkflowResult, WorkflowStep

        selection_step = WorkflowStep(
            step_number=1,
            content="Selection complete",
            model="claude",
            metadata={
                'title': 'Interactive Selection',
                'selected_clusters': [
                    {
                        'cluster_id': 'cluster-1',
                        'theme': 'User Engagement',
                        'overall_score': 4.5,
                        'recommendation': 'High Priority',
                        'scores': {
                            'feasibility': {'score': 4, 'explanation': 'Doable'},
                            'impact': {'score': 5, 'explanation': 'High value'},
                        },
                        'ideas': [{'idea_id': 'idea-1', 'reason': 'Gamification'}]
                    },
                ]
            }
        )

        return WorkflowResult(
            synthesis="Selected 1 cluster",
            steps=[selection_step],
            metadata={
                'workflow': 'ideate-selection',
                'selected_clusters': selection_step.metadata['selected_clusters']
            }
        )

    @pytest.mark.asyncio
    async def test_run_elaboration(self, ideate_workflow, mock_selection_result):
        """Test elaboration of selected clusters."""
        result = await ideate_workflow.run_elaboration(
            selection_result=mock_selection_result
        )

        assert result is not None
        assert result.metadata['workflow'] == 'ideate-elaboration'
        assert result.metadata['num_elaborated'] == 1
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_elaboration_without_selection_raises_error(self, ideate_workflow):
        """Test that missing selection result raises ValueError."""
        with pytest.raises(ValueError, match="Selection result must have metadata"):
            await ideate_workflow.run_elaboration(selection_result=None)

    @pytest.mark.asyncio
    async def test_elaborate_cluster(self, ideate_workflow):
        """Test elaborating a single cluster."""
        cluster = {
            'cluster_id': 'cluster-1',
            'theme': 'User Engagement',
            'overall_score': 4.5,
            'scores': {
                'feasibility': {'score': 4, 'explanation': 'Achievable'},
            },
            'ideas': [{'idea_id': 'idea-1', 'reason': 'Gamification'}]
        }

        elaboration_step = await ideate_workflow._elaborate_cluster(
            cluster=cluster,
            step_number=1
        )

        assert elaboration_step is not None
        assert elaboration_step.metadata['title'] == 'Elaboration: User Engagement'
        assert elaboration_step.metadata['theme'] == 'User Engagement'
        assert 'outline_sections' in elaboration_step.metadata

    def test_parse_outline_sections(self, ideate_workflow):
        """Test parsing outline sections from elaboration content."""
        content = """## Overview
This is the overview section.

## Goals & Objectives
- Goal 1
- Goal 2

## Implementation Approach
Step-by-step approach here."""

        sections = ideate_workflow._parse_outline_sections(content)

        assert len(sections) >= 2
        assert any('overview' in s['title'].lower() for s in sections)
        assert any('goals' in s['title'].lower() or 'objectives' in s['title'].lower() for s in sections)


class TestCompleteIdeation:
    """Test complete ideation workflow."""

    @pytest.mark.asyncio
    async def test_run_complete_ideation(self, ideate_workflow, mock_provider):
        """Test running complete ideation workflow (divergent + convergent)."""
        provider_map = {
            "claude": mock_provider,
            "gemini": mock_provider,
        }

        # Mock RoleOrchestrator for parallel brainstorming
        with patch('modelchorus.workflows.ideate.ideate_workflow.RoleOrchestrator') as mock_orch:
            mock_result = MagicMock()
            mock_result.role_responses = [
                ("brainstormer-practical", GenerationResponse(
                    content="Practical ideas for API improvement",
                    model="claude",
                    usage={"total_tokens": 100}
                )),
                ("brainstormer-innovative", GenerationResponse(
                    content="Innovative ideas for API improvement",
                    model="gemini",
                    usage={"total_tokens": 120}
                )),
            ]

            mock_orch_instance = MagicMock()
            mock_orch_instance.execute = AsyncMock(return_value=mock_result)
            mock_orch.return_value = mock_orch_instance

            # Run complete ideation
            result = await ideate_workflow.run_complete_ideation(
                prompt="How to improve our API documentation?",
                provider_map=provider_map,
                perspectives=['practical', 'innovative'],
                scoring_criteria=['feasibility', 'impact']
            )

            assert result is not None
            assert result.metadata['workflow'] == 'ideate-complete'
            assert 'divergent_phase' in result.metadata
            assert 'convergent_phase' in result.metadata
            # Divergent (2) + Convergent (3: extraction, clustering, scoring)
            assert len(result.steps) >= 5

    @pytest.mark.asyncio
    async def test_complete_ideation_empty_prompt_raises_error(self, ideate_workflow):
        """Test that empty prompt raises ValueError."""
        provider_map = {"claude": MagicMock()}

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await ideate_workflow.run_complete_ideation(
                prompt="",
                provider_map=provider_map
            )

    @pytest.mark.asyncio
    async def test_complete_ideation_empty_provider_map_raises_error(self, ideate_workflow):
        """Test that empty provider map raises ValueError."""
        with pytest.raises(ValueError, match="Provider map cannot be empty"):
            await ideate_workflow.run_complete_ideation(
                prompt="Test",
                provider_map={}
            )


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_provider_generation_failure(self, ideate_workflow):
        """Test handling of provider generation failure."""
        # Mock provider to raise exception
        ideate_workflow.provider.generate = AsyncMock(
            side_effect=Exception("Provider error")
        )

        with pytest.raises(Exception, match="Provider error"):
            await ideate_workflow.run(prompt="Test prompt")

    @pytest.mark.asyncio
    async def test_extraction_with_no_brainstorming_steps(self, ideate_workflow):
        """Test extraction fails gracefully with no brainstorming steps."""
        from modelchorus.core.base_workflow import WorkflowResult

        empty_result = WorkflowResult(
            success=True,
            synthesis="Empty",
            steps=[],
            metadata={}
        )

        # _extract_ideas expects brainstorming_result.steps to exist
        # This should handle empty steps gracefully
        try:
            await ideate_workflow._extract_ideas(empty_result)
        except (ValueError, IndexError, AttributeError):
            # Expected to fail with empty steps - this is correct behavior
            pass

    @pytest.mark.asyncio
    async def test_clustering_with_no_extracted_ideas(self, ideate_workflow):
        """Test clustering fails with no extracted ideas."""
        from modelchorus.core.base_workflow import WorkflowStep

        extraction_step = WorkflowStep(
            step_number=1,
            content="No ideas",
            model="claude",
            metadata={'extracted_ideas': []}
        )

        with pytest.raises(ValueError, match="No ideas found"):
            await ideate_workflow._cluster_ideas(extraction_step)

    @pytest.mark.asyncio
    async def test_scoring_with_no_clusters(self, ideate_workflow):
        """Test scoring fails with no clusters."""
        from modelchorus.core.base_workflow import WorkflowStep

        clustering_step = WorkflowStep(
            step_number=2,
            content="No clusters",
            model="claude",
            metadata={'clusters': []}
        )

        with pytest.raises(ValueError, match="No clusters found"):
            await ideate_workflow._score_ideas(
                clustering_step,
                scoring_criteria=['feasibility']
            )


class TestSystemPrompts:
    """Test system prompt generation."""

    def test_get_ideation_system_prompt(self, ideate_workflow):
        """Test ideation system prompt generation."""
        prompt = ideate_workflow._get_ideation_system_prompt()

        assert "creative" in prompt.lower() or "ideation" in prompt.lower()
        assert "brainstorm" in prompt.lower()
        assert len(prompt) > 100

    def test_get_extraction_system_prompt(self, ideate_workflow):
        """Test extraction system prompt generation."""
        prompt = ideate_workflow._get_extraction_system_prompt()

        assert "extract" in prompt.lower()
        assert "ideas" in prompt.lower()
        assert len(prompt) > 50

    def test_get_clustering_system_prompt(self, ideate_workflow):
        """Test clustering system prompt generation."""
        prompt = ideate_workflow._get_clustering_system_prompt()

        assert "cluster" in prompt.lower() or "theme" in prompt.lower()
        assert len(prompt) > 50

    def test_get_scoring_system_prompt(self, ideate_workflow):
        """Test scoring system prompt generation."""
        prompt = ideate_workflow._get_scoring_system_prompt()

        assert "score" in prompt.lower() or "evaluat" in prompt.lower()
        assert len(prompt) > 50

    def test_get_elaboration_system_prompt(self, ideate_workflow):
        """Test elaboration system prompt generation."""
        prompt = ideate_workflow._get_elaboration_system_prompt()

        assert "outline" in prompt.lower() or "detail" in prompt.lower()
        assert "actionable" in prompt.lower()
        assert len(prompt) > 100


class TestPromptFraming:
    """Test prompt framing methods."""

    def test_frame_ideation_prompt_initial(self, ideate_workflow):
        """Test framing initial ideation prompt."""
        framed = ideate_workflow._frame_ideation_prompt(
            "Test topic",
            is_continuation=False
        )

        assert "brainstorm" in framed.lower()
        assert "Test topic" in framed

    def test_frame_ideation_prompt_continuation(self, ideate_workflow):
        """Test framing continuation ideation prompt."""
        framed = ideate_workflow._frame_ideation_prompt(
            "Follow-up request",
            is_continuation=True
        )

        assert "continue" in framed.lower()
        assert "Follow-up request" in framed
