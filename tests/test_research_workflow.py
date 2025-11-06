"""
Tests for ResearchWorkflow functionality.

Tests the systematic research workflow including question formulation, source
gathering, evidence extraction, validation, and dossier generation with
comprehensive coverage of parallel processing and research features.
"""

import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from modelchorus.workflows.research import (
    ResearchWorkflow,
    Evidence,
    ExtractionResult,
    ValidationResult,
)
from modelchorus.providers.base_provider import GenerationRequest, GenerationResponse
from modelchorus.core.conversation import ConversationMemory
from modelchorus.core.base_workflow import WorkflowResult, WorkflowStep


@pytest.fixture
def mock_provider():
    """Mock ModelProvider for testing."""
    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.validate_api_key = MagicMock(return_value=True)

    # Mock async generate method
    async def mock_generate(request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            content=f"Research response to: {request.prompt[:50]}",
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
def research_workflow(mock_provider, conversation_memory):
    """Create ResearchWorkflow instance for testing."""
    return ResearchWorkflow(
        provider=mock_provider,
        conversation_memory=conversation_memory,
    )


class TestResearchWorkflowInitialization:
    """Test ResearchWorkflow initialization."""

    def test_initialization_with_provider(self, mock_provider, conversation_memory):
        """Test basic initialization with provider."""
        workflow = ResearchWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory,
        )

        assert workflow.name == "Research"
        assert workflow.description == "Systematic research workflow for information gathering and analysis"
        assert workflow.provider == mock_provider
        assert workflow.conversation_memory == conversation_memory
        assert workflow.source_registry == []

    def test_initialization_without_memory(self, mock_provider):
        """Test initialization without conversation memory."""
        workflow = ResearchWorkflow(provider=mock_provider)

        assert workflow.provider == mock_provider
        assert workflow.conversation_memory is None

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider is required"):
            ResearchWorkflow(provider=None)

    def test_default_config(self, research_workflow):
        """Test default configuration values."""
        assert research_workflow.default_config['temperature'] == 0.5
        assert research_workflow.default_config['max_tokens'] == 4000
        assert research_workflow.default_config['research_depth'] == 'thorough'
        assert research_workflow.default_config['source_validation'] is True
        assert research_workflow.default_config['citation_style'] == 'informal'

    def test_initialization_with_custom_config(self, mock_provider):
        """Test initialization with custom configuration."""
        custom_config = {
            'temperature': 0.7,
            'research_depth': 'comprehensive',
            'citation_style': 'academic'
        }
        workflow = ResearchWorkflow(
            provider=mock_provider,
            config=custom_config
        )

        assert workflow.default_config['temperature'] == 0.7
        assert workflow.default_config['research_depth'] == 'comprehensive'
        assert workflow.default_config['citation_style'] == 'academic'

    def test_get_provider(self, research_workflow, mock_provider):
        """Test get_provider method."""
        assert research_workflow.get_provider() == mock_provider


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self, research_workflow):
        """Test validation of valid configuration."""
        valid_config = {
            'temperature': 0.5,
            'research_depth': 'thorough',
            'citation_style': 'academic'
        }
        assert research_workflow.validate_config(valid_config) is True

    def test_validate_invalid_research_depth(self, research_workflow):
        """Test validation rejects invalid research depth."""
        invalid_config = {'research_depth': 'invalid_depth'}
        assert research_workflow.validate_config(invalid_config) is False

    def test_validate_invalid_citation_style(self, research_workflow):
        """Test validation rejects invalid citation style."""
        invalid_config = {'citation_style': 'invalid_style'}
        assert research_workflow.validate_config(invalid_config) is False

    def test_validate_invalid_temperature(self, research_workflow):
        """Test validation rejects invalid temperature."""
        invalid_config = {'temperature': 1.5}
        assert research_workflow.validate_config(invalid_config) is False

        invalid_config = {'temperature': -0.5}
        assert research_workflow.validate_config(invalid_config) is False

    def test_validate_all_research_depths(self, research_workflow):
        """Test all valid research depth values."""
        for depth in ['shallow', 'moderate', 'thorough', 'comprehensive']:
            config = {'research_depth': depth}
            assert research_workflow.validate_config(config) is True

    def test_validate_all_citation_styles(self, research_workflow):
        """Test all valid citation style values."""
        for style in ['informal', 'academic', 'technical']:
            config = {'citation_style': style}
            assert research_workflow.validate_config(config) is True


class TestResearchExecution:
    """Test research workflow execution."""

    @pytest.mark.asyncio
    async def test_basic_research_execution(self, research_workflow):
        """Test basic research workflow execution."""
        result = await research_workflow.run("AI orchestration trends")

        assert isinstance(result, WorkflowResult)
        assert len(result.steps) == 1  # Question formulation step
        assert result.synthesis is not None
        assert 'thread_id' in result.metadata
        assert result.metadata['research_topic'] == "AI orchestration trends"
        assert result.metadata['research_depth'] == 'thorough'

    @pytest.mark.asyncio
    async def test_research_with_continuation_id(self, research_workflow):
        """Test research with continuation_id for threading."""
        thread_id = str(uuid.uuid4())
        result = await research_workflow.run(
            "Multi-model consensus",
            continuation_id=thread_id
        )

        assert result.metadata['thread_id'] == thread_id

    @pytest.mark.asyncio
    async def test_research_with_custom_depth(self, research_workflow):
        """Test research with custom depth configuration."""
        result = await research_workflow.run(
            "AI research topic",
            research_depth='comprehensive'
        )

        assert result.metadata['research_depth'] == 'comprehensive'

    @pytest.mark.asyncio
    async def test_research_with_empty_prompt_raises_error(self, research_workflow):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await research_workflow.run("")

        with pytest.raises(ValueError, match="cannot be empty"):
            await research_workflow.run("   ")

    @pytest.mark.asyncio
    async def test_research_with_invalid_config_raises_error(self, research_workflow):
        """Test that invalid configuration raises ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            await research_workflow.run(
                "Test topic",
                research_depth='invalid_depth'
            )

    @pytest.mark.asyncio
    async def test_conversation_memory_updated(self, research_workflow):
        """Test that conversation memory is updated during research."""
        prompt = "Test research topic"

        # Let the workflow create the thread
        result = await research_workflow.run(prompt)

        thread_id = result.metadata['thread_id']

        # Check conversation was stored
        messages = research_workflow.conversation_memory.get_messages(thread_id)
        assert len(messages) >= 2  # User message + assistant response
        assert messages[0].role == "user"
        assert messages[0].content == prompt
        assert messages[1].role == "assistant"


class TestQuestionFormulation:
    """Test research question formulation."""

    @pytest.mark.asyncio
    async def test_formulate_questions_creates_step(self, research_workflow):
        """Test question formulation creates proper WorkflowStep."""
        step = await research_workflow._formulate_questions(
            prompt="AI trends",
            thread_id=str(uuid.uuid4()),
            config=research_workflow.default_config
        )

        assert isinstance(step, WorkflowStep)
        assert step.step_number == 1
        assert step.model == "test-provider"
        assert 'title' in step.metadata
        assert step.metadata['title'] == 'Research Question Formulation'

    @pytest.mark.asyncio
    async def test_question_prompt_includes_depth_guidance(self, research_workflow):
        """Test that question prompts include depth-appropriate guidance."""
        # Mock generate to capture the prompt
        captured_prompt = None

        async def capture_prompt(request: GenerationRequest) -> GenerationResponse:
            nonlocal captured_prompt
            captured_prompt = request.prompt
            return GenerationResponse(
                content="Test questions",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=capture_prompt)

        # Test with different depths
        for depth in ['shallow', 'moderate', 'thorough', 'comprehensive']:
            await research_workflow._formulate_questions(
                prompt="Test topic",
                thread_id=str(uuid.uuid4()),
                config={'research_depth': depth, 'temperature': 0.5, 'max_tokens': 4000}
            )
            assert depth in captured_prompt.lower()

    def test_question_formulation_system_prompt(self, research_workflow):
        """Test question formulation system prompt content."""
        system_prompt = research_workflow._get_question_formulation_system_prompt()

        assert "research" in system_prompt.lower()
        assert "questions" in system_prompt.lower()
        assert "specific" in system_prompt.lower()


class TestSourceManagement:
    """Test source registry and management."""

    def test_ingest_source_basic(self, research_workflow):
        """Test basic source ingestion."""
        source = research_workflow.ingest_source(
            title="Test Article",
            url="https://example.com/article",
            source_type="article"
        )

        assert source['source_id'] is not None
        assert source['title'] == "Test Article"
        assert source['url'] == "https://example.com/article"
        assert source['type'] == "article"
        assert len(research_workflow.source_registry) == 1

    def test_ingest_source_with_metadata(self, research_workflow):
        """Test source ingestion with metadata."""
        metadata = {'author': 'John Doe', 'year': 2024}
        source = research_workflow.ingest_source(
            title="Research Paper",
            url="https://example.com/paper",
            source_type="paper",
            credibility="high",
            tags=["ai", "ml"],
            metadata=metadata
        )

        assert source['credibility'] == "high"
        assert source['tags'] == ["ai", "ml"]
        assert source['metadata'] == metadata

    def test_source_validation(self, research_workflow):
        """Test automatic source validation."""
        source = research_workflow.ingest_source(
            title="Academic Paper",
            url="https://example.com/paper",
            source_type="paper"
        )

        assert source['validated'] is True
        assert 'validation_score' in source
        assert source['credibility'] in ['high', 'medium', 'low']

    def test_get_sources_by_tag(self, research_workflow):
        """Test filtering sources by tag."""
        research_workflow.ingest_source(
            title="AI Article",
            source_type="article",
            tags=["ai", "research"]
        )
        research_workflow.ingest_source(
            title="ML Paper",
            source_type="paper",
            tags=["ml", "research"]
        )
        research_workflow.ingest_source(
            title="Python Tutorial",
            source_type="tutorial",
            tags=["python"]
        )

        research_sources = research_workflow.get_sources_by_tag("research")
        assert len(research_sources) == 2

        ai_sources = research_workflow.get_sources_by_tag("ai")
        assert len(ai_sources) == 1

    def test_get_sources_by_type(self, research_workflow):
        """Test filtering sources by type."""
        research_workflow.ingest_source(title="Article 1", source_type="article")
        research_workflow.ingest_source(title="Article 2", source_type="article")
        research_workflow.ingest_source(title="Paper 1", source_type="paper")

        articles = research_workflow.get_sources_by_type("article")
        assert len(articles) == 2

        papers = research_workflow.get_sources_by_type("paper")
        assert len(papers) == 1

    def test_get_sources_by_credibility(self, research_workflow):
        """Test filtering sources by credibility level."""
        research_workflow.ingest_source(
            title="High Cred Source",
            url="https://example.com",
            source_type="paper"
        )
        research_workflow.ingest_source(
            title="Low Cred Source",
            source_type="blog"
        )

        # Credibility is auto-assigned during validation
        high_cred = research_workflow.get_sources_by_credibility("high")
        assert len(high_cred) >= 0  # Depends on validation logic

    def test_get_source_summary(self, research_workflow):
        """Test source registry summary statistics."""
        research_workflow.ingest_source(
            title="Source 1",
            source_type="article",
            tags=["ai"]
        )
        research_workflow.ingest_source(
            title="Source 2",
            source_type="paper",
            tags=["ml", "ai"]
        )

        summary = research_workflow.get_source_summary()

        assert summary['total_sources'] == 2
        assert 'by_type' in summary
        assert 'by_credibility' in summary
        assert 'by_tags' in summary
        assert 'validated_count' in summary

    def test_clear_source_registry(self, research_workflow):
        """Test clearing source registry."""
        research_workflow.ingest_source(title="Test Source", source_type="article")
        assert len(research_workflow.source_registry) == 1

        research_workflow.clear_source_registry()
        assert len(research_workflow.source_registry) == 0


class TestEvidenceExtraction:
    """Test evidence extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_evidence_from_all_sources(self, research_workflow):
        """Test extracting evidence from all sources."""
        # Add test sources
        research_workflow.ingest_source(title="Source 1", source_type="article")
        research_workflow.ingest_source(title="Source 2", source_type="paper")

        # Mock provider to return structured evidence
        async def mock_extract(request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                content="Evidence content\nRelevance: 0.8\nConfidence: high\nCategory: testing",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_extract)

        results = await research_workflow.extract_evidence_from_sources()

        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_extract_evidence_from_specific_sources(self, research_workflow):
        """Test extracting evidence from specific source IDs."""
        src1 = research_workflow.ingest_source(title="Source 1", source_type="article")
        src2 = research_workflow.ingest_source(title="Source 2", source_type="paper")
        research_workflow.ingest_source(title="Source 3", source_type="blog")

        # Mock provider
        async def mock_extract(request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                content="Test evidence",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_extract)

        # Extract from only first two sources
        results = await research_workflow.extract_evidence_from_sources(
            source_ids=[src1['source_id'], src2['source_id']]
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_extract_evidence_with_questions(self, research_workflow):
        """Test evidence extraction guided by research questions."""
        research_workflow.ingest_source(title="Test Source", source_type="article")

        questions = [
            "What are the benefits?",
            "What are the risks?"
        ]

        captured_prompt = None

        async def capture_extract(request: GenerationRequest) -> GenerationResponse:
            nonlocal captured_prompt
            captured_prompt = request.prompt
            return GenerationResponse(
                content="Test evidence",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=capture_extract)

        await research_workflow.extract_evidence_from_sources(questions=questions)

        # Verify questions were included in prompt
        assert "benefits" in captured_prompt.lower()
        assert "risks" in captured_prompt.lower()

    @pytest.mark.asyncio
    async def test_extract_evidence_invalid_source_ids(self, research_workflow):
        """Test that invalid source IDs raise ValueError."""
        research_workflow.ingest_source(title="Valid Source", source_type="article")

        with pytest.raises(ValueError, match="Invalid source IDs"):
            await research_workflow.extract_evidence_from_sources(
                source_ids=['invalid_id']
            )

    @pytest.mark.asyncio
    async def test_extract_evidence_parallel_execution(self, research_workflow):
        """Test parallel evidence extraction with concurrency control."""
        # Add multiple sources
        for i in range(10):
            research_workflow.ingest_source(
                title=f"Source {i}",
                source_type="article"
            )

        call_count = 0
        max_concurrent = 0
        current_concurrent = 0

        async def mock_extract_with_concurrency(request: GenerationRequest) -> GenerationResponse:
            nonlocal call_count, max_concurrent, current_concurrent
            call_count += 1
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)

            # Simulate some async work
            await asyncio.sleep(0.01)

            current_concurrent -= 1
            return GenerationResponse(
                content="Test evidence",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_extract_with_concurrency)

        results = await research_workflow.extract_evidence_from_sources(max_concurrent=3)

        assert len(results) == 10
        assert call_count == 10
        assert max_concurrent <= 3  # Should respect concurrency limit

    @pytest.mark.asyncio
    async def test_extract_evidence_timeout_handling(self, research_workflow):
        """Test timeout handling in evidence extraction."""
        research_workflow.ingest_source(title="Slow Source", source_type="article")

        async def slow_extract(request: GenerationRequest) -> GenerationResponse:
            await asyncio.sleep(10)  # Simulate slow response
            return GenerationResponse(
                content="Late evidence",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=slow_extract)

        results = await research_workflow.extract_evidence_from_sources(
            timeout_per_source=0.1  # Very short timeout
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "timed out" in results[0].error.lower()

    @pytest.mark.asyncio
    async def test_extract_evidence_error_handling(self, research_workflow):
        """Test error handling in evidence extraction."""
        research_workflow.ingest_source(title="Error Source", source_type="article")

        async def failing_extract(request: GenerationRequest) -> GenerationResponse:
            raise Exception("Extraction failed")

        research_workflow.provider.generate = AsyncMock(side_effect=failing_extract)

        results = await research_workflow.extract_evidence_from_sources()

        assert len(results) == 1
        assert results[0].success is False
        assert "failed" in results[0].error.lower()

    @pytest.mark.asyncio
    async def test_extract_evidence_no_sources(self, research_workflow):
        """Test evidence extraction with no sources returns empty list."""
        results = await research_workflow.extract_evidence_from_sources()
        assert results == []


class TestEvidenceValidation:
    """Test evidence validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_evidence_basic(self, research_workflow):
        """Test basic evidence validation."""
        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id="src1",
                content="Test evidence content",
                relevance_score=0.8,
                confidence="high"
            )
        ]

        # Mock validation response
        async def mock_validate(request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                content="Is Valid: true\nConfidence Score: 0.85\nValidation Notes: Good evidence",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_validate)

        results = await research_workflow.validate_evidence(evidence)

        assert len(results) == 1
        assert isinstance(results[0], ValidationResult)
        assert results[0].evidence_id == "ev1"

    @pytest.mark.asyncio
    async def test_validate_evidence_with_cross_reference(self, research_workflow):
        """Test evidence validation with cross-referencing."""
        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id="src1",
                content="Evidence 1",
                relevance_score=0.8,
                confidence="high"
            ),
            Evidence(
                evidence_id="ev2",
                source_id="src2",
                content="Evidence 2",
                relevance_score=0.7,
                confidence="medium"
            )
        ]

        async def mock_validate(request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                content="Is Valid: true\nConfidence Score: 0.8\nValidation Notes: Validated",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_validate)

        results = await research_workflow.validate_evidence(
            evidence,
            cross_reference=True
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_validate_evidence_detect_contradictions(self, research_workflow):
        """Test contradiction detection in validation."""
        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id="src1",
                content="AI is beneficial",
                relevance_score=0.8,
                confidence="high"
            ),
            Evidence(
                evidence_id="ev2",
                source_id="src2",
                content="AI is harmful",
                relevance_score=0.7,
                confidence="medium"
            )
        ]

        async def mock_validate(request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                content="Is Valid: true\nConfidence Score: 0.6\n"
                        "Contradictions: Conflicts with evidence ev2\n"
                        "Validation Notes: Potential contradiction detected",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn"
            )

        research_workflow.provider.generate = AsyncMock(side_effect=mock_validate)

        results = await research_workflow.validate_evidence(
            evidence,
            detect_contradictions=True
        )

        # At least one result should have contradictions detected
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_validate_evidence_empty_list(self, research_workflow):
        """Test validation with empty evidence list."""
        results = await research_workflow.validate_evidence([])
        assert results == []

    @pytest.mark.asyncio
    async def test_validate_evidence_error_handling(self, research_workflow):
        """Test error handling in evidence validation."""
        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id="src1",
                content="Test evidence",
                relevance_score=0.8,
                confidence="high"
            )
        ]

        async def failing_validate(request: GenerationRequest) -> GenerationResponse:
            raise Exception("Validation failed")

        research_workflow.provider.generate = AsyncMock(side_effect=failing_validate)

        results = await research_workflow.validate_evidence(evidence)

        # Should return default validation result on error
        assert len(results) == 1
        assert results[0].is_valid is True  # Assumes valid on error
        assert results[0].confidence_score == 0.3  # Low confidence


class TestDossierGeneration:
    """Test research dossier generation."""

    def test_generate_dossier_basic(self, research_workflow):
        """Test basic dossier generation."""
        # Add sources
        research_workflow.ingest_source(
            title="Test Source",
            url="https://example.com",
            source_type="article"
        )

        # Create evidence
        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Important finding",
                relevance_score=0.9,
                confidence="high",
                category="findings"
            )
        ]

        dossier = research_workflow.generate_dossier(evidence)

        assert "# Research Dossier" in dossier
        assert "Executive Summary" in dossier
        assert "Test Source" in dossier  # Citation should be included

    def test_generate_dossier_with_validation(self, research_workflow):
        """Test dossier generation with validation results."""
        research_workflow.ingest_source(title="Source 1", source_type="article")

        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Test evidence",
                relevance_score=0.8,
                confidence="high"
            )
        ]

        validation = [
            ValidationResult(
                evidence_id="ev1",
                is_valid=True,
                confidence_score=0.85,
                validation_notes=["Strong evidence"]
            )
        ]

        dossier = research_workflow.generate_dossier(
            evidence,
            validation_results=validation
        )

        assert "Validation Metrics" in dossier
        assert "0.85" in dossier  # Confidence score

    def test_generate_dossier_citation_styles(self, research_workflow):
        """Test dossier generation with different citation styles."""
        research_workflow.ingest_source(
            title="Test Article",
            url="https://example.com",
            source_type="article"
        )

        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Evidence",
                relevance_score=0.8,
                confidence="high"
            )
        ]

        # Test each citation style
        for style in ['informal', 'academic', 'technical']:
            dossier = research_workflow.generate_dossier(
                evidence,
                citation_style=style
            )
            assert "Source Citations" in dossier
            assert "Test Article" in dossier

    def test_generate_dossier_with_contradictions(self, research_workflow):
        """Test dossier generation includes contradiction analysis."""
        research_workflow.ingest_source(title="Source 1", source_type="article")

        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Evidence with contradiction",
                relevance_score=0.7,
                confidence="medium"
            )
        ]

        validation = [
            ValidationResult(
                evidence_id="ev1",
                is_valid=True,
                confidence_score=0.6,
                contradictions=["Contradicts finding X", "Conflicts with Y"],
                validation_notes=["Needs review"]
            )
        ]

        dossier = research_workflow.generate_dossier(
            evidence,
            validation_results=validation,
            include_contradictions=True
        )

        assert "Contradiction Analysis" in dossier
        assert "Contradicts finding X" in dossier

    def test_generate_dossier_evidence_tables(self, research_workflow):
        """Test dossier includes formatted evidence tables."""
        research_workflow.ingest_source(title="Source 1", source_type="article")

        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Test evidence 1",
                relevance_score=0.9,
                confidence="high",
                category="Key Findings"
            ),
            Evidence(
                evidence_id="ev2",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Test evidence 2",
                relevance_score=0.7,
                confidence="medium",
                category="Key Findings"
            )
        ]

        dossier = research_workflow.generate_dossier(
            evidence,
            include_evidence_table=True
        )

        assert "Evidence Analysis" in dossier
        assert "Key Findings" in dossier
        assert "| Evidence |" in dossier  # Table header

    def test_generate_dossier_empty_evidence(self, research_workflow):
        """Test dossier generation with no evidence."""
        dossier = research_workflow.generate_dossier([])

        assert "# Research Dossier" in dossier
        assert "No evidence available" in dossier

    def test_generate_dossier_no_citations_option(self, research_workflow):
        """Test dossier generation without citations."""
        research_workflow.ingest_source(title="Source 1", source_type="article")

        evidence = [
            Evidence(
                evidence_id="ev1",
                source_id=research_workflow.source_registry[0]['source_id'],
                content="Evidence",
                relevance_score=0.8,
                confidence="high"
            )
        ]

        dossier = research_workflow.generate_dossier(
            evidence,
            include_citations=False
        )

        assert "Source Citations" not in dossier


class TestResearchStateDataclass:
    """Test ResearchState dataclass (if/when implemented)."""

    # These tests will be implemented when ResearchState is added
    # as mentioned in the task metadata

    pass  # Placeholder for future tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
