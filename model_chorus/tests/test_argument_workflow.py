"""
Tests for ArgumentWorkflow functionality.

Tests the complete dialectical analysis workflow including Creator (thesis),
Skeptic (rebuttal), and Moderator (synthesis) roles, ArgumentMap generation,
and conversation threading.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers.base_provider import GenerationRequest, GenerationResponse
from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import ArgumentMap, ArgumentPerspective
from model_chorus.core.role_orchestration import OrchestrationResult, OrchestrationPattern

# Note: mock_provider and conversation_memory fixtures are now in conftest.py


@pytest.fixture
def argument_workflow(mock_provider, conversation_memory):
    """Create ArgumentWorkflow instance for testing."""
    return ArgumentWorkflow(
        provider=mock_provider,
        conversation_memory=conversation_memory,
    )


class TestArgumentWorkflowInitialization:
    """Test ArgumentWorkflow initialization."""

    def test_initialization_with_provider(self, mock_provider, conversation_memory):
        """Test basic initialization with provider."""
        workflow = ArgumentWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory,
        )

        assert workflow.name == "Argument"
        assert workflow.description == "Structured argument analysis with dialectical reasoning"
        assert workflow.provider == mock_provider
        assert workflow.conversation_memory == conversation_memory

    def test_initialization_without_memory(self, mock_provider):
        """Test initialization without conversation memory."""
        workflow = ArgumentWorkflow(provider=mock_provider)

        assert workflow.provider == mock_provider
        assert workflow.conversation_memory is None

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            ArgumentWorkflow(provider=None)

    def test_validate_config(self, argument_workflow):
        """Test config validation."""
        assert argument_workflow.validate_config() is True

    def test_get_provider(self, argument_workflow, mock_provider):
        """Test get_provider method."""
        assert argument_workflow.get_provider() == mock_provider


class TestRoleCreation:
    """Test role creation methods."""

    def test_create_creator_role(self, argument_workflow):
        """Test Creator role creation."""
        creator = argument_workflow._create_creator_role()

        assert creator.role == "creator"
        assert creator.stance == "for"
        assert creator.model == "test-provider"
        assert creator.temperature == 0.7
        assert creator.metadata["step"] == 1
        assert creator.metadata["step_name"] == "Thesis Generation"
        assert creator.metadata["role_type"] == "advocate"

    def test_create_skeptic_role(self, argument_workflow):
        """Test Skeptic role creation."""
        skeptic = argument_workflow._create_skeptic_role()

        assert skeptic.role == "skeptic"
        assert skeptic.stance == "against"
        assert skeptic.model == "test-provider"
        assert skeptic.temperature == 0.7
        assert skeptic.metadata["step"] == 2
        assert skeptic.metadata["step_name"] == "Critical Rebuttal"
        assert skeptic.metadata["role_type"] == "critic"

    def test_create_moderator_role(self, argument_workflow):
        """Test Moderator role creation."""
        moderator = argument_workflow._create_moderator_role()

        assert moderator.role == "moderator"
        assert moderator.stance == "neutral"
        assert moderator.model == "test-provider"
        assert moderator.temperature == 0.7
        assert moderator.metadata["step"] == 3
        assert moderator.metadata["step_name"] == "Balanced Synthesis"
        assert moderator.metadata["role_type"] == "synthesizer"


class TestArgumentMapGeneration:
    """Test ArgumentMap generation."""

    def test_generate_argument_map(self, argument_workflow):
        """Test ArgumentMap generation from role responses."""

        # Create mock responses
        class MockResponse:
            def __init__(self, content, model):
                self.content = content
                self.model = model

        creator_resp = MockResponse("Thesis content supporting the position", "test-model")
        skeptic_resp = MockResponse("Rebuttal content against the position", "test-model")
        moderator_resp = MockResponse("Synthesis balancing both perspectives", "test-model")

        # Generate ArgumentMap
        arg_map = argument_workflow._generate_argument_map(
            prompt="Universal basic income reduces poverty",
            creator_response=creator_resp,
            skeptic_response=skeptic_resp,
            moderator_response=moderator_resp,
            synthesis="Final balanced synthesis",
            metadata={"thread_id": "test123"},
        )

        # Verify ArgumentMap structure
        assert isinstance(arg_map, ArgumentMap)
        assert arg_map.topic == "Universal basic income reduces poverty"
        assert len(arg_map.perspectives) == 3
        assert arg_map.synthesis == "Final balanced synthesis"
        assert arg_map.metadata["thread_id"] == "test123"

    def test_argument_map_creator_perspective(self, argument_workflow):
        """Test Creator perspective in ArgumentMap."""

        class MockResponse:
            def __init__(self, content, model):
                self.content = content
                self.model = model

        creator_resp = MockResponse("Thesis content", "test-model")
        skeptic_resp = MockResponse("Rebuttal content", "test-model")
        moderator_resp = MockResponse("Synthesis content", "test-model")

        arg_map = argument_workflow._generate_argument_map(
            prompt="Test topic",
            creator_response=creator_resp,
            skeptic_response=skeptic_resp,
            moderator_response=moderator_resp,
            synthesis="Test synthesis",
            metadata={},
        )

        creator = arg_map.get_perspective("creator")
        assert creator is not None
        assert creator.role == "creator"
        assert creator.stance == "for"
        assert creator.content == "Thesis content"
        assert creator.model == "test-model"
        assert creator.metadata["step"] == 1
        assert creator.metadata["step_name"] == "Thesis Generation"

    def test_argument_map_skeptic_perspective(self, argument_workflow):
        """Test Skeptic perspective in ArgumentMap."""

        class MockResponse:
            def __init__(self, content, model):
                self.content = content
                self.model = model

        creator_resp = MockResponse("Thesis content", "test-model")
        skeptic_resp = MockResponse("Rebuttal content", "test-model")
        moderator_resp = MockResponse("Synthesis content", "test-model")

        arg_map = argument_workflow._generate_argument_map(
            prompt="Test topic",
            creator_response=creator_resp,
            skeptic_response=skeptic_resp,
            moderator_response=moderator_resp,
            synthesis="Test synthesis",
            metadata={},
        )

        skeptic = arg_map.get_perspective("skeptic")
        assert skeptic is not None
        assert skeptic.role == "skeptic"
        assert skeptic.stance == "against"
        assert skeptic.content == "Rebuttal content"
        assert skeptic.model == "test-model"
        assert skeptic.metadata["step"] == 2
        assert skeptic.metadata["step_name"] == "Critical Rebuttal"

    def test_argument_map_moderator_perspective(self, argument_workflow):
        """Test Moderator perspective in ArgumentMap."""

        class MockResponse:
            def __init__(self, content, model):
                self.content = content
                self.model = model

        creator_resp = MockResponse("Thesis content", "test-model")
        skeptic_resp = MockResponse("Rebuttal content", "test-model")
        moderator_resp = MockResponse("Synthesis content", "test-model")

        arg_map = argument_workflow._generate_argument_map(
            prompt="Test topic",
            creator_response=creator_resp,
            skeptic_response=skeptic_resp,
            moderator_response=moderator_resp,
            synthesis="Test synthesis",
            metadata={},
        )

        moderator = arg_map.get_perspective("moderator")
        assert moderator is not None
        assert moderator.role == "moderator"
        assert moderator.stance == "neutral"
        assert moderator.content == "Synthesis content"
        assert moderator.model == "test-model"
        assert moderator.metadata["step"] == 3
        assert moderator.metadata["step_name"] == "Balanced Synthesis"


class TestArgumentWorkflowExecution:
    """Test ArgumentWorkflow execution."""

    @pytest.mark.asyncio
    async def test_workflow_execution_with_mocked_orchestrator(self, argument_workflow):
        """Test workflow execution with mocked RoleOrchestrator."""
        # Create mock role responses as GenerationResponse objects
        creator_response = GenerationResponse(
            content="Strong thesis supporting the position with evidence",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason="end_turn",
        )

        skeptic_response = GenerationResponse(
            content="Critical rebuttal challenging the thesis with counter-arguments",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason="end_turn",
        )

        moderator_response = GenerationResponse(
            content="Balanced synthesis integrating both perspectives",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason="end_turn",
        )

        # Create mock orchestration result
        # role_responses is a list of (role_name, response) tuples
        mock_orchestration_result = OrchestrationResult(
            role_responses=[
                ("creator", creator_response),
                ("skeptic", skeptic_response),
                ("moderator", moderator_response),
            ],
            pattern_used=OrchestrationPattern.SEQUENTIAL,
            execution_order=["creator", "skeptic", "moderator"],
        )

        # Patch RoleOrchestrator.execute
        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(return_value=mock_orchestration_result)

            # Execute workflow
            result = await argument_workflow.run(
                prompt="Universal basic income would reduce poverty"
            )

            # Verify result
            assert result.success is True
            assert len(result.steps) == 3

            # Verify Creator step
            assert result.steps[0].step_number == 1
            assert result.steps[0].metadata["role"] == "creator"
            assert "Strong thesis" in result.steps[0].content

            # Verify Skeptic step
            assert result.steps[1].step_number == 2
            assert result.steps[1].metadata["role"] == "skeptic"
            assert "Critical rebuttal" in result.steps[1].content

            # Verify Moderator step
            assert result.steps[2].step_number == 3
            assert result.steps[2].metadata["role"] == "moderator"
            assert "Balanced synthesis" in result.steps[2].content

            # Verify metadata
            assert result.metadata["roles_executed"] == ["creator", "skeptic", "moderator"]
            assert result.metadata["steps_completed"] == 3
            assert "thread_id" in result.metadata

            # Verify ArgumentMap
            assert "argument_map" in result.metadata
            arg_map = result.metadata["argument_map"]
            assert isinstance(arg_map, ArgumentMap)
            assert len(arg_map.perspectives) == 3

    @pytest.mark.asyncio
    async def test_workflow_metadata(self, argument_workflow):
        """Test workflow metadata generation."""
        # Create mock role responses
        creator_response = GenerationResponse(
            content="Thesis", model="test-model", usage={}, stop_reason="end_turn"
        )

        skeptic_response = GenerationResponse(
            content="Rebuttal", model="test-model", usage={}, stop_reason="end_turn"
        )

        moderator_response = GenerationResponse(
            content="Synthesis", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_orchestration_result = OrchestrationResult(
            role_responses=[
                ("creator", creator_response),
                ("skeptic", skeptic_response),
                ("moderator", moderator_response),
            ],
            pattern_used=OrchestrationPattern.SEQUENTIAL,
            execution_order=["creator", "skeptic", "moderator"],
        )

        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(return_value=mock_orchestration_result)

            result = await argument_workflow.run(prompt="Test topic")

            # Verify metadata structure
            assert result.metadata["provider"] == "test-provider"
            assert result.metadata["model"] == "test-model"
            assert result.metadata["workflow_pattern"] == "role_orchestration"
            assert result.metadata["orchestration_pattern"] == "sequential"
            assert result.metadata["is_continuation"] is False


class TestConversationThreading:
    """Test conversation threading and continuation."""

    @pytest.mark.asyncio
    async def test_new_conversation_creates_thread(self, argument_workflow):
        """Test that new conversation creates a thread ID."""
        creator_response = GenerationResponse(
            content="Thesis", model="test-model", usage={}, stop_reason="end_turn"
        )
        skeptic_response = GenerationResponse(
            content="Rebuttal", model="test-model", usage={}, stop_reason="end_turn"
        )
        moderator_response = GenerationResponse(
            content="Synthesis", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_result = OrchestrationResult(
            role_responses=[
                ("creator", creator_response),
                ("skeptic", skeptic_response),
                ("moderator", moderator_response),
            ],
            pattern_used=OrchestrationPattern.SEQUENTIAL,
            execution_order=["creator", "skeptic", "moderator"],
        )

        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(return_value=mock_result)

            result = await argument_workflow.run(prompt="Test topic")

            assert "thread_id" in result.metadata
            assert result.metadata["is_continuation"] is False
            assert isinstance(uuid.UUID(result.metadata["thread_id"]), uuid.UUID)

    @pytest.mark.asyncio
    async def test_continuation_uses_existing_thread(self, argument_workflow):
        """Test that continuation uses existing thread ID."""
        creator_response = GenerationResponse(
            content="Thesis", model="test-model", usage={}, stop_reason="end_turn"
        )
        skeptic_response = GenerationResponse(
            content="Rebuttal", model="test-model", usage={}, stop_reason="end_turn"
        )
        moderator_response = GenerationResponse(
            content="Synthesis", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_result = OrchestrationResult(
            role_responses=[
                ("creator", creator_response),
                ("skeptic", skeptic_response),
                ("moderator", moderator_response),
            ],
            pattern_used=OrchestrationPattern.SEQUENTIAL,
            execution_order=["creator", "skeptic", "moderator"],
        )

        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(return_value=mock_result)

            # First conversation
            result1 = await argument_workflow.run(prompt="First topic")
            thread_id = result1.metadata["thread_id"]

            # Continuation
            result2 = await argument_workflow.run(
                prompt="Follow-up question", continuation_id=thread_id
            )

            assert result2.metadata["thread_id"] == thread_id
            assert result2.metadata["is_continuation"] is True


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_workflow_handles_orchestration_failure(self, argument_workflow):
        """Test that workflow handles orchestration failures gracefully."""
        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(
                side_effect=Exception("Orchestration failed")
            )

            result = await argument_workflow.run(prompt="Test topic")

            assert result.success is False
            assert result.error == "Orchestration failed"
            assert "thread_id" in result.metadata

    @pytest.mark.asyncio
    async def test_workflow_handles_insufficient_responses(self, argument_workflow):
        """Test that workflow handles insufficient role responses."""
        # Create mock with only 2 responses instead of 3
        creator_response = GenerationResponse(
            content="Thesis", model="test-model", usage={}, stop_reason="end_turn"
        )
        skeptic_response = GenerationResponse(
            content="Rebuttal", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_result = OrchestrationResult(
            role_responses=[
                ("creator", creator_response),
                ("skeptic", skeptic_response),
            ],  # Missing Moderator
            pattern_used=OrchestrationPattern.SEQUENTIAL,
            execution_order=["creator", "skeptic"],
        )

        with patch(
            "model_chorus.workflows.argument.argument_workflow.RoleOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator_instance = MockOrchestrator.return_value
            mock_orchestrator_instance.execute = AsyncMock(return_value=mock_result)

            result = await argument_workflow.run(prompt="Test topic")

            assert result.success is False
            assert "Expected 3 role responses" in result.error
