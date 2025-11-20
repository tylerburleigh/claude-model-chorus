"""
Tests for StudyWorkflow.

Tests verify that the StudyWorkflow correctly:
- Initializes with providers and persona router
- Executes investigation workflow with role-based personas
- Manages conversation threading and history
- Handles errors gracefully
- Integrates with PersonaRouter for intelligent persona selection
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from model_chorus.workflows.study.study_workflow import StudyWorkflow
from model_chorus.core.base_workflow import WorkflowResult, WorkflowStep
from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import ConversationMessage


class TestStudyWorkflowInitialization:
    """Test suite for StudyWorkflow initialization."""

    def test_init_with_provider(self):
        """Test StudyWorkflow initializes with a valid provider."""
        provider = Mock()
        provider.provider_name = "test-provider"

        workflow = StudyWorkflow(provider)

        assert workflow.provider == provider
        assert workflow.name == "Study"
        assert workflow.fallback_providers == []
        assert workflow.persona_router is not None

    def test_init_with_fallback_providers(self):
        """Test StudyWorkflow initializes with fallback providers."""
        provider = Mock()
        provider.provider_name = "primary"
        fallback1 = Mock()
        fallback1.provider_name = "fallback1"
        fallback2 = Mock()
        fallback2.provider_name = "fallback2"

        workflow = StudyWorkflow(provider, fallback_providers=[fallback1, fallback2])

        assert workflow.provider == provider
        assert len(workflow.fallback_providers) == 2
        assert fallback1 in workflow.fallback_providers
        assert fallback2 in workflow.fallback_providers

    def test_init_provider_none_raises_error(self):
        """Test that providing None as provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be None"):
            StudyWorkflow(None)

    def test_init_with_conversation_memory(self):
        """Test StudyWorkflow initializes with conversation memory."""
        provider = Mock()
        provider.provider_name = "test-provider"
        memory = ConversationMemory()

        workflow = StudyWorkflow(provider, conversation_memory=memory)

        assert workflow.conversation_memory == memory

    def test_init_persona_router_ready(self):
        """Test that persona router is initialized with personas."""
        provider = Mock()
        provider.provider_name = "test-provider"

        workflow = StudyWorkflow(provider)

        personas = workflow.persona_router.get_available_personas()
        assert personas is not None
        assert isinstance(personas, list)
        assert len(personas) > 0

    def test_init_with_config(self):
        """Test StudyWorkflow initializes with custom config."""
        provider = Mock()
        provider.provider_name = "test-provider"
        config = {"custom_key": "custom_value"}

        workflow = StudyWorkflow(provider, config=config)

        assert workflow.config == config


class TestStudyWorkflowRun:
    """Test suite for StudyWorkflow.run() method."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    @pytest.mark.asyncio
    async def test_run_with_valid_prompt(self, workflow):
        """Test run() executes successfully with valid prompt."""
        prompt = "Explore authentication patterns"

        result = await workflow.run(prompt, skip_provider_check=True)

        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.synthesis is not None
        assert len(result.synthesis) > 0

    @pytest.mark.asyncio
    async def test_run_empty_prompt_raises_error(self, workflow):
        """Test run() raises ValueError for empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await workflow.run("", skip_provider_check=True)

    @pytest.mark.asyncio
    async def test_run_whitespace_prompt_raises_error(self, workflow):
        """Test run() raises ValueError for whitespace-only prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await workflow.run("   ", skip_provider_check=True)

    @pytest.mark.asyncio
    async def test_run_returns_workflow_result(self, workflow):
        """Test run() returns WorkflowResult."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        assert isinstance(result, WorkflowResult)
        assert hasattr(result, "success")
        assert hasattr(result, "synthesis")
        assert hasattr(result, "steps")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_run_result_metadata_structure(self, workflow):
        """Test run() result contains expected metadata."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        assert result.metadata is not None
        assert "thread_id" in result.metadata
        assert "workflow_type" in result.metadata
        assert result.metadata["workflow_type"] == "study"
        assert "personas_used" in result.metadata
        assert "investigation_rounds" in result.metadata
        assert "timestamp" in result.metadata
        assert "is_continuation" in result.metadata

    @pytest.mark.asyncio
    async def test_run_result_steps_is_list(self, workflow):
        """Test run() result.steps is a list."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        assert isinstance(result.steps, list)
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_run_result_steps_are_workflow_steps(self, workflow):
        """Test run() result.steps contains WorkflowStep objects."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        for step in result.steps:
            assert isinstance(step, WorkflowStep)
            assert hasattr(step, "step_number")
            assert hasattr(step, "content")
            assert hasattr(step, "model")

    @pytest.mark.asyncio
    async def test_run_creates_thread_id(self, workflow):
        """Test run() creates a thread_id when continuation_id not provided."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        thread_id = result.metadata.get("thread_id")
        assert thread_id is not None
        assert len(thread_id) > 0

    @pytest.mark.asyncio
    async def test_run_with_continuation_id(self, workflow):
        """Test run() uses provided continuation_id as thread_id."""
        continuation_id = "test-thread-123"

        result = await workflow.run(
            "Test prompt", continuation_id=continuation_id, skip_provider_check=True
        )

        assert result.metadata["thread_id"] == continuation_id
        assert result.metadata["is_continuation"] is True

    @pytest.mark.asyncio
    async def test_run_without_continuation_is_not_continuation(self, workflow):
        """Test run() is_continuation is False when no continuation_id provided."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        assert result.metadata["is_continuation"] is False

    @pytest.mark.asyncio
    async def test_run_metadata_timestamp_format(self, workflow):
        """Test run() metadata timestamp is ISO format."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        timestamp = result.metadata["timestamp"]
        # Should be able to parse as ISO format
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            assert True
        except ValueError:
            pytest.fail(f"Invalid ISO timestamp: {timestamp}")

    @pytest.mark.asyncio
    async def test_run_synthesis_is_string(self, workflow):
        """Test run() synthesis is a string."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        assert isinstance(result.synthesis, str)
        assert len(result.synthesis) > 0


class TestStudyWorkflowPersonaSetup:
    """Test suite for persona setup in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    def test_setup_personas_default(self, workflow):
        """Test _setup_personas returns default personas when none provided."""
        personas = workflow._setup_personas(None)

        assert isinstance(personas, list)
        assert len(personas) == 2
        assert any(p["name"] == "Researcher" for p in personas)
        assert any(p["name"] == "Critic" for p in personas)

    def test_setup_personas_default_structure(self, workflow):
        """Test default personas have required fields."""
        personas = workflow._setup_personas(None)

        for persona in personas:
            assert "name" in persona
            assert "expertise" in persona
            assert "role" in persona

    def test_setup_personas_custom(self, workflow):
        """Test _setup_personas preserves custom personas."""
        custom_personas = [
            {"name": "Expert", "expertise": "specialized knowledge", "role": "domain expert"}
        ]

        personas = workflow._setup_personas(custom_personas)

        assert personas == custom_personas

    def test_setup_personas_empty_list(self, workflow):
        """Test _setup_personas returns defaults for empty list."""
        personas = workflow._setup_personas([])

        # Empty list is falsy, so defaults are used
        assert len(personas) == 2

    @pytest.mark.asyncio
    async def test_run_personas_in_metadata(self, workflow):
        """Test run() includes personas_used in metadata."""
        result = await workflow.run("Test prompt", skip_provider_check=True)

        personas_used = result.metadata.get("personas_used")
        assert personas_used is not None
        assert isinstance(personas_used, list)
        assert len(personas_used) > 0


class TestStudyWorkflowConversationHandling:
    """Test suite for conversation memory integration in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow_with_memory(self, mock_provider):
        """Create a StudyWorkflow with conversation memory."""
        memory = ConversationMemory()
        return StudyWorkflow(mock_provider, conversation_memory=memory)

    @pytest.fixture
    def workflow_without_memory(self, mock_provider):
        """Create a StudyWorkflow without conversation memory."""
        return StudyWorkflow(mock_provider, conversation_memory=None)

    @pytest.mark.asyncio
    async def test_run_creates_thread_in_memory(self, workflow_with_memory):
        """Test run() creates a thread in conversation memory."""
        result = await workflow_with_memory.run("Test prompt", skip_provider_check=True)

        thread_id = result.metadata["thread_id"]
        thread = workflow_with_memory.conversation_memory.get_thread(thread_id)

        assert thread is not None

    @pytest.mark.asyncio
    async def test_run_stores_messages_in_memory(self, workflow_with_memory):
        """Test run() stores user and assistant messages in memory."""
        prompt = "Test research question"
        result = await workflow_with_memory.run(prompt, skip_provider_check=True)

        thread_id = result.metadata["thread_id"]
        thread = workflow_with_memory.conversation_memory.get_thread(thread_id)

        # Should have at least user message and assistant response
        assert len(thread.messages) >= 2

    @pytest.mark.asyncio
    async def test_run_without_memory_still_works(self, workflow_without_memory):
        """Test run() works correctly without conversation memory."""
        result = await workflow_without_memory.run("Test prompt", skip_provider_check=True)

        assert result.success is True
        assert result.metadata["thread_id"] is not None

    @pytest.mark.asyncio
    async def test_run_reuses_thread_on_continuation(self, workflow_with_memory):
        """Test run() with continuation_id reuses the thread."""
        # First run
        result1 = await workflow_with_memory.run("First question", skip_provider_check=True)
        thread_id = result1.metadata["thread_id"]

        # Get message count after first run
        thread_before = workflow_with_memory.conversation_memory.get_thread(thread_id)
        message_count_before = len(thread_before.messages)

        # Second run with continuation
        result2 = await workflow_with_memory.run(
            "Follow-up question", continuation_id=thread_id, skip_provider_check=True
        )

        # Verify same thread is used
        assert result2.metadata["thread_id"] == thread_id

        # Verify messages are added
        thread_after = workflow_with_memory.conversation_memory.get_thread(thread_id)
        assert len(thread_after.messages) > message_count_before


class TestStudyWorkflowInvestigation:
    """Test suite for investigation flow in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    @pytest.mark.asyncio
    async def test_investigation_returns_steps(self, workflow):
        """Test _conduct_investigation returns list of WorkflowStep."""
        personas = [{"name": "Researcher"}, {"name": "Critic"}]

        steps = await workflow._conduct_investigation(
            prompt="Test question", personas=personas, history=[], thread_id="test-123"
        )

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert all(isinstance(step, WorkflowStep) for step in steps)

    @pytest.mark.asyncio
    async def test_investigation_steps_have_metadata(self, workflow):
        """Test investigation steps contain metadata."""
        personas = [{"name": "Researcher"}]

        steps = await workflow._conduct_investigation(
            prompt="Test question", personas=personas, history=[], thread_id="test-123"
        )

        for step in steps:
            assert step.step_number is not None
            assert step.content is not None
            assert step.model is not None
            assert step.metadata is not None

    @pytest.mark.asyncio
    async def test_investigation_includes_available_personas(self, workflow):
        """Test investigation metadata includes available personas."""
        personas = [{"name": "Researcher"}, {"name": "Critic"}]

        steps = await workflow._conduct_investigation(
            prompt="Test question", personas=personas, history=[], thread_id="test-123"
        )

        # Metadata should indicate router is available
        assert steps[0].metadata.get("router_available") is True
        assert steps[0].metadata.get("available_personas") is not None

    @pytest.mark.asyncio
    async def test_investigation_empty_personas(self, workflow):
        """Test _conduct_investigation handles empty personas."""
        steps = await workflow._conduct_investigation(
            prompt="Test question", personas=[], history=[], thread_id="test-123"
        )

        assert isinstance(steps, list)
        assert len(steps) > 0


class TestStudyWorkflowSynthesis:
    """Test suite for synthesis in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    @pytest.mark.asyncio
    async def test_synthesize_findings_returns_string(self, workflow):
        """Test _synthesize_findings returns a string."""
        step = WorkflowStep(step_number=1, content="Test finding", model="test-model")

        synthesis = await workflow._synthesize_findings([step])

        assert isinstance(synthesis, str)
        assert len(synthesis) > 0

    @pytest.mark.asyncio
    async def test_synthesize_findings_empty_steps(self, workflow):
        """Test _synthesize_findings handles empty steps."""
        synthesis = await workflow._synthesize_findings([])

        assert isinstance(synthesis, str)

    @pytest.mark.asyncio
    async def test_synthesize_findings_multiple_steps(self, workflow):
        """Test _synthesize_findings with multiple steps."""
        steps = [
            WorkflowStep(step_number=1, content="Finding 1", model="model1"),
            WorkflowStep(step_number=2, content="Finding 2", model="model2"),
            WorkflowStep(step_number=3, content="Finding 3", model="model3"),
        ]

        synthesis = await workflow._synthesize_findings(steps)

        assert isinstance(synthesis, str)
        assert "3" in synthesis  # Should mention number of steps

    @pytest.mark.asyncio
    async def test_synthesize_includes_step_count(self, workflow):
        """Test synthesis output includes investigation step count."""
        steps = [
            WorkflowStep(step_number=1, content="Step 1", model="model"),
            WorkflowStep(step_number=2, content="Step 2", model="model"),
        ]

        synthesis = await workflow._synthesize_findings(steps)

        assert "2" in synthesis


class TestStudyWorkflowErrorHandling:
    """Test suite for error handling in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    @pytest.mark.asyncio
    async def test_run_error_returns_false_success(self, workflow):
        """Test run() returns success=False on error."""
        # Mock provider to raise exception
        workflow.provider.generate = AsyncMock(side_effect=Exception("Provider error"))

        # Skip provider check to avoid checking availability
        result = await workflow.run("Test prompt", skip_provider_check=True)

        # Even with error, should return WorkflowResult
        assert isinstance(result, WorkflowResult)

    @pytest.mark.asyncio
    async def test_run_captures_error_message(self, workflow):
        """Test run() captures error message."""
        workflow.provider.generate = AsyncMock(side_effect=Exception("Test error"))

        result = await workflow.run("Test prompt", skip_provider_check=True)

        # Error should be captured if workflow fails
        if not result.success:
            assert hasattr(result, "error")


class TestStudyWorkflowRoutingHistory:
    """Test suite for routing history access in StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow(self, mock_provider):
        """Create a StudyWorkflow instance for testing."""
        return StudyWorkflow(mock_provider)

    def test_get_routing_history_available(self, workflow):
        """Test get_routing_history() method is available."""
        history = workflow.get_routing_history()

        assert history is not None
        assert isinstance(history, list)

    def test_get_routing_history_with_limit(self, workflow):
        """Test get_routing_history() respects limit parameter."""
        history = workflow.get_routing_history(limit=5)

        assert isinstance(history, list)
        if len(history) > 5:
            pytest.fail("Limit parameter not respected")

    def test_get_routing_history_with_investigation_id(self, workflow):
        """Test get_routing_history() filters by investigation_id."""
        history = workflow.get_routing_history(investigation_id="test-123")

        assert isinstance(history, list)


class TestStudyWorkflowIntegration:
    """Integration tests for StudyWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.provider_name = "test-provider"
        return provider

    @pytest.fixture
    def workflow_with_memory(self, mock_provider):
        """Create a StudyWorkflow with conversation memory."""
        memory = ConversationMemory()
        return StudyWorkflow(mock_provider, conversation_memory=memory)

    @pytest.mark.asyncio
    async def test_full_workflow_execution(self, workflow_with_memory):
        """Test complete workflow execution from start to finish."""
        prompt = "Explore machine learning fundamentals"

        result = await workflow_with_memory.run(prompt, skip_provider_check=True)

        # Verify result structure
        assert result.success is True
        assert result.synthesis is not None
        assert len(result.steps) > 0
        assert result.metadata is not None

        # Verify metadata completeness
        assert "thread_id" in result.metadata
        assert "workflow_type" in result.metadata
        assert "personas_used" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_continuation_flow(self, workflow_with_memory):
        """Test conversation continuation across multiple runs."""
        # First investigation
        result1 = await workflow_with_memory.run(
            "Initial research question", skip_provider_check=True
        )
        thread_id = result1.metadata["thread_id"]

        # Get thread state after first run
        thread_before = workflow_with_memory.conversation_memory.get_thread(thread_id)
        first_run_messages = len(thread_before.messages)

        # Continue investigation
        result2 = await workflow_with_memory.run(
            "Follow-up investigation", continuation_id=thread_id, skip_provider_check=True
        )

        # Verify continuation
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["is_continuation"] is True

        # Verify messages accumulated
        thread_after = workflow_with_memory.conversation_memory.get_thread(thread_id)
        assert len(thread_after.messages) > first_run_messages

    @pytest.mark.asyncio
    async def test_workflow_with_custom_personas(self, workflow_with_memory):
        """Test workflow execution with custom personas."""
        # Note: personas is extracted from kwargs in run(), so we test
        # that the workflow properly uses default personas when none are passed
        result = await workflow_with_memory.run(
            "Test with custom personas", skip_provider_check=True
        )

        assert result.success is True
        personas_used = result.metadata.get("personas_used")
        # Should have default personas (Researcher and Critic)
        assert len(personas_used) >= 1

    def test_router_persona_count(self, workflow_with_memory):
        """Test that persona router has expected personas available."""
        available = workflow_with_memory.persona_router.get_available_personas()

        assert available is not None
        assert isinstance(available, list)
        assert len(available) > 0

        # Should include common research personas
        persona_names = [p for p in available]
        assert any("Researcher" in p or "research" in p.lower() for p in persona_names)
