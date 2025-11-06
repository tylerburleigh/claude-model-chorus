"""
Integration tests for ThinkDeep workflow.

Tests verify ThinkDeepWorkflow functionality including:
- Investigation step execution and tracking
- Multi-step investigation with state management
- Conversation threading and continuation
- Hypothesis management through workflow
- Confidence progression tracking
- File examination tracking
- Expert validation integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from modelchorus.workflows.thinkdeep import ThinkDeepWorkflow
from modelchorus.providers.base_provider import GenerationResponse, GenerationRequest
from modelchorus.core.conversation import ConversationMemory
from modelchorus.core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)


class TestInvestigationStepExecution:
    """Test suite for investigation step execution in ThinkDeepWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def mock_expert_provider(self):
        """Create a mock expert provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "expert_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_single_investigation_step_execution(
        self, mock_provider, conversation_memory
    ):
        """Test executing a single investigation step."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Found async/await pattern in authentication service. The AuthService uses modern async patterns.",
            model="test-model",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation step
        result = await workflow.run(
            prompt="Investigate authentication patterns in the codebase",
            files=["src/auth.py"]
        )

        # Verify result success
        assert result.success is True
        assert result.synthesis is not None
        assert "async/await" in result.synthesis

        # Verify metadata
        assert "thread_id" in result.metadata
        assert result.metadata["provider"] == "test_provider"
        assert result.metadata["model"] == "test-model"
        assert result.metadata["investigation_step"] == 2  # State has 1 step, metadata shows next
        assert result.metadata["is_continuation"] is False

        # Verify provider was called
        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args
        request = call_args[0][0]
        assert isinstance(request, GenerationRequest)
        assert "authentication patterns" in request.prompt.lower()

    @pytest.mark.asyncio
    async def test_investigation_step_creates_investigation_step_object(
        self, mock_provider, conversation_memory
    ):
        """Test that running workflow creates InvestigationStep in state."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings: Database uses connection pooling with size limit of 5.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation step
        result = await workflow.run(
            prompt="Check database configuration",
            files=["config/database.py"]
        )

        # Get investigation state
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)

        # Verify InvestigationStep was created
        assert state is not None
        assert len(state.steps) == 1

        step = state.steps[0]
        assert isinstance(step, InvestigationStep)
        assert step.step_number == 1
        assert "database" in step.findings.lower() or "connection pooling" in step.findings.lower()
        assert "config/database.py" in step.files_checked
        assert step.confidence == ConfidenceLevel.EXPLORING.value  # Default confidence

    @pytest.mark.asyncio
    async def test_multi_step_investigation_progression(
        self, mock_provider, conversation_memory
    ):
        """Test multiple investigation steps with state tracking."""
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Initial investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Step 1: Found potential memory leak in cache layer. Need more investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result1 = await workflow.run(
            prompt="Analyze memory usage patterns",
            files=["src/cache.py"]
        )
        thread_id = result1.metadata["thread_id"]

        # Verify step 1
        state1 = workflow.get_investigation_state(thread_id)
        assert len(state1.steps) == 1
        assert state1.steps[0].step_number == 1

        # Step 2: Continue investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Step 2: Confirmed no eviction policy configured. Cache grows unbounded.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Check cache eviction policy",
            continuation_id=thread_id,
            files=["src/cache/eviction.py"]
        )

        # Verify step 2
        state2 = workflow.get_investigation_state(thread_id)
        assert len(state2.steps) == 2
        assert state2.steps[0].step_number == 1
        assert state2.steps[1].step_number == 2

        # Step 3: Conclude investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Step 3: Root cause identified. Memory leak due to unbounded cache.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result3 = await workflow.run(
            prompt="Finalize investigation findings",
            continuation_id=thread_id
        )

        # Verify final state
        state3 = workflow.get_investigation_state(thread_id)
        assert len(state3.steps) == 3
        assert state3.steps[2].step_number == 3

        # Verify metadata progression
        assert result1.metadata["investigation_step"] == 2
        assert result2.metadata["investigation_step"] == 3
        assert result3.metadata["investigation_step"] == 4

    @pytest.mark.asyncio
    async def test_investigation_step_tracks_files_checked(
        self, mock_provider, conversation_memory
    ):
        """Test that investigation steps track files examined."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Analyzed authentication and authorization modules.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute with multiple files
        files_to_check = [
            "src/auth/authentication.py",
            "src/auth/authorization.py",
            "tests/test_auth.py"
        ]

        result = await workflow.run(
            prompt="Review authentication system",
            files=files_to_check
        )

        # Get state and verify files tracked
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)

        assert len(state.steps) == 1
        step = state.steps[0]

        # Verify all files were tracked in the step
        assert len(step.files_checked) == 3
        for file_path in files_to_check:
            assert file_path in step.files_checked

        # Verify files also tracked in state.relevant_files
        for file_path in files_to_check:
            assert file_path in state.relevant_files

    @pytest.mark.asyncio
    async def test_investigation_step_with_no_files(
        self, mock_provider, conversation_memory
    ):
        """Test investigation step execution without file context."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Conceptual analysis of system architecture completed.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute without files
        result = await workflow.run(
            prompt="Describe the system architecture at a high level"
        )

        # Get state and verify
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)

        assert len(state.steps) == 1
        step = state.steps[0]

        # Verify files_checked is empty
        assert step.files_checked == []
        assert state.relevant_files == []

    @pytest.mark.asyncio
    async def test_investigation_step_confidence_tracking(
        self, mock_provider, conversation_memory
    ):
        """Test that investigation steps track confidence levels."""
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Initial step - exploring
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial exploration shows potential issue.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Check initial confidence
        state1 = workflow.get_investigation_state(thread_id)
        assert state1.current_confidence == ConfidenceLevel.EXPLORING.value
        assert state1.steps[0].confidence == ConfidenceLevel.EXPLORING.value

        # Update confidence manually (simulating workflow logic)
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Next step with updated confidence
        mock_provider.generate.return_value = GenerationResponse(
            content="Evidence gathered, hypothesis forming.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Continue investigation",
            continuation_id=thread_id
        )

        # Check updated confidence
        state2 = workflow.get_investigation_state(thread_id)
        assert state2.current_confidence == ConfidenceLevel.MEDIUM.value

        # New step should have medium confidence
        assert len(state2.steps) == 2
        assert state2.steps[1].confidence == ConfidenceLevel.MEDIUM.value

    @pytest.mark.asyncio
    async def test_investigation_step_with_hypothesis_integration(
        self, mock_provider, conversation_memory
    ):
        """Test investigation steps work with hypothesis management."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Database connection pool size is insufficient for peak load.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute first step
        result = await workflow.run(
            prompt="Investigate performance issues",
            files=["config/database.py"]
        )

        thread_id = result.metadata["thread_id"]

        # Add hypothesis manually (simulating workflow logic)
        hypothesis_added = workflow.add_hypothesis(
            thread_id,
            "Database pool size causes performance bottleneck",
            evidence=["Pool size is 5", "Peak load requires 20 connections"]
        )
        assert hypothesis_added is True

        # Continue investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Confirmed: timeout errors correlate with peak traffic.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Analyze timeout patterns",
            continuation_id=thread_id
        )

        # Get state and verify
        state = workflow.get_investigation_state(thread_id)

        # Verify steps and hypotheses coexist
        assert len(state.steps) == 2
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].hypothesis == "Database pool size causes performance bottleneck"
        assert len(state.hypotheses[0].evidence) == 2

    @pytest.mark.asyncio
    async def test_investigation_step_findings_extraction(
        self, mock_provider, conversation_memory
    ):
        """Test that findings are properly extracted from responses."""
        # Setup mock response with multiple paragraphs
        full_response = """The investigation reveals a critical memory leak in the cache layer.

The cache implementation does not have an eviction policy configured, allowing entries to accumulate indefinitely. This leads to continuous memory growth over time.

Additional analysis shows that the cache hit rate is 95%, indicating the cache is effective but needs proper memory management.

Recommended action: Implement LRU eviction policy with maximum size limit."""

        mock_provider.generate.return_value = GenerationResponse(
            content=full_response,
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation
        result = await workflow.run(
            prompt="Investigate memory leak",
            files=["src/cache.py"]
        )

        # Get state and check findings
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)

        assert len(state.steps) == 1
        step = state.steps[0]

        # Findings should be first paragraph (or truncated to 500 chars)
        assert "memory leak" in step.findings.lower()
        assert len(step.findings) <= 500

    @pytest.mark.asyncio
    async def test_investigation_step_with_expert_validation(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test investigation step execution with expert validation."""
        # Setup primary provider response
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis: Security vulnerability in authentication flow.",
            model="primary-model",
            usage={},
            stop_reason="end_turn"
        )

        # Setup expert provider response
        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert validation: Confirmed vulnerability. Recommend immediate fix.",
            model="expert-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow with expert provider
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True}
        )

        # Set confidence to non-certain to trigger expert validation
        result = await workflow.run(
            prompt="Review authentication security",
            files=["src/auth.py"]
        )

        thread_id = result.metadata["thread_id"]

        # Update confidence to medium (triggers expert validation)
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Run another step to trigger expert validation
        result2 = await workflow.run(
            prompt="Verify security findings",
            continuation_id=thread_id
        )

        # Verify expert validation was performed
        assert result2.metadata.get("expert_validation_performed") is True

        # Verify both providers were called
        assert mock_provider.generate.call_count >= 1
        assert mock_expert_provider.generate.call_count >= 1

    @pytest.mark.asyncio
    async def test_investigation_step_metadata_completeness(
        self, mock_provider, conversation_memory
    ):
        """Test that investigation step result includes complete metadata."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation complete.",
            model="test-model-v1",
            usage={"input_tokens": 150, "output_tokens": 75},
            stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation
        result = await workflow.run(
            prompt="Test investigation",
            files=["test.py"]
        )

        # Verify all expected metadata fields
        assert "thread_id" in result.metadata
        assert "provider" in result.metadata
        assert "model" in result.metadata
        assert "usage" in result.metadata
        assert "stop_reason" in result.metadata
        assert "is_continuation" in result.metadata
        assert "investigation_step" in result.metadata
        assert "hypotheses_count" in result.metadata
        assert "confidence" in result.metadata
        assert "files_examined" in result.metadata
        assert "expert_validation_performed" in result.metadata

        # Verify metadata values
        assert result.metadata["provider"] == "test_provider"
        assert result.metadata["model"] == "test-model-v1"
        assert result.metadata["usage"]["input_tokens"] == 150
        assert result.metadata["usage"]["output_tokens"] == 75
        assert result.metadata["stop_reason"] == "end_turn"
        assert result.metadata["is_continuation"] is False
        assert result.metadata["investigation_step"] == 2
        assert result.metadata["hypotheses_count"] == 0
        assert result.metadata["confidence"] == "exploring"
        assert result.metadata["files_examined"] == 1

    @pytest.mark.asyncio
    async def test_investigation_step_error_handling(
        self, mock_provider, conversation_memory
    ):
        """Test investigation step handles provider errors gracefully."""
        # Setup provider to raise error
        mock_provider.generate.side_effect = Exception("API timeout")

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation (should not raise)
        result = await workflow.run(prompt="Test investigation")

        # Verify error handling
        assert result.success is False
        assert result.error == "API timeout"
        assert "thread_id" in result.metadata

    @pytest.mark.asyncio
    async def test_investigation_without_conversation_memory(
        self, mock_provider
    ):
        """Test investigation step execution without conversation memory."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation complete without memory.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        # Create workflow without conversation memory
        workflow = ThinkDeepWorkflow(provider=mock_provider)

        # Execute investigation
        result = await workflow.run(prompt="Test investigation")

        # Verify success
        assert result.success is True
        assert result.synthesis is not None

        # State should not be persisted without memory
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)
        assert state is None or len(state.steps) == 0
