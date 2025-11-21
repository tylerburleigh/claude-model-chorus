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

from unittest.mock import AsyncMock

import pytest

from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import (
    ConfidenceLevel,
    InvestigationStep,
)
from model_chorus.providers.base_provider import GenerationRequest, GenerationResponse
from model_chorus.workflows.thinkdeep import ThinkDeepWorkflow


class TestInvestigationStepExecution:
    """Test suite for investigation step execution in ThinkDeepWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        provider.check_availability = AsyncMock(return_value=(True, ""))
        return provider

    @pytest.fixture
    def mock_expert_provider(self):
        """Create a mock expert provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "expert_provider"
        provider.validate_api_key.return_value = True
        provider.check_availability = AsyncMock(return_value=(True, ""))
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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute investigation step
        result = await workflow.run(
            step="Investigate authentication patterns in the codebase",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Checking authentication patterns",
            files=["src/auth.py"],
        )

        # Verify result success
        assert result.success is True
        assert result.synthesis is not None
        assert "async/await" in result.synthesis

        # Verify metadata
        assert "thread_id" in result.metadata
        assert result.metadata["provider"] == "test_provider"
        assert result.metadata["model"] == "test-model"
        assert (
            result.metadata["investigation_step"] == 2
        )  # State has 1 step, metadata shows next
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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute investigation step
        result = await workflow.run(
            step="Check database configuration",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Analyzing database configuration",
            files=["config/database.py"],
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
        assert (
            "database" in step.findings.lower()
            or "connection pooling" in step.findings.lower()
        )
        assert "config/database.py" in step.files_checked
        assert step.confidence == ConfidenceLevel.EXPLORING.value  # Default confidence

    @pytest.mark.asyncio
    async def test_multi_step_investigation_progression(
        self, mock_provider, conversation_memory
    ):
        """Test multiple investigation steps with state tracking."""
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Step 1: Initial investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Step 1: Found potential memory leak in cache layer. Need more investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Analyze memory usage patterns",
            step_number=1,
            total_steps=3,
            next_step_required=True,
            findings="Initial memory analysis",
            files=["src/cache.py"],
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
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Check cache eviction policy",
            step_number=2,
            total_steps=3,
            next_step_required=True,
            findings="Found no eviction policy configured",
            continuation_id=thread_id,
            files=["src/cache/eviction.py"],
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
            stop_reason="end_turn",
        )

        result3 = await workflow.run(
            step="Finalize investigation findings",
            step_number=3,
            total_steps=3,
            next_step_required=False,
            findings="Root cause identified - unbounded cache growth",
            continuation_id=thread_id,
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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute with multiple files
        files_to_check = [
            "src/auth/authentication.py",
            "src/auth/authorization.py",
            "tests/test_auth.py",
        ]

        result = await workflow.run(
            step="Review authentication system",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Reviewing authentication implementation",
            files=files_to_check,
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
    async def test_relevant_files_merge_into_state_and_metadata(
        self, mock_provider, conversation_memory
    ):
        """Explicitly supplied relevant files should be tracked and deduplicated."""
        mock_provider.generate.side_effect = [
            GenerationResponse(
                content="Initial investigation synthesis.",
                model="test-model",
                usage={"input_tokens": 25, "output_tokens": 40},
                stop_reason="end_turn",
            ),
            GenerationResponse(
                content="Follow-up investigation synthesis.",
                model="test-model",
                usage={"input_tokens": 30, "output_tokens": 45},
                stop_reason="end_turn",
            ),
        ]

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Step 1: Provide relevant files without opening them
        result1 = await workflow.run(
            step="Scope regression impact",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Identified two modules to review.",
            confidence="exploring",
            files=None,
            relevant_files=[
                "src/api.py",
                "tests/test_api.py",
                "src/api.py",  # duplicate should be removed
            ],
        )

        assert result1.success is True
        assert result1.metadata["relevant_files_this_step"] == [
            "src/api.py",
            "tests/test_api.py",
        ]

        first_prompt: GenerationRequest = mock_provider.generate.call_args_list[0][0][0]
        assert "Additional Relevant Files Referenced This Step" in first_prompt.prompt
        assert "src/api.py" in first_prompt.prompt
        assert "tests/test_api.py" in first_prompt.prompt

        thread_id = result1.metadata["thread_id"]
        state_after_step_1 = workflow.get_investigation_state(thread_id)
        assert state_after_step_1.relevant_files == ["src/api.py", "tests/test_api.py"]

        # Step 2: Provide overlapping files via both inputs to ensure deduplication
        result2 = await workflow.run(
            step="Validate hypothesis with targeted checks",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Confirmed impact is isolated.",
            confidence="medium",
            continuation_id=thread_id,
            files=["src/api.py"],
            relevant_files=["tests/test_api.py", "src/services/auth.py"],
        )

        assert result2.success is True
        assert result2.metadata["relevant_files_this_step"] == [
            "tests/test_api.py",
            "src/services/auth.py",
        ]

        final_state = workflow.get_investigation_state(thread_id)
        assert final_state.relevant_files == [
            "src/api.py",
            "tests/test_api.py",
            "src/services/auth.py",
        ]
        assert result2.metadata["relevant_files"] == final_state.relevant_files
        assert result2.metadata["total_files_examined"] == len(
            final_state.relevant_files
        )

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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute without files
        result = await workflow.run(
            step="Describe the system architecture at a high level",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="System architecture analysis",
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
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Initial step - exploring
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial exploration shows potential issue.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Starting confidence tracking investigation",
        )
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
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Continue investigation",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Evidence gathered, hypothesis forming",
            continuation_id=thread_id,
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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute first step
        result = await workflow.run(
            step="Investigate performance issues",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Investigating performance issues with database",
            files=["config/database.py"],
        )

        thread_id = result.metadata["thread_id"]

        # Add hypothesis manually (simulating workflow logic)
        hypothesis_added = workflow.add_hypothesis(
            thread_id,
            "Database pool size causes performance bottleneck",
            evidence=["Pool size is 5", "Peak load requires 20 connections"],
        )
        assert hypothesis_added is True

        # Continue investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Confirmed: timeout errors correlate with peak traffic.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Analyze timeout patterns",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Confirmed: timeout errors correlate with peak traffic",
            continuation_id=thread_id,
        )

        # Get state and verify
        state = workflow.get_investigation_state(thread_id)

        # Verify steps and hypotheses coexist
        assert len(state.steps) == 2
        assert len(state.hypotheses) == 1
        assert (
            state.hypotheses[0].hypothesis
            == "Database pool size causes performance bottleneck"
        )
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
            content=full_response, model="test-model", usage={}, stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute investigation
        result = await workflow.run(
            step="Investigate memory leak",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Investigating memory leak in cache layer",
            files=["src/cache.py"],
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
            stop_reason="end_turn",
        )

        # Setup expert provider response
        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert validation: Confirmed vulnerability. Recommend immediate fix.",
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow with expert provider
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Set confidence to non-certain to trigger expert validation
        result = await workflow.run(
            step="Review authentication security",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Reviewing authentication security patterns",
            files=["src/auth.py"],
        )

        thread_id = result.metadata["thread_id"]

        # Update confidence to medium (triggers expert validation)
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Run another step to trigger expert validation
        result2 = await workflow.run(
            step="Verify security findings",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Verifying security findings with expert validation",
            continuation_id=thread_id,
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
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute investigation
        result = await workflow.run(
            step="Test investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Testing investigation metadata completeness",
            files=["test.py"],
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
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Execute investigation (should not raise)
        result = await workflow.run(
            step="Test investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Testing error handling",
        )

        # Verify error handling
        assert result.success is False
        assert result.error == "API timeout"
        assert "thread_id" in result.metadata

    @pytest.mark.asyncio
    async def test_empty_provider_response_reports_error(
        self, mock_provider, conversation_memory
    ):
        """Provider empty response should surface error without storing assistant message."""
        mock_provider.generate.return_value = GenerationResponse(
            content="", model="test-model", usage={}, stop_reason="end_turn"
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Investigate empty response scenario",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Initial findings",
            skip_provider_check=True,
        )

        assert result.success is False
        assert result.error == "Provider 'test_provider' returned an empty response"
        assert "thread_id" in result.metadata

        thread = conversation_memory.get_thread(result.metadata["thread_id"])
        assert thread is not None
        assert len(thread.messages) == 0

        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_investigation_without_conversation_memory(self, mock_provider):
        """Test investigation step execution without conversation memory."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation complete without memory.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow without conversation memory
        workflow = ThinkDeepWorkflow(provider=mock_provider)

        # Execute investigation
        result = await workflow.run(
            step="Test investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Testing investigation without conversation memory",
        )

        # Verify success
        assert result.success is True
        assert result.synthesis is not None

        # State should not be persisted without memory
        thread_id = result.metadata["thread_id"]
        state = workflow.get_investigation_state(thread_id)
        assert state is None or len(state.steps) == 0


class TestHypothesisEvolution:
    """Test suite for hypothesis evolution in ThinkDeepWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_add_hypothesis_to_investigation(
        self, mock_provider, conversation_memory
    ):
        """Test adding a hypothesis to an ongoing investigation."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow and start investigation
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to add hypothesis",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypothesis
        success = workflow.add_hypothesis(
            thread_id,
            "Database connection pool is undersized",
            evidence=["Pool size is 5", "Peak connections exceed 20"],
        )

        assert success is True

        # Verify hypothesis was added
        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 1

        hyp = state.hypotheses[0]
        assert hyp.hypothesis == "Database connection pool is undersized"
        assert len(hyp.evidence) == 2
        assert hyp.status == "active"

    @pytest.mark.asyncio
    async def test_update_hypothesis_with_evidence(
        self, mock_provider, conversation_memory
    ):
        """Test adding evidence to an existing hypothesis."""
        # Setup workflow and investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to update hypothesis with evidence",
        )
        thread_id = result.metadata["thread_id"]

        # Add initial hypothesis
        workflow.add_hypothesis(
            thread_id, "Memory leak in cache layer", evidence=["Memory grows over time"]
        )

        # Update with new evidence
        success = workflow.update_hypothesis(
            thread_id,
            "Memory leak in cache layer",
            new_evidence=["No eviction policy found", "Cache size unbounded"],
        )

        assert success is True

        # Verify evidence was added
        state = workflow.get_investigation_state(thread_id)
        hyp = state.hypotheses[0]
        assert len(hyp.evidence) == 3
        assert "Memory grows over time" in hyp.evidence
        assert "No eviction policy found" in hyp.evidence
        assert "Cache size unbounded" in hyp.evidence

    @pytest.mark.asyncio
    async def test_validate_hypothesis(self, mock_provider, conversation_memory):
        """Test marking a hypothesis as validated."""
        # Setup workflow and investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation complete.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test hypothesis validation",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypothesis
        workflow.add_hypothesis(
            thread_id,
            "API uses async/await pattern",
            evidence=["Found async def in code"],
        )

        # Validate hypothesis
        success = workflow.validate_hypothesis(
            thread_id, "API uses async/await pattern"
        )

        assert success is True

        # Verify status changed
        state = workflow.get_investigation_state(thread_id)
        hyp = state.hypotheses[0]
        assert hyp.status == "validated"

    @pytest.mark.asyncio
    async def test_disprove_hypothesis(self, mock_provider, conversation_memory):
        """Test marking a hypothesis as disproven."""
        # Setup workflow and investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test hypothesis disprove",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypothesis
        workflow.add_hypothesis(
            thread_id, "Network latency causes timeout", evidence=["Initial suspicion"]
        )

        # Disprove hypothesis
        success = workflow.disprove_hypothesis(
            thread_id, "Network latency causes timeout"
        )

        assert success is True

        # Verify status changed
        state = workflow.get_investigation_state(thread_id)
        hyp = state.hypotheses[0]
        assert hyp.status == "disproven"

    @pytest.mark.asyncio
    async def test_multiple_hypothesis_evolution(
        self, mock_provider, conversation_memory
    ):
        """Test evolution of multiple competing hypotheses."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test multiple hypothesis evolution",
        )
        thread_id = result.metadata["thread_id"]

        # Add multiple hypotheses
        workflow.add_hypothesis(
            thread_id,
            "H1: Database connection issue",
            evidence=["Slow queries observed"],
        )

        workflow.add_hypothesis(
            thread_id, "H2: Network congestion", evidence=["High latency reported"]
        )

        workflow.add_hypothesis(
            thread_id, "H3: Application code bug", evidence=["Error logs present"]
        )

        # Evolve hypotheses over investigation
        # Add evidence to H1
        workflow.update_hypothesis(
            thread_id,
            "H1: Database connection issue",
            new_evidence=["Connection pool exhausted", "Timeout errors in logs"],
        )

        # Disprove H2
        workflow.disprove_hypothesis(thread_id, "H2: Network congestion")

        # Validate H1
        workflow.validate_hypothesis(thread_id, "H1: Database connection issue")

        # Verify final state
        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 3

        # H1 should be validated with 3 pieces of evidence
        h1 = [h for h in state.hypotheses if "H1" in h.hypothesis][0]
        assert h1.status == "validated"
        assert len(h1.evidence) == 3

        # H2 should be disproven
        h2 = [h for h in state.hypotheses if "H2" in h.hypothesis][0]
        assert h2.status == "disproven"

        # H3 should still be active
        h3 = [h for h in state.hypotheses if "H3" in h.hypothesis][0]
        assert h3.status == "active"

    @pytest.mark.asyncio
    async def test_get_active_hypotheses(self, mock_provider, conversation_memory):
        """Test filtering active hypotheses."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test active hypotheses filtering",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypotheses with different statuses
        workflow.add_hypothesis(thread_id, "H1: Active hypothesis 1")
        workflow.add_hypothesis(thread_id, "H2: Active hypothesis 2")
        workflow.add_hypothesis(thread_id, "H3: To be validated")
        workflow.add_hypothesis(thread_id, "H4: To be disproven")

        # Change statuses
        workflow.validate_hypothesis(thread_id, "H3: To be validated")
        workflow.disprove_hypothesis(thread_id, "H4: To be disproven")

        # Get active hypotheses
        active = workflow.get_active_hypotheses(thread_id)

        assert len(active) == 2
        assert all(h.status == "active" for h in active)
        assert any("H1" in h.hypothesis for h in active)
        assert any("H2" in h.hypothesis for h in active)

    @pytest.mark.asyncio
    async def test_get_all_hypotheses(self, mock_provider, conversation_memory):
        """Test getting all hypotheses regardless of status."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test getting all hypotheses",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypotheses with different statuses
        workflow.add_hypothesis(thread_id, "H1: Active")
        workflow.add_hypothesis(thread_id, "H2: Validated")
        workflow.add_hypothesis(thread_id, "H3: Disproven")

        workflow.validate_hypothesis(thread_id, "H2: Validated")
        workflow.disprove_hypothesis(thread_id, "H3: Disproven")

        # Get all hypotheses
        all_hyps = workflow.get_all_hypotheses(thread_id)

        assert len(all_hyps) == 3

        statuses = [h.status for h in all_hyps]
        assert "active" in statuses
        assert "validated" in statuses
        assert "disproven" in statuses

    @pytest.mark.asyncio
    async def test_hypothesis_persistence_across_turns(
        self, mock_provider, conversation_memory
    ):
        """Test that hypotheses persist across investigation turns."""
        # Setup workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Turn 1: Start investigation and add hypothesis
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=3,
            next_step_required=True,
            findings="Initial investigation findings",
        )
        thread_id = result1.metadata["thread_id"]

        workflow.add_hypothesis(
            thread_id,
            "Performance bottleneck in database",
            evidence=["Slow query times"],
        )

        # Turn 2: Continue investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Additional findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Continue investigation",
            step_number=2,
            total_steps=3,
            next_step_required=True,
            findings="Additional findings during investigation",
            continuation_id=thread_id,
        )

        # Add more evidence
        workflow.update_hypothesis(
            thread_id,
            "Performance bottleneck in database",
            new_evidence=["Missing index found"],
        )

        # Turn 3: Conclude investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Final analysis.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result3 = await workflow.run(
            step="Finalize",
            step_number=3,
            total_steps=3,
            next_step_required=False,
            findings="Final analysis and conclusion",
            continuation_id=thread_id,
        )

        # Validate hypothesis
        workflow.validate_hypothesis(thread_id, "Performance bottleneck in database")

        # Verify hypothesis persisted and evolved
        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 1

        hyp = state.hypotheses[0]
        assert hyp.hypothesis == "Performance bottleneck in database"
        assert len(hyp.evidence) == 2
        assert "Slow query times" in hyp.evidence
        assert "Missing index found" in hyp.evidence
        assert hyp.status == "validated"

    @pytest.mark.asyncio
    async def test_hypothesis_update_with_status_change(
        self, mock_provider, conversation_memory
    ):
        """Test updating hypothesis evidence and status simultaneously."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test hypothesis update with status change",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypothesis
        workflow.add_hypothesis(
            thread_id,
            "Cache eviction policy missing",
            evidence=["Memory grows unbounded"],
        )

        # Update with evidence and status
        success = workflow.update_hypothesis(
            thread_id,
            "Cache eviction policy missing",
            new_evidence=["Confirmed no LRU implementation", "No max size configured"],
            new_status="validated",
        )

        assert success is True

        # Verify both updates applied
        state = workflow.get_investigation_state(thread_id)
        hyp = state.hypotheses[0]
        assert len(hyp.evidence) == 3
        assert hyp.status == "validated"

    @pytest.mark.asyncio
    async def test_hypothesis_not_found_handling(
        self, mock_provider, conversation_memory
    ):
        """Test handling of operations on non-existent hypothesis."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test hypothesis not found handling",
        )
        thread_id = result.metadata["thread_id"]

        # Try to update non-existent hypothesis
        success = workflow.update_hypothesis(
            thread_id, "Non-existent hypothesis", new_evidence=["Some evidence"]
        )

        assert success is False

        # Try to validate non-existent hypothesis
        success = workflow.validate_hypothesis(thread_id, "Non-existent hypothesis")

        assert success is False

        # Try to disprove non-existent hypothesis
        success = workflow.disprove_hypothesis(thread_id, "Non-existent hypothesis")

        assert success is False

    @pytest.mark.asyncio
    async def test_hypothesis_metadata_tracking(
        self, mock_provider, conversation_memory
    ):
        """Test that hypothesis count is tracked in result metadata."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Initial investigation - no hypotheses
        result1 = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Starting investigation with no hypotheses",
        )
        assert result1.metadata["hypotheses_count"] == 0

        thread_id = result1.metadata["thread_id"]

        # Add hypotheses
        workflow.add_hypothesis(thread_id, "H1")
        workflow.add_hypothesis(thread_id, "H2")

        # Continue investigation - should show hypothesis count
        result2 = await workflow.run(
            step="Continue",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Continuing investigation with hypotheses",
            continuation_id=thread_id,
        )

        assert result2.metadata["hypotheses_count"] == 2


class TestConfidenceProgression:
    """Test suite for confidence level progression in ThinkDeepWorkflow."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_initial_confidence_level(self, mock_provider, conversation_memory):
        """Test that investigations start with 'exploring' confidence."""
        # Setup mock response
        mock_provider.generate.return_value = GenerationResponse(
            content="Starting investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Start investigation
        result = await workflow.run(
            step="Begin investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Beginning investigation to test initial confidence",
        )
        thread_id = result.metadata["thread_id"]

        # Verify initial confidence
        state = workflow.get_investigation_state(thread_id)
        assert state.current_confidence == ConfidenceLevel.EXPLORING.value

        # Also verify in result metadata
        assert result.metadata["confidence"] == ConfidenceLevel.EXPLORING.value

    @pytest.mark.asyncio
    async def test_update_confidence_level(self, mock_provider, conversation_memory):
        """Test manually updating confidence level."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test confidence level updates",
        )
        thread_id = result.metadata["thread_id"]

        # Update confidence through all levels
        confidence_levels = [
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
            ConfidenceLevel.ALMOST_CERTAIN,
            ConfidenceLevel.CERTAIN,
        ]

        for level in confidence_levels:
            success = workflow.update_confidence(thread_id, level.value)
            assert success is True

            state = workflow.get_investigation_state(thread_id)
            assert state.current_confidence == level.value

    @pytest.mark.asyncio
    async def test_get_confidence_level(self, mock_provider, conversation_memory):
        """Test retrieving current confidence level."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test confidence retrieval",
        )
        thread_id = result.metadata["thread_id"]

        # Initial confidence
        confidence = workflow.get_confidence(thread_id)
        assert confidence == ConfidenceLevel.EXPLORING.value

        # After update
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
        confidence = workflow.get_confidence(thread_id)
        assert confidence == ConfidenceLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_confidence_progression_across_steps(
        self, mock_provider, conversation_memory
    ):
        """Test confidence increases as investigation progresses."""
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Step 1: Initial exploration
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial exploration shows potential issue.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=4,
            next_step_required=True,
            findings="Initial exploration shows potential issue",
        )
        thread_id = result1.metadata["thread_id"]

        # Verify exploring confidence
        state1 = workflow.get_investigation_state(thread_id)
        assert state1.current_confidence == ConfidenceLevel.EXPLORING.value

        # Step 2: Hypothesis formed - increase to low
        workflow.update_confidence(thread_id, ConfidenceLevel.LOW.value)

        mock_provider.generate.return_value = GenerationResponse(
            content="Hypothesis: Database connection pool issue.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Form hypothesis",
            step_number=2,
            total_steps=4,
            next_step_required=True,
            findings="Hypothesis: Database connection pool issue",
            continuation_id=thread_id,
        )

        state2 = workflow.get_investigation_state(thread_id)
        assert state2.current_confidence == ConfidenceLevel.LOW.value

        # Step 3: Evidence gathered - increase to medium
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        mock_provider.generate.return_value = GenerationResponse(
            content="Evidence supports hypothesis.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result3 = await workflow.run(
            step="Gather evidence",
            step_number=3,
            total_steps=4,
            next_step_required=True,
            findings="Evidence supports hypothesis",
            continuation_id=thread_id,
        )

        state3 = workflow.get_investigation_state(thread_id)
        assert state3.current_confidence == ConfidenceLevel.MEDIUM.value

        # Step 4: Hypothesis validated - increase to high
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        mock_provider.generate.return_value = GenerationResponse(
            content="Hypothesis validated with strong evidence.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result4 = await workflow.run(
            step="Validate hypothesis",
            step_number=4,
            total_steps=4,
            next_step_required=False,
            findings="Hypothesis validated with strong evidence",
            continuation_id=thread_id,
        )

        state4 = workflow.get_investigation_state(thread_id)
        assert state4.current_confidence == ConfidenceLevel.HIGH.value

        # Verify progression in steps
        assert len(state4.steps) == 4
        assert state4.steps[0].confidence == ConfidenceLevel.EXPLORING.value
        assert state4.steps[1].confidence == ConfidenceLevel.LOW.value
        assert state4.steps[2].confidence == ConfidenceLevel.MEDIUM.value
        assert state4.steps[3].confidence == ConfidenceLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_confidence_tracked_in_metadata(
        self, mock_provider, conversation_memory
    ):
        """Test confidence level appears in result metadata."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Initial investigation
        result1 = await workflow.run(
            step="Start",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Starting investigation to test confidence in metadata",
        )
        assert result1.metadata["confidence"] == ConfidenceLevel.EXPLORING.value

        thread_id = result1.metadata["thread_id"]

        # Update confidence
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Continue investigation
        result2 = await workflow.run(
            step="Continue",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Continuing investigation with updated confidence",
            continuation_id=thread_id,
        )
        assert result2.metadata["confidence"] == ConfidenceLevel.MEDIUM.value

    @pytest.mark.asyncio
    async def test_invalid_confidence_level_rejected(
        self, mock_provider, conversation_memory
    ):
        """Test that invalid confidence levels are rejected."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test invalid confidence rejection",
        )
        thread_id = result.metadata["thread_id"]

        # Try invalid confidence level
        success = workflow.update_confidence(thread_id, "invalid_level")
        assert success is False

        # Verify confidence unchanged
        state = workflow.get_investigation_state(thread_id)
        assert state.current_confidence == ConfidenceLevel.EXPLORING.value

    @pytest.mark.asyncio
    async def test_confidence_complete_progression(
        self, mock_provider, conversation_memory
    ):
        """Test complete confidence progression from exploring to certain."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test complete confidence progression",
        )
        thread_id = result.metadata["thread_id"]

        # Progress through all confidence levels
        progression = [
            ConfidenceLevel.EXPLORING,  # Start
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
            ConfidenceLevel.ALMOST_CERTAIN,
            ConfidenceLevel.CERTAIN,
        ]

        # Verify initial state
        state = workflow.get_investigation_state(thread_id)
        assert state.current_confidence == progression[0].value

        # Progress through levels
        for i in range(1, len(progression)):
            success = workflow.update_confidence(thread_id, progression[i].value)
            assert success is True

            state = workflow.get_investigation_state(thread_id)
            assert state.current_confidence == progression[i].value

        # Verify reached certain
        final_state = workflow.get_investigation_state(thread_id)
        assert final_state.current_confidence == ConfidenceLevel.CERTAIN.value

    @pytest.mark.asyncio
    async def test_investigation_completion_criteria(
        self, mock_provider, conversation_memory
    ):
        """Test investigation completion based on confidence level."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test completion criteria",
        )
        thread_id = result.metadata["thread_id"]

        # Not complete initially (no hypotheses, low confidence)
        assert workflow.is_investigation_complete(thread_id) is False

        # Add hypothesis but confidence still exploring
        workflow.add_hypothesis(thread_id, "Test hypothesis")
        assert workflow.is_investigation_complete(thread_id) is False

        # Increase confidence but not high enough
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)
        assert workflow.is_investigation_complete(thread_id) is False

        # Increase to high - still not enough
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
        assert workflow.is_investigation_complete(thread_id) is False

        # Increase to almost certain - should be complete
        workflow.update_confidence(thread_id, ConfidenceLevel.ALMOST_CERTAIN.value)
        assert workflow.is_investigation_complete(thread_id) is True

        # Certain should also be complete
        workflow.update_confidence(thread_id, ConfidenceLevel.CERTAIN.value)
        assert workflow.is_investigation_complete(thread_id) is True

    @pytest.mark.asyncio
    async def test_investigation_summary_includes_confidence(
        self, mock_provider, conversation_memory
    ):
        """Test that investigation summary includes confidence level."""
        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test summary includes confidence",
        )
        thread_id = result.metadata["thread_id"]

        # Get summary with initial confidence
        summary1 = workflow.get_investigation_summary(thread_id)
        assert summary1 is not None
        assert "confidence" in summary1
        assert summary1["confidence"] == ConfidenceLevel.EXPLORING.value

        # Update confidence and get new summary
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
        summary2 = workflow.get_investigation_summary(thread_id)
        assert summary2["confidence"] == ConfidenceLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_confidence_cannot_decrease(self, mock_provider, conversation_memory):
        """Test that confidence can be updated to lower values (no restriction)."""
        # Note: The workflow doesn't enforce that confidence only increases,
        # which allows for scenarios where new evidence might reduce confidence.
        # This test verifies that behavior is allowed.

        # Setup workflow
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        result = await workflow.run(
            step="Start investigation",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Starting investigation to test confidence cannot decrease",
        )
        thread_id = result.metadata["thread_id"]

        # Increase confidence
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
        state1 = workflow.get_investigation_state(thread_id)
        assert state1.current_confidence == ConfidenceLevel.HIGH.value

        # Decrease confidence (allowed - new evidence might reduce confidence)
        success = workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)
        assert success is True

        state2 = workflow.get_investigation_state(thread_id)
        assert state2.current_confidence == ConfidenceLevel.MEDIUM.value

    @pytest.mark.asyncio
    async def test_confidence_persistence_across_turns(
        self, mock_provider, conversation_memory
    ):
        """Test that confidence persists across investigation turns."""
        # Setup workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Turn 1: Start with exploring
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial exploration.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Start",
            step_number=1,
            total_steps=3,
            next_step_required=True,
            findings="Initial exploration",
        )
        thread_id = result1.metadata["thread_id"]

        # Turn 2: Update to medium
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        mock_provider.generate.return_value = GenerationResponse(
            content="Continuing investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Continue",
            step_number=2,
            total_steps=3,
            next_step_required=True,
            findings="Continuing investigation",
            continuation_id=thread_id,
        )

        # Turn 3: Update to high
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        mock_provider.generate.return_value = GenerationResponse(
            content="Final analysis.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result3 = await workflow.run(
            step="Finalize",
            step_number=3,
            total_steps=3,
            next_step_required=False,
            findings="Final analysis",
            continuation_id=thread_id,
        )

        # Verify confidence persisted and progressed
        state = workflow.get_investigation_state(thread_id)
        assert state.current_confidence == ConfidenceLevel.HIGH.value

        # Verify each turn had correct confidence in metadata
        assert result1.metadata["confidence"] == ConfidenceLevel.EXPLORING.value
        assert result2.metadata["confidence"] == ConfidenceLevel.MEDIUM.value
        assert result3.metadata["confidence"] == ConfidenceLevel.HIGH.value


class TestEndToEndIntegration:
    """End-to-end integration tests for complete investigation scenarios."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_five_step_investigation_with_hypothesis_evolution(
        self, mock_provider, conversation_memory
    ):
        """
        Verify that multi-step investigation (5+ steps) works end-to-end.

        This test simulates a complete investigation scenario with:
        - 5+ investigation steps
        - Multiple hypotheses with status changes
        - Confidence progression from exploring to high
        - Evidence accumulation
        - File tracking across steps
        """
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Step 1: Initial exploration
        mock_provider.generate.return_value = GenerationResponse(
            content="Initial exploration: System shows intermittent timeout errors during peak load.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result1 = await workflow.run(
            step="Investigate why application has intermittent timeout errors",
            step_number=1,
            total_steps=6,
            next_step_required=True,
            findings="Initial exploration: System shows intermittent timeout errors during peak load",
            files=["app.py"],
        )

        assert result1.success is True
        thread_id = result1.metadata["thread_id"]
        assert result1.metadata["confidence"] == ConfidenceLevel.EXPLORING.value

        # Add first hypothesis
        workflow.add_hypothesis(
            thread_id,
            "Database connection pool is undersized for peak load",
            evidence=["Timeout errors correlate with peak traffic"],
        )

        # Step 2: Test first hypothesis
        mock_provider.generate.return_value = GenerationResponse(
            content="Database configuration shows pool size of 5. Peak load analysis suggests 20+ concurrent connections needed.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result2 = await workflow.run(
            step="Check database connection pool configuration",
            step_number=2,
            total_steps=6,
            next_step_required=True,
            findings="Database configuration shows pool size of 5. Peak load analysis suggests 20+ concurrent connections needed",
            continuation_id=thread_id,
            files=["config/database.py"],
        )

        assert result2.success is True

        # Add evidence and update confidence
        workflow.update_hypothesis(
            thread_id,
            "Database connection pool is undersized for peak load",
            new_evidence=["Pool size: 5", "Peak concurrent connections: 20+"],
        )
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Add second hypothesis
        workflow.add_hypothesis(
            thread_id,
            "Network latency causes timeouts",
            evidence=["Geographic distance to database server"],
        )

        # Step 3: Test second hypothesis
        mock_provider.generate.return_value = GenerationResponse(
            content="Network metrics show consistent low latency. Average: 2ms, p99: 5ms. No correlation with timeout events.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result3 = await workflow.run(
            step="Analyze network latency metrics",
            step_number=3,
            total_steps=6,
            next_step_required=True,
            findings="Network metrics show consistent low latency. Average: 2ms, p99: 5ms. No correlation with timeout events",
            continuation_id=thread_id,
            files=["logs/network_metrics.log"],
        )

        assert result3.success is True

        # Disprove second hypothesis
        workflow.update_hypothesis(
            thread_id,
            "Network latency causes timeouts",
            new_evidence=["Network latency consistently low (2ms avg, 5ms p99)"],
            new_status="disproven",
        )

        # Step 4: Gather more evidence for first hypothesis
        mock_provider.generate.return_value = GenerationResponse(
            content="Load testing confirms: timeout rate increases linearly with connection pool exhaustion. Clear correlation.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result4 = await workflow.run(
            step="Run load tests to confirm pool exhaustion hypothesis",
            step_number=4,
            total_steps=6,
            next_step_required=True,
            findings="Load testing confirms: timeout rate increases linearly with connection pool exhaustion. Clear correlation",
            continuation_id=thread_id,
            files=["tests/load_test_results.log"],
        )

        assert result4.success is True

        # Validate first hypothesis and increase confidence
        workflow.update_hypothesis(
            thread_id,
            "Database connection pool is undersized for peak load",
            new_evidence=[
                "Load tests confirm correlation",
                "Timeout rate linear with pool exhaustion",
            ],
            new_status="validated",
        )
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Step 5: Verify solution approach
        mock_provider.generate.return_value = GenerationResponse(
            content="Solution validated: Increasing pool size to 25 eliminates timeouts under peak load in staging environment.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result5 = await workflow.run(
            step="Test proposed solution: increase pool size to 25",
            step_number=5,
            total_steps=6,
            next_step_required=True,
            findings="Solution validated: Increasing pool size to 25 eliminates timeouts under peak load in staging environment",
            continuation_id=thread_id,
            files=["staging/validation_results.log"],
        )

        assert result5.success is True
        workflow.update_confidence(thread_id, ConfidenceLevel.VERY_HIGH.value)

        # Step 6: Final conclusion
        mock_provider.generate.return_value = GenerationResponse(
            content="Root cause confirmed: Database connection pool size insufficient. Solution verified in staging.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result6 = await workflow.run(
            step="Summarize findings and confidence in root cause identification",
            step_number=6,
            total_steps=6,
            next_step_required=False,
            findings="Root cause confirmed: Database connection pool size insufficient. Solution verified in staging",
            continuation_id=thread_id,
        )

        assert result6.success is True

        # Verify final state
        state = workflow.get_investigation_state(thread_id)

        # Verify 6 steps completed
        assert len(state.steps) == 6

        # Verify confidence progression
        assert state.steps[0].confidence == ConfidenceLevel.EXPLORING.value
        assert (
            state.steps[1].confidence == ConfidenceLevel.EXPLORING.value
        )  # Not updated yet
        assert state.current_confidence == ConfidenceLevel.VERY_HIGH.value

        # Verify hypotheses
        assert len(state.hypotheses) == 2

        # Find hypotheses by status
        validated_hyps = [h for h in state.hypotheses if h.status == "validated"]
        disproven_hyps = [h for h in state.hypotheses if h.status == "disproven"]

        assert len(validated_hyps) == 1
        assert len(disproven_hyps) == 1

        # Verify validated hypothesis has evidence
        validated_hyp = validated_hyps[0]
        assert "Database connection pool" in validated_hyp.hypothesis
        assert len(validated_hyp.evidence) >= 4  # Initial + 3 updates

        # Verify files tracked
        assert len(state.relevant_files) == 5  # 5 different files examined

        # Verify investigation completion criteria
        summary = workflow.get_investigation_summary(thread_id)
        assert summary is not None
        assert summary["total_steps"] == 6
        assert summary["validated_hypotheses"] == 1
        assert summary["disproven_hypotheses"] == 1
        assert summary["confidence"] == ConfidenceLevel.VERY_HIGH.value
        assert summary["files_examined"] == 5

        print("\n✅ End-to-End Verification Complete:")
        print(f"   - Investigation Steps: {summary['total_steps']}")
        print(
            f"   - Hypotheses: {summary['total_hypotheses']} (1 validated, 1 disproven)"
        )
        print(f"   - Final Confidence: {summary['confidence']}")
        print(f"   - Files Examined: {summary['files_examined']}")
        print(f"   - Investigation Complete: {summary['is_complete']}")

    @pytest.mark.asyncio
    async def test_complete_investigation_workflow(
        self, mock_provider, conversation_memory
    ):
        """
        Test complete investigation workflow from start to completion.

        Verifies:
        - Investigation can be started
        - Hypotheses can be added and evolved
        - Confidence can progress to completion
        - Investigation completion criteria work correctly
        """
        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Start investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Starting systematic investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        result = await workflow.run(
            step="Investigate the issue",
            step_number=1,
            total_steps=5,
            next_step_required=True,
            findings="Starting systematic investigation",
        )
        thread_id = result.metadata["thread_id"]

        # Add hypothesis
        workflow.add_hypothesis(
            thread_id, "Root cause identified", evidence=["Strong evidence"]
        )

        # Progress through investigation
        for step in range(2, 6):
            mock_provider.generate.return_value = GenerationResponse(
                content=f"Investigation step {step} findings.",
                model="test-model",
                usage={},
                stop_reason="end_turn",
            )

            result = await workflow.run(
                step=f"Step {step} investigation",
                step_number=step,
                total_steps=5,
                next_step_required=(step < 5),
                findings=f"Investigation step {step} findings",
                continuation_id=thread_id,
            )

            assert result.success is True

        # Validate hypothesis and reach completion
        workflow.validate_hypothesis(thread_id, "Root cause identified")
        workflow.update_confidence(thread_id, ConfidenceLevel.CERTAIN.value)

        # Verify investigation complete
        assert workflow.is_investigation_complete(thread_id) is True

        # Verify summary
        summary = workflow.get_investigation_summary(thread_id)
        assert summary["is_complete"] is True
        assert summary["confidence"] == ConfidenceLevel.CERTAIN.value
        assert summary["total_steps"] == 5
        assert summary["validated_hypotheses"] == 1
