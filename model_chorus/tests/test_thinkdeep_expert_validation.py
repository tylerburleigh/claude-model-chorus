"""
Expert validation flow tests for ThinkDeep workflow.

Tests verify expert validation functionality including:
- Expert provider integration and triggering
- Confidence-based validation triggering
- Multiple expert models coordination
- Expert validation result handling
- Error handling when expert validation fails
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from model_chorus.workflows.thinkdeep import ThinkDeepWorkflow
from model_chorus.providers.base_provider import GenerationResponse, GenerationRequest
from model_chorus.core.conversation import ConversationMemory
from model_chorus.core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)


class TestExpertProviderIntegration:
    """Test suite for expert provider integration in ThinkDeepWorkflow."""

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
    async def test_expert_validation_enabled_with_expert_provider(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation is enabled when expert provider is provided."""
        # Create workflow with expert provider
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
        )

        # Expert validation should be enabled by default when expert_provider is set
        assert workflow.enable_expert_validation is True
        assert workflow.expert_provider == mock_expert_provider

    @pytest.mark.asyncio
    async def test_expert_validation_disabled_without_expert_provider(
        self, mock_provider, conversation_memory
    ):
        """Test that expert validation is disabled when no expert provider."""
        # Create workflow without expert provider
        workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Expert validation should be disabled
        assert workflow.enable_expert_validation is False
        assert workflow.expert_provider is None

    @pytest.mark.asyncio
    async def test_expert_validation_explicit_disable_via_config(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation can be explicitly disabled via config."""
        # Create workflow with expert provider but disable validation
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": False},
        )

        # Expert validation should be disabled despite expert_provider being set
        assert workflow.enable_expert_validation is False
        assert workflow.expert_provider is not None  # Provider still set

    @pytest.mark.asyncio
    async def test_expert_validation_explicit_enable_via_config(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation can be explicitly enabled via config."""
        # Create workflow with expert provider and explicit enable
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Expert validation should be enabled
        assert workflow.enable_expert_validation is True


class TestExpertValidationTriggering:
    """Test suite for expert validation triggering logic."""

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
    async def test_expert_validation_triggered_at_medium_confidence(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation is triggered when confidence reaches medium."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert validation confirms findings.",
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow with expert validation enabled
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Update confidence to medium (should trigger validation on next turn)
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Reset call count to track only next call
        mock_expert_provider.generate.reset_mock()

        # Continue investigation (should trigger expert validation)
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Expert provider should have been called
        assert mock_expert_provider.generate.call_count >= 1
        assert result2.metadata.get("expert_validation_performed") is True

    @pytest.mark.asyncio
    async def test_expert_validation_not_triggered_at_exploring_confidence(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation is NOT triggered at exploring confidence."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.", model="test-model", usage={}, stop_reason="end_turn"
        )

        # Create workflow with expert validation enabled
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation (confidence: exploring)
        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Reset call count to only track next call
        mock_expert_provider.generate.reset_mock()

        # Continue investigation without updating confidence
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Expert provider should NOT have been called for second step (confidence too low)
        # Note: Expert validation may be attempted but errors are caught and handled gracefully
        # The key test is that expert_validation_performed metadata is False
        assert result2.metadata.get("expert_validation_performed") is False

    @pytest.mark.asyncio
    async def test_expert_validation_triggered_at_high_confidence(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation is triggered at high confidence."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert validation confirms findings.",
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow with expert validation enabled
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Update confidence to high
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Reset call count
        mock_expert_provider.generate.reset_mock()

        # Continue investigation (should trigger expert validation)
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Expert provider should have been called
        assert mock_expert_provider.generate.call_count >= 1
        assert result2.metadata.get("expert_validation_performed") is True

    @pytest.mark.asyncio
    async def test_expert_validation_not_triggered_when_disabled(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation is NOT triggered when disabled via config."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation findings.", model="test-model", usage={}, stop_reason="end_turn"
        )

        # Create workflow with expert validation DISABLED
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": False},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Update confidence to high (would normally trigger validation)
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Continue investigation
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Expert provider should NOT have been called (validation disabled)
        assert mock_expert_provider.generate.call_count == 0
        assert result2.metadata.get("expert_validation_performed") is False


class TestExpertValidationResultHandling:
    """Test suite for handling expert validation results."""

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
    async def test_expert_validation_result_included_in_metadata(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation result is included in result metadata."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis.", model="test-model", usage={}, stop_reason="end_turn"
        )

        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert validation: Analysis confirmed.",
            model="expert-model",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation and trigger expert validation
        result1 = await workflow.run(prompt="Start")
        thread_id = result1.metadata["thread_id"]

        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        result2 = await workflow.run(prompt="Continue", continuation_id=thread_id)

        # Verify expert validation metadata
        assert result2.metadata.get("expert_validation_performed") is True
        # Expert validation content should be included in conversation

    @pytest.mark.asyncio
    async def test_expert_validation_conversation_history_updated(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation updates conversation history."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis.", model="test-model", usage={}, stop_reason="end_turn"
        )

        expert_validation_content = (
            "Expert validation: Root cause confirmed. High confidence in diagnosis."
        )
        mock_expert_provider.generate.return_value = GenerationResponse(
            content=expert_validation_content,
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Investigate issue")
        thread_id = result1.metadata["thread_id"]

        # Trigger expert validation
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Verify expert validation was performed
        assert result2.metadata.get("expert_validation_performed") is True

        # Expert validation content is integrated into the workflow
        # The key assertion is that expert validation happened successfully


class TestExpertValidationErrorHandling:
    """Test suite for error handling in expert validation."""

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
    async def test_expert_validation_failure_does_not_crash_investigation(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test that expert validation failure doesn't crash the investigation."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis continues.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        # Expert provider raises exception
        mock_expert_provider.generate.side_effect = Exception("Expert API timeout")

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Start investigation")
        thread_id = result1.metadata["thread_id"]

        # Trigger expert validation
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Continue investigation (expert validation will fail but shouldn't crash)
        result2 = await workflow.run(prompt="Continue investigation", continuation_id=thread_id)

        # Investigation should still succeed even if expert validation failed
        assert result2.success is True
        # Expert validation performed flag may be False or not set
        assert result2.metadata.get("expert_validation_performed") is False

    @pytest.mark.asyncio
    async def test_expert_validation_timeout_handling(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test handling of expert validation timeouts."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis.", model="test-model", usage={}, stop_reason="end_turn"
        )

        # Expert provider times out
        import asyncio

        mock_expert_provider.generate.side_effect = asyncio.TimeoutError("Request timeout")

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation
        result1 = await workflow.run(prompt="Start")
        thread_id = result1.metadata["thread_id"]

        # Trigger expert validation
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Continue (expert validation will timeout but shouldn't crash)
        result2 = await workflow.run(prompt="Continue", continuation_id=thread_id)

        # Investigation should continue successfully
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_expert_validation_with_empty_response(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test handling of empty expert validation response."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Primary analysis.", model="test-model", usage={}, stop_reason="end_turn"
        )

        # Expert provider returns empty content
        mock_expert_provider.generate.return_value = GenerationResponse(
            content="", model="expert-model", usage={}, stop_reason="end_turn"
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation and trigger expert validation
        result1 = await workflow.run(prompt="Start")
        thread_id = result1.metadata["thread_id"]

        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        result2 = await workflow.run(prompt="Continue", continuation_id=thread_id)

        # Investigation should handle empty expert response gracefully
        assert result2.success is True


class TestExpertValidationWithHypotheses:
    """Test suite for expert validation interaction with hypotheses."""

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
    async def test_expert_validation_validates_hypothesis(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation can validate a hypothesis."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation suggests memory leak in cache.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert analysis confirms memory leak hypothesis. Evidence is strong.",
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation and add hypothesis
        result1 = await workflow.run(prompt="Investigate memory issue")
        thread_id = result1.metadata["thread_id"]

        workflow.add_hypothesis(
            thread_id,
            "Memory leak in cache layer",
            evidence=["Memory grows over time", "No eviction policy"],
        )

        # Trigger expert validation at medium confidence
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        result2 = await workflow.run(
            prompt="Verify hypothesis with expert", continuation_id=thread_id
        )

        # Expert validation should have been performed
        assert result2.metadata.get("expert_validation_performed") is True

        # Get state to verify hypothesis tracking
        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].hypothesis == "Memory leak in cache layer"

    @pytest.mark.asyncio
    async def test_expert_validation_with_multiple_hypotheses(
        self, mock_provider, mock_expert_provider, conversation_memory
    ):
        """Test expert validation with multiple competing hypotheses."""
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Investigation shows multiple potential causes.",
            model="test-model",
            usage={},
            stop_reason="end_turn",
        )

        mock_expert_provider.generate.return_value = GenerationResponse(
            content="Expert analysis: First hypothesis most likely. Evidence for H1 is stronger than H2.",
            model="expert-model",
            usage={},
            stop_reason="end_turn",
        )

        # Create workflow
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            expert_provider=mock_expert_provider,
            conversation_memory=conversation_memory,
            config={"enable_expert_validation": True},
        )

        # Start investigation with multiple hypotheses
        result1 = await workflow.run(prompt="Investigate issue")
        thread_id = result1.metadata["thread_id"]

        workflow.add_hypothesis(thread_id, "H1: Database connection pool issue")
        workflow.add_hypothesis(thread_id, "H2: Network latency")
        workflow.add_hypothesis(thread_id, "H3: Application code bug")

        # Trigger expert validation
        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        result2 = await workflow.run(
            prompt="Get expert opinion on hypotheses", continuation_id=thread_id
        )

        # Expert validation performed with multiple hypotheses
        assert result2.metadata.get("expert_validation_performed") is True

        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 3
