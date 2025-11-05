"""
Tests for Consensus workflow.
"""

import pytest
from unittest.mock import AsyncMock

from modelchorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from modelchorus.providers.base_provider import GenerationResponse, GenerationRequest


class TestConsensusWorkflow:
    """Test suite for ConsensusWorkflow."""

    def test_initialization(self):
        """Test ConsensusWorkflow initialization."""
        mock_provider = AsyncMock()
        mock_provider.provider_name = "test"

        workflow = ConsensusWorkflow([mock_provider])
        assert workflow.strategy == ConsensusStrategy.ALL_RESPONSES
        assert len(workflow.provider_configs) == 1

    def test_initialization_multiple_providers(self):
        """Test initialization with multiple providers."""
        mock_claude = AsyncMock()
        mock_claude.provider_name = "claude"

        mock_codex = AsyncMock()
        mock_codex.provider_name = "codex"

        workflow = ConsensusWorkflow([mock_claude, mock_codex])
        assert len(workflow.provider_configs) == 2

    def test_initialization_with_strategy(self):
        """Test initialization with custom strategy."""
        mock_provider = AsyncMock()
        mock_provider.provider_name = "test"

        workflow = ConsensusWorkflow([mock_provider], strategy=ConsensusStrategy.FIRST_VALID)
        assert workflow.strategy == ConsensusStrategy.FIRST_VALID

    @pytest.mark.asyncio
    async def test_execute_all_responses_strategy(self):
        """Test consensus with all_responses strategy."""
        # Mock providers
        mock_claude = AsyncMock()
        mock_claude.provider_name = "claude"
        mock_claude.generate.return_value = GenerationResponse(
            content="Claude response",
            model="claude-sonnet",
            usage={},
        )

        mock_codex = AsyncMock()
        mock_codex.provider_name = "codex"
        mock_codex.generate.return_value = GenerationResponse(
            content="Codex response",
            model="gpt-5-codex",
            usage={},
        )

        workflow = ConsensusWorkflow([mock_claude, mock_codex], strategy=ConsensusStrategy.ALL_RESPONSES)

        request = GenerationRequest(prompt="Test prompt")
        result = await workflow.execute(request)

        assert result.consensus_response is not None
        assert "claude" in result.provider_results
        assert "codex" in result.provider_results
        assert len(result.provider_results) == 2

    @pytest.mark.asyncio
    async def test_execute_first_valid_strategy(self):
        """Test consensus with first_valid strategy."""
        # Mock providers - first fails, second succeeds
        mock_claude = AsyncMock()
        mock_claude.provider_name = "claude"
        mock_claude.generate.side_effect = Exception("API error")

        mock_codex = AsyncMock()
        mock_codex.provider_name = "codex"
        mock_codex.generate.return_value = GenerationResponse(
            content="Codex response",
            model="gpt-5-codex",
            usage={},
        )

        workflow = ConsensusWorkflow([mock_claude, mock_codex], strategy=ConsensusStrategy.FIRST_VALID)

        request = GenerationRequest(prompt="Test prompt")
        result = await workflow.execute(request)

        assert result.consensus_response == "Codex response"
        assert result.provider_results["codex"].content == "Codex response"
        assert "claude" in result.failed_providers

    @pytest.mark.asyncio
    async def test_execute_with_parameters(self):
        """Test consensus execution with custom parameters."""
        mock_provider = AsyncMock()
        mock_provider.provider_name = "claude"
        mock_provider.generate.return_value = GenerationResponse(
            content="Response",
            model="claude-sonnet",
            usage={},
        )

        workflow = ConsensusWorkflow([mock_provider])

        request = GenerationRequest(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=500,
            system_prompt="Custom system",
        )
        await workflow.execute(request)

        # Verify generate was called with correct request
        call_args = mock_provider.generate.call_args
        passed_request = call_args[0][0]

        assert passed_request.prompt == "Test prompt"
        assert passed_request.temperature == 0.5
        assert passed_request.max_tokens == 500
        assert passed_request.system_prompt == "Custom system"

    @pytest.mark.asyncio
    async def test_execute_all_providers_fail(self):
        """Test consensus when all providers fail."""
        mock_provider = AsyncMock()
        mock_provider.provider_name = "claude"
        mock_provider.generate.side_effect = Exception("API error")

        workflow = ConsensusWorkflow([mock_provider])

        request = GenerationRequest(prompt="Test prompt")
        result = await workflow.execute(request)

        assert result.consensus_response is None
        assert len(result.failed_providers) == 1
        assert "claude" in result.failed_providers
