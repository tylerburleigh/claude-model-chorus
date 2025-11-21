"""
Integration tests for ModelChorus.

These tests verify that components work together correctly.
Note: These tests use mocks to avoid calling actual CLI tools.
"""

from unittest.mock import AsyncMock, patch

import pytest

from model_chorus.providers.base_provider import GenerationRequest
from model_chorus.providers.claude_provider import ClaudeProvider
from model_chorus.providers.codex_provider import CodexProvider
from model_chorus.workflows.consensus import ConsensusStrategy, ConsensusWorkflow


class TestIntegration:
    """Integration test suite."""

    @pytest.mark.asyncio
    async def test_end_to_end_consensus(
        self, mock_claude_response, mock_codex_response
    ):
        """Test end-to-end consensus workflow with multiple providers."""
        # Create providers
        claude = ClaudeProvider()
        codex = CodexProvider()

        workflow = ConsensusWorkflow(
            [claude, codex], strategy=ConsensusStrategy.ALL_RESPONSES
        )

        # Mock CLI command execution
        with patch(
            "model_chorus.providers.cli_provider.asyncio.create_subprocess_exec"
        ) as mock_exec:
            import json

            async def mock_process(*args, **kwargs):
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                if args[0] == "claude":
                    mock_proc.communicate.return_value = (
                        json.dumps(mock_claude_response).encode(),
                        b"",
                    )
                elif args[0] == "codex":
                    mock_proc.communicate.return_value = (
                        mock_codex_response.encode(),
                        b"",
                    )
                return mock_proc

            mock_exec.side_effect = mock_process

            # Execute workflow
            request = GenerationRequest(prompt="What is 2+2?")
            result = await workflow.execute(request)

            # Verify results
            assert len(result.provider_results) == 2
            assert "claude" in result.provider_results
            assert "codex" in result.provider_results
            assert result.consensus_response is not None

    @pytest.mark.asyncio
    async def test_provider_initialization_and_generation(self, mock_claude_response):
        """Test provider initialization and text generation."""
        provider = ClaudeProvider()

        # Mock subprocess execution
        with patch(
            "model_chorus.providers.cli_provider.asyncio.create_subprocess_exec"
        ) as mock_exec:
            import json

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (
                json.dumps(mock_claude_response).encode(),
                b"",
            )
            mock_exec.return_value = mock_proc

            request = GenerationRequest(prompt="Test prompt")
            response = await provider.generate(request)

            assert response.content == "This is a test response from Claude."
            assert response.model == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_error_handling_across_workflow(self):
        """Test error handling propagation through workflow."""
        claude = ClaudeProvider()
        workflow = ConsensusWorkflow([claude])

        # Mock provider that fails
        with patch(
            "model_chorus.providers.cli_provider.asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"Error: API key missing")
            mock_exec.return_value = mock_proc

            request = GenerationRequest(prompt="Test prompt")

            # Should raise RuntimeError when all providers fail
            with pytest.raises(RuntimeError, match="only 0/1 providers succeeded"):
                await workflow.execute(request)

    @pytest.mark.asyncio
    async def test_multiple_strategy_comparison(
        self, mock_claude_response, mock_codex_response
    ):
        """Test different consensus strategies with same data."""
        import json

        # Mock CLI execution
        with patch(
            "model_chorus.providers.cli_provider.asyncio.create_subprocess_exec"
        ) as mock_exec:

            async def mock_process(*args, **kwargs):
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                if args[0] == "claude":
                    mock_proc.communicate.return_value = (
                        json.dumps(mock_claude_response).encode(),
                        b"",
                    )
                else:  # codex
                    mock_proc.communicate.return_value = (
                        mock_codex_response.encode(),
                        b"",
                    )
                return mock_proc

            mock_exec.side_effect = mock_process

            # Test ALL_RESPONSES strategy
            claude_all = ClaudeProvider()
            codex_all = CodexProvider()
            workflow_all = ConsensusWorkflow(
                [claude_all, codex_all], strategy=ConsensusStrategy.ALL_RESPONSES
            )
            request = GenerationRequest(prompt="Test")
            result_all = await workflow_all.execute(request)

            assert len(result_all.provider_results) == 2

            # Test FIRST_VALID strategy
            claude_first = ClaudeProvider()
            codex_first = CodexProvider()
            workflow_first = ConsensusWorkflow(
                [claude_first, codex_first], strategy=ConsensusStrategy.FIRST_VALID
            )
            result_first = await workflow_first.execute(request)

            # First valid should return after first success
            assert result_first.consensus_response is not None

    @pytest.mark.asyncio
    async def test_concurrent_provider_execution(
        self, mock_claude_response, mock_codex_response
    ):
        """Test that providers execute concurrently, not sequentially."""
        import json
        import time

        claude = ClaudeProvider()
        codex = CodexProvider()
        workflow = ConsensusWorkflow([claude, codex])

        with patch(
            "model_chorus.providers.cli_provider.asyncio.create_subprocess_exec"
        ) as mock_exec:
            call_times = []

            async def mock_process(*args, **kwargs):
                call_times.append(time.time())
                import asyncio

                await asyncio.sleep(0.1)  # Simulate some delay

                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                if args[0] == "claude":
                    mock_proc.communicate.return_value = (
                        json.dumps(mock_claude_response).encode(),
                        b"",
                    )
                else:
                    mock_proc.communicate.return_value = (
                        mock_codex_response.encode(),
                        b"",
                    )
                return mock_proc

            mock_exec.side_effect = mock_process

            start_time = time.time()
            request = GenerationRequest(prompt="Test")
            await workflow.execute(request)
            total_time = time.time() - start_time

            # If concurrent, should take ~0.1s, not ~0.2s
            assert total_time < 0.3  # Allow some overhead
            # Both providers should start around the same time
            if len(call_times) >= 2:
                assert abs(call_times[0] - call_times[1]) < 0.1
