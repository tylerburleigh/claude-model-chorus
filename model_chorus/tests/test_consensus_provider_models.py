"""
Tests for provider-level metadata handling within the consensus workflow.
"""

from unittest.mock import AsyncMock

import pytest

from model_chorus.providers.base_provider import GenerationRequest, GenerationResponse
from model_chorus.workflows.consensus import ConsensusWorkflow


def _make_provider(name: str, response_text: str) -> AsyncMock:
    provider = AsyncMock()
    provider.provider_name = name

    async def _generate(request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            content=response_text,
            model=f"{name}-model",
            metadata=request.metadata.copy(),
        )

    provider.generate.side_effect = _generate
    return provider


@pytest.mark.asyncio
async def test_consensus_applies_provider_model_override_without_mutation():
    """Ensure provider-specific metadata adds model override and preserves shared metadata."""
    gemini_provider = _make_provider("gemini", "Gemini response")
    claude_provider = _make_provider("claude", "Claude response")

    workflow = ConsensusWorkflow([gemini_provider, claude_provider])
    for config in workflow.provider_configs:
        if config.provider.provider_name == "gemini":
            config.metadata["model"] = "gemini-2.5-pro"

    shared_request = GenerationRequest(
        prompt="Test prompt",
        metadata={"trace_id": "abc-123"},
    )

    await workflow.execute(shared_request)

    gemini_request = gemini_provider.generate.call_args[0][0]
    claude_request = claude_provider.generate.call_args[0][0]

    assert gemini_request is not shared_request
    assert claude_request is not shared_request
    assert gemini_request.metadata is not shared_request.metadata
    assert claude_request.metadata is not shared_request.metadata

    assert gemini_request.metadata["model"] == "gemini-2.5-pro"
    assert gemini_request.metadata["trace_id"] == "abc-123"
    assert "model" not in claude_request.metadata
    assert claude_request.metadata["trace_id"] == "abc-123"

    # Original request remains untouched
    assert shared_request.metadata == {"trace_id": "abc-123"}


@pytest.mark.asyncio
async def test_shared_request_model_overrides_provider_metadata():
    """Shared request metadata should take precedence over provider-level overrides."""
    gemini_provider = _make_provider("gemini", "Gemini response")

    workflow = ConsensusWorkflow([gemini_provider])
    workflow.provider_configs[0].metadata["model"] = "gemini-2.5-pro"

    shared_request = GenerationRequest(
        prompt="Test prompt",
        metadata={"model": "shared-preferred-model", "trace_id": "trace-xyz"},
    )

    await workflow.execute(shared_request)

    gemini_request = gemini_provider.generate.call_args[0][0]

    assert gemini_request.metadata["model"] == "shared-preferred-model"
    assert gemini_request.metadata["trace_id"] == "trace-xyz"
    assert workflow.provider_configs[0].metadata["model"] == "gemini-2.5-pro"
