"""
Pytest configuration and fixtures for ModelChorus tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_claude_response():
    """Mock response from Claude CLI --output-format json."""
    return {
        "type": "result",
        "subtype": "success",
        "result": "This is a test response from Claude.",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "modelUsage": {
            "claude-sonnet-4-5-20250929": {
                "inputTokens": 10,
                "outputTokens": 50,
                "costUSD": 0.001,
            }
        },
        "duration_ms": 1500,
        "total_cost_usd": 0.001,
    }


@pytest.fixture
def mock_codex_response():
    """Mock JSONL response from Codex CLI --json."""
    return """{"type":"thread.started","thread_id":"test-thread-123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"reasoning","text":"Thinking about the request"}}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"This is a test response from Codex."}}
{"type":"turn.completed","usage":{"input_tokens":15,"cached_input_tokens":0,"output_tokens":45}}"""


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for CLI command execution."""
    mock = MagicMock()
    mock.return_value.returncode = 0
    mock.return_value.stdout = ""
    mock.return_value.stderr = ""
    return mock


@pytest.fixture
def sample_generation_request():
    """Sample GenerationRequest for testing."""
    from modelchorus.providers.base_provider import GenerationRequest

    return GenerationRequest(
        prompt="What is 2+2?",
        temperature=0.7,
        max_tokens=100,
        system_prompt="You are a helpful assistant.",
        metadata={},
    )
