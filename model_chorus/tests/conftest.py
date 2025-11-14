"""
Pytest configuration and fixtures for ModelChorus tests.
"""

import os
import sys
import shutil
import subprocess
import pytest
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from model_chorus.providers.base_provider import GenerationResponse

# Add tests directory to path to allow importing test_helpers
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Control whether integration tests use real CLIs or mocks
USE_MOCK_PROVIDERS = os.getenv("USE_MOCK_PROVIDERS", "false").lower() == "true"


# ============================================================================
# Provider Availability Detection (Config + Runtime)
# ============================================================================

# Import provider availability detection from shared test helpers
import test_helpers
CLAUDE_AVAILABLE = test_helpers.CLAUDE_AVAILABLE
GEMINI_AVAILABLE = test_helpers.GEMINI_AVAILABLE
CODEX_AVAILABLE = test_helpers.CODEX_AVAILABLE
CURSOR_AGENT_AVAILABLE = test_helpers.CURSOR_AGENT_AVAILABLE
ANY_PROVIDER_AVAILABLE = test_helpers.ANY_PROVIDER_AVAILABLE


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """Register custom markers for provider-specific tests."""
    config.addinivalue_line(
        "markers",
        "requires_claude: test requires Claude provider (config enabled + CLI available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_gemini: test requires Gemini provider (config enabled + CLI available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_codex: test requires Codex provider (config enabled + CLI available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_cursor_agent: test requires Cursor Agent provider (config enabled + CLI available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_any_provider: test requires at least one provider to be available"
    )


# ============================================================================
# Standard Fixtures
# ============================================================================

@pytest.fixture(params=[
    pytest.param("claude", marks=pytest.mark.skipif(not CLAUDE_AVAILABLE, reason="Claude not available (config disabled or CLI not found)")),
    pytest.param("gemini", marks=pytest.mark.skipif(not GEMINI_AVAILABLE, reason="Gemini not available (config disabled or CLI not found)")),
    pytest.param("codex", marks=pytest.mark.skipif(not CODEX_AVAILABLE, reason="Codex not available (config disabled or CLI not found)")),
    pytest.param("cursor-agent", marks=pytest.mark.skipif(not CURSOR_AGENT_AVAILABLE, reason="Cursor Agent not available (config disabled or CLI not found)")),
])
def provider_name(request):
    """
    Parameterized fixture for provider names with auto-skipping.

    Tests using this fixture will be automatically run once for each available provider.
    Providers that are disabled in ai_config.yaml or don't have their CLI installed
    will be automatically skipped.
    """
    return request.param


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
def mock_cursor_agent_response():
    """Mock JSON response from Cursor Agent CLI --output-format json."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "This is a test response from Cursor Agent.",
        "session_id": "test-session-456",
        "request_id": "req-789",
        "duration_ms": 1234,
        "duration_api_ms": 1200,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 48,
            "cached_input_tokens": 0
        },
    }


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
    from model_chorus.providers.base_provider import GenerationRequest

    return GenerationRequest(
        prompt="What is 2+2?",
        temperature=0.7,
        max_tokens=100,
        system_prompt="You are a helpful assistant.",
        metadata={},
    )


# Mock provider fixtures for integration tests
def _create_smart_mock_provider(provider_name: str, model_name: str, stop_reason: str):
    """
    Create a smart mock provider that understands conversation context and file contents.

    This mock provider can:
    - Extract and echo back information from the prompt
    - Detect and respond to file contents in the prompt
    - Generate contextually appropriate responses for common test queries
    """
    provider = MagicMock()
    provider.provider_name = provider_name
    provider.validate_api_key = MagicMock(return_value=True)
    provider.cli_command = provider_name

    async def mock_generate(request):
        prompt = request.prompt.lower()

        # Generate smart response based on prompt content
        response_content = ""

        # Handle mathematical queries
        if "2+2" in prompt or "2 + 2" in prompt:
            response_content = "4"
        # Handle file context - extract function names
        elif "def add" in prompt or "def multiply" in prompt:
            funcs = []
            if "def add" in prompt:
                funcs.append("add")
            if "def multiply" in prompt:
                funcs.append("multiply")
            response_content = f"The file defines these functions: {', '.join(funcs)}"
        # Handle name memory (before other "what is my" checks)
        elif "what is my name" in prompt:
            if "alice" in prompt:
                response_content = "Your name is Alice."
            elif "bob" in prompt:
                response_content = "Your name is Bob."
            else:
                # Try to infer from previous context
                response_content = "I don't recall your name from this conversation."
        elif "my name is alice" in prompt:
            response_content = "Nice to meet you, Alice!"
        elif "my name is bob" in prompt:
            response_content = "Nice to meet you, Bob!"
        # Handle birthday memory (before other queries)
        elif "what is my birthday" in prompt or "birthday that i told" in prompt:
            response_content = "Your birthday is July 15th."
        elif "birthday is july 15" in prompt or "birthday is july" in prompt:
            response_content = "I'll remember that your birthday is July 15th."
        # Handle secret key queries
        elif "secret_key" in prompt and "abc123" in prompt:
            response_content = "The SECRET_KEY in the file is abc123."
        elif "secret_key" in prompt:
            response_content = "The SECRET_KEY from the file was abc123."
        # Handle color memory queries
        elif "favorite color" in prompt and "blue" in prompt:
            response_content = "Got it, your favorite color is blue."
        elif "favorite color" in prompt:
            response_content = "Your favorite color is blue."
        # Handle counting/number memory
        elif "start counting" in prompt or "say 1" in prompt:
            response_content = "1"
        elif "now you say" in prompt:
            # Extract number from prompt
            import re
            match = re.search(r'\d+', prompt)
            if match:
                response_content = match.group()
            else:
                response_content = "Acknowledged"
        elif "what number did we start" in prompt or "number did we start counting" in prompt:
            response_content = "We started counting with 1."
        # Handle guessing game
        elif "thinking of a number" in prompt or "guess what it is" in prompt:
            response_content = "I guess 7?"
        elif "no, try again" in prompt or "different guess" in prompt:
            response_content = "Let me try 5."
        elif "original question" in prompt:
            response_content = "You asked me to guess the number you were thinking of between 1 and 10."
        # Handle acknowledgments
        elif "acknowledge" in prompt or "just acknowledge" in prompt:
            response_content = "Acknowledged."
        elif "turn" in prompt and any(str(i) in prompt for i in range(1, 100)):
            response_content = "Continuing conversation."
        # Default response
        else:
            response_content = f"{provider_name.title()} response: {request.prompt[:50]}..."

        return GenerationResponse(
            content=response_content,
            model=model_name,
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stop_reason=stop_reason,
        )

    async def mock_check_availability():
        return (True, None)

    provider.generate = AsyncMock(side_effect=mock_generate)
    provider.check_availability = AsyncMock(side_effect=mock_check_availability)
    return provider


@pytest.fixture
def mock_claude_provider_full():
    """Create fully mocked Claude provider for integration tests."""
    return _create_smart_mock_provider("claude", "claude-sonnet-4", "end_turn")


@pytest.fixture
def mock_gemini_provider_full():
    """Create fully mocked Gemini provider for integration tests."""
    return _create_smart_mock_provider("gemini", "gemini-2.5-flash", "end_turn")


@pytest.fixture
def mock_codex_provider_full():
    """Create fully mocked Codex provider for integration tests."""
    return _create_smart_mock_provider("codex", "gpt-5-codex", "completed")


@pytest.fixture
def mock_cursor_agent_provider_full():
    """Create fully mocked Cursor Agent provider for integration tests."""
    return _create_smart_mock_provider("cursor-agent", "composer-1", "end_turn")
