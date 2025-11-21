import json

from model_chorus.providers.base_provider import TokenUsage
from model_chorus.providers.claude_provider import ClaudeProvider
from model_chorus.providers.gemini_provider import GeminiProvider


class TestStandardization:
    def test_claude_standardization(self):
        provider = ClaudeProvider()
        mock_response = {
            "result": "Test Content",
            "usage": {"input_tokens": 10, "output_tokens": 20, "cached_tokens": 5},
            "modelUsage": {"claude-3-opus": {}},
            "subtype": "success",
            "duration_ms": 123,
            "session_id": "sess_abc",
            "total_cost_usd": 0.002,
        }
        stdout = json.dumps(mock_response)
        stderr = "Some warning"

        response = provider.parse_response(stdout, stderr, 0)

        assert response.provider == "claude"
        assert response.thread_id == "sess_abc"
        assert response.stderr == "Some warning"
        assert response.duration_ms == 123
        assert isinstance(response.usage, TokenUsage)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.cached_input_tokens == 5
        assert response.usage.total_tokens == 30
        assert response.raw_response == mock_response

    def test_gemini_standardization(self):
        provider = GeminiProvider()
        mock_response = {
            "response": "Test Content",
            "stats": {
                "models": {
                    "gemini-2.5-pro": {
                        "tokens": {"prompt": 15, "candidates": 25, "total": 40}
                    }
                }
            },
        }
        stdout = json.dumps(mock_response)
        stderr = "Some gemini warning"

        response = provider.parse_response(stdout, stderr, 0)

        assert response.provider == "gemini"
        assert response.thread_id is None
        assert response.stderr == "Some gemini warning"
        assert response.duration_ms is None
        assert isinstance(response.usage, TokenUsage)
        assert response.usage.input_tokens == 15
        assert response.usage.output_tokens == 25
        assert response.usage.total_tokens == 40
        assert response.raw_response == mock_response
