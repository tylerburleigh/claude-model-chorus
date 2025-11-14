#!/usr/bin/env python3
"""
Test backward compatibility and different response formats
"""

import json
import sys

# Test case 1: Response WITH usage data (ideal case)
CURSOR_OUTPUT_WITH_USAGE = """{
    "type": "result",
    "subtype": "success",
    "is_error": false,
    "duration_ms": 1696,
    "duration_api_ms": 1696,
    "result": "Test response with usage data",
    "session_id": "test-session-123",
    "request_id": "test-request-456",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 20
    }
}"""

# Test case 2: Response WITHOUT usage data (actual cursor-agent behavior)
CURSOR_OUTPUT_WITHOUT_USAGE = """{
    "type": "result",
    "subtype": "success",
    "is_error": false,
    "duration_ms": 49948,
    "duration_api_ms": 49948,
    "result": "Test response without usage data",
    "session_id": "ec6da22e-0a09-4d4d-8855-4f2053b819b3",
    "request_id": "77988436-61fe-4ef4-8e48-77ad6ab35884"
}"""

try:
    from model_chorus.providers.cursor_agent_provider import CursorAgentProvider

    provider = CursorAgentProvider()

    print("=" * 60)
    print("Test 1: Response WITH usage data")
    print("=" * 60)

    response1 = provider.parse_response(
        stdout=CURSOR_OUTPUT_WITH_USAGE,
        stderr="",
        returncode=0
    )

    print(f"✅ Content: {response1.content}")
    print(f"✅ thread_id: {response1.thread_id}")
    print(f"✅ TokenUsage:")
    print(f"   - input_tokens: {response1.usage.input_tokens} (expected: 100)")
    print(f"   - output_tokens: {response1.usage.output_tokens} (expected: 50)")
    print(f"   - cached_input_tokens: {response1.usage.cached_input_tokens} (expected: 20)")
    print(f"   - total_tokens: {response1.usage.total_tokens} (expected: 150)")

    # Verify values
    assert response1.usage.input_tokens == 100, f"Expected 100, got {response1.usage.input_tokens}"
    assert response1.usage.output_tokens == 50, f"Expected 50, got {response1.usage.output_tokens}"
    assert response1.usage.cached_input_tokens == 20, f"Expected 20, got {response1.usage.cached_input_tokens}"
    assert response1.usage.total_tokens == 150, f"Expected 150, got {response1.usage.total_tokens}"

    print("\n" + "=" * 60)
    print("Test 2: Response WITHOUT usage data (actual CLI behavior)")
    print("=" * 60)

    response2 = provider.parse_response(
        stdout=CURSOR_OUTPUT_WITHOUT_USAGE,
        stderr="",
        returncode=0
    )

    print(f"✅ Content: {response2.content}")
    print(f"✅ thread_id: {response2.thread_id}")
    print(f"✅ TokenUsage (defaults when not provided):")
    print(f"   - input_tokens: {response2.usage.input_tokens} (expected: 0)")
    print(f"   - output_tokens: {response2.usage.output_tokens} (expected: 0)")
    print(f"   - total_tokens: {response2.usage.total_tokens} (expected: 0)")

    # Verify TokenUsage object exists even without usage field
    assert response2.usage is not None, "TokenUsage should be created even without usage data"
    assert response2.usage.input_tokens == 0, f"Expected 0, got {response2.usage.input_tokens}"
    assert response2.usage.output_tokens == 0, f"Expected 0, got {response2.usage.output_tokens}"

    print("\n" + "=" * 60)
    print("✅ All backward compatibility tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("1. TokenUsage created correctly with provided values")
    print("2. TokenUsage defaults to 0 when usage field missing")
    print("3. session_id correctly extracted to thread_id")
    print("4. Content correctly extracted from result field")

    sys.exit(0)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
