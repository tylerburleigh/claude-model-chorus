#!/usr/bin/env python3
"""
Test that code_blocks and language metadata are preserved in raw_response
"""

import json
import sys

# Test case: Response with code_blocks and language metadata
CURSOR_OUTPUT_WITH_METADATA = """{
    "type": "result",
    "subtype": "success",
    "is_error": false,
    "duration_ms": 1696,
    "duration_api_ms": 1696,
    "result": "Here's a Python example:\\n\\n```python\\ndef hello():\\n    print('Hello')\\n```",
    "session_id": "test-session-789",
    "request_id": "test-request-abc",
    "usage": {
        "input_tokens": 50,
        "output_tokens": 25
    },
    "code_blocks": [
        {
            "language": "python",
            "code": "def hello():\\n    print('Hello')"
        }
    ],
    "language": "python",
    "extra_metadata": {
        "model_version": "v1.2.3",
        "provider_specific": "data"
    }
}"""

try:
    from model_chorus.providers.cursor_agent_provider import CursorAgentProvider

    provider = CursorAgentProvider()

    print("=" * 60)
    print("Test: Preserve code_blocks and language metadata")
    print("=" * 60)

    response = provider.parse_response(
        stdout=CURSOR_OUTPUT_WITH_METADATA,
        stderr="",
        returncode=0
    )

    print(f"\n✅ Basic fields:")
    print(f"   - content: {response.content[:50]}...")
    print(f"   - thread_id: {response.thread_id}")

    print(f"\n✅ raw_response preserved:")
    print(f"   - Type: {type(response.raw_response)}")
    print(f"   - Keys: {list(response.raw_response.keys())}")

    # Verify specific metadata fields are preserved
    checks = []

    if "code_blocks" in response.raw_response:
        code_blocks = response.raw_response["code_blocks"]
        checks.append(("✅", f"code_blocks preserved: {len(code_blocks)} block(s)"))
        if code_blocks:
            checks.append(("✅", f"  - language: {code_blocks[0]['language']}"))
            checks.append(("✅", f"  - code: {len(code_blocks[0]['code'])} chars"))
    else:
        checks.append(("❌", "code_blocks field not found in raw_response"))

    if "language" in response.raw_response:
        checks.append(("✅", f"language field preserved: {response.raw_response['language']}"))
    else:
        checks.append(("❌", "language field not found in raw_response"))

    if "extra_metadata" in response.raw_response:
        checks.append(("✅", f"extra_metadata preserved: {response.raw_response['extra_metadata']}"))
    else:
        checks.append(("⚠️ ", "extra_metadata not in response (optional)"))

    # Check that standard fields are also there
    if "session_id" in response.raw_response:
        checks.append(("✅", f"session_id in raw_response: {response.raw_response['session_id']}"))

    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)

    for status, message in checks:
        print(f"{status} {message}")

    # Summary
    passed = sum(1 for status, _ in checks if status == "✅")
    total = len([c for c in checks if c[0] != "⚠️ "])

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} required checks passed")
    print("=" * 60)

    if passed == total:
        print("\n✅ All metadata fields preserved in raw_response!")
        sys.exit(0)
    else:
        print("\n❌ Some metadata fields missing")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
