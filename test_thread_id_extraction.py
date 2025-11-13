#!/usr/bin/env python3
"""
Comprehensive test for thread_id extraction across all providers
Verifies that each provider correctly extracts and maps thread_id
"""

import sys

def test_cursor_agent_thread_id():
    """Test Cursor Agent: session_id → thread_id"""
    from model_chorus.providers.cursor_agent_provider import CursorAgentProvider

    sample_json = '{"type":"result","subtype":"success","is_error":false,"result":"Test","session_id":"cursor-session-123"}'

    provider = CursorAgentProvider()
    response = provider.parse_response(stdout=sample_json, stderr="", returncode=0)

    expected = "cursor-session-123"
    if response.thread_id == expected:
        return ("✅", f"Cursor Agent: session_id → thread_id ({expected})")
    else:
        return ("❌", f"Cursor Agent: Expected '{expected}', got '{response.thread_id}'")


def test_codex_thread_id():
    """Test Codex: thread_id → thread_id"""
    from model_chorus.providers.codex_provider import CodexProvider

    sample_jsonl = """{"type":"thread.started","thread_id":"codex-thread-456"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Test"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}"""

    provider = CodexProvider()
    response = provider.parse_response(stdout=sample_jsonl, stderr="", returncode=0)

    expected = "codex-thread-456"
    if response.thread_id == expected:
        return ("✅", f"Codex: thread_id → thread_id ({expected})")
    else:
        return ("❌", f"Codex: Expected '{expected}', got '{response.thread_id}'")


try:
    print("=" * 60)
    print("Thread ID Extraction Verification - All Providers")
    print("=" * 60)

    checks = []

    # Test each provider
    print("\n🔍 Testing Cursor Agent Provider...")
    checks.append(test_cursor_agent_thread_id())

    print("\n🔍 Testing Codex Provider...")
    checks.append(test_codex_thread_id())

    # Summary
    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)

    for status, message in checks:
        print(f"{status} {message}")

    passed = sum(1 for status, _ in checks if status == "✅")
    total = len(checks)

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} providers verified")
    print("=" * 60)

    if passed == total:
        print("\n✅ All provider thread_id extractions verified!")
        print("\nMapping Summary:")
        print("  - Cursor Agent: session_id → thread_id")
        print("  - Codex: thread_id → thread_id")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} providers failed verification")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
