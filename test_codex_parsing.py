#!/usr/bin/env python3
"""
Test script to verify CodexProvider.parse_response() implementation
Uses mock JSONL data based on the documented format
"""

import json
import sys

# Mock Codex JSONL output (based on documented format in codex_provider.py)
CODEX_JSONL_OUTPUT = """{"type":"thread.started","thread_id":"019a7e99-1150-7c21-87b9-9ec1ee2e4da4"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Hello! How can I help you today?"}}
{"type":"turn.completed","usage":{"input_tokens":15,"output_tokens":8,"cached_input_tokens":5}}"""

# Test case 2: Without cached tokens
CODEX_JSONL_NO_CACHE = """{"type":"thread.started","thread_id":"test-thread-456"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"Response without cache"}}
{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":10}}"""

try:
    from model_chorus.providers.codex_provider import CodexProvider

    provider = CodexProvider()

    print("=" * 60)
    print("Test 1: Parse Codex JSONL with full usage data")
    print("=" * 60)

    response1 = provider.parse_response(
        stdout=CODEX_JSONL_OUTPUT,
        stderr="",
        returncode=0
    )

    print(f"\n✅ Parsing succeeded!")
    print(f"\nGenerationResponse fields:")
    print(f"  - content: {response1.content}")
    print(f"  - thread_id: {response1.thread_id}")
    print(f"  - model: {response1.model}")
    print(f"  - provider: {response1.provider}")

    print(f"\nTokenUsage fields:")
    print(f"  - input_tokens: {response1.usage.input_tokens} (expected: 15)")
    print(f"  - output_tokens: {response1.usage.output_tokens} (expected: 8)")
    print(f"  - cached_input_tokens: {response1.usage.cached_input_tokens} (expected: 5)")
    print(f"  - total_tokens: {response1.usage.total_tokens} (expected: 23)")

    print(f"\nraw_response:")
    print(f"  - Type: {type(response1.raw_response)}")
    print(f"  - Events count: {len(response1.raw_response.get('events', []))}")

    # Verification checks
    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)

    checks = []

    # Check 1: TokenUsage object created
    if response1.usage is not None:
        checks.append(("✅", "TokenUsage object created"))
    else:
        checks.append(("❌", "TokenUsage object is None"))

    # Check 2: thread_id extracted from thread.started event
    if response1.thread_id == "019a7e99-1150-7c21-87b9-9ec1ee2e4da4":
        checks.append(("✅", f"thread_id extracted: {response1.thread_id}"))
    else:
        checks.append(("❌", f"thread_id mismatch: got '{response1.thread_id}'"))

    # Check 3: Token counts correct
    if response1.usage.input_tokens == 15:
        checks.append(("✅", "input_tokens correct: 15"))
    else:
        checks.append(("❌", f"input_tokens wrong: {response1.usage.input_tokens}"))

    if response1.usage.output_tokens == 8:
        checks.append(("✅", "output_tokens correct: 8"))
    else:
        checks.append(("❌", f"output_tokens wrong: {response1.usage.output_tokens}"))

    if response1.usage.cached_input_tokens == 5:
        checks.append(("✅", "cached_input_tokens correct: 5"))
    else:
        checks.append(("❌", f"cached_input_tokens wrong: {response1.usage.cached_input_tokens}"))

    if response1.usage.total_tokens == 23:
        checks.append(("✅", "total_tokens correct: 23"))
    else:
        checks.append(("❌", f"total_tokens wrong: {response1.usage.total_tokens}"))

    # Check 4: Content extracted
    if response1.content == "Hello! How can I help you today?":
        checks.append(("✅", "Content extracted correctly"))
    else:
        checks.append(("❌", f"Content mismatch: {response1.content}"))

    # Check 5: Provider field
    if response1.provider == "codex":
        checks.append(("✅", "Provider field: codex"))
    else:
        checks.append(("❌", f"Provider wrong: {response1.provider}"))

    # Check 6: Raw response preserved
    if "events" in response1.raw_response and len(response1.raw_response["events"]) == 4:
        checks.append(("✅", "Raw response preserved (4 events)"))
    else:
        checks.append(("❌", "Raw response not properly preserved"))

    for status, message in checks:
        print(f"{status} {message}")

    # Test 2: Without cached tokens
    print("\n" + "=" * 60)
    print("Test 2: Parse without cached_input_tokens field")
    print("=" * 60)

    response2 = provider.parse_response(
        stdout=CODEX_JSONL_NO_CACHE,
        stderr="",
        returncode=0
    )

    print(f"\n✅ Parsing succeeded!")
    print(f"  - input_tokens: {response2.usage.input_tokens} (expected: 20)")
    print(f"  - output_tokens: {response2.usage.output_tokens} (expected: 10)")
    print(f"  - cached_input_tokens: {response2.usage.cached_input_tokens} (expected: 0)")
    print(f"  - total_tokens: {response2.usage.total_tokens} (expected: 30)")

    if response2.usage.cached_input_tokens == 0:
        checks.append(("✅", "Defaults to 0 when cached_input_tokens missing"))
    else:
        checks.append(("❌", f"Should default to 0, got {response2.usage.cached_input_tokens}"))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for status, _ in checks if status == "✅")
    total = len(checks)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("\n✅ All Codex parsing tests passed!")
        print("\nNote: Live CLI testing not performed due to authentication issues.")
        print("Verification based on mock JSONL data matching documented format.")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} checks failed")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
