#!/usr/bin/env python3
"""
Test backward compatibility - verify existing code patterns still work
Tests both attribute access (new) and dict-like access (backward compat)
"""

import sys

# Simulated old workflow code that uses dict-like access
def old_workflow_pattern(response):
    """Simulates existing workflow code using dict access."""
    # Old pattern: access usage as dict
    input_tokens = response.usage['input_tokens']
    output_tokens = response.usage['output_tokens']
    total = response.usage['total_tokens']

    return {
        'input': input_tokens,
        'output': output_tokens,
        'total': total
    }

# New workflow code using attribute access
def new_workflow_pattern(response):
    """New pattern: access usage as attributes."""
    return {
        'input': response.usage.input_tokens,
        'output': response.usage.output_tokens,
        'total': response.usage.total_tokens
    }

try:
    from model_chorus.providers.cursor_agent_provider import CursorAgentProvider

    # Sample response
    SAMPLE_JSON = '{"type":"result","subtype":"success","is_error":false,"result":"Test","session_id":"test-123","usage":{"input_tokens":100,"output_tokens":50,"cached_input_tokens":10}}'

    provider = CursorAgentProvider()
    response = provider.parse_response(
        stdout=SAMPLE_JSON,
        stderr="",
        returncode=0
    )

    print("=" * 60)
    print("Backward Compatibility Test")
    print("=" * 60)

    # Test 1: Old pattern (dict-like access)
    print("\n✅ Test 1: Old workflow pattern (dict access)")
    old_result = old_workflow_pattern(response)
    print(f"  usage['input_tokens'] = {old_result['input']}")
    print(f"  usage['output_tokens'] = {old_result['output']}")
    print(f"  usage['total_tokens'] = {old_result['total']}")

    # Test 2: New pattern (attribute access)
    print("\n✅ Test 2: New workflow pattern (attribute access)")
    new_result = new_workflow_pattern(response)
    print(f"  usage.input_tokens = {new_result['input']}")
    print(f"  usage.output_tokens = {new_result['output']}")
    print(f"  usage.total_tokens = {new_result['total']}")

    # Test 3: Verify both give same results
    print("\n✅ Test 3: Both patterns return same values")
    if old_result == new_result:
        print("  ✅ Dict and attribute access are equivalent")
    else:
        print(f"  ❌ Mismatch: {old_result} != {new_result}")
        sys.exit(1)

    # Test 4: Verify specific values
    checks = []

    if old_result['input'] == 100:
        checks.append(("✅", "input_tokens = 100"))
    else:
        checks.append(("❌", f"input_tokens wrong: {old_result['input']}"))

    if old_result['output'] == 50:
        checks.append(("✅", "output_tokens = 50"))
    else:
        checks.append(("❌", f"output_tokens wrong: {old_result['output']}"))

    if old_result['total'] == 150:
        checks.append(("✅", "total_tokens = 150"))
    else:
        checks.append(("❌", f"total_tokens wrong: {old_result['total']}"))

    # Test 5: Test KeyError for invalid key
    print("\n✅ Test 4: Invalid key raises KeyError")
    try:
        _ = response.usage['invalid_key']
        checks.append(("❌", "Should raise KeyError for invalid key"))
    except KeyError:
        checks.append(("✅", "KeyError raised for invalid key (expected)"))

    # Test 6: Verify cached_input_tokens accessible both ways
    print("\n✅ Test 5: cached_input_tokens accessible")
    if response.usage['cached_input_tokens'] == response.usage.cached_input_tokens == 10:
        checks.append(("✅", "cached_input_tokens accessible both ways (10)"))
    else:
        checks.append(("❌", "cached_input_tokens access mismatch"))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)

    for status, message in checks:
        print(f"{status} {message}")

    passed = sum(1 for status, _ in checks if status == "✅")
    total = len(checks)

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("\n✅ Backward compatibility verified!")
        print("Both old dict-like access and new attribute access work correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} checks failed")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
