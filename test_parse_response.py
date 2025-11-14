#!/usr/bin/env python3
"""
Test script to verify CursorAgentProvider.parse_response() implementation
"""

import json
import sys

# Sample cursor-agent JSON output (from actual CLI run)
CURSOR_OUTPUT = """{"type":"result","subtype":"success","is_error":false,"duration_ms":49948,"duration_api_ms":49948,"result":"\\nChecking the test setup to determine how to run tests.\\n\\nRunning the test suite with pytest:\\n\\nImport errors and cache conflicts detected. Cleaning pytest cache and rerunning tests:\\n\\nRunning pytest from the project root as configured:\\n\\nChecking a test file to understand the import structure:\\n\\nTests import `model_chorus`, but pytest's rootdir is `/Users/tylerburleigh` instead of the project root. Installing the package in development mode so imports resolve:\\n\\nRunning only the model_chorus tests that appear to work:\\n\\nTests are running. Summary:\\n\\n- 477 tests collected\\n- Many integration tests skipped (likely require API keys)\\n- Some tests passing (citation, config, conversation, etc.)\\n- Some tests failing (chat workflow, provider features)\\n\\nThe main issue is that tests in the `/tests/` directory have import errors because `model_chorus` isn't installed as a package. The tests in `model_chorus/tests/` run because pytest uses that directory as the root.\\n\\nTo fix the import issues, either:\\n1. Install the package in development mode: `pip install -e model_chorus/`\\n2. Set `PYTHONPATH` to include the project root\\n\\nShould I fix the import issues, investigate specific failures, or run a subset of tests?","session_id":"ec6da22e-0a09-4d4d-8855-4f2053b819b3","request_id":"77988436-61fe-4ef4-8e48-77ad6ab35884"}"""

try:
    # Import the provider
    from model_chorus.providers.cursor_agent_provider import CursorAgentProvider

    # Create provider instance
    provider = CursorAgentProvider()

    # Parse the response (matching the actual signature: stdout, stderr, returncode)
    print("=" * 60)
    print("Testing CursorAgentProvider.parse_response()")
    print("=" * 60)

    response = provider.parse_response(
        stdout=CURSOR_OUTPUT,
        stderr="",
        returncode=0
    )

    print("\n✅ Parsing succeeded!")
    print("\nGenerationResponse fields:")
    print(f"  - content: {response.content[:100]}..." if len(response.content) > 100 else f"  - content: {response.content}")
    print(f"  - thread_id: {response.thread_id}")
    print(f"  - model: {response.model}")

    print("\nTokenUsage fields:")
    if response.usage:
        print(f"  - input_tokens: {response.usage.input_tokens}")
        print(f"  - output_tokens: {response.usage.output_tokens}")
        print(f"  - total_tokens: {response.usage.total_tokens}")
    else:
        print("  ❌ TokenUsage is None!")

    # Verify critical requirements
    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)

    checks = []

    # Check 1: TokenUsage object created
    if response.usage is not None:
        checks.append(("✅", "TokenUsage object created"))
    else:
        checks.append(("❌", "TokenUsage object is None"))

    # Check 2: session_id extracted to thread_id
    expected_session_id = "ec6da22e-0a09-4d4d-8855-4f2053b819b3"
    if response.thread_id == expected_session_id:
        checks.append(("✅", f"session_id extracted to thread_id: {response.thread_id}"))
    else:
        checks.append(("❌", f"thread_id mismatch: expected '{expected_session_id}', got '{response.thread_id}'"))

    # Check 3: Text content extracted
    if response.content and len(response.content) > 0:
        checks.append(("✅", f"Response content extracted ({len(response.content)} chars)"))
    else:
        checks.append(("❌", "Response content is empty"))

    # Print all checks
    for status, message in checks:
        print(f"{status} {message}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for status, _ in checks if status == "✅")
    total = len(checks)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
