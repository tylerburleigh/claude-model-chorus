"""Quick test for RetryMiddleware implementation."""

import asyncio
from model_chorus.providers.middleware import (
    RetryMiddleware,
    RetryConfig,
    Middleware,
)
from model_chorus.providers.base_provider import (
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
)


class MockProvider(ModelProvider):
    """Mock provider for testing."""

    def __init__(self, fail_count: int = 0):
        super().__init__("mock", None, {})
        self.fail_count = fail_count
        self.attempts = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate with configurable failures."""
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise Exception("Transient error")
        return GenerationResponse(content="Success!", model="mock")

    def supports_vision(self, model_id: str) -> bool:
        return False


async def test_retry_success_after_failures():
    """Test that retry middleware succeeds after transient failures."""
    print("Test 1: Success after 2 failures...")
    provider = MockProvider(fail_count=2)
    retry_middleware = RetryMiddleware(provider, RetryConfig(max_retries=3))

    request = GenerationRequest(prompt="Test prompt")
    response = await retry_middleware.generate(request)

    assert response.content == "Success!", f"Expected 'Success!', got {response.content}"
    assert provider.attempts == 3, f"Expected 3 attempts, got {provider.attempts}"
    print("✓ Test 1 passed!")


async def test_no_failures():
    """Test that middleware works when no retries needed."""
    print("\nTest 2: No failures (immediate success)...")
    provider = MockProvider(fail_count=0)
    retry_middleware = RetryMiddleware(provider)

    request = GenerationRequest(prompt="Test prompt")
    response = await retry_middleware.generate(request)

    assert response.content == "Success!", f"Expected 'Success!', got {response.content}"
    assert provider.attempts == 1, f"Expected 1 attempt, got {provider.attempts}"
    print("✓ Test 2 passed!")


async def test_permanent_error():
    """Test that permanent errors fail immediately."""
    print("\nTest 3: Permanent error (no retry)...")

    class PermanentErrorProvider(ModelProvider):
        def __init__(self):
            super().__init__("test", None, {})
            self.attempts = 0

        async def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.attempts += 1
            raise Exception("401 Unauthorized")

        def supports_vision(self, model_id: str) -> bool:
            return False

    provider = PermanentErrorProvider()
    retry_middleware = RetryMiddleware(provider)

    request = GenerationRequest(prompt="Test prompt")

    try:
        await retry_middleware.generate(request)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "401" in str(e), f"Expected '401' in error, got {e}"
        assert provider.attempts == 1, f"Expected 1 attempt (no retry), got {provider.attempts}"
    print("✓ Test 3 passed!")


async def test_all_retries_exhausted():
    """Test behavior when all retries are exhausted."""
    print("\nTest 4: All retries exhausted...")
    provider = MockProvider(fail_count=10)  # Will always fail
    retry_middleware = RetryMiddleware(provider, RetryConfig(max_retries=2))

    request = GenerationRequest(prompt="Test prompt")

    try:
        await retry_middleware.generate(request)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "All 3 attempts failed" in str(e), f"Expected 'All 3 attempts failed', got {e}"
        assert provider.attempts == 3, f"Expected 3 attempts, got {provider.attempts}"
    print("✓ Test 4 passed!")


async def main():
    """Run all tests."""
    print("Running RetryMiddleware tests...\n")
    await test_retry_success_after_failures()
    await test_no_failures()
    await test_permanent_error()
    await test_all_retries_exhausted()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
