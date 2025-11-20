"""
Concurrency and performance tests for ConversationMemory.

Tests verify that ConversationMemory can handle:
- 100+ concurrent conversation threads
- Thread-safe operations under parallel access
- Memory management under high load
- Thread isolation (no cross-contamination between conversations)
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock

import pytest

from model_chorus.core.conversation import ConversationMemory
from model_chorus.providers.base_provider import GenerationResponse
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.workflows.thinkdeep import ThinkDeepWorkflow


class TestConcurrentConversationHandling:
    """
    Test suite for concurrent conversation handling.

    Validates that ConversationMemory and workflows can handle high concurrency
    scenarios with proper thread isolation and performance.
    """

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create shared conversation memory for concurrent testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_100_concurrent_chat_conversations(
        self, mock_provider, conversation_memory
    ):
        """
        Test handling 100 concurrent chat conversations.

        Verifies:
        - All conversations execute successfully
        - Thread isolation (no message cross-contamination)
        - Unique thread IDs for each conversation
        - Performance remains acceptable under load
        """
        num_conversations = 100

        # Setup mock to return unique responses
        def generate_response(*args, **kwargs):
            return GenerationResponse(
                content=f"Response {uuid.uuid4()}",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn",
            )

        mock_provider.generate.side_effect = generate_response

        # Create chat workflow (shared across all conversations)
        chat_workflow = ChatWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Define concurrent conversation tasks
        async def run_conversation(conversation_id: int):
            """Run a single conversation."""
            result = await chat_workflow.run(
                prompt=f"Conversation {conversation_id}: What is AI?"
            )
            return {
                "conversation_id": conversation_id,
                "thread_id": result.metadata.get("thread_id"),
                "success": result.success,
                "content": result.synthesis,
            }

        # Execute all conversations concurrently
        start_time = time.time()
        tasks = [run_conversation(i) for i in range(num_conversations)]
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Verify all conversations succeeded
        assert len(results) == num_conversations
        assert all(r["success"] for r in results), "All conversations should succeed"

        # Verify unique thread IDs
        thread_ids = [r["thread_id"] for r in results]
        assert (
            len(set(thread_ids)) == num_conversations
        ), "All thread IDs should be unique"

        # Verify thread isolation - each thread should have exactly 2 messages
        for result in results:
            thread = conversation_memory.get_thread(result["thread_id"])
            assert thread is not None, f"Thread {result['thread_id']} should exist"
            assert (
                len(thread.messages) == 2
            ), "Each thread should have user + assistant message"

            # Verify message content contains conversation ID
            user_message = thread.messages[0]
            assert user_message.role == "user"
            assert f"Conversation {result['conversation_id']}" in user_message.content

        # Verify no cross-contamination
        all_messages = []
        for result in results:
            thread = conversation_memory.get_thread(result["thread_id"])
            all_messages.extend(thread.messages)

        # Count unique conversation IDs in messages
        conversation_ids_in_messages = set()
        for msg in all_messages:
            if msg.role == "user" and "Conversation" in msg.content:
                # Extract conversation ID from message
                parts = msg.content.split()
                if len(parts) > 1:
                    try:
                        conv_id = int(parts[1].rstrip(":"))
                        conversation_ids_in_messages.add(conv_id)
                    except ValueError:
                        pass

        assert (
            len(conversation_ids_in_messages) == num_conversations
        ), "All conversation IDs should be present in messages"

        # Performance check - should complete in reasonable time
        # 100 conversations with mocked provider should be fast (< 5 seconds)
        assert (
            execution_time < 5.0
        ), f"100 concurrent conversations took {execution_time:.2f}s, expected < 5s"

        print(
            f"\n✓ Executed {num_conversations} concurrent conversations in {execution_time:.2f}s"
        )
        print(
            f"✓ Average time per conversation: {execution_time/num_conversations*1000:.1f}ms"
        )

    @pytest.mark.asyncio
    async def test_concurrent_multi_turn_conversations(
        self, mock_provider, conversation_memory
    ):
        """
        Test concurrent multi-turn conversations.

        Verifies:
        - Multiple conversations can continue concurrently
        - Context preserved correctly for each conversation
        - No interference between concurrent continuations
        """
        num_conversations = 50
        turns_per_conversation = 3

        # Setup mock
        response_counter = 0

        def generate_response(*args, **kwargs):
            nonlocal response_counter
            response_counter += 1
            return GenerationResponse(
                content=f"Turn response {response_counter}",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                stop_reason="end_turn",
            )

        mock_provider.generate.side_effect = generate_response

        # Create chat workflow
        chat_workflow = ChatWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Track thread IDs for each conversation
        conversation_threads = {}

        # Execute multiple turns for each conversation concurrently
        for turn in range(turns_per_conversation):

            async def run_turn(conversation_id: int):
                """Run a single turn of a conversation."""
                # Get existing thread ID if this is a continuation
                continuation_id = conversation_threads.get(conversation_id)

                result = await chat_workflow.run(
                    prompt=f"Conversation {conversation_id}, Turn {turn}",
                    continuation_id=continuation_id,
                )

                # Store thread ID for next turn
                thread_id = result.metadata.get("thread_id")
                conversation_threads[conversation_id] = thread_id

                return {
                    "conversation_id": conversation_id,
                    "turn": turn,
                    "thread_id": thread_id,
                    "success": result.success,
                }

            # Execute all conversations for this turn concurrently
            tasks = [run_turn(i) for i in range(num_conversations)]
            turn_results = await asyncio.gather(*tasks)

            # Verify all turns succeeded
            assert all(r["success"] for r in turn_results)

        # Verify final state
        # Each conversation should have (turns_per_conversation * 2) messages
        # (user + assistant for each turn)
        expected_messages = turns_per_conversation * 2

        for conversation_id, thread_id in conversation_threads.items():
            thread = conversation_memory.get_thread(thread_id)
            assert thread is not None
            assert len(thread.messages) == expected_messages, (
                f"Conversation {conversation_id} should have {expected_messages} messages, "
                f"got {len(thread.messages)}"
            )

            # Verify all turns are present in order
            user_messages = [msg for msg in thread.messages if msg.role == "user"]
            for turn in range(turns_per_conversation):
                assert f"Turn {turn}" in user_messages[turn].content

        print(
            f"\n✓ Executed {num_conversations} conversations × {turns_per_conversation} turns = "
            f"{num_conversations * turns_per_conversation} total turns"
        )

    @pytest.mark.asyncio
    async def test_mixed_workflow_concurrent_execution(
        self, mock_provider, conversation_memory
    ):
        """
        Test concurrent execution of different workflow types.

        Verifies:
        - Chat and ThinkDeep workflows can run concurrently
        - ConversationMemory handles multiple workflow types
        - No interference between workflow types
        """
        num_chat = 50
        num_thinkdeep = 50

        # Setup mock
        mock_provider.generate.return_value = GenerationResponse(
            content="Mixed workflow response",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            stop_reason="end_turn",
        )

        # Create workflows
        chat_workflow = ChatWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        thinkdeep_workflow = ThinkDeepWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        # Define tasks
        async def run_chat(i: int):
            result = await chat_workflow.run(prompt=f"Chat {i}")
            return ("chat", result.metadata.get("thread_id"), result.success)

        async def run_thinkdeep(i: int):
            result = await thinkdeep_workflow.run(
                prompt=f"ThinkDeep investigation {i}", files=[]
            )
            return ("thinkdeep", result.metadata.get("thread_id"), result.success)

        # Execute mixed workflows concurrently
        tasks = []
        tasks.extend([run_chat(i) for i in range(num_chat)])
        tasks.extend([run_thinkdeep(i) for i in range(num_thinkdeep)])

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Verify all executions succeeded
        assert len(results) == num_chat + num_thinkdeep
        assert all(r[2] for r in results), "All workflow executions should succeed"

        # Verify unique thread IDs
        thread_ids = [r[1] for r in results]
        assert len(set(thread_ids)) == len(
            thread_ids
        ), "All thread IDs should be unique"

        # Verify workflow type separation
        chat_threads = [r[1] for r in results if r[0] == "chat"]
        thinkdeep_threads = [r[1] for r in results if r[0] == "thinkdeep"]

        assert len(chat_threads) == num_chat
        assert len(thinkdeep_threads) == num_thinkdeep

        # Verify threads exist and have correct structure
        for thread_id in chat_threads:
            thread = conversation_memory.get_thread(thread_id)
            assert thread is not None
            assert len(thread.messages) == 2  # user + assistant

        for thread_id in thinkdeep_threads:
            thread = conversation_memory.get_thread(thread_id)
            assert thread is not None
            assert len(thread.messages) == 2  # user + assistant

        print(
            f"\n✓ Executed {num_chat} chat + {num_thinkdeep} thinkdeep = "
            f"{len(results)} mixed workflows in {execution_time:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_performance_scalability(self, mock_provider, conversation_memory):
        """
        Test performance scalability with increasing load.

        Measures execution time for different concurrency levels to verify
        that performance scales reasonably.
        """
        # Setup mock
        mock_provider.generate.return_value = GenerationResponse(
            content="Scalability test response",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            stop_reason="end_turn",
        )

        chat_workflow = ChatWorkflow(
            provider=mock_provider, conversation_memory=conversation_memory
        )

        async def run_batch(batch_size: int):
            """Run a batch of conversations and return execution time."""
            start = time.time()

            tasks = [
                chat_workflow.run(prompt=f"Request {i}") for i in range(batch_size)
            ]
            await asyncio.gather(*tasks)

            return time.time() - start

        # Test different batch sizes
        batch_sizes = [10, 50, 100]
        execution_times = {}

        for size in batch_sizes:
            exec_time = await run_batch(size)
            execution_times[size] = exec_time

            avg_time = exec_time / size * 1000  # ms per conversation
            print(
                f"\n✓ Batch size {size}: {exec_time:.3f}s total, {avg_time:.1f}ms average"
            )

        # Verify performance doesn't degrade exponentially
        # Time for 100 should be < 15x time for 10 (allowing for overhead)
        time_ratio = execution_times[100] / execution_times[10]
        assert (
            time_ratio < 15.0
        ), f"Performance degradation too high: {time_ratio:.1f}x for 10x load"

        print(
            f"\n✓ Performance scaling factor: {time_ratio:.1f}x for 10x load increase"
        )
