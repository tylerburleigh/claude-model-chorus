"""
Integration tests for workflow chaining patterns.

Tests verify that different workflows can be chained together effectively,
demonstrating ModelChorus's orchestration capabilities:
- Consensus → ThinkDeep → Chat workflow integration
- Context preservation across workflow boundaries
- Continuation ID management across workflow types
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "model_chorus" / "src"
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.workflows.thinkdeep import ThinkDeepWorkflow
from model_chorus.workflows.chat import ChatWorkflow
from model_chorus.providers.base_provider import GenerationResponse, GenerationRequest
from model_chorus.core.conversation import ConversationMemory


class TestConsensusThinkDeepChatChaining:
    """
    Test suite for consensus → thinkdeep → chat workflow integration.

    This pattern demonstrates using multiple orchestration strategies in sequence:
    1. Consensus: Gather multi-model opinions on a decision
    2. ThinkDeep: Investigate specific concerns raised
    3. Chat: Refine understanding and get practical recommendations
    """

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        provider.check_availability = AsyncMock(return_value=(True, None))
        return provider

    @pytest.fixture
    def mock_provider_2(self):
        """Create a second mock provider for consensus testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider_2"
        provider.validate_api_key.return_value = True
        provider.check_availability = AsyncMock(return_value=(True, None))
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create shared conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_consensus_to_thinkdeep_to_chat_workflow(
        self, mock_provider, mock_provider_2, conversation_memory
    ):
        """
        Test chaining consensus → thinkdeep → chat workflows.

        Scenario: Technology decision requiring multi-perspective analysis,
        deep investigation, and practical refinement.

        Example use case: "Should we use Redis or Memcached for caching?"
        1. Consensus: Get opinions from multiple models
        2. ThinkDeep: Investigate memory usage patterns (concern raised)
        3. Chat: Refine deployment recommendations
        """
        # Stage 1: Consensus - Multi-model decision making
        # ================================================

        # Setup consensus mock responses (simulating two models with different opinions)
        mock_provider.generate.return_value = GenerationResponse(
            content="Redis is recommended for this use case. It provides persistence, "
                   "richer data structures (lists, sets, sorted sets), and built-in "
                   "replication. The main concern is memory usage - Redis keeps all "
                   "data in memory which could be expensive at scale.",
            model="test-model-1",
            usage={"input_tokens": 50, "output_tokens": 100},
            stop_reason="end_turn"
        )

        mock_provider_2.generate.return_value = GenerationResponse(
            content="Memcached is simpler and more memory-efficient for pure caching. "
                   "However, Redis offers more features that could be valuable: "
                   "persistence, pub/sub, transactions. Both are excellent choices, "
                   "but Redis has more long-term flexibility.",
            model="test-model-2",
            usage={"input_tokens": 50, "output_tokens": 95},
            stop_reason="end_turn"
        )

        # Create and execute consensus workflow
        consensus_workflow = ConsensusWorkflow(
            providers=[mock_provider, mock_provider_2],
            strategy=ConsensusStrategy.ALL_RESPONSES
        )

        consensus_request = GenerationRequest(
            prompt="Should we use Redis or Memcached for session caching in our web application? "
                  "Consider: performance, scalability, operational complexity, and cost."
        )

        consensus_result = await consensus_workflow.execute(consensus_request)

        # Verify consensus execution
        assert consensus_result is not None
        assert len(consensus_result.all_responses) == 2
        assert "Redis" in consensus_result.all_responses[0].content
        assert "Memcached" in consensus_result.all_responses[1].content

        # Extract key concern from consensus (memory usage)
        consensus_concern = "Redis memory usage at scale"

        # Stage 2: ThinkDeep - Investigate specific concern
        # ==================================================

        # Reset mock for thinkdeep investigation
        mock_provider.generate.return_value = GenerationResponse(
            content=f"Investigating {consensus_concern}. Redis uses a memory allocator "
                   "that can lead to fragmentation over time. For 1M sessions (~1KB each), "
                   "expect ~1.5GB actual usage due to allocator overhead. Mitigation: "
                   "Use maxmemory policies, monitor with INFO memory, consider Redis "
                   "clustering for horizontal scaling.",
            model="test-model-1",
            usage={"input_tokens": 120, "output_tokens": 150},
            stop_reason="end_turn"
        )

        # Create ThinkDeep workflow (uses same conversation memory)
        thinkdeep_workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute investigation (no continuation_id - this is a new investigation)
        thinkdeep_result = await thinkdeep_workflow.run(
            prompt=f"Based on earlier consensus discussion about Redis vs Memcached, "
                  f"investigate: {consensus_concern}. Analyze memory footprint patterns, "
                  f"fragmentation issues, and mitigation strategies.",
            files=[]
        )

        # Verify thinkdeep execution
        assert thinkdeep_result.success is True
        assert thinkdeep_result.synthesis is not None
        assert "memory" in thinkdeep_result.synthesis.lower()
        assert "maxmemory" in thinkdeep_result.synthesis or "fragmentation" in thinkdeep_result.synthesis

        # Get thread ID for continuation
        thinkdeep_thread_id = thinkdeep_result.metadata.get("thread_id")
        assert thinkdeep_thread_id is not None

        # Stage 3: Chat - Refine practical recommendations
        # =================================================

        # Reset mock for chat refinement
        mock_provider.generate.return_value = GenerationResponse(
            content="For deployment, I recommend starting with Redis in a single-instance "
                   "configuration with maxmemory-policy allkeys-lru set to 2GB. Monitor "
                   "memory usage with CloudWatch/Prometheus. Once you hit 70% memory usage, "
                   "plan for Redis Cluster with 3 masters. This gives you time to validate "
                   "the architecture while keeping costs low initially.",
            model="test-model-1",
            usage={"input_tokens": 200, "output_tokens": 120},
            stop_reason="end_turn"
        )

        # Create Chat workflow (uses same conversation memory)
        chat_workflow = ChatWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute chat with continuation from thinkdeep
        chat_result = await chat_workflow.run(
            prompt="Given the memory analysis, what specific deployment configuration "
                  "would you recommend for a production environment handling 100K "
                  "concurrent users?",
            continuation_id=thinkdeep_thread_id  # Continue from thinkdeep thread
        )

        # Verify chat execution
        assert chat_result.success is True
        assert chat_result.synthesis is not None
        assert "deployment" in chat_result.synthesis.lower() or "configuration" in chat_result.synthesis.lower()

        # Verify continuation worked
        assert chat_result.metadata.get("is_continuation") is True
        assert chat_result.metadata.get("thread_id") == thinkdeep_thread_id

        # Stage 4: Verify full workflow chain
        # ====================================

        # Verify conversation thread contains all interactions
        thread = conversation_memory.get_thread(thinkdeep_thread_id)
        assert thread is not None

        # Should have messages from thinkdeep + chat (2 user prompts + 2 assistant responses = 4 messages)
        # ThinkDeep: user prompt + assistant response
        # Chat: user prompt + assistant response
        assert len(thread.messages) >= 4, f"Expected at least 4 messages, got {len(thread.messages)}"

        # Verify message sequence
        messages = thread.messages
        assert any("investigate" in msg.content.lower() for msg in messages if msg.role == "user")
        assert any("deployment" in msg.content.lower() for msg in messages if msg.role == "user")

        # Verify all three workflow stages produced results
        assert consensus_result.all_responses  # Stage 1: consensus opinions
        assert thinkdeep_result.synthesis  # Stage 2: investigation findings
        assert chat_result.synthesis  # Stage 3: practical recommendations

    @pytest.mark.asyncio
    async def test_workflow_chain_context_isolation(
        self, mock_provider, mock_provider_2, conversation_memory
    ):
        """
        Test that workflows can run independently or share context intentionally.

        Verifies:
        - Workflows without continuation_id start fresh threads
        - Workflows with continuation_id share context properly
        - Conversation memory isolates different workflow chains
        """
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Response from independent workflow execution",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            stop_reason="end_turn"
        )

        # Create two independent thinkdeep workflows
        thinkdeep_1 = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        thinkdeep_2 = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Execute first workflow
        result_1 = await thinkdeep_1.run(
            prompt="First independent investigation",
            files=[]
        )
        thread_1_id = result_1.metadata.get("thread_id")

        # Execute second workflow (independent - no continuation)
        result_2 = await thinkdeep_2.run(
            prompt="Second independent investigation",
            files=[]
        )
        thread_2_id = result_2.metadata.get("thread_id")

        # Verify different threads
        assert thread_1_id != thread_2_id

        # Verify each thread has only its own messages
        thread_1 = conversation_memory.get_thread(thread_1_id)
        thread_2 = conversation_memory.get_thread(thread_2_id)

        assert len(thread_1.messages) == 2  # user + assistant
        assert len(thread_2.messages) == 2  # user + assistant

        # Now create a chat that continues from thinkdeep_1
        chat = ChatWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        result_3 = await chat.run(
            prompt="Follow-up question",
            continuation_id=thread_1_id
        )

        # Verify chat continued thread_1
        assert result_3.metadata.get("thread_id") == thread_1_id
        assert result_3.metadata.get("is_continuation") is True

        # Verify thread_1 now has both thinkdeep and chat messages
        thread_1_updated = conversation_memory.get_thread(thread_1_id)
        assert len(thread_1_updated.messages) == 4  # thinkdeep (user+assistant) + chat (user+assistant)

        # Verify thread_2 unchanged
        thread_2_unchanged = conversation_memory.get_thread(thread_2_id)
        assert len(thread_2_unchanged.messages) == 2

    @pytest.mark.asyncio
    async def test_consensus_without_continuation_support(
        self, mock_provider, mock_provider_2
    ):
        """
        Test that ConsensusWorkflow works independently without continuation.

        Note: ConsensusWorkflow doesn't inherit from BaseWorkflow and doesn't
        support continuation_id. This test verifies it works as a starting point
        for workflow chains even though it can't be continued itself.
        """
        # Setup mock responses
        mock_provider.generate.return_value = GenerationResponse(
            content="Consensus response 1",
            model="model-1",
            usage={"input_tokens": 10, "output_tokens": 20},
            stop_reason="end_turn"
        )

        mock_provider_2.generate.return_value = GenerationResponse(
            content="Consensus response 2",
            model="model-2",
            usage={"input_tokens": 10, "output_tokens": 20},
            stop_reason="end_turn"
        )

        # Create consensus workflow
        consensus = ConsensusWorkflow(
            providers=[mock_provider, mock_provider_2],
            strategy=ConsensusStrategy.ALL_RESPONSES
        )

        # Execute consensus
        result = await consensus.execute(
            GenerationRequest(prompt="Test question")
        )

        # Verify consensus executed successfully
        assert result is not None
        assert len(result.all_responses) == 2
        assert result.provider_results is not None

        # Note: ConsensusWorkflow doesn't have thread_id in metadata
        # because it doesn't use ConversationMemory. This is expected
        # and correct - consensus is a one-shot multi-model query.

        # The pattern for using consensus in chains is:
        # 1. Run consensus (no continuation)
        # 2. Extract key findings programmatically
        # 3. Pass findings to next workflow (thinkdeep/chat)
        # 4. ThinkDeep/Chat create their own threads
