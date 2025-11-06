"""
Complex problem investigation tests for ThinkDeep workflow.

Tests verify ThinkDeepWorkflow handles complex scenarios including:
- Architectural decision making with pros/cons analysis
- Bug investigation with systematic debugging
- Multi-step reasoning and hypothesis evolution
- Long-form problem analysis
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from modelchorus.workflows.thinkdeep import ThinkDeepWorkflow
from modelchorus.providers.base_provider import GenerationResponse, GenerationRequest
from modelchorus.core.conversation import ConversationMemory
from modelchorus.core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)


class TestArchitecturalDecisionScenarios:
    """Test suite for architectural decision making scenarios."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_architectural_decision_rest_vs_graphql(
        self, mock_provider, conversation_memory
    ):
        """
        Test architectural decision scenario: REST vs GraphQL API design.

        Verifies multi-step reasoning with:
        - Initial requirements analysis
        - Multiple competing approaches (hypotheses)
        - Pros/cons evaluation (evidence gathering)
        - Trade-off analysis
        - Final recommendation with confidence
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Understand requirements
        mock_provider.generate.return_value = GenerationResponse(
            content="Requirements: Building API for mobile app with complex data relationships. Need efficient data fetching.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result1 = await workflow.run(
            prompt="Should we use REST or GraphQL for our new API? Analyze requirements first.",
            files=["docs/requirements.md"]
        )

        assert result1.success is True
        thread_id = result1.metadata["thread_id"]

        # Add two competing hypotheses
        workflow.add_hypothesis(
            thread_id,
            "REST API is the better choice",
            evidence=["Simpler to implement", "Team familiar with REST", "Good caching support"]
        )

        workflow.add_hypothesis(
            thread_id,
            "GraphQL API is the better choice",
            evidence=["Complex data relationships", "Flexible querying", "Reduced over-fetching"]
        )

        # Step 2: Analyze REST approach
        mock_provider.generate.return_value = GenerationResponse(
            content="REST analysis: Simple endpoints, established patterns. However, mobile clients need multiple endpoints for single views, causing over-fetching.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Analyze the pros and cons of REST API for our use case",
            continuation_id=thread_id
        )

        # Update REST hypothesis with analysis
        workflow.update_hypothesis(
            thread_id,
            "REST API is the better choice",
            new_evidence=[
                "PRO: Simple implementation",
                "PRO: Team expertise",
                "CON: Over-fetching data for mobile",
                "CON: Multiple round trips needed"
            ]
        )

        workflow.update_confidence(thread_id, ConfidenceLevel.LOW.value)

        # Step 3: Analyze GraphQL approach
        mock_provider.generate.return_value = GenerationResponse(
            content="GraphQL analysis: Solves over-fetching, single endpoint. Learning curve for team, requires new tooling and caching strategy.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result3 = await workflow.run(
            prompt="Analyze the pros and cons of GraphQL for our use case",
            continuation_id=thread_id
        )

        # Update GraphQL hypothesis with analysis
        workflow.update_hypothesis(
            thread_id,
            "GraphQL API is the better choice",
            new_evidence=[
                "PRO: Solves over-fetching problem",
                "PRO: Single endpoint, flexible queries",
                "PRO: Strong typing with schema",
                "CON: Team learning curve",
                "CON: Complex caching",
                "CON: New tooling needed"
            ]
        )

        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Step 4: Evaluate trade-offs and make recommendation
        mock_provider.generate.return_value = GenerationResponse(
            content="Trade-off analysis: Mobile data efficiency critical, over-fetching causes performance issues. GraphQL's benefits outweigh learning curve for this use case.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result4 = await workflow.run(
            prompt="Weigh the trade-offs and make a recommendation",
            continuation_id=thread_id
        )

        # Validate GraphQL hypothesis as the recommendation
        workflow.validate_hypothesis(thread_id, "GraphQL API is the better choice")
        workflow.disprove_hypothesis(thread_id, "REST API is the better choice")
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Verify final state
        state = workflow.get_investigation_state(thread_id)

        # Should have 4 investigation steps
        assert len(state.steps) == 4

        # Should have 2 hypotheses with opposite conclusions
        assert len(state.hypotheses) == 2
        validated = [h for h in state.hypotheses if h.status == "validated"]
        disproven = [h for h in state.hypotheses if h.status == "disproven"]

        assert len(validated) == 1
        assert len(disproven) == 1
        assert "GraphQL" in validated[0].hypothesis

        # Final confidence should be high
        assert state.current_confidence == ConfidenceLevel.HIGH.value

        # Summary should show decision process
        summary = workflow.get_investigation_summary(thread_id)
        assert summary["total_steps"] == 4
        assert summary["validated_hypotheses"] == 1
        assert summary["disproven_hypotheses"] == 1

    @pytest.mark.asyncio
    async def test_architectural_decision_database_selection(
        self, mock_provider, conversation_memory
    ):
        """
        Test architectural decision: Database technology selection.

        Scenario: Choosing between PostgreSQL, MongoDB, and Cassandra
        for a high-traffic application.
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Requirements analysis
        mock_provider.generate.return_value = GenerationResponse(
            content="Requirements: High write throughput, eventual consistency acceptable, horizontal scaling needed.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result = await workflow.run(
            prompt="Choose database: PostgreSQL vs MongoDB vs Cassandra. Analyze requirements.",
            files=["docs/requirements.md", "docs/scale_projections.md"]
        )

        thread_id = result.metadata["thread_id"]

        # Add three competing hypotheses
        workflow.add_hypothesis(thread_id, "PostgreSQL is best choice")
        workflow.add_hypothesis(thread_id, "MongoDB is best choice")
        workflow.add_hypothesis(thread_id, "Cassandra is best choice")

        # Simulate analysis steps
        for i in range(2, 5):
            mock_provider.generate.return_value = GenerationResponse(
                content=f"Analysis step {i}: Evaluating option {i-1}",
                model="test-model",
                usage={},
                stop_reason="end_turn"
            )

            await workflow.run(
                prompt=f"Analyze option {i-1}",
                continuation_id=thread_id
            )

        # Validate Cassandra based on requirements
        workflow.validate_hypothesis(thread_id, "Cassandra is best choice")
        workflow.disprove_hypothesis(thread_id, "PostgreSQL is best choice")
        workflow.disprove_hypothesis(thread_id, "MongoDB is best choice")
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Verify decision process
        state = workflow.get_investigation_state(thread_id)
        assert len(state.hypotheses) == 3
        assert len([h for h in state.hypotheses if h.status == "validated"]) == 1
        assert len([h for h in state.hypotheses if h.status == "disproven"]) == 2


class TestBugInvestigationScenarios:
    """Test suite for systematic bug investigation scenarios."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_bug_investigation_api_slowness(
        self, mock_provider, conversation_memory
    ):
        """
        Test bug investigation: API endpoint responding slowly.

        Verifies systematic debugging with:
        - Initial symptom analysis
        - Multiple hypothesis generation
        - Evidence-based hypothesis elimination
        - Root cause identification
        - Confidence progression to certain
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Symptom analysis
        mock_provider.generate.return_value = GenerationResponse(
            content="Symptom: /api/users endpoint responding in 5+ seconds. Started after recent deployment. Peak traffic times worse.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result1 = await workflow.run(
            prompt="Investigate why /api/users endpoint is slow. Start with symptom analysis.",
            files=["logs/api_performance.log"]
        )

        assert result1.success is True
        thread_id = result1.metadata["thread_id"]

        # Generate multiple hypotheses for slow API
        workflow.add_hypothesis(
            thread_id,
            "Database query inefficiency (missing index)",
            evidence=["Slow response correlates with database calls"]
        )

        workflow.add_hypothesis(
            thread_id,
            "N+1 query problem in ORM",
            evidence=["Recent code changes added eager loading"]
        )

        workflow.add_hypothesis(
            thread_id,
            "Network latency to database",
            evidence=["Problem started after deployment to new region"]
        )

        workflow.add_hypothesis(
            thread_id,
            "Memory leak causing GC pressure",
            evidence=["Performance degrades over time"]
        )

        # Step 2: Check database queries
        mock_provider.generate.return_value = GenerationResponse(
            content="Database slow query log shows SELECT * FROM users taking 4.5s. EXPLAIN shows full table scan, no index on email column used in WHERE clause.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Analyze database query performance and execution plans",
            continuation_id=thread_id,
            files=["logs/slow_query.log", "db/schema.sql"]
        )

        # Strong evidence for missing index hypothesis
        workflow.update_hypothesis(
            thread_id,
            "Database query inefficiency (missing index)",
            new_evidence=[
                "Full table scan detected in EXPLAIN",
                "No index on email column",
                "Query takes 4.5s of total 5s response time"
            ]
        )

        workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

        # Step 3: Rule out N+1 problem
        mock_provider.generate.return_value = GenerationResponse(
            content="Code analysis: Recent changes use select_related() correctly. Only 3 queries total, not N+1 pattern.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result3 = await workflow.run(
            prompt="Check for N+1 query patterns in recent code changes",
            continuation_id=thread_id,
            files=["src/api/users.py"]
        )

        # Disprove N+1 hypothesis
        workflow.update_hypothesis(
            thread_id,
            "N+1 query problem in ORM",
            new_evidence=["Only 3 queries executed", "select_related() used correctly"],
            new_status="disproven"
        )

        # Step 4: Rule out network latency
        mock_provider.generate.return_value = GenerationResponse(
            content="Network metrics: Database connection latency 2ms average, 5ms p99. Not significant contributor.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result4 = await workflow.run(
            prompt="Measure network latency to database server",
            continuation_id=thread_id,
            files=["logs/network_metrics.log"]
        )

        # Disprove network latency hypothesis
        workflow.disprove_hypothesis(thread_id, "Network latency to database")
        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Step 5: Rule out memory leak
        mock_provider.generate.return_value = GenerationResponse(
            content="Memory analysis: Heap usage stable, GC pauses under 10ms. Memory not the issue.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result5 = await workflow.run(
            prompt="Analyze memory usage and GC metrics",
            continuation_id=thread_id,
            files=["logs/gc.log", "logs/heap_dump.hprof"]
        )

        # Disprove memory leak hypothesis
        workflow.disprove_hypothesis(thread_id, "Memory leak causing GC pressure")

        # Step 6: Confirm root cause with fix verification
        mock_provider.generate.return_value = GenerationResponse(
            content="Added index on users.email column. Query time reduced from 4.5s to 15ms. Response time now 200ms total.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result6 = await workflow.run(
            prompt="Verify that adding index fixes the performance issue",
            continuation_id=thread_id,
            files=["staging/performance_test_results.log"]
        )

        # Validate root cause with high confidence
        workflow.validate_hypothesis(thread_id, "Database query inefficiency (missing index)")
        workflow.update_confidence(thread_id, ConfidenceLevel.VERY_HIGH.value)

        # Verify investigation completeness
        state = workflow.get_investigation_state(thread_id)

        # 6 investigation steps
        assert len(state.steps) == 6

        # 4 hypotheses: 1 validated, 3 disproven
        assert len(state.hypotheses) == 4
        validated = [h for h in state.hypotheses if h.status == "validated"]
        disproven = [h for h in state.hypotheses if h.status == "disproven"]

        assert len(validated) == 1
        assert len(disproven) == 3
        assert "missing index" in validated[0].hypothesis.lower()

        # High confidence in root cause
        assert state.current_confidence == ConfidenceLevel.VERY_HIGH.value

        # Files from different investigation angles (8 total: 1+2+1+1+2+1)
        assert len(state.relevant_files) == 8

        # Summary shows systematic investigation
        summary = workflow.get_investigation_summary(thread_id)
        assert summary["total_steps"] == 6
        assert summary["validated_hypotheses"] == 1
        assert summary["disproven_hypotheses"] == 3
        assert summary["confidence"] == ConfidenceLevel.VERY_HIGH.value

    @pytest.mark.asyncio
    async def test_bug_investigation_intermittent_crash(
        self, mock_provider, conversation_memory
    ):
        """
        Test bug investigation: Intermittent application crash.

        Scenario: App crashes occasionally, hard to reproduce.
        Requires analyzing logs, stack traces, and environmental factors.
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Analyze crash symptoms
        mock_provider.generate.return_value = GenerationResponse(
            content="Crash pattern: NullPointerException in background thread. Happens ~10% of requests. No obvious pattern in timing.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result1 = await workflow.run(
            prompt="Investigate intermittent application crashes. Analyze stack traces and patterns.",
            files=["logs/crash_reports.log"]
        )

        thread_id = result1.metadata["thread_id"]

        # Generate hypotheses for intermittent crash
        workflow.add_hypothesis(
            thread_id,
            "Race condition in background processing",
            evidence=["Crash in background thread", "Intermittent nature"]
        )

        workflow.add_hypothesis(
            thread_id,
            "Null value from external API",
            evidence=["NullPointerException in code that processes API response"]
        )

        workflow.add_hypothesis(
            thread_id,
            "Memory corruption under high load",
            evidence=["Only happens under production load"]
        )

        # Step 2: Analyze thread interactions
        mock_provider.generate.return_value = GenerationResponse(
            content="Thread dump analysis: No obvious race condition. Synchronized blocks used correctly. Thread count stable.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Analyze thread dumps for race conditions",
            continuation_id=thread_id,
            files=["logs/thread_dumps.log"]
        )

        # Weak evidence against race condition
        workflow.update_hypothesis(
            thread_id,
            "Race condition in background processing",
            new_evidence=["No race detected in thread dumps", "Synchronization appears correct"]
        )

        # Step 3: Check external API responses
        mock_provider.generate.return_value = GenerationResponse(
            content="API response analysis: External service occasionally returns null for 'metadata' field. Code assumes non-null.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result3 = await workflow.run(
            prompt="Analyze external API responses for null values",
            continuation_id=thread_id,
            files=["logs/api_responses.json", "src/integrations/external_api.py"]
        )

        # Strong evidence for null API response
        workflow.update_hypothesis(
            thread_id,
            "Null value from external API",
            new_evidence=[
                "API returns null for 'metadata' field in some responses",
                "Code accesses metadata.user without null check",
                "Crash rate matches API null response rate (10%)"
            ]
        )

        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Step 4: Rule out memory corruption
        mock_provider.generate.return_value = GenerationResponse(
            content="Memory analysis shows no corruption. Stack traces consistent with null pointer, not memory issues.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result4 = await workflow.run(
            prompt="Check for memory corruption",
            continuation_id=thread_id
        )

        workflow.disprove_hypothesis(thread_id, "Memory corruption under high load")

        # Step 5: Confirm fix
        mock_provider.generate.return_value = GenerationResponse(
            content="Added null check for metadata field. Tested with 1000 requests, no crashes. Problem resolved.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result5 = await workflow.run(
            prompt="Verify that null check fixes the crash",
            continuation_id=thread_id,
            files=["tests/integration_test_results.log"]
        )

        # Validate root cause
        workflow.validate_hypothesis(thread_id, "Null value from external API")
        workflow.disprove_hypothesis(thread_id, "Race condition in background processing")
        workflow.update_confidence(thread_id, ConfidenceLevel.VERY_HIGH.value)

        # Verify systematic investigation
        state = workflow.get_investigation_state(thread_id)
        assert len(state.steps) == 5
        assert len(state.hypotheses) == 3

        validated = [h for h in state.hypotheses if h.status == "validated"]
        assert len(validated) == 1
        assert "Null value from external API" in validated[0].hypothesis


class TestComplexMultiStepReasoning:
    """Test suite for complex multi-step reasoning scenarios."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = AsyncMock()
        provider.provider_name = "test_provider"
        provider.validate_api_key.return_value = True
        return provider

    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory for testing."""
        return ConversationMemory()

    @pytest.mark.asyncio
    async def test_long_investigation_with_hypothesis_pivots(
        self, mock_provider, conversation_memory
    ):
        """
        Test long investigation (10+ steps) with hypothesis pivots.

        Verifies that workflow can handle:
        - Extended investigation (10+ steps)
        - Hypothesis revisions based on new evidence
        - Confidence changes (both increase and decrease)
        - Complex evidence accumulation
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Initial investigation
        mock_provider.generate.return_value = GenerationResponse(
            content="Starting complex investigation.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result = await workflow.run(
            prompt="Investigate complex system performance degradation"
        )

        thread_id = result.metadata["thread_id"]

        # Add initial hypothesis
        workflow.add_hypothesis(
            thread_id,
            "Initial hypothesis: CPU bottleneck",
            evidence=["High CPU usage observed"]
        )

        # Simulate 10+ investigation steps with hypothesis evolution
        for step_num in range(2, 12):
            mock_provider.generate.return_value = GenerationResponse(
                content=f"Investigation step {step_num}: Gathering evidence and refining understanding.",
                model="test-model",
                usage={},
                stop_reason="end_turn"
            )

            await workflow.run(
                prompt=f"Investigation step {step_num}",
                continuation_id=thread_id
            )

            # Simulate hypothesis evolution
            if step_num == 4:
                # Disprove initial hypothesis, add new one
                workflow.disprove_hypothesis(thread_id, "Initial hypothesis: CPU bottleneck")
                workflow.add_hypothesis(
                    thread_id,
                    "Revised hypothesis: I/O bottleneck",
                    evidence=["Disk I/O wait time high"]
                )
                workflow.update_confidence(thread_id, ConfidenceLevel.LOW.value)

            elif step_num == 7:
                # Refine hypothesis with more evidence
                workflow.update_hypothesis(
                    thread_id,
                    "Revised hypothesis: I/O bottleneck",
                    new_evidence=["Database queries show high I/O", "Disk queue depth elevated"]
                )
                workflow.update_confidence(thread_id, ConfidenceLevel.MEDIUM.value)

            elif step_num == 10:
                # Validate final hypothesis
                workflow.validate_hypothesis(thread_id, "Revised hypothesis: I/O bottleneck")
                workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)

        # Verify long investigation state
        state = workflow.get_investigation_state(thread_id)

        # Should have 11 steps total (initial + 10 more)
        assert len(state.steps) >= 11

        # Should have 2 hypotheses (initial disproven, revised validated)
        assert len(state.hypotheses) == 2

        # Verify hypothesis evolution
        disproven = [h for h in state.hypotheses if h.status == "disproven"]
        validated = [h for h in state.hypotheses if h.status == "validated"]

        assert len(disproven) == 1
        assert len(validated) == 1
        assert "CPU bottleneck" in disproven[0].hypothesis
        assert "I/O bottleneck" in validated[0].hypothesis

        # Final confidence should be high
        assert state.current_confidence == ConfidenceLevel.HIGH.value

        # Summary should reflect complex investigation
        summary = workflow.get_investigation_summary(thread_id)
        assert summary["total_steps"] >= 11
        assert summary["total_hypotheses"] == 2
        assert summary["validated_hypotheses"] == 1
        assert summary["disproven_hypotheses"] == 1

    @pytest.mark.asyncio
    async def test_investigation_with_multiple_evidence_types(
        self, mock_provider, conversation_memory
    ):
        """
        Test investigation that gathers multiple types of evidence.

        Verifies:
        - Code analysis evidence
        - Log file evidence
        - Performance metrics evidence
        - User reports evidence
        - All integrated into coherent investigation
        """
        workflow = ThinkDeepWorkflow(
            provider=mock_provider,
            conversation_memory=conversation_memory
        )

        # Step 1: Analyze user reports
        mock_provider.generate.return_value = GenerationResponse(
            content="User reports: Form submission fails 30% of the time. Error message: 'Request timeout'.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result = await workflow.run(
            prompt="Investigate form submission failures reported by users",
            files=["docs/user_reports.md"]
        )

        thread_id = result.metadata["thread_id"]

        workflow.add_hypothesis(
            thread_id,
            "Backend processing timeout causing failures",
            evidence=["User reports show timeout errors"]
        )

        # Step 2: Analyze logs
        mock_provider.generate.return_value = GenerationResponse(
            content="Logs show: form submission endpoint has 504 Gateway Timeout errors. Backend processing takes >30s.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result2 = await workflow.run(
            prompt="Check application logs for form submission errors",
            continuation_id=thread_id,
            files=["logs/application.log"]
        )

        workflow.update_hypothesis(
            thread_id,
            "Backend processing timeout causing failures",
            new_evidence=["504 Gateway Timeout in logs", "Processing exceeds 30s timeout"]
        )

        # Step 3: Analyze code
        mock_provider.generate.return_value = GenerationResponse(
            content="Code analysis: Form handler makes synchronous API call to external service with no timeout config.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result3 = await workflow.run(
            prompt="Analyze form submission code for timeout handling",
            continuation_id=thread_id,
            files=["src/forms/submission_handler.py"]
        )

        workflow.update_hypothesis(
            thread_id,
            "Backend processing timeout causing failures",
            new_evidence=[
                "Synchronous API call to external service",
                "No timeout configured on API client",
                "External service response time varies"
            ]
        )

        # Step 4: Check performance metrics
        mock_provider.generate.return_value = GenerationResponse(
            content="Metrics: External API p50=2s, p95=25s, p99=45s. Highly variable response times.",
            model="test-model",
            usage={},
            stop_reason="end_turn"
        )

        result4 = await workflow.run(
            prompt="Review external API performance metrics",
            continuation_id=thread_id,
            files=["metrics/external_api_latency.json"]
        )

        workflow.update_hypothesis(
            thread_id,
            "Backend processing timeout causing failures",
            new_evidence=[
                "External API p99 latency: 45s",
                "30% of requests exceed 30s (matches failure rate)"
            ]
        )

        workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
        workflow.validate_hypothesis(thread_id, "Backend processing timeout causing failures")

        # Verify multi-source evidence integration
        state = workflow.get_investigation_state(thread_id)

        # Evidence from 4 different sources
        validated_hyp = [h for h in state.hypotheses if h.status == "validated"][0]
        assert len(validated_hyp.evidence) >= 6  # Multiple pieces from each source

        # Files from different evidence types
        assert len(state.relevant_files) == 4
        assert any("user_reports" in f for f in state.relevant_files)
        assert any("logs" in f for f in state.relevant_files)
        assert any("forms" in f for f in state.relevant_files)
        assert any("metrics" in f for f in state.relevant_files)

        # High confidence from multiple evidence sources
        assert state.current_confidence == ConfidenceLevel.HIGH.value
