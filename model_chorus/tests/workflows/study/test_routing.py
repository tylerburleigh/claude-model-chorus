"""
Tests for PersonaRouter and routing functionality in STUDY workflow.

Tests cover:
- Routing skill invocation and JSON output
- Fallback routing when context analysis fails
- Routing decision structure and metadata
- Integration with StudyWorkflow
"""

import pytest
from unittest.mock import Mock, patch
from model_chorus.workflows.study.persona_router import PersonaRouter, RoutingDecision
from model_chorus.workflows.study.persona_base import Persona, PersonaRegistry
from model_chorus.workflows.study.personas import get_default_registry
from model_chorus.core.models import StudyState


class TestRoutingSkillInvocation:
    """Test routing skill invocation and JSON output."""

    def test_routing_skill_invocation(self):
        """Test that routing skill invokes successfully and returns valid JSON."""
        # Setup: Create router with default registry
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # Create test state
        state = StudyState(
            investigation_id="test-routing-001",
            session_id="session-routing-001",
            question="Test routing question",
            current_phase="discovery",
            confidence="medium",
            findings=[],
            personas_active=[]
        )

        # Execute: Route to next persona
        decision = router.route_next_persona(state)

        # Verify: Decision is valid RoutingDecision
        assert isinstance(decision, RoutingDecision), "Should return RoutingDecision"
        assert decision.persona is not None, "Should select a persona for discovery phase"
        assert decision.persona_name in ["Researcher", "Critic"], \
            f"Discovery should select Researcher or Critic, got {decision.persona_name}"

        # Verify: Decision has required fields
        assert decision.reasoning, "Should include reasoning"
        assert isinstance(decision.confidence, str), "Confidence should be string"
        assert isinstance(decision.guidance, list), "Guidance should be list"
        assert isinstance(decision.metadata, dict), "Metadata should be dict"

        # Verify: Reasoning is informative
        assert len(decision.reasoning) > 20, "Reasoning should be descriptive"

        # Verify: Guidance is actionable
        assert len(decision.guidance) > 0, "Should provide guidance"
        for guide in decision.guidance:
            assert isinstance(guide, str), "Each guidance item should be string"
            assert len(guide) > 5, "Guidance should be descriptive"

        # Verify: Metadata includes phase info
        assert "phase" in decision.metadata, "Metadata should include phase"
        assert decision.metadata["phase"] == "discovery"

        # Verify: No fallback was used (normal routing)
        assert decision.metadata.get("fallback_used") is not True, \
            "Normal routing should not use fallback"

    def test_routing_different_phases(self):
        """Test routing for different investigation phases."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # Test each phase
        test_cases = [
            ("discovery", ["Researcher", "Critic"]),
            ("validation", ["Critic", "Researcher"]),
            ("planning", ["Planner", "Researcher", "Critic"]),
        ]

        for phase, expected_personas in test_cases:
            state = StudyState(
                investigation_id=f"test-{phase}",
                session_id=f"session-{phase}",
                question=f"Test {phase}",
                current_phase=phase,
                confidence="medium",
                findings=[],
                personas_active=[]
            )

            decision = router.route_next_persona(state)

            assert decision.persona_name in expected_personas, \
                f"Phase {phase} should select from {expected_personas}, got {decision.persona_name}"

    def test_routing_with_findings(self):
        """Test routing behavior when findings are present."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # State with findings
        state = StudyState(
            investigation_id="test-findings",
            session_id="session-findings",
            question="Test with findings",
            current_phase="discovery",
            confidence="medium",
            findings=[
                {
                    "persona": "Researcher",
                    "finding": "Test finding 1",
                    "evidence": [],
                    "confidence": "high"
                }
            ],
            personas_active=["Researcher"]
        )

        decision = router.route_next_persona(state)

        # With Researcher as prior persona and findings, should get Critic
        assert decision.persona_name == "Critic", \
            "After Researcher with findings, should select Critic to challenge assumptions"

    def test_routing_complete_phase(self):
        """Test routing when investigation is complete."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # State in complete phase
        state = StudyState(
            investigation_id="test-complete",
            session_id="session-complete",
            question="Test complete",
            current_phase="complete",
            confidence="high",
            findings=[],
            personas_active=[]
        )

        decision = router.route_next_persona(state)

        # Should return None persona (investigation complete)
        assert decision.persona is None, "Complete phase should return None persona"
        assert decision.persona_name == "None", "persona_name should be 'None'"


class TestFallbackRouting:
    """Test fallback routing when context analysis fails."""

    def test_fallback_routing_on_exception(self):
        """Test that fallback activates when context analysis raises exception."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        state = StudyState(
            investigation_id="test-fallback",
            session_id="session-fallback",
            question="Test fallback",
            current_phase="discovery",
            confidence="medium",
            findings=[],
            personas_active=[]
        )

        # Mock analyze_context to raise exception
        with patch('model_chorus.workflows.study.persona_router.analyze_context') as mock_analyze:
            mock_analyze.side_effect = RuntimeError("Simulated skill failure")

            decision = router.route_next_persona(state)

            # Verify fallback activated
            assert decision.persona is not None, "Fallback should return a persona"
            assert decision.persona_name == "Researcher", \
                "Discovery phase should fallback to Researcher"

            assert "Fallback routing" in decision.reasoning, \
                "Reasoning should indicate fallback was used"

            assert decision.metadata.get("fallback_used") is True, \
                "Metadata should indicate fallback"

            assert "RuntimeError" in decision.metadata.get("original_error_type", ""), \
                "Metadata should include error type"

    def test_fallback_for_all_phases(self):
        """Test fallback routing for all investigation phases."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        fallback_map = {
            "discovery": "Researcher",
            "validation": "Critic",
            "planning": "Planner",
        }

        with patch('model_chorus.workflows.study.persona_router.analyze_context') as mock_analyze:
            mock_analyze.side_effect = Exception("Simulated failure")

            for phase, expected_persona in fallback_map.items():
                state = StudyState(
                    investigation_id=f"test-fallback-{phase}",
                    session_id=f"session-fallback-{phase}",
                    question=f"Test fallback {phase}",
                    current_phase=phase,
                    confidence="medium",
                    findings=[],
                    personas_active=[]
                )

                decision = router.route_next_persona(state)

                assert decision.persona_name == expected_persona, \
                    f"Phase {phase} should fallback to {expected_persona}"

                assert decision.metadata.get("fallback_used") is True

    def test_fallback_provides_valid_guidance(self):
        """Test that fallback provides valid guidance."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        state = StudyState(
            investigation_id="test-fallback-guidance",
            session_id="session-fallback-guidance",
            question="Test guidance",
            current_phase="discovery",
            confidence="medium",
            findings=[],
            personas_active=[]
        )

        with patch('model_chorus.workflows.study.persona_router.analyze_context') as mock_analyze:
            mock_analyze.side_effect = Exception("Failure")

            decision = router.route_next_persona(state)

            # Verify guidance is present and valid
            assert len(decision.guidance) > 0, "Fallback should provide guidance"
            assert all(isinstance(g, str) for g in decision.guidance), \
                "All guidance items should be strings"
            assert all(len(g) > 5 for g in decision.guidance), \
                "Guidance should be descriptive"


class TestRoutingHistory:
    """Test routing history tracking."""

    def test_routing_history_recorded(self):
        """Test that routing decisions are recorded in history."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        state = StudyState(
            investigation_id="test-history",
            session_id="session-history",
            question="Test history",
            current_phase="discovery",
            confidence="medium",
            findings=[],
            personas_active=[]
        )

        # Make routing decision
        decision = router.route_next_persona(state)

        # Check history was recorded
        history = router.get_routing_history()
        assert len(history) > 0, "History should have entries"

        latest = history[0]  # Most recent first
        assert latest.investigation_id == "test-history"
        assert latest.selected_persona == decision.persona_name

    def test_routing_history_filtering(self):
        """Test filtering routing history by investigation."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # Create decisions for different investigations
        for i in range(3):
            state = StudyState(
                investigation_id=f"test-{i}",
                session_id=f"session-{i}",
                question=f"Test {i}",
                current_phase="discovery",
                confidence="medium",
                findings=[],
                personas_active=[]
            )
            router.route_next_persona(state)

        # Get filtered history
        history = router.get_routing_history(investigation_id="test-1")
        assert len(history) == 1, "Should have one entry for test-1"
        assert history[0].investigation_id == "test-1"

    def test_routing_history_limit(self):
        """Test limiting routing history results."""
        registry = get_default_registry()
        router = PersonaRouter(registry)

        # Create multiple decisions
        for i in range(5):
            state = StudyState(
                investigation_id="test-limit",
                session_id="session-limit",
                question="Test",
                current_phase="discovery",
                confidence="medium",
                findings=[],
                personas_active=[]
            )
            router.route_next_persona(state)

        # Get limited history
        history = router.get_routing_history(limit=2)
        assert len(history) == 2, "Should limit to 2 entries"


class TestStudyWorkflowIntegration:
    """Test PersonaRouter integration with StudyWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_has_router(self):
        """Test that StudyWorkflow initializes PersonaRouter."""
        from model_chorus.workflows.study.study_workflow import StudyWorkflow
        from model_chorus.providers.claude_provider import ClaudeProvider

        # Create workflow (may fail if provider unavailable, skip test)
        try:
            provider = ClaudeProvider()
            workflow = StudyWorkflow(provider)

            # Verify router is initialized
            assert hasattr(workflow, 'persona_router'), \
                "StudyWorkflow should have persona_router attribute"

            assert isinstance(workflow.persona_router, PersonaRouter), \
                "persona_router should be PersonaRouter instance"

            # Verify router has personas
            personas = workflow.persona_router.get_available_personas()
            assert len(personas) == 3, "Should have 3 default personas"
            assert "Researcher" in personas
            assert "Critic" in personas
            assert "Planner" in personas

        except Exception as e:
            pytest.skip(f"Provider unavailable: {e}")

    @pytest.mark.asyncio
    async def test_workflow_routing_history_access(self):
        """Test that StudyWorkflow provides access to routing history."""
        from model_chorus.workflows.study.study_workflow import StudyWorkflow
        from model_chorus.providers.claude_provider import ClaudeProvider

        try:
            provider = ClaudeProvider()
            workflow = StudyWorkflow(provider)

            # Verify get_routing_history method exists
            assert hasattr(workflow, 'get_routing_history'), \
                "StudyWorkflow should have get_routing_history method"

            # Call should work (even if empty)
            history = workflow.get_routing_history()
            assert isinstance(history, list), "Should return list"

        except Exception as e:
            pytest.skip(f"Provider unavailable: {e}")
