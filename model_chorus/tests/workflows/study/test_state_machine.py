"""
Tests for Study workflow state machine.

Tests verify that the InvestigationStateMachine correctly implements
valid phase transitions and confidence escalation logic.
"""

import pytest

from model_chorus.core.models import ConfidenceLevel, InvestigationPhase, StudyState
from model_chorus.workflows.study.state_machine import InvestigationStateMachine


class TestInvestigationStateMachineTransitions:
    """Test suite for state machine phase transitions."""

    @pytest.fixture
    def initial_state(self):
        """Create initial study state."""
        return StudyState(
            investigation_id="test-123",
            session_id="session-456",
            current_phase=InvestigationPhase.DISCOVERY.value,
            confidence=ConfidenceLevel.EXPLORING.value,
        )

    @pytest.fixture
    def state_machine(self, initial_state):
        """Create state machine instance."""
        return InvestigationStateMachine(initial_state)

    def test_discovery_to_validation_transition(self, state_machine):
        """Test valid transition from DISCOVERY to VALIDATION."""
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

        # Should be able to transition to VALIDATION
        assert state_machine.can_transition(InvestigationPhase.VALIDATION)

        # Perform transition
        result = state_machine.transition(InvestigationPhase.VALIDATION)
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.VALIDATION
        assert state_machine.state.current_phase == InvestigationPhase.VALIDATION.value

    def test_validation_to_planning_transition(self, state_machine):
        """Test valid transition from VALIDATION to PLANNING."""
        # Move to VALIDATION first
        state_machine.transition(InvestigationPhase.VALIDATION)
        assert state_machine.current_phase == InvestigationPhase.VALIDATION

        # Should be able to transition to PLANNING
        assert state_machine.can_transition(InvestigationPhase.PLANNING)

        # Perform transition
        result = state_machine.transition(InvestigationPhase.PLANNING)
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.PLANNING

    def test_planning_to_complete_transition(self, state_machine):
        """Test valid transition from PLANNING to COMPLETE."""
        # Move through phases to PLANNING
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        assert state_machine.current_phase == InvestigationPhase.PLANNING

        # Should be able to transition to COMPLETE
        assert state_machine.can_transition(InvestigationPhase.COMPLETE)

        # Perform transition
        result = state_machine.transition(InvestigationPhase.COMPLETE)
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.COMPLETE
        assert state_machine.is_terminal()

    def test_validation_back_to_discovery(self, state_machine):
        """Test allowed backtrack from VALIDATION to DISCOVERY."""
        # Move to VALIDATION
        state_machine.transition(InvestigationPhase.VALIDATION)

        # Should be able to go back to DISCOVERY
        assert state_machine.can_transition(InvestigationPhase.DISCOVERY)

        # Perform transition
        result = state_machine.transition(
            InvestigationPhase.DISCOVERY, reason="Need more evidence"
        )
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_planning_back_to_discovery(self, state_machine):
        """Test allowed backtrack from PLANNING to DISCOVERY."""
        # Move through to PLANNING
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)

        # Should be able to go back to DISCOVERY
        assert state_machine.can_transition(InvestigationPhase.DISCOVERY)

        # Perform transition
        result = state_machine.transition(
            InvestigationPhase.DISCOVERY, reason="Found gaps"
        )
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_invalid_discovery_to_planning(self, state_machine):
        """Test invalid transition from DISCOVERY directly to PLANNING."""
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

        # Should NOT be able to transition directly to PLANNING
        assert not state_machine.can_transition(InvestigationPhase.PLANNING)

        # Attempting transition should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            state_machine.transition(InvestigationPhase.PLANNING)

        assert "Invalid phase transition" in str(excinfo.value)
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_invalid_discovery_to_complete(self, state_machine):
        """Test invalid transition from DISCOVERY directly to COMPLETE."""
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

        # Should NOT be able to transition directly to COMPLETE
        assert not state_machine.can_transition(InvestigationPhase.COMPLETE)

        # Attempting transition should raise ValueError
        with pytest.raises(ValueError):
            state_machine.transition(InvestigationPhase.COMPLETE)

        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_invalid_transition_from_complete(self, state_machine):
        """Test that COMPLETE is a terminal state."""
        # Move through to COMPLETE
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        state_machine.transition(InvestigationPhase.COMPLETE)

        # Should be terminal
        assert state_machine.is_terminal()

        # Should not be able to transition anywhere
        assert not state_machine.can_transition(InvestigationPhase.DISCOVERY)
        assert not state_machine.can_transition(InvestigationPhase.VALIDATION)
        assert not state_machine.can_transition(InvestigationPhase.PLANNING)

        # Attempting transition should raise ValueError
        with pytest.raises(ValueError):
            state_machine.transition(InvestigationPhase.DISCOVERY)

    def test_get_next_phase(self, state_machine):
        """Test getting the next phase in typical progression."""
        # From DISCOVERY, next should be VALIDATION
        next_phase = state_machine.get_next_phase()
        assert next_phase == InvestigationPhase.VALIDATION

        # From VALIDATION, next should be PLANNING (first option)
        state_machine.transition(InvestigationPhase.VALIDATION)
        next_phase = state_machine.get_next_phase()
        assert next_phase == InvestigationPhase.PLANNING

        # From PLANNING, next should be COMPLETE
        state_machine.transition(InvestigationPhase.PLANNING)
        next_phase = state_machine.get_next_phase()
        assert next_phase == InvestigationPhase.COMPLETE

        # From COMPLETE, next should be None
        state_machine.transition(InvestigationPhase.COMPLETE)
        next_phase = state_machine.get_next_phase()
        assert next_phase is None

    def test_get_valid_transitions(self, state_machine):
        """Test getting all valid transitions from each phase."""
        # From DISCOVERY
        valid = state_machine.get_valid_transitions()
        assert valid == [InvestigationPhase.VALIDATION]

        # From VALIDATION
        state_machine.transition(InvestigationPhase.VALIDATION)
        valid = state_machine.get_valid_transitions()
        assert InvestigationPhase.PLANNING in valid
        assert InvestigationPhase.DISCOVERY in valid

        # From PLANNING
        state_machine.transition(InvestigationPhase.PLANNING)
        valid = state_machine.get_valid_transitions()
        assert InvestigationPhase.COMPLETE in valid
        assert InvestigationPhase.DISCOVERY in valid

        # From COMPLETE (terminal)
        state_machine.transition(InvestigationPhase.COMPLETE)
        valid = state_machine.get_valid_transitions()
        assert valid == []

    def test_advance_to_next(self, state_machine):
        """Test convenience method to advance to next phase."""
        # From DISCOVERY, should advance to VALIDATION
        result = state_machine.advance_to_next("Testing")
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.VALIDATION

        # Continue advancing
        state_machine.advance_to_next()
        assert state_machine.current_phase == InvestigationPhase.PLANNING

        state_machine.advance_to_next()
        assert state_machine.current_phase == InvestigationPhase.COMPLETE

        # From terminal, should return False
        result = state_machine.advance_to_next()
        assert result is False
        assert state_machine.current_phase == InvestigationPhase.COMPLETE

    def test_reset_to_discovery(self, state_machine):
        """Test resetting investigation back to DISCOVERY phase."""
        # Move to PLANNING
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        assert state_machine.current_phase == InvestigationPhase.PLANNING

        # Reset to DISCOVERY
        result = state_machine.reset_to_discovery("Found gaps")
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

        # Can also reset from VALIDATION
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.reset_to_discovery()
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_reset_from_complete_raises_error(self, state_machine):
        """Test that resetting from COMPLETE phase raises error."""
        # Move to COMPLETE
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        state_machine.transition(InvestigationPhase.COMPLETE)

        # Reset should fail from COMPLETE
        with pytest.raises(ValueError) as excinfo:
            state_machine.reset_to_discovery()

        assert "Cannot reset from COMPLETE" in str(excinfo.value)
        assert state_machine.current_phase == InvestigationPhase.COMPLETE

    def test_reset_from_discovery_is_noop(self, state_machine):
        """Test that resetting from DISCOVERY returns True (no-op)."""
        # Already in DISCOVERY
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

        # Reset should return True but not change state
        result = state_machine.reset_to_discovery()
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY

    def test_transition_with_reason(self, state_machine):
        """Test that transition reason is logged."""
        # Transition with reason (just verify it doesn't error)
        result = state_machine.transition(
            InvestigationPhase.VALIDATION, reason="Sufficient evidence gathered"
        )
        assert result is True
        assert state_machine.current_phase == InvestigationPhase.VALIDATION


class TestConfidenceLevelProgression:
    """Test suite for confidence level progression logic."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine with initial state."""
        state = StudyState(
            investigation_id="test-123",
            session_id="session-456",
            current_phase=InvestigationPhase.DISCOVERY.value,
            confidence=ConfidenceLevel.EXPLORING.value,
        )
        return InvestigationStateMachine(state)

    def test_update_confidence(self, state_machine):
        """Test updating confidence level."""
        assert state_machine.state.confidence == ConfidenceLevel.EXPLORING.value

        # Update to LOW
        state_machine.update_confidence(ConfidenceLevel.LOW)
        assert state_machine.state.confidence == ConfidenceLevel.LOW.value

        # Update to HIGH
        state_machine.update_confidence(ConfidenceLevel.HIGH)
        assert state_machine.state.confidence == ConfidenceLevel.HIGH.value

    def test_should_escalate_from_discovery_low_confidence(self, state_machine):
        """Test that LOW confidence blocks escalation from DISCOVERY."""
        state_machine.state.confidence = ConfidenceLevel.LOW.value
        assert not state_machine.should_escalate_phase()

    def test_should_escalate_from_discovery_medium_confidence(self, state_machine):
        """Test that MEDIUM confidence allows escalation from DISCOVERY."""
        state_machine.state.confidence = ConfidenceLevel.MEDIUM.value
        assert state_machine.should_escalate_phase()

    def test_should_escalate_from_discovery_high_confidence(self, state_machine):
        """Test that HIGH confidence allows escalation from DISCOVERY."""
        state_machine.state.confidence = ConfidenceLevel.HIGH.value
        assert state_machine.should_escalate_phase()

    def test_should_escalate_from_validation_high_confidence(self, state_machine):
        """Test that HIGH confidence allows escalation from VALIDATION."""
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.state.confidence = ConfidenceLevel.HIGH.value
        assert state_machine.should_escalate_phase()

    def test_should_escalate_from_validation_medium_confidence(self, state_machine):
        """Test that MEDIUM confidence blocks escalation from VALIDATION."""
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.state.confidence = ConfidenceLevel.MEDIUM.value
        assert not state_machine.should_escalate_phase()

    def test_should_escalate_from_planning(self, state_machine):
        """Test escalation logic from PLANNING phase."""
        # Move to PLANNING
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)

        # MEDIUM confidence should not escalate
        state_machine.state.confidence = ConfidenceLevel.MEDIUM.value
        assert not state_machine.should_escalate_phase()

        # HIGH confidence should escalate
        state_machine.state.confidence = ConfidenceLevel.HIGH.value
        assert state_machine.should_escalate_phase()

    def test_should_escalate_from_complete(self, state_machine):
        """Test that escalation always returns False from COMPLETE."""
        # Move to COMPLETE
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        state_machine.transition(InvestigationPhase.COMPLETE)

        # Even with HIGH confidence, should not escalate (terminal state)
        state_machine.state.confidence = ConfidenceLevel.CERTAIN.value
        assert not state_machine.should_escalate_phase()

    def test_confidence_threshold_discovery(self, state_machine):
        """Test getting confidence threshold for DISCOVERY phase."""
        threshold = state_machine.get_confidence_threshold(InvestigationPhase.DISCOVERY)
        assert threshold == ConfidenceLevel.MEDIUM

    def test_confidence_threshold_validation(self, state_machine):
        """Test getting confidence threshold for VALIDATION phase."""
        threshold = state_machine.get_confidence_threshold(
            InvestigationPhase.VALIDATION
        )
        assert threshold == ConfidenceLevel.HIGH

    def test_confidence_threshold_planning(self, state_machine):
        """Test getting confidence threshold for PLANNING phase."""
        threshold = state_machine.get_confidence_threshold(InvestigationPhase.PLANNING)
        assert threshold == ConfidenceLevel.HIGH

    def test_confidence_threshold_complete(self, state_machine):
        """Test that COMPLETE phase has no escalation threshold."""
        threshold = state_machine.get_confidence_threshold(InvestigationPhase.COMPLETE)
        assert threshold is None

    def test_confidence_threshold_current_phase(self, state_machine):
        """Test getting threshold for current phase."""
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY
        threshold = state_machine.get_confidence_threshold()
        assert threshold == ConfidenceLevel.MEDIUM

        # Move to VALIDATION
        state_machine.transition(InvestigationPhase.VALIDATION)
        threshold = state_machine.get_confidence_threshold()
        assert threshold == ConfidenceLevel.HIGH


class TestStateTransitionIntegration:
    """Integration tests for state transitions with confidence."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine with initial state."""
        state = StudyState(
            investigation_id="test-123",
            session_id="session-456",
            current_phase=InvestigationPhase.DISCOVERY.value,
            confidence=ConfidenceLevel.EXPLORING.value,
        )
        return InvestigationStateMachine(state)

    def test_full_investigation_flow(self, state_machine):
        """Test complete investigation flow through all phases."""
        # DISCOVERY phase: gather initial evidence
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY
        state_machine.update_confidence(ConfidenceLevel.MEDIUM)

        # Ready to move to VALIDATION
        assert state_machine.should_escalate_phase()
        state_machine.transition(InvestigationPhase.VALIDATION)

        # VALIDATION phase: validate findings
        assert state_machine.current_phase == InvestigationPhase.VALIDATION
        state_machine.update_confidence(ConfidenceLevel.HIGH)

        # Ready to move to PLANNING
        assert state_machine.should_escalate_phase()
        state_machine.transition(InvestigationPhase.PLANNING)

        # PLANNING phase: synthesize and create plan
        assert state_machine.current_phase == InvestigationPhase.PLANNING

        # Complete investigation
        state_machine.transition(InvestigationPhase.COMPLETE)
        assert state_machine.is_terminal()

    def test_investigation_with_backtrack(self, state_machine):
        """Test investigation that backtracks and revisits phases."""
        # Progress to PLANNING
        state_machine.update_confidence(ConfidenceLevel.MEDIUM)
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.update_confidence(ConfidenceLevel.HIGH)
        state_machine.transition(InvestigationPhase.PLANNING)

        # Discover gap, backtrack to DISCOVERY
        state_machine.reset_to_discovery("Found inconsistency")

        # Re-investigate
        assert state_machine.current_phase == InvestigationPhase.DISCOVERY
        state_machine.update_confidence(ConfidenceLevel.HIGH)

        # Return to VALIDATION
        state_machine.transition(InvestigationPhase.VALIDATION)
        state_machine.transition(InvestigationPhase.PLANNING)
        state_machine.transition(InvestigationPhase.COMPLETE)

        assert state_machine.is_terminal()

    def test_multiple_confidence_updates_in_phase(self, state_machine):
        """Test updating confidence multiple times within a phase."""
        # In DISCOVERY phase, progressively increase confidence
        state_machine.update_confidence(ConfidenceLevel.EXPLORING)
        assert not state_machine.should_escalate_phase()

        state_machine.update_confidence(ConfidenceLevel.LOW)
        assert not state_machine.should_escalate_phase()

        state_machine.update_confidence(ConfidenceLevel.MEDIUM)
        assert state_machine.should_escalate_phase()  # Now ready

        # Transition and continue in new phase
        state_machine.transition(InvestigationPhase.VALIDATION)

        # In VALIDATION, need to reach HIGH
        state_machine.update_confidence(ConfidenceLevel.MEDIUM)
        assert not state_machine.should_escalate_phase()

        state_machine.update_confidence(ConfidenceLevel.HIGH)
        assert state_machine.should_escalate_phase()
