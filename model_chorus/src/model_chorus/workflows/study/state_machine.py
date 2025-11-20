"""
State machine for Study workflow investigation phases.

This module implements phase transition logic for persona-based investigations,
ensuring systematic progression through DISCOVERY → VALIDATION → PLANNING → COMPLETE.
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum

from ...core.models import InvestigationPhase, StudyState, ConfidenceLevel

logger = logging.getLogger(__name__)


class InvestigationStateMachine:
    """
    State machine for managing investigation phase transitions.

    Enforces valid phase transitions and provides methods for checking
    transition validity and progressing through investigation phases.

    Valid transitions:
        DISCOVERY → VALIDATION
        VALIDATION → PLANNING or VALIDATION → DISCOVERY (if more discovery needed)
        PLANNING → COMPLETE or PLANNING → DISCOVERY (if gaps found)
        COMPLETE → (terminal state)

    Attributes:
        current_phase: Current investigation phase
        state: StudyState instance being managed
    """

    # Define valid phase transitions
    VALID_TRANSITIONS: Dict[InvestigationPhase, List[InvestigationPhase]] = {
        InvestigationPhase.DISCOVERY: [InvestigationPhase.VALIDATION],
        InvestigationPhase.VALIDATION: [
            InvestigationPhase.PLANNING,
            InvestigationPhase.DISCOVERY,  # Allow returning to discovery if needed
        ],
        InvestigationPhase.PLANNING: [
            InvestigationPhase.COMPLETE,
            InvestigationPhase.DISCOVERY,  # Allow returning if gaps found
        ],
        InvestigationPhase.COMPLETE: [],  # Terminal state
    }

    def __init__(self, state: StudyState):
        """
        Initialize state machine with a StudyState instance.

        Args:
            state: StudyState instance to manage
        """
        self.state = state
        self.current_phase = InvestigationPhase(state.current_phase)
        logger.info(f"InvestigationStateMachine initialized in phase: {self.current_phase.value}")

    def can_transition(self, target_phase: InvestigationPhase) -> bool:
        """
        Check if transition to target phase is valid.

        Args:
            target_phase: Phase to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_targets = self.VALID_TRANSITIONS.get(self.current_phase, [])
        is_valid = target_phase in valid_targets

        if not is_valid:
            logger.warning(
                f"Invalid transition attempted: {self.current_phase.value} → {target_phase.value}"
            )

        return is_valid

    def transition(self, target_phase: InvestigationPhase, reason: Optional[str] = None) -> bool:
        """
        Transition to target phase if valid.

        Args:
            target_phase: Phase to transition to
            reason: Optional reason for transition (for logging)

        Returns:
            True if transition succeeded, False otherwise

        Raises:
            ValueError: If transition is invalid
        """
        if not self.can_transition(target_phase):
            raise ValueError(
                f"Invalid phase transition: {self.current_phase.value} → {target_phase.value}. "
                f"Valid transitions from {self.current_phase.value}: "
                f"{[p.value for p in self.VALID_TRANSITIONS.get(self.current_phase, [])]}"
            )

        old_phase = self.current_phase
        self.current_phase = target_phase
        self.state.current_phase = target_phase.value

        log_msg = f"Phase transition: {old_phase.value} → {target_phase.value}"
        if reason:
            log_msg += f" (Reason: {reason})"
        logger.info(log_msg)

        return True

    def get_next_phase(self) -> Optional[InvestigationPhase]:
        """
        Get the primary next phase in the investigation flow.

        Returns the first valid transition option, representing the
        typical "forward" progression path.

        Returns:
            Next phase in typical progression, or None if in terminal state
        """
        valid_targets = self.VALID_TRANSITIONS.get(self.current_phase, [])

        if not valid_targets:
            # Terminal state (COMPLETE)
            return None

        # Return first option as the "primary" next phase
        return valid_targets[0]

    def get_valid_transitions(self) -> List[InvestigationPhase]:
        """
        Get all valid transitions from current phase.

        Returns:
            List of phases that can be transitioned to from current phase
        """
        return self.VALID_TRANSITIONS.get(self.current_phase, [])

    def is_terminal(self) -> bool:
        """
        Check if current phase is a terminal state.

        Returns:
            True if current phase is COMPLETE (terminal), False otherwise
        """
        return self.current_phase == InvestigationPhase.COMPLETE

    def advance_to_next(self, reason: Optional[str] = None) -> bool:
        """
        Convenience method to advance to the primary next phase.

        Args:
            reason: Optional reason for advancement

        Returns:
            True if advanced successfully, False if already in terminal state

        Raises:
            ValueError: If no valid next phase exists (shouldn't happen with proper logic)
        """
        next_phase = self.get_next_phase()

        if next_phase is None:
            logger.info("Already in terminal state (COMPLETE)")
            return False

        return self.transition(next_phase, reason)

    def reset_to_discovery(self, reason: Optional[str] = None) -> bool:
        """
        Reset investigation to DISCOVERY phase.

        Can be called from VALIDATION or PLANNING phases when additional
        exploration is needed.

        Args:
            reason: Reason for resetting to discovery

        Returns:
            True if reset succeeded

        Raises:
            ValueError: If resetting from DISCOVERY or COMPLETE (invalid)
        """
        if self.current_phase == InvestigationPhase.DISCOVERY:
            logger.warning("Already in DISCOVERY phase, no reset needed")
            return True

        if self.current_phase == InvestigationPhase.COMPLETE:
            raise ValueError("Cannot reset from COMPLETE phase")

        return self.transition(
            InvestigationPhase.DISCOVERY, reason or "Resetting for additional exploration"
        )

    def update_confidence(self, new_confidence: ConfidenceLevel) -> None:
        """
        Update the confidence level in the investigation state.

        Args:
            new_confidence: New confidence level to set
        """
        old_confidence = ConfidenceLevel(self.state.confidence)
        self.state.confidence = new_confidence.value

        logger.info(f"Confidence updated: {old_confidence.value} → {new_confidence.value}")

    def should_escalate_phase(self) -> bool:
        """
        Determine if confidence level is sufficient to escalate to next phase.

        Uses confidence thresholds appropriate for each phase:
        - DISCOVERY: Requires MEDIUM or higher to move to VALIDATION
        - VALIDATION: Requires HIGH or higher to move to PLANNING
        - PLANNING: Requires HIGH or higher to move to COMPLETE
        - COMPLETE: Already terminal, returns False

        Returns:
            True if confidence is sufficient for phase escalation, False otherwise
        """
        current_confidence = ConfidenceLevel(self.state.confidence)

        # Define confidence thresholds for phase transitions
        thresholds = {
            InvestigationPhase.DISCOVERY: ConfidenceLevel.MEDIUM,
            InvestigationPhase.VALIDATION: ConfidenceLevel.HIGH,
            InvestigationPhase.PLANNING: ConfidenceLevel.HIGH,
        }

        # COMPLETE is terminal, no escalation possible
        if self.current_phase == InvestigationPhase.COMPLETE:
            return False

        required_confidence = thresholds.get(self.current_phase)
        if required_confidence is None:
            return False

        # Get confidence level ordering for comparison
        confidence_order = {
            ConfidenceLevel.EXPLORING: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.VERY_HIGH: 4,
            ConfidenceLevel.ALMOST_CERTAIN: 5,
            ConfidenceLevel.CERTAIN: 6,
        }

        current_level = confidence_order.get(current_confidence, 0)
        required_level = confidence_order.get(required_confidence, 0)

        should_escalate = current_level >= required_level

        if should_escalate:
            logger.info(
                f"Confidence level {current_confidence.value} meets threshold "
                f"for escalation from {self.current_phase.value}"
            )
        else:
            logger.debug(
                f"Confidence level {current_confidence.value} below threshold "
                f"{required_confidence.value} for {self.current_phase.value} escalation"
            )

        return should_escalate

    def get_confidence_threshold(
        self, phase: Optional[InvestigationPhase] = None
    ) -> Optional[ConfidenceLevel]:
        """
        Get the confidence threshold required for escalation from a given phase.

        Args:
            phase: Phase to get threshold for (defaults to current phase)

        Returns:
            Required confidence level for escalation, or None if phase is terminal
        """
        target_phase = phase or self.current_phase

        thresholds = {
            InvestigationPhase.DISCOVERY: ConfidenceLevel.MEDIUM,
            InvestigationPhase.VALIDATION: ConfidenceLevel.HIGH,
            InvestigationPhase.PLANNING: ConfidenceLevel.HIGH,
        }

        return thresholds.get(target_phase)
