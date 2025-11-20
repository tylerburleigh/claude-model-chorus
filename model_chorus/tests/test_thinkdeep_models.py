"""
Unit tests for ThinkDeep workflow models.

Tests verify ThinkDeep model functionality including:
- ConfidenceLevel enum values and progression
- Hypothesis model creation, validation, and evidence tracking
- InvestigationStep model with findings and confidence
- ThinkDeepState model with hypothesis and step management
- Model serialization and deserialization
- Edge cases and validation errors
"""

import json
import pytest
from pydantic import ValidationError

from model_chorus.core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)


class TestConfidenceLevel:
    """Test suite for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """Test all confidence level enum values exist."""
        assert ConfidenceLevel.EXPLORING.value == "exploring"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.ALMOST_CERTAIN.value == "almost_certain"
        assert ConfidenceLevel.CERTAIN.value == "certain"

    def test_confidence_level_count(self):
        """Test that we have exactly 7 confidence levels."""
        levels = list(ConfidenceLevel)
        assert len(levels) == 7

    def test_confidence_level_progression(self):
        """Test logical progression of confidence levels."""
        progression = [
            ConfidenceLevel.EXPLORING,
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
            ConfidenceLevel.ALMOST_CERTAIN,
            ConfidenceLevel.CERTAIN,
        ]

        # Verify we can iterate through progression
        for i, level in enumerate(progression):
            assert isinstance(level, ConfidenceLevel)
            assert level.value in [
                "exploring",
                "low",
                "medium",
                "high",
                "very_high",
                "almost_certain",
                "certain",
            ]

    def test_confidence_level_string_representation(self):
        """Test string conversion of confidence levels."""
        assert str(ConfidenceLevel.EXPLORING.value) == "exploring"
        assert str(ConfidenceLevel.CERTAIN.value) == "certain"


class TestHypothesis:
    """Test suite for Hypothesis model."""

    def test_hypothesis_creation(self):
        """Test basic hypothesis creation."""
        hyp = Hypothesis(
            hypothesis="API uses async/await pattern",
            evidence=["Found async def in auth.py"],
            status="active",
        )

        assert hyp.hypothesis == "API uses async/await pattern"
        assert len(hyp.evidence) == 1
        assert hyp.evidence[0] == "Found async def in auth.py"
        assert hyp.status == "active"

    def test_hypothesis_default_values(self):
        """Test hypothesis with default values."""
        hyp = Hypothesis(hypothesis="Test hypothesis")

        assert hyp.hypothesis == "Test hypothesis"
        assert hyp.evidence == []  # Default empty list
        assert hyp.status == "active"  # Default status

    def test_hypothesis_all_statuses(self):
        """Test all valid hypothesis statuses."""
        statuses = ["active", "disproven", "validated"]

        for status in statuses:
            hyp = Hypothesis(hypothesis="Test", status=status)
            assert hyp.status == status

    def test_hypothesis_invalid_status(self):
        """Test that invalid status raises validation error."""
        with pytest.raises(ValidationError):
            Hypothesis(hypothesis="Test", status="invalid_status")  # Not in allowed values

    def test_hypothesis_empty_hypothesis_text(self):
        """Test that empty hypothesis text fails validation."""
        with pytest.raises(ValidationError):
            Hypothesis(hypothesis="")  # min_length=1 constraint

    def test_hypothesis_with_multiple_evidence(self):
        """Test hypothesis with multiple evidence items."""
        evidence_items = [
            "Found async def in auth.py line 45",
            "Tests use asyncio.run() in test_auth.py",
            "No callback patterns found in service layer",
            "Database client uses async connection pool",
        ]

        hyp = Hypothesis(
            hypothesis="System uses async/await throughout",
            evidence=evidence_items,
            status="validated",
        )

        assert len(hyp.evidence) == 4
        assert hyp.evidence == evidence_items

    def test_hypothesis_serialization(self):
        """Test hypothesis can be serialized to dict."""
        hyp = Hypothesis(
            hypothesis="Memory leak in cache layer",
            evidence=["Cache grows unbounded", "No eviction policy"],
            status="active",
        )

        data = hyp.model_dump()

        assert data["hypothesis"] == "Memory leak in cache layer"
        assert len(data["evidence"]) == 2
        assert data["status"] == "active"

    def test_hypothesis_json_serialization(self):
        """Test hypothesis can be serialized to JSON."""
        hyp = Hypothesis(
            hypothesis="Test hypothesis", evidence=["Evidence 1", "Evidence 2"], status="validated"
        )

        json_str = hyp.model_dump_json()
        assert isinstance(json_str, str)

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert parsed["hypothesis"] == "Test hypothesis"
        assert len(parsed["evidence"]) == 2

    def test_hypothesis_from_dict(self):
        """Test creating hypothesis from dictionary."""
        data = {
            "hypothesis": "Rate limiting not working",
            "evidence": ["Requests exceed limit", "No 429 errors seen"],
            "status": "active",
        }

        hyp = Hypothesis(**data)

        assert hyp.hypothesis == data["hypothesis"]
        assert hyp.evidence == data["evidence"]
        assert hyp.status == data["status"]


class TestInvestigationStep:
    """Test suite for InvestigationStep model."""

    def test_investigation_step_creation(self):
        """Test basic investigation step creation."""
        step = InvestigationStep(
            step_number=1,
            findings="Found async/await pattern in auth service",
            files_checked=["src/auth.py"],
            confidence="medium",
        )

        assert step.step_number == 1
        assert step.findings == "Found async/await pattern in auth service"
        assert step.files_checked == ["src/auth.py"]
        assert step.confidence == "medium"

    def test_investigation_step_default_files(self):
        """Test investigation step with default empty files list."""
        step = InvestigationStep(
            step_number=1, findings="Initial analysis complete", confidence="low"
        )

        assert step.files_checked == []  # Default empty list

    def test_investigation_step_multiple_files(self):
        """Test investigation step with multiple files."""
        files = [
            "src/services/auth.py",
            "src/api/users.py",
            "tests/test_auth.py",
            "config/database.py",
        ]

        step = InvestigationStep(
            step_number=2,
            findings="Analyzed authentication flow across components",
            files_checked=files,
            confidence="high",
        )

        assert len(step.files_checked) == 4
        assert step.files_checked == files

    def test_investigation_step_number_validation(self):
        """Test that step_number must be >= 1."""
        # Valid step numbers
        step1 = InvestigationStep(step_number=1, findings="Test", confidence="low")
        assert step1.step_number == 1

        step100 = InvestigationStep(step_number=100, findings="Test", confidence="high")
        assert step100.step_number == 100

        # Invalid step numbers
        with pytest.raises(ValidationError):
            InvestigationStep(step_number=0, findings="Test", confidence="low")  # Less than 1

        with pytest.raises(ValidationError):
            InvestigationStep(step_number=-1, findings="Test", confidence="low")  # Negative

    def test_investigation_step_empty_findings(self):
        """Test that empty findings fails validation."""
        with pytest.raises(ValidationError):
            InvestigationStep(
                step_number=1, findings="", confidence="low"  # min_length=1 constraint
            )

    def test_investigation_step_empty_confidence(self):
        """Test that empty confidence fails validation."""
        with pytest.raises(ValidationError):
            InvestigationStep(
                step_number=1, findings="Test findings", confidence=""  # min_length=1 constraint
            )

    def test_investigation_step_serialization(self):
        """Test investigation step serialization to dict."""
        step = InvestigationStep(
            step_number=3,
            findings="Identified root cause in database layer",
            files_checked=["src/db/connection.py", "src/db/pool.py"],
            confidence="very_high",
        )

        data = step.model_dump()

        assert data["step_number"] == 3
        assert data["findings"] == "Identified root cause in database layer"
        assert len(data["files_checked"]) == 2
        assert data["confidence"] == "very_high"

    def test_investigation_step_json_roundtrip(self):
        """Test investigation step JSON serialization roundtrip."""
        original = InvestigationStep(
            step_number=5,
            findings="Comprehensive analysis completed",
            files_checked=["file1.py", "file2.py", "file3.py"],
            confidence="almost_certain",
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        data = json.loads(json_str)
        restored = InvestigationStep(**data)

        assert restored.step_number == original.step_number
        assert restored.findings == original.findings
        assert restored.files_checked == original.files_checked
        assert restored.confidence == original.confidence


class TestThinkDeepState:
    """Test suite for ThinkDeepState model."""

    def test_thinkdeep_state_creation(self):
        """Test basic ThinkDeepState creation."""
        hyp = Hypothesis(hypothesis="Test hypothesis", evidence=["Evidence 1"], status="active")

        step = InvestigationStep(step_number=1, findings="Initial findings", confidence="low")

        state = ThinkDeepState(
            hypotheses=[hyp], steps=[step], current_confidence="low", relevant_files=["test.py"]
        )

        assert len(state.hypotheses) == 1
        assert len(state.steps) == 1
        assert state.current_confidence == "low"
        assert state.relevant_files == ["test.py"]

    def test_thinkdeep_state_default_values(self):
        """Test ThinkDeepState with default empty values."""
        state = ThinkDeepState()

        assert state.hypotheses == []
        assert state.steps == []
        assert state.current_confidence == "exploring"  # Default
        assert state.relevant_files == []

    def test_thinkdeep_state_multiple_hypotheses(self):
        """Test state with multiple hypotheses in different statuses."""
        hypotheses = [
            Hypothesis(hypothesis="H1", status="active"),
            Hypothesis(hypothesis="H2", status="validated"),
            Hypothesis(hypothesis="H3", status="disproven"),
            Hypothesis(hypothesis="H4", status="active"),
        ]

        state = ThinkDeepState(hypotheses=hypotheses)

        assert len(state.hypotheses) == 4
        assert state.hypotheses[1].status == "validated"
        assert state.hypotheses[2].status == "disproven"

    def test_thinkdeep_state_multiple_steps(self):
        """Test state with progression of investigation steps."""
        steps = [
            InvestigationStep(
                step_number=1, findings="Initial exploration", confidence="exploring"
            ),
            InvestigationStep(step_number=2, findings="Hypothesis formed", confidence="low"),
            InvestigationStep(step_number=3, findings="Evidence gathered", confidence="medium"),
            InvestigationStep(step_number=4, findings="Hypothesis validated", confidence="high"),
        ]

        state = ThinkDeepState(steps=steps, current_confidence="high")

        assert len(state.steps) == 4
        assert state.steps[0].confidence == "exploring"
        assert state.steps[-1].confidence == "high"
        assert state.current_confidence == "high"

    def test_thinkdeep_state_file_accumulation(self):
        """Test accumulation of relevant files over investigation."""
        files = [
            "src/auth.py",
            "src/database.py",
            "tests/test_auth.py",
            "config/settings.py",
            "src/middleware/auth.py",
        ]

        state = ThinkDeepState(relevant_files=files)

        assert len(state.relevant_files) == 5
        assert "src/auth.py" in state.relevant_files
        assert "config/settings.py" in state.relevant_files

    def test_thinkdeep_state_complex_scenario(self):
        """Test complete investigation scenario with all components."""
        # Create multiple hypotheses
        hypotheses = [
            Hypothesis(
                hypothesis="Database connection pooling issue",
                evidence=[
                    "Connection pool size is 5",
                    "Peak load requires 20 connections",
                    "Timeout errors during peak hours",
                ],
                status="validated",
            ),
            Hypothesis(
                hypothesis="Memory leak in cache layer",
                evidence=["Memory usage grows over time", "Cache has no eviction policy"],
                status="active",
            ),
        ]

        # Create investigation steps
        steps = [
            InvestigationStep(
                step_number=1,
                findings="Examined database configuration",
                files_checked=["config/database.py"],
                confidence="low",
            ),
            InvestigationStep(
                step_number=2,
                findings="Analyzed connection pool behavior under load",
                files_checked=["src/db/pool.py", "tests/load/test_db.py"],
                confidence="medium",
            ),
            InvestigationStep(
                step_number=3,
                findings="Confirmed pool size insufficient for peak load",
                files_checked=["logs/production.log"],
                confidence="high",
            ),
        ]

        # Create state
        state = ThinkDeepState(
            hypotheses=hypotheses,
            steps=steps,
            current_confidence="high",
            relevant_files=[
                "config/database.py",
                "src/db/pool.py",
                "tests/load/test_db.py",
                "logs/production.log",
            ],
        )

        # Verify complete state
        assert len(state.hypotheses) == 2
        assert len(state.steps) == 3
        assert state.current_confidence == "high"
        assert len(state.relevant_files) == 4

        # Verify hypothesis details
        validated_hyps = [h for h in state.hypotheses if h.status == "validated"]
        assert len(validated_hyps) == 1
        assert len(validated_hyps[0].evidence) == 3

        # Verify step progression
        assert state.steps[0].confidence == "low"
        assert state.steps[1].confidence == "medium"
        assert state.steps[2].confidence == "high"

    def test_thinkdeep_state_serialization(self):
        """Test ThinkDeepState serialization to dict."""
        hyp = Hypothesis(hypothesis="Test", status="active")
        step = InvestigationStep(step_number=1, findings="Test", confidence="low")

        state = ThinkDeepState(
            hypotheses=[hyp], steps=[step], current_confidence="medium", relevant_files=["test.py"]
        )

        data = state.model_dump()

        assert "hypotheses" in data
        assert "steps" in data
        assert data["current_confidence"] == "medium"
        assert data["relevant_files"] == ["test.py"]

    def test_thinkdeep_state_json_roundtrip(self):
        """Test complete JSON serialization roundtrip."""
        # Create original state
        original = ThinkDeepState(
            hypotheses=[
                Hypothesis(hypothesis="H1", evidence=["E1", "E2"], status="validated"),
                Hypothesis(hypothesis="H2", status="active"),
            ],
            steps=[
                InvestigationStep(
                    step_number=1, findings="F1", files_checked=["f1.py"], confidence="low"
                ),
                InvestigationStep(
                    step_number=2,
                    findings="F2",
                    files_checked=["f2.py", "f3.py"],
                    confidence="medium",
                ),
            ],
            current_confidence="medium",
            relevant_files=["f1.py", "f2.py", "f3.py"],
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        data = json.loads(json_str)
        restored = ThinkDeepState(**data)

        # Verify hypotheses
        assert len(restored.hypotheses) == 2
        assert restored.hypotheses[0].hypothesis == "H1"
        assert len(restored.hypotheses[0].evidence) == 2
        assert restored.hypotheses[0].status == "validated"

        # Verify steps
        assert len(restored.steps) == 2
        assert restored.steps[0].step_number == 1
        assert restored.steps[1].step_number == 2
        assert len(restored.steps[1].files_checked) == 2

        # Verify other fields
        assert restored.current_confidence == "medium"
        assert len(restored.relevant_files) == 3

    def test_thinkdeep_state_nested_validation(self):
        """Test that nested model validation works correctly."""
        # Invalid hypothesis in state
        with pytest.raises(ValidationError):
            ThinkDeepState(
                hypotheses=[
                    Hypothesis(hypothesis="Valid"),
                    Hypothesis(hypothesis="", status="active"),  # Empty hypothesis
                ]
            )

        # Invalid step in state
        with pytest.raises(ValidationError):
            ThinkDeepState(
                steps=[
                    InvestigationStep(
                        step_number=0, findings="Test", confidence="low"  # Invalid: < 1
                    )
                ]
            )


class TestModelIntegration:
    """Test integration scenarios using multiple models together."""

    def test_hypothesis_lifecycle(self):
        """Test typical hypothesis lifecycle in an investigation."""
        # Step 1: Form hypothesis
        hyp = Hypothesis(
            hypothesis="API timeout caused by slow database queries", evidence=[], status="active"
        )

        # Step 2: Add evidence
        hyp.evidence.append("Found 5-second query in logs")
        hyp.evidence.append("Query lacks proper index")

        # Step 3: Validate
        hyp.status = "validated"

        assert len(hyp.evidence) == 2
        assert hyp.status == "validated"

    def test_investigation_progression(self):
        """Test complete investigation from start to completion."""
        state = ThinkDeepState()

        # Initial state
        assert state.current_confidence == "exploring"
        assert len(state.hypotheses) == 0
        assert len(state.steps) == 0

        # Step 1: Form hypothesis
        hyp1 = Hypothesis(hypothesis="Memory leak in cache", status="active")
        state.hypotheses.append(hyp1)
        state.steps.append(
            InvestigationStep(step_number=1, findings="Observed memory growth", confidence="low")
        )
        state.current_confidence = "low"

        # Step 2: Gather evidence
        hyp1.evidence.append("Memory increases 10% per hour")
        hyp1.evidence.append("No eviction policy configured")
        state.steps.append(
            InvestigationStep(
                step_number=2, findings="Confirmed no cache eviction", confidence="medium"
            )
        )
        state.current_confidence = "medium"

        # Step 3: Validate and conclude
        hyp1.status = "validated"
        state.steps.append(
            InvestigationStep(
                step_number=3, findings="Root cause identified and confirmed", confidence="high"
            )
        )
        state.current_confidence = "high"

        # Verify final state
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].status == "validated"
        assert len(state.hypotheses[0].evidence) == 2
        assert len(state.steps) == 3
        assert state.current_confidence == "high"

    def test_multiple_hypothesis_tracking(self):
        """Test tracking multiple competing hypotheses."""
        state = ThinkDeepState()

        # Add multiple hypotheses
        state.hypotheses.extend(
            [
                Hypothesis(hypothesis="H1: Database issue", status="active"),
                Hypothesis(hypothesis="H2: Network latency", status="active"),
                Hypothesis(hypothesis="H3: Code bug", status="active"),
            ]
        )

        # Investigation proves H1, disproves H2, H3 remains active
        state.hypotheses[0].status = "validated"
        state.hypotheses[0].evidence.append("Confirmed DB slow queries")

        state.hypotheses[1].status = "disproven"
        state.hypotheses[1].evidence.append("Network metrics normal")

        # Count by status
        validated = [h for h in state.hypotheses if h.status == "validated"]
        disproven = [h for h in state.hypotheses if h.status == "disproven"]
        active = [h for h in state.hypotheses if h.status == "active"]

        assert len(validated) == 1
        assert len(disproven) == 1
        assert len(active) == 1
