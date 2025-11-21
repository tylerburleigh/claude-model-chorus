"""
Unit tests for WorkflowRegistry module.

Tests verify workflow registration, retrieval, and metadata management including:
- Workflow registration via decorator and programmatic methods
- Workflow retrieval and listing
- Metadata registration and querying
- Error handling for invalid registrations
- Registry state management (clear, unregister)
"""

import inspect

import pytest

from model_chorus.core.base_workflow import BaseWorkflow, WorkflowResult
from model_chorus.core.models import WorkflowMetadata
from model_chorus.core.registry import WorkflowRegistry


# Helper workflow implementations for testing (renamed to avoid pytest collection)
class MockWorkflow(BaseWorkflow):
    """Minimal mock workflow implementation."""

    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        """Minimal run implementation for testing."""
        result = WorkflowResult(success=True, synthesis=f"Test response to: {prompt}")
        result.add_step(1, "Test step", "test-model")
        return result


class AnotherMockWorkflow(BaseWorkflow):
    """Another mock workflow for multiple registration tests."""

    async def run(self, prompt: str, **kwargs) -> WorkflowResult:
        """Minimal run implementation for testing."""
        return WorkflowResult(success=True, synthesis="Another test response")


class InvalidWorkflow:
    """Invalid workflow that doesn't inherit from BaseWorkflow."""

    async def run(self, prompt: str, **kwargs):
        """This doesn't return WorkflowResult."""
        return "invalid"


class TestWorkflowRegistryBasics:
    """Test suite for basic WorkflowRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        WorkflowRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        WorkflowRegistry.clear()

    # ========================================================================
    # Registration Tests (Decorator)
    # ========================================================================

    def test_register_decorator_basic(self):
        """Test registering workflow with decorator."""

        @WorkflowRegistry.register("test-workflow")
        class DecoratorWorkflow(BaseWorkflow):
            async def run(self, prompt: str, **kwargs) -> WorkflowResult:
                return WorkflowResult(success=True)

        assert WorkflowRegistry.is_registered("test-workflow")
        assert WorkflowRegistry.get("test-workflow") == DecoratorWorkflow

    def test_register_decorator_returns_class(self):
        """Test that decorator returns the original class."""

        @WorkflowRegistry.register("test-workflow")
        class DecoratorWorkflow(BaseWorkflow):
            async def run(self, prompt: str, **kwargs) -> WorkflowResult:
                return WorkflowResult(success=True)

        # Should be able to instantiate directly
        workflow = DecoratorWorkflow("test", "description")
        assert workflow.name == "test"
        assert workflow.description == "description"

    def test_register_decorator_duplicate_name_raises_error(self):
        """Test that registering duplicate workflow name raises ValueError."""

        @WorkflowRegistry.register("duplicate")
        class FirstWorkflow(BaseWorkflow):
            async def run(self, prompt: str, **kwargs) -> WorkflowResult:
                return WorkflowResult(success=True)

        with pytest.raises(ValueError, match="already registered"):

            @WorkflowRegistry.register("duplicate")
            class SecondWorkflow(BaseWorkflow):
                async def run(self, prompt: str, **kwargs) -> WorkflowResult:
                    return WorkflowResult(success=True)

    def test_register_decorator_non_baseflow_raises_error(self):
        """Test that registering non-BaseWorkflow class raises TypeError."""
        with pytest.raises(TypeError, match="must inherit from BaseWorkflow"):

            @WorkflowRegistry.register("invalid")
            class InvalidClass:
                pass

    def test_register_decorator_non_class_raises_error(self):
        """Test that registering non-class raises TypeError."""
        with pytest.raises(TypeError, match="Expected a class"):

            @WorkflowRegistry.register("invalid")
            def not_a_class():
                pass

    # ========================================================================
    # Programmatic Registration Tests
    # ========================================================================

    def test_register_workflow_programmatic(self):
        """Test programmatic workflow registration."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)

        assert WorkflowRegistry.is_registered("test")
        assert WorkflowRegistry.get("test") == MockWorkflow

    def test_register_workflow_duplicate_raises_error(self):
        """Test programmatic registration with duplicate name raises ValueError."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)

        with pytest.raises(ValueError, match="already registered"):
            WorkflowRegistry.register_workflow("test", AnotherMockWorkflow)

    def test_register_workflow_non_baseflow_raises_error(self):
        """Test programmatic registration of non-BaseWorkflow raises TypeError."""
        with pytest.raises(TypeError, match="must inherit from BaseWorkflow"):
            WorkflowRegistry.register_workflow("invalid", InvalidWorkflow)

    def test_register_workflow_non_class_raises_error(self):
        """Test programmatic registration of non-class raises TypeError."""
        with pytest.raises(TypeError, match="Expected a class"):
            WorkflowRegistry.register_workflow("invalid", "not-a-class")

    # ========================================================================
    # Workflow Retrieval Tests
    # ========================================================================

    def test_get_workflow_exists(self):
        """Test retrieving registered workflow."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)

        workflow_class = WorkflowRegistry.get("test")
        assert workflow_class == MockWorkflow

        # Should be able to instantiate
        workflow = workflow_class("name", "desc")
        assert isinstance(workflow, BaseWorkflow)

    def test_get_workflow_not_exists_raises_error(self):
        """Test retrieving non-existent workflow raises KeyError."""
        with pytest.raises(KeyError, match="No workflow registered with name 'nonexistent'"):
            WorkflowRegistry.get("nonexistent")

    def test_get_workflow_error_lists_available(self):
        """Test that KeyError lists available workflows."""
        WorkflowRegistry.register_workflow("workflow-1", MockWorkflow)
        WorkflowRegistry.register_workflow("workflow-2", AnotherMockWorkflow)

        with pytest.raises(KeyError, match="Available workflows: workflow-1, workflow-2"):
            WorkflowRegistry.get("nonexistent")

    def test_get_workflow_error_when_empty_registry(self):
        """Test that KeyError shows 'none' when registry is empty."""
        with pytest.raises(KeyError, match="Available workflows: none"):
            WorkflowRegistry.get("any")

    # ========================================================================
    # Workflow Listing Tests
    # ========================================================================

    def test_list_workflows_empty(self):
        """Test listing workflows when registry is empty."""
        workflows = WorkflowRegistry.list_workflows()
        assert workflows == []

    def test_list_workflows_single(self):
        """Test listing workflows with one registered."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)

        workflows = WorkflowRegistry.list_workflows()
        assert workflows == ["test"]

    def test_list_workflows_multiple_sorted(self):
        """Test that list_workflows returns sorted names."""
        WorkflowRegistry.register_workflow("zebra", MockWorkflow)
        WorkflowRegistry.register_workflow("alpha", AnotherMockWorkflow)
        WorkflowRegistry.register_workflow("beta", MockWorkflow)

        workflows = WorkflowRegistry.list_workflows()
        assert workflows == ["alpha", "beta", "zebra"]

    # ========================================================================
    # is_registered Tests
    # ========================================================================

    def test_is_registered_true(self):
        """Test is_registered returns True for registered workflow."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        assert WorkflowRegistry.is_registered("test") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered workflow."""
        assert WorkflowRegistry.is_registered("nonexistent") is False

    # ========================================================================
    # Unregister Tests
    # ========================================================================

    def test_unregister_workflow(self):
        """Test unregistering a workflow."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        assert WorkflowRegistry.is_registered("test")

        WorkflowRegistry.unregister("test")
        assert not WorkflowRegistry.is_registered("test")

    def test_unregister_nonexistent_raises_error(self):
        """Test unregistering non-existent workflow raises KeyError."""
        with pytest.raises(KeyError, match="No workflow registered with name 'nonexistent'"):
            WorkflowRegistry.unregister("nonexistent")

    def test_unregister_removes_metadata(self):
        """Test that unregister also removes associated metadata."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        WorkflowRegistry.register_metadata("test", "Test workflow", version="1.0.0")

        assert WorkflowRegistry.get_workflow_info("test") is not None

        WorkflowRegistry.unregister("test")

        assert not WorkflowRegistry.is_registered("test")
        assert WorkflowRegistry.get_workflow_info("test") is None

    # ========================================================================
    # Clear Tests
    # ========================================================================

    def test_clear_removes_all_workflows(self):
        """Test that clear removes all registered workflows."""
        WorkflowRegistry.register_workflow("workflow-1", MockWorkflow)
        WorkflowRegistry.register_workflow("workflow-2", AnotherMockWorkflow)

        assert len(WorkflowRegistry.list_workflows()) == 2

        WorkflowRegistry.clear()

        assert len(WorkflowRegistry.list_workflows()) == 0
        assert not WorkflowRegistry.is_registered("workflow-1")
        assert not WorkflowRegistry.is_registered("workflow-2")

    def test_clear_removes_all_metadata(self):
        """Test that clear removes all metadata."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        WorkflowRegistry.register_metadata("test", "Test workflow")

        assert WorkflowRegistry.get_workflow_info("test") is not None

        WorkflowRegistry.clear()

        assert WorkflowRegistry.get_workflow_info("test") is None


class TestWorkflowMetadata:
    """Test suite for workflow metadata functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        WorkflowRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        WorkflowRegistry.clear()

    # ========================================================================
    # Metadata Registration Tests
    # ========================================================================

    def test_register_metadata_basic(self):
        """Test registering basic metadata."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        WorkflowRegistry.register_metadata("test", "Test workflow description")

        metadata = WorkflowRegistry.get_workflow_info("test")

        assert metadata is not None
        assert metadata.name == "test"
        assert metadata.description == "Test workflow description"
        assert metadata.version == "1.0.0"  # Default
        assert metadata.author == "Unknown"  # Default

    def test_register_metadata_full(self):
        """Test registering complete metadata with all fields."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        WorkflowRegistry.register_metadata(
            "test",
            description="Full workflow description",
            version="2.5.1",
            author="Test Author",
            category="testing",
            parameters=["param1", "param2"],
            examples=["example 1", "example 2"],
        )

        metadata = WorkflowRegistry.get_workflow_info("test")

        assert metadata is not None
        assert metadata.name == "test"
        assert metadata.description == "Full workflow description"
        assert metadata.version == "2.5.1"
        assert metadata.author == "Test Author"
        assert metadata.category == "testing"
        assert metadata.parameters == ["param1", "param2"]
        assert metadata.examples == ["example 1", "example 2"]

    def test_register_metadata_without_workflow(self):
        """Test that metadata can be registered independently of workflow."""
        # Metadata can be registered without workflow being registered first
        WorkflowRegistry.register_metadata("orphan", "Orphan workflow")

        metadata = WorkflowRegistry.get_workflow_info("orphan")
        assert metadata is not None
        assert metadata.name == "orphan"

    def test_register_metadata_updates_existing(self):
        """Test that registering metadata updates existing metadata."""
        WorkflowRegistry.register_metadata("test", "First description", version="1.0.0")

        metadata = WorkflowRegistry.get_workflow_info("test")
        assert metadata.description == "First description"
        assert metadata.version == "1.0.0"

        # Update metadata
        WorkflowRegistry.register_metadata("test", "Updated description", version="2.0.0")

        metadata = WorkflowRegistry.get_workflow_info("test")
        assert metadata.description == "Updated description"
        assert metadata.version == "2.0.0"

    # ========================================================================
    # Metadata Retrieval Tests
    # ========================================================================

    def test_get_workflow_info_exists(self):
        """Test retrieving existing workflow metadata."""
        WorkflowRegistry.register_metadata("test", "Test workflow")

        metadata = WorkflowRegistry.get_workflow_info("test")

        assert isinstance(metadata, WorkflowMetadata)
        assert metadata.name == "test"
        assert metadata.description == "Test workflow"

    def test_get_workflow_info_not_exists(self):
        """Test retrieving non-existent workflow metadata returns None."""
        metadata = WorkflowRegistry.get_workflow_info("nonexistent")
        assert metadata is None

    def test_get_workflow_info_after_registration(self):
        """Test metadata is None before registration, present after."""
        assert WorkflowRegistry.get_workflow_info("test") is None

        WorkflowRegistry.register_metadata("test", "Test workflow")

        assert WorkflowRegistry.get_workflow_info("test") is not None


class TestWorkflowRegistryIntegration:
    """Test suite for integrated workflow and metadata scenarios."""

    def setup_method(self):
        """Clear registry before each test."""
        WorkflowRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        WorkflowRegistry.clear()

    # ========================================================================
    # Combined Workflow and Metadata Tests
    # ========================================================================

    def test_workflow_with_metadata(self):
        """Test registering workflow with metadata."""
        WorkflowRegistry.register_workflow("consensus", MockWorkflow)
        WorkflowRegistry.register_metadata(
            "consensus",
            description="Multi-model consultation with configurable synthesis",
            version="2.0.0",
            author="ModelChorus Team",
            category="consultation",
            parameters=["strategy", "providers"],
            examples=["model-chorus consensus 'prompt' --strategy vote"],
        )

        # Verify workflow is registered
        assert WorkflowRegistry.is_registered("consensus")
        workflow_class = WorkflowRegistry.get("consensus")
        assert workflow_class == MockWorkflow

        # Verify metadata is registered
        metadata = WorkflowRegistry.get_workflow_info("consensus")
        assert metadata is not None
        assert metadata.name == "consensus"
        assert metadata.description == "Multi-model consultation with configurable synthesis"
        assert metadata.version == "2.0.0"
        assert metadata.category == "consultation"

    def test_multiple_workflows_with_metadata(self):
        """Test registering multiple workflows with metadata."""
        # Register first workflow
        WorkflowRegistry.register_workflow("thinkdeep", MockWorkflow)
        WorkflowRegistry.register_metadata(
            "thinkdeep",
            description="Extended reasoning workflow",
            version="1.0.0",
            author="Team A",
        )

        # Register second workflow
        WorkflowRegistry.register_workflow("consensus", AnotherMockWorkflow)
        WorkflowRegistry.register_metadata(
            "consensus",
            description="Consensus building workflow",
            version="2.0.0",
            author="Team B",
        )

        # Verify both workflows exist
        workflows = WorkflowRegistry.list_workflows()
        assert len(workflows) == 2
        assert "thinkdeep" in workflows
        assert "consensus" in workflows

        # Verify metadata for both
        thinkdeep_meta = WorkflowRegistry.get_workflow_info("thinkdeep")
        consensus_meta = WorkflowRegistry.get_workflow_info("consensus")

        assert thinkdeep_meta.author == "Team A"
        assert consensus_meta.author == "Team B"

    def test_workflow_instantiation_after_registry(self):
        """Test that registered workflows can be instantiated properly."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)

        workflow_class = WorkflowRegistry.get("test")
        workflow = workflow_class("Test Workflow", "A test workflow instance")

        assert isinstance(workflow, BaseWorkflow)
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow instance"

    # ========================================================================
    # Real-World Usage Pattern Tests
    # ========================================================================

    def test_decorator_pattern_with_metadata(self):
        """Test realistic decorator usage pattern with metadata."""

        @WorkflowRegistry.register("advanced-workflow")
        class AdvancedWorkflow(BaseWorkflow):
            """Advanced workflow with comprehensive metadata."""

            async def run(self, prompt: str, **kwargs) -> WorkflowResult:
                return WorkflowResult(success=True, synthesis="Advanced result")

        # Register metadata separately
        WorkflowRegistry.register_metadata(
            "advanced-workflow",
            description="Advanced workflow with extended capabilities",
            version="3.0.0",
            author="Advanced Team",
            category="advanced",
            parameters=["mode", "depth", "thoroughness"],
            examples=[
                "model-chorus advanced-workflow 'prompt' --mode deep",
                "model-chorus advanced-workflow 'prompt' --depth comprehensive",
            ],
        )

        # Verify complete registration
        assert WorkflowRegistry.is_registered("advanced-workflow")

        workflow_class = WorkflowRegistry.get("advanced-workflow")
        assert workflow_class == AdvancedWorkflow

        metadata = WorkflowRegistry.get_workflow_info("advanced-workflow")
        assert metadata.version == "3.0.0"
        assert len(metadata.parameters) == 3
        assert len(metadata.examples) == 2

    def test_list_workflows_discovery(self):
        """Test workflow discovery pattern for CLI help."""
        # Register multiple workflows with metadata
        workflows_config = [
            ("chat", "Single-model conversational interaction", "1.0.0"),
            ("consensus", "Multi-model consultation", "2.0.0"),
            ("thinkdeep", "Extended reasoning investigation", "1.5.0"),
            ("argument", "Three-role dialectical analysis", "1.0.0"),
        ]

        for name, description, version in workflows_config:
            WorkflowRegistry.register_workflow(name, MockWorkflow)
            WorkflowRegistry.register_metadata(name, description, version=version)

        # Discover all workflows
        workflow_names = WorkflowRegistry.list_workflows()
        assert len(workflow_names) == 4

        # Get metadata for each
        for name in workflow_names:
            metadata = WorkflowRegistry.get_workflow_info(name)
            assert metadata is not None
            assert metadata.description
            assert metadata.version

    def test_registry_isolation(self):
        """Test that registry maintains proper isolation between workflows."""
        # Register two workflows
        WorkflowRegistry.register_workflow("workflow-1", MockWorkflow)
        WorkflowRegistry.register_workflow("workflow-2", AnotherMockWorkflow)

        # Add metadata only to first
        WorkflowRegistry.register_metadata("workflow-1", "First workflow", version="1.0.0")

        # Verify isolation
        meta1 = WorkflowRegistry.get_workflow_info("workflow-1")
        meta2 = WorkflowRegistry.get_workflow_info("workflow-2")

        assert meta1 is not None
        assert meta2 is None

        # Unregister first shouldn't affect second
        WorkflowRegistry.unregister("workflow-1")

        assert not WorkflowRegistry.is_registered("workflow-1")
        assert WorkflowRegistry.is_registered("workflow-2")


class TestWorkflowRegistryEdgeCases:
    """Test suite for edge cases and error conditions."""

    def setup_method(self):
        """Clear registry before each test."""
        WorkflowRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        WorkflowRegistry.clear()

    # ========================================================================
    # Edge Case Tests
    # ========================================================================

    def test_empty_workflow_name(self):
        """Test that empty workflow name is handled."""
        # Pydantic WorkflowMetadata requires min_length=1, but registry doesn't validate
        # This tests registry behavior with empty string
        WorkflowRegistry.register_workflow("", MockWorkflow)
        assert WorkflowRegistry.is_registered("")

    def test_special_characters_in_name(self):
        """Test workflow names with special characters."""
        special_names = [
            "workflow-with-dashes",
            "workflow_with_underscores",
            "workflow.with.dots",
            "workflow:with:colons",
        ]

        for name in special_names:
            WorkflowRegistry.register_workflow(name, MockWorkflow)
            assert WorkflowRegistry.is_registered(name)

    def test_clear_on_empty_registry(self):
        """Test that clear on empty registry doesn't raise error."""
        WorkflowRegistry.clear()  # Should not raise
        assert len(WorkflowRegistry.list_workflows()) == 0

    def test_unregister_after_clear(self):
        """Test unregister after clear raises appropriate error."""
        WorkflowRegistry.register_workflow("test", MockWorkflow)
        WorkflowRegistry.clear()

        with pytest.raises(KeyError):
            WorkflowRegistry.unregister("test")

    def test_metadata_version_format(self):
        """Test that metadata validates version format."""
        # WorkflowMetadata should validate semantic versioning
        WorkflowRegistry.register_metadata("test", "Test", version="1.2.3")
        metadata = WorkflowRegistry.get_workflow_info("test")
        assert metadata.version == "1.2.3"

        # Invalid version should raise validation error
        with pytest.raises(ValueError):
            WorkflowRegistry.register_metadata("test2", "Test", version="invalid")
