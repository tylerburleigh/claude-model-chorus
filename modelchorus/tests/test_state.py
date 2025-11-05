"""
Unit tests for state persistence layer.

Tests verify StateManager functionality including:
- In-memory state storage and retrieval
- Thread-safe concurrent operations
- JSON serialization and deserialization
- File-based persistence (when enabled)
- Error handling and edge cases
"""

import json
import pytest
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

from modelchorus.core.state import StateManager, get_default_state_manager
from modelchorus.core.models import ConversationState


class TestStateManager:
    """Test suite for StateManager class."""

    # ========================================================================
    # In-Memory State Tests
    # ========================================================================

    def test_set_and_get_state(self):
        """Test basic state storage and retrieval."""
        manager = StateManager()

        test_data = {
            "current_step": 1,
            "models_consulted": ["claude", "gpt-5"],
            "consensus_reached": False
        }

        manager.set_state("test_workflow", test_data)
        retrieved = manager.get_state("test_workflow")

        assert retrieved is not None
        assert retrieved["current_step"] == 1
        assert retrieved["models_consulted"] == ["claude", "gpt-5"]
        assert retrieved["consensus_reached"] is False

    def test_get_state_object_with_metadata(self):
        """Test retrieving complete state object with metadata."""
        manager = StateManager()

        test_data = {"step": 1}
        manager.set_state("test_workflow", test_data, schema_version="2.0")

        state_obj = manager.get_state_object("test_workflow")

        assert state_obj is not None
        assert state_obj.workflow_name == "test_workflow"
        assert state_obj.data == test_data
        assert state_obj.schema_version == "2.0"
        assert state_obj.created_at is not None
        assert state_obj.updated_at is not None

    def test_update_state_existing(self):
        """Test merging updates into existing state."""
        manager = StateManager()

        # Set initial state
        initial_data = {
            "step": 1,
            "count": 0,
            "status": "active"
        }
        manager.set_state("test_workflow", initial_data)

        # Update with new fields
        updates = {
            "step": 2,
            "count": 5
        }
        manager.update_state("test_workflow", updates)

        # Verify merge
        result = manager.get_state("test_workflow")
        assert result["step"] == 2  # Updated
        assert result["count"] == 5  # Updated
        assert result["status"] == "active"  # Preserved

    def test_update_state_nonexistent(self):
        """Test creating state via update when it doesn't exist."""
        manager = StateManager()

        updates = {"step": 1, "value": "new"}
        success = manager.update_state("new_workflow", updates)

        assert success is True
        result = manager.get_state("new_workflow")
        assert result is not None
        assert result["step"] == 1
        assert result["value"] == "new"

    def test_delete_state(self):
        """Test state deletion."""
        manager = StateManager()

        manager.set_state("test_workflow", {"data": "value"})
        assert manager.get_state("test_workflow") is not None

        # Delete existing state
        result = manager.delete_state("test_workflow")
        assert result is True
        assert manager.get_state("test_workflow") is None

        # Delete non-existent state
        result = manager.delete_state("nonexistent")
        assert result is False

    def test_list_workflows(self):
        """Test listing all workflows with stored state."""
        manager = StateManager()

        # Empty initially
        assert manager.list_workflows() == []

        # Add multiple workflows
        manager.set_state("workflow_a", {"data": 1})
        manager.set_state("workflow_b", {"data": 2})
        manager.set_state("workflow_c", {"data": 3})

        workflows = manager.list_workflows()
        assert len(workflows) == 3
        assert "workflow_a" in workflows
        assert "workflow_b" in workflows
        assert "workflow_c" in workflows

    def test_clear_all(self):
        """Test clearing all workflow state."""
        manager = StateManager()

        # Add multiple states
        manager.set_state("workflow_1", {"data": 1})
        manager.set_state("workflow_2", {"data": 2})
        manager.set_state("workflow_3", {"data": 3})

        assert len(manager.list_workflows()) == 3

        # Clear all
        count = manager.clear_all()
        assert count == 3
        assert manager.list_workflows() == []
        assert manager.get_state("workflow_1") is None

    # ========================================================================
    # Thread-Safety Tests
    # ========================================================================

    def test_concurrent_set_state(self):
        """Test thread-safety with concurrent writes."""
        manager = StateManager()
        num_threads = 10
        operations_per_thread = 20

        def write_state(thread_id):
            for i in range(operations_per_thread):
                manager.set_state(
                    f"workflow_{thread_id}",
                    {"thread": thread_id, "iteration": i}
                )

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=write_state, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all workflows were created
        workflows = manager.list_workflows()
        assert len(workflows) == num_threads

        # Verify final state for each workflow
        for i in range(num_threads):
            state = manager.get_state(f"workflow_{i}")
            assert state is not None
            assert state["thread"] == i

    def test_concurrent_read_write(self):
        """Test thread-safety with mixed read/write operations."""
        manager = StateManager()
        manager.set_state("shared_workflow", {"counter": 0})

        num_readers = 5
        num_writers = 5
        iterations = 30
        read_count = threading.Event()
        write_count = threading.Event()

        def reader():
            for _ in range(iterations):
                state = manager.get_state("shared_workflow")
                assert state is not None
                assert "counter" in state
                time.sleep(0.001)  # Small delay to increase contention

        def writer(thread_id):
            for i in range(iterations):
                manager.update_state(
                    "shared_workflow",
                    {"counter": thread_id * 1000 + i}
                )
                time.sleep(0.001)

        threads = []

        # Start readers
        for i in range(num_readers):
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        # Start writers
        for i in range(num_writers):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify state is still consistent
        final_state = manager.get_state("shared_workflow")
        assert final_state is not None
        assert "counter" in final_state

    def test_state_isolation(self):
        """Test that different workflows don't interfere with each other."""
        manager = StateManager()

        # Create multiple workflows
        manager.set_state("workflow_a", {"name": "A", "value": 100})
        manager.set_state("workflow_b", {"name": "B", "value": 200})
        manager.set_state("workflow_c", {"name": "C", "value": 300})

        # Modify one workflow
        manager.update_state("workflow_b", {"value": 999})

        # Verify others unchanged
        state_a = manager.get_state("workflow_a")
        state_b = manager.get_state("workflow_b")
        state_c = manager.get_state("workflow_c")

        assert state_a["value"] == 100  # Unchanged
        assert state_b["value"] == 999  # Modified
        assert state_c["value"] == 300  # Unchanged

    # ========================================================================
    # Serialization Tests
    # ========================================================================

    def test_serialize_state(self):
        """Test serializing state to JSON string."""
        manager = StateManager()

        test_data = {
            "step": 1,
            "models": ["claude", "gpt-5"],
            "config": {"temperature": 0.7}
        }
        manager.set_state("test_workflow", test_data)

        json_str = manager.serialize_state("test_workflow")
        assert json_str is not None

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert parsed["workflow_name"] == "test_workflow"
        assert parsed["data"]["step"] == 1
        assert parsed["data"]["models"] == ["claude", "gpt-5"]
        assert "created_at" in parsed
        assert "updated_at" in parsed

    def test_deserialize_state(self):
        """Test deserializing JSON string to state object."""
        manager = StateManager()

        json_str = """
        {
            "workflow_name": "test_workflow",
            "data": {"step": 1, "value": "test"},
            "schema_version": "1.0",
            "created_at": "2025-11-05T12:00:00Z",
            "updated_at": "2025-11-05T12:00:00Z"
        }
        """

        state = manager.deserialize_state(json_str)
        assert isinstance(state, ConversationState)
        assert state.workflow_name == "test_workflow"
        assert state.data["step"] == 1
        assert state.data["value"] == "test"
        assert state.schema_version == "1.0"

    def test_roundtrip_serialization(self):
        """Test serialize then deserialize maintains data integrity."""
        manager = StateManager()

        original_data = {
            "complex_field": {
                "nested": {"deeply": ["a", "b", "c"]},
                "numbers": [1, 2, 3],
                "boolean": True
            },
            "string": "test value"
        }

        manager.set_state("test_workflow", original_data, schema_version="2.5")

        # Serialize
        json_str = manager.serialize_state("test_workflow")

        # Deserialize
        state = manager.deserialize_state(json_str)

        # Verify data integrity
        assert state.data == original_data
        assert state.schema_version == "2.5"
        assert state.workflow_name == "test_workflow"

    def test_export_state(self, tmp_path):
        """Test exporting state to JSON file."""
        manager = StateManager()

        test_data = {"exported": True, "value": 42}
        manager.set_state("export_test", test_data)

        output_file = tmp_path / "exported_state.json"
        success = manager.export_state("export_test", output_file)

        assert success is True
        assert output_file.exists()

        # Verify file content
        content = json.loads(output_file.read_text())
        assert content["workflow_name"] == "export_test"
        assert content["data"]["exported"] is True
        assert content["data"]["value"] == 42

    def test_import_state(self, tmp_path):
        """Test importing state from JSON file."""
        manager = StateManager()

        # Create state file
        state_data = {
            "workflow_name": "imported_workflow",
            "data": {"imported": True, "count": 10},
            "schema_version": "1.5",
            "created_at": "2025-11-05T10:00:00Z",
            "updated_at": "2025-11-05T11:00:00Z"
        }

        import_file = tmp_path / "import_state.json"
        import_file.write_text(json.dumps(state_data, indent=2))

        # Import
        workflow_name = manager.import_state(import_file)

        assert workflow_name == "imported_workflow"

        # Verify imported data
        state = manager.get_state("imported_workflow")
        assert state is not None
        assert state["imported"] is True
        assert state["count"] == 10

    def test_deserialize_invalid_json(self):
        """Test error handling for invalid JSON."""
        manager = StateManager()

        invalid_json = "{ this is not valid json }"

        with pytest.raises(Exception):  # Pydantic ValidationError
            manager.deserialize_state(invalid_json)

    # ========================================================================
    # File Persistence Tests
    # ========================================================================

    def test_file_persistence_on_set(self, tmp_path):
        """Test automatic file save when persistence enabled."""
        state_dir = tmp_path / "state"
        manager = StateManager(state_dir=state_dir, enable_file_persistence=True)

        test_data = {"persisted": True}
        manager.set_state("persistent_workflow", test_data)

        # Verify file created
        state_file = state_dir / "persistent_workflow.json"
        assert state_file.exists()

        # Verify file content
        content = json.loads(state_file.read_text())
        assert content["workflow_name"] == "persistent_workflow"
        assert content["data"]["persisted"] is True

    def test_load_from_disk(self, tmp_path):
        """Test loading state from disk into memory."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Create state file manually
        state_data = {
            "workflow_name": "disk_workflow",
            "data": {"loaded": True, "source": "disk"},
            "schema_version": "1.0",
            "created_at": "2025-11-05T10:00:00Z",
            "updated_at": "2025-11-05T11:00:00Z"
        }

        state_file = state_dir / "disk_workflow.json"
        state_file.write_text(json.dumps(state_data))

        # Load into new manager
        manager = StateManager(state_dir=state_dir, enable_file_persistence=True)
        success = manager.load_from_disk("disk_workflow")

        assert success is True

        # Verify in memory
        state = manager.get_state("disk_workflow")
        assert state is not None
        assert state["loaded"] is True
        assert state["source"] == "disk"

    def test_load_all_from_disk(self, tmp_path):
        """Test bulk loading of all state files from disk."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Create multiple state files
        for i in range(5):
            state_data = {
                "workflow_name": f"workflow_{i}",
                "data": {"id": i},
                "schema_version": "1.0",
                "created_at": "2025-11-05T10:00:00Z",
                "updated_at": "2025-11-05T11:00:00Z"
            }
            state_file = state_dir / f"workflow_{i}.json"
            state_file.write_text(json.dumps(state_data))

        # Load all
        manager = StateManager(state_dir=state_dir, enable_file_persistence=True)
        count = manager.load_all_from_disk()

        assert count == 5
        assert len(manager.list_workflows()) == 5

        # Verify each loaded
        for i in range(5):
            state = manager.get_state(f"workflow_{i}")
            assert state is not None
            assert state["id"] == i

    def test_sync_to_disk(self, tmp_path):
        """Test manual sync of in-memory state to disk."""
        state_dir = tmp_path / "state"
        manager = StateManager(state_dir=state_dir, enable_file_persistence=True)

        # Create states in memory only (disable auto-persist for test)
        manager.enable_file_persistence = False
        manager.set_state("workflow_1", {"data": 1})
        manager.set_state("workflow_2", {"data": 2})
        manager.set_state("workflow_3", {"data": 3})

        # Re-enable and sync
        manager.enable_file_persistence = True
        count = manager.sync_to_disk()

        assert count == 3

        # Verify files created
        assert (state_dir / "workflow_1.json").exists()
        assert (state_dir / "workflow_2.json").exists()
        assert (state_dir / "workflow_3.json").exists()

    def test_delete_removes_file(self, tmp_path):
        """Test that delete also removes file when persistence enabled."""
        state_dir = tmp_path / "state"
        manager = StateManager(state_dir=state_dir, enable_file_persistence=True)

        manager.set_state("temp_workflow", {"temporary": True})

        state_file = state_dir / "temp_workflow.json"
        assert state_file.exists()

        # Delete state
        manager.delete_state("temp_workflow")

        # Verify file removed
        assert not state_file.exists()

    def test_persistence_disabled_no_files(self, tmp_path):
        """Test that no files are created when persistence disabled."""
        state_dir = tmp_path / "state"
        manager = StateManager(state_dir=state_dir, enable_file_persistence=False)

        manager.set_state("memory_only", {"in_memory": True})

        # Verify no state directory created
        assert not state_dir.exists()

        # State should still work in memory
        state = manager.get_state("memory_only")
        assert state is not None
        assert state["in_memory"] is True

    # ========================================================================
    # Edge Case Tests
    # ========================================================================

    def test_get_nonexistent_workflow(self):
        """Test getting state for workflow that doesn't exist."""
        manager = StateManager()

        result = manager.get_state("nonexistent_workflow")
        assert result is None

    def test_delete_nonexistent_workflow(self):
        """Test deleting workflow that doesn't exist."""
        manager = StateManager()

        result = manager.delete_state("nonexistent_workflow")
        assert result is False

    def test_serialize_nonexistent(self):
        """Test serializing workflow that doesn't exist."""
        manager = StateManager()

        result = manager.serialize_state("nonexistent_workflow")
        assert result is None

    def test_import_malformed_json(self, tmp_path):
        """Test importing file with malformed JSON."""
        manager = StateManager()

        bad_file = tmp_path / "malformed.json"
        bad_file.write_text("{ this is not valid json }")

        workflow_name = manager.import_state(bad_file)
        assert workflow_name is None  # Should fail gracefully

    def test_state_timestamps_update(self):
        """Test that created_at persists but updated_at changes."""
        manager = StateManager()

        # Create initial state
        manager.set_state("timestamp_test", {"version": 1})
        state1 = manager.get_state_object("timestamp_test")
        created_at = state1.created_at
        updated_at_1 = state1.updated_at

        # Wait a tiny bit
        time.sleep(0.01)

        # Update state
        manager.update_state("timestamp_test", {"version": 2})
        state2 = manager.get_state_object("timestamp_test")
        updated_at_2 = state2.updated_at

        # created_at should be same, updated_at should differ
        assert state2.created_at == created_at
        assert updated_at_2 != updated_at_1

    def test_schema_version_preservation(self):
        """Test that schema version is set and retrievable."""
        manager = StateManager()

        manager.set_state("versioned", {"data": "test"}, schema_version="3.2.1")

        # Get and verify
        state = manager.get_state_object("versioned")
        assert state.schema_version == "3.2.1"

        # Note: update_state uses default schema_version (1.0)
        # This is current implementation behavior
        # To preserve custom schema version, use set_state directly
        manager.set_state("versioned", {"data": "updated"}, schema_version="3.2.1")
        state = manager.get_state_object("versioned")
        assert state.schema_version == "3.2.1"

    def test_get_default_state_manager(self):
        """Test singleton default state manager."""
        manager1 = get_default_state_manager()
        manager2 = get_default_state_manager()

        # Should be same instance
        assert manager1 is manager2

        # Should work normally
        manager1.set_state("singleton_test", {"shared": True})
        state = manager2.get_state("singleton_test")
        assert state is not None
        assert state["shared"] is True


class TestStateManagerExportImportRoundtrip:
    """Test complete export/import workflow."""

    def test_export_import_roundtrip(self, tmp_path):
        """Test complete workflow: set -> export -> import to new manager."""
        # Create and populate first manager
        manager1 = StateManager()

        original_data = {
            "workflow_state": "completed",
            "models": ["claude", "gpt-5", "gemini"],
            "results": {
                "consensus": "Approach A",
                "confidence": 0.85
            }
        }

        manager1.set_state("roundtrip_test", original_data, schema_version="2.0")

        # Export to file
        export_file = tmp_path / "exported.json"
        manager1.export_state("roundtrip_test", export_file)

        # Create new manager and import
        manager2 = StateManager()
        workflow_name = manager2.import_state(export_file)

        assert workflow_name == "roundtrip_test"

        # Verify data integrity
        imported_data = manager2.get_state("roundtrip_test")
        assert imported_data == original_data

        # Verify schema version preserved
        state_obj = manager2.get_state_object("roundtrip_test")
        assert state_obj.schema_version == "2.0"


class TestStateManagerFileRecovery:
    """Test state recovery after simulated process restart."""

    def test_process_restart_recovery(self, tmp_path):
        """Test recovering state after simulated restart with file persistence."""
        state_dir = tmp_path / "persistent_state"

        # First "process" - create and persist states
        manager1 = StateManager(state_dir=state_dir, enable_file_persistence=True)

        manager1.set_state("workflow_a", {"step": 5, "status": "in_progress"})
        manager1.set_state("workflow_b", {"step": 10, "status": "completed"})
        manager1.set_state("workflow_c", {"step": 2, "status": "pending"})

        # Simulate process end (manager goes out of scope)
        del manager1

        # Second "process" - new manager, recover from disk
        manager2 = StateManager(state_dir=state_dir, enable_file_persistence=True)
        count = manager2.load_all_from_disk()

        assert count == 3

        # Verify all states recovered
        state_a = manager2.get_state("workflow_a")
        state_b = manager2.get_state("workflow_b")
        state_c = manager2.get_state("workflow_c")

        assert state_a["step"] == 5
        assert state_a["status"] == "in_progress"

        assert state_b["step"] == 10
        assert state_b["status"] == "completed"

        assert state_c["step"] == 2
        assert state_c["status"] == "pending"
