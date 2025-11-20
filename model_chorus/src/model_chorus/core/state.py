"""
State persistence layer for ModelChorus workflows.

Provides thread-safe in-memory and optional file-based persistence
for workflow state. Complements ConversationMemory by focusing on
workflow execution state rather than conversation history.

Key Features:
- Thread-safe in-memory state storage
- JSON serialization for Pydantic models
- Optional file-based persistence
- State isolation per workflow instance
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .models import ConversationState


logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_STATE_DIR = Path.home() / ".model-chorus" / "state"


class StateManager:
    """
    Thread-safe state persistence manager for workflows.

    Manages workflow execution state separately from conversation history.
    Provides in-memory storage with optional file-based persistence.

    Workflow state typically includes:
    - Current step/phase information
    - Intermediate results
    - Configuration and settings
    - Workflow-specific metadata

    Attributes:
        state_dir: Directory for file-based state persistence
        enable_file_persistence: Whether to persist state to disk
        _state_store: Thread-safe in-memory state storage
        _lock: Lock for thread-safe operations
    """

    def __init__(self, state_dir: Path = DEFAULT_STATE_DIR, enable_file_persistence: bool = False):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for file-based state storage
            enable_file_persistence: Enable disk persistence
        """
        self.state_dir = state_dir
        self.enable_file_persistence = enable_file_persistence

        # Thread-safe in-memory storage (task-1-4-1)
        self._state_store: Dict[str, ConversationState] = {}
        self._lock = threading.RLock()

        # Create state directory if file persistence enabled
        if self.enable_file_persistence:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"StateManager initialized with file persistence: {state_dir}")
        else:
            logger.info("StateManager initialized (in-memory only)")

    # ========================================================================
    # In-Memory State Storage (task-1-4-1)
    # ========================================================================

    def set_state(
        self, workflow_name: str, state_data: Dict[str, Any], schema_version: str = "1.0"
    ) -> None:
        """
        Store workflow state in memory.

        Thread-safe operation. If file persistence enabled, also writes to disk.

        Args:
            workflow_name: Workflow identifier
            state_data: State data to store
            schema_version: State schema version

        Example:
            >>> manager = StateManager()
            >>> manager.set_state("consensus", {
            ...     "current_model_index": 2,
            ...     "models_consulted": ["gpt-5", "claude"],
            ...     "consensus_reached": False
            ... })
        """
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()

            # Check if state exists (for update vs create)
            existing_state = self._state_store.get(workflow_name)
            created_at = existing_state.created_at if existing_state else now

            # Create state object
            state = ConversationState(
                workflow_name=workflow_name,
                data=state_data,
                schema_version=schema_version,
                created_at=created_at,
                updated_at=now,
            )

            # Store in memory
            self._state_store[workflow_name] = state

            # Persist to file if enabled
            if self.enable_file_persistence:
                self._save_to_file(workflow_name, state)

            logger.debug(f"State stored for workflow '{workflow_name}' ({len(state_data)} fields)")

    def get_state(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve workflow state from memory.

        Thread-safe operation. Returns state data dict or None if not found.

        Args:
            workflow_name: Workflow identifier

        Returns:
            State data dict or None

        Example:
            >>> state_data = manager.get_state("consensus")
            >>> if state_data:
            ...     print(f"Current model: {state_data.get('current_model_index')}")
        """
        with self._lock:
            state = self._state_store.get(workflow_name)
            if state:
                logger.debug(f"Retrieved state for workflow '{workflow_name}'")
                return state.data
            else:
                logger.debug(f"No state found for workflow '{workflow_name}'")
                return None

    def get_state_object(self, workflow_name: str) -> Optional[ConversationState]:
        """
        Retrieve complete state object (including metadata).

        Args:
            workflow_name: Workflow identifier

        Returns:
            ConversationState object or None

        Example:
            >>> state_obj = manager.get_state_object("consensus")
            >>> if state_obj:
            ...     print(f"Last updated: {state_obj.updated_at}")
            ...     print(f"Schema version: {state_obj.schema_version}")
        """
        with self._lock:
            return self._state_store.get(workflow_name)

    def update_state(self, workflow_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields in workflow state.

        Merges updates into existing state. Creates new state if doesn't exist.

        Args:
            workflow_name: Workflow identifier
            updates: Fields to update

        Returns:
            True if successful

        Example:
            >>> manager.update_state("consensus", {
            ...     "current_model_index": 3,
            ...     "consensus_reached": True
            ... })
        """
        with self._lock:
            existing_state = self.get_state(workflow_name)

            if existing_state:
                # Merge updates
                existing_state.update(updates)
                state_data = existing_state
            else:
                # Create new state with updates
                state_data = updates

            self.set_state(workflow_name, state_data)
            return True

    def delete_state(self, workflow_name: str) -> bool:
        """
        Delete workflow state from memory and disk.

        Args:
            workflow_name: Workflow identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> manager.delete_state("consensus")
        """
        with self._lock:
            if workflow_name in self._state_store:
                del self._state_store[workflow_name]

                # Delete file if file persistence enabled
                if self.enable_file_persistence:
                    self._delete_file(workflow_name)

                logger.info(f"Deleted state for workflow '{workflow_name}'")
                return True
            else:
                logger.debug(f"No state to delete for workflow '{workflow_name}'")
                return False

    def list_workflows(self) -> list[str]:
        """
        List all workflows with stored state.

        Returns:
            List of workflow names

        Example:
            >>> workflows = manager.list_workflows()
            >>> print(f"Active workflows: {', '.join(workflows)}")
        """
        with self._lock:
            return list(self._state_store.keys())

    def clear_all(self) -> int:
        """
        Clear all workflow state from memory.

        Returns:
            Number of states cleared

        Example:
            >>> count = manager.clear_all()
            >>> print(f"Cleared {count} workflow states")
        """
        with self._lock:
            count = len(self._state_store)
            self._state_store.clear()
            logger.info(f"Cleared all workflow state ({count} workflows)")
            return count

    # ========================================================================
    # State Serialization/Deserialization (task-1-4-2)
    # ========================================================================

    def serialize_state(self, workflow_name: str) -> Optional[str]:
        """
        Serialize workflow state to JSON string.

        Uses Pydantic's JSON serialization for proper type handling.

        Args:
            workflow_name: Workflow identifier

        Returns:
            JSON string or None if not found

        Example:
            >>> json_str = manager.serialize_state("consensus")
            >>> print(json_str)
            '{"workflow_name":"consensus","data":{...},...}'
        """
        with self._lock:
            state = self._state_store.get(workflow_name)
            if state:
                return state.model_dump_json(indent=2)
            return None

    def deserialize_state(self, json_str: str) -> ConversationState:
        """
        Deserialize JSON string to state object.

        Args:
            json_str: JSON string representation

        Returns:
            ConversationState object

        Raises:
            ValidationError: If JSON doesn't match ConversationState schema

        Example:
            >>> state = manager.deserialize_state(json_str)
            >>> manager.set_state(state.workflow_name, state.data, state.schema_version)
        """
        return ConversationState.model_validate_json(json_str)

    def export_state(self, workflow_name: str, output_path: Path) -> bool:
        """
        Export workflow state to JSON file.

        Args:
            workflow_name: Workflow identifier
            output_path: Output file path

        Returns:
            True if successful, False if state not found

        Example:
            >>> manager.export_state("consensus", Path("state_export.json"))
        """
        json_str = self.serialize_state(workflow_name)
        if json_str:
            output_path.write_text(json_str)
            logger.info(f"Exported state for '{workflow_name}' to {output_path}")
            return True
        return False

    def import_state(self, input_path: Path) -> Optional[str]:
        """
        Import workflow state from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Workflow name if successful, None otherwise

        Example:
            >>> workflow_name = manager.import_state(Path("state_export.json"))
            >>> print(f"Imported state for workflow: {workflow_name}")
        """
        try:
            json_str = input_path.read_text()
            state = self.deserialize_state(json_str)

            # Store in memory
            with self._lock:
                self._state_store[state.workflow_name] = state

                # Persist if file persistence enabled
                if self.enable_file_persistence:
                    self._save_to_file(state.workflow_name, state)

            logger.info(f"Imported state for workflow '{state.workflow_name}'")
            return state.workflow_name

        except Exception as e:
            logger.error(f"Error importing state from {input_path}: {e}")
            return None

    # ========================================================================
    # File-Based Persistence (task-1-4-3)
    # ========================================================================

    def _save_to_file(self, workflow_name: str, state: ConversationState) -> None:
        """
        Save state to disk (internal method).

        Args:
            workflow_name: Workflow identifier
            state: State object to save
        """
        if not self.enable_file_persistence:
            return

        try:
            state_file = self.state_dir / f"{workflow_name}.json"
            state_file.write_text(state.model_dump_json(indent=2))
            logger.debug(f"Persisted state to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state to file for '{workflow_name}': {e}")

    def _delete_file(self, workflow_name: str) -> None:
        """
        Delete state file from disk (internal method).

        Args:
            workflow_name: Workflow identifier
        """
        if not self.enable_file_persistence:
            return

        try:
            state_file = self.state_dir / f"{workflow_name}.json"
            if state_file.exists():
                state_file.unlink()
                logger.debug(f"Deleted state file {state_file}")
        except Exception as e:
            logger.error(f"Error deleting state file for '{workflow_name}': {e}")

    def load_from_disk(self, workflow_name: str) -> bool:
        """
        Load state from disk into memory.

        Useful for recovering state after process restart when file
        persistence is enabled.

        Args:
            workflow_name: Workflow identifier

        Returns:
            True if loaded successfully, False otherwise

        Example:
            >>> manager = StateManager(enable_file_persistence=True)
            >>> if manager.load_from_disk("consensus"):
            ...     state = manager.get_state("consensus")
            ...     print("State recovered from disk")
        """
        if not self.enable_file_persistence:
            logger.warning("File persistence not enabled")
            return False

        try:
            state_file = self.state_dir / f"{workflow_name}.json"
            if not state_file.exists():
                logger.debug(f"No state file found for '{workflow_name}'")
                return False

            json_str = state_file.read_text()
            state = self.deserialize_state(json_str)

            with self._lock:
                self._state_store[workflow_name] = state

            logger.info(f"Loaded state for '{workflow_name}' from disk")
            return True

        except Exception as e:
            logger.error(f"Error loading state from disk for '{workflow_name}': {e}")
            return False

    def load_all_from_disk(self) -> int:
        """
        Load all state files from disk into memory.

        Useful for initialization when file persistence is enabled.

        Returns:
            Number of states loaded

        Example:
            >>> manager = StateManager(enable_file_persistence=True)
            >>> count = manager.load_all_from_disk()
            >>> print(f"Loaded {count} workflow states from disk")
        """
        if not self.enable_file_persistence:
            logger.warning("File persistence not enabled")
            return 0

        loaded = 0
        for state_file in self.state_dir.glob("*.json"):
            try:
                workflow_name = state_file.stem
                if self.load_from_disk(workflow_name):
                    loaded += 1
            except Exception as e:
                logger.error(f"Error loading state file {state_file}: {e}")

        logger.info(f"Loaded {loaded} workflow states from disk")
        return loaded

    def sync_to_disk(self) -> int:
        """
        Sync all in-memory state to disk.

        Useful for ensuring persistence before shutdown.

        Returns:
            Number of states persisted

        Example:
            >>> count = manager.sync_to_disk()
            >>> print(f"Persisted {count} workflow states to disk")
        """
        if not self.enable_file_persistence:
            logger.warning("File persistence not enabled")
            return 0

        count = 0
        with self._lock:
            for workflow_name, state in self._state_store.items():
                try:
                    self._save_to_file(workflow_name, state)
                    count += 1
                except Exception as e:
                    logger.error(f"Error syncing state for '{workflow_name}': {e}")

        logger.info(f"Synced {count} workflow states to disk")
        return count


# Singleton instance for convenience
_default_manager: Optional[StateManager] = None


def get_default_state_manager() -> StateManager:
    """
    Get singleton default state manager instance.

    Returns:
        Default StateManager instance

    Example:
        >>> from model_chorus.core.state import get_default_state_manager
        >>> manager = get_default_state_manager()
        >>> manager.set_state("my_workflow", {"step": 1})
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = StateManager()
    return _default_manager
