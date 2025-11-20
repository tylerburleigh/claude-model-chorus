"""
Conversation memory management for ModelChorus.

Provides file-based persistence for multi-turn conversations with continuation support.
Based on Zen MCP patterns but adapted for CLI-based orchestration architecture.

Key Features:
- UUID-based thread identification (continuation_id)
- File-based persistence (survives process restarts)
- Thread-safe operations with file locking
- TTL-based automatic cleanup
- Context window management for long conversations
"""

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import filelock

from .models import ConversationMessage, ConversationThread

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONVERSATIONS_DIR = Path.home() / ".model-chorus" / "conversations"
DEFAULT_TTL_HOURS = 3
DEFAULT_MAX_MESSAGES_PER_THREAD = 50


class ConversationMemory:
    """
    Manages conversation threads with file-based persistence.

    Provides thread-safe storage and retrieval of conversation history,
    enabling multi-turn conversations across workflow executions.

    Architecture:
        - Each thread stored as JSON file: ~/.model-chorus/conversations/{thread_id}.json
        - File locking prevents concurrent access corruption
        - TTL-based cleanup removes expired threads
        - Supports conversation chains via parent_thread_id

    Attributes:
        conversations_dir: Directory where conversation files are stored
        ttl_hours: Time-to-live for conversation threads in hours
        max_messages: Maximum messages per thread before truncation
    """

    def __init__(
        self,
        conversations_dir: Path = DEFAULT_CONVERSATIONS_DIR,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        max_messages: int = DEFAULT_MAX_MESSAGES_PER_THREAD,
    ):
        """
        Initialize conversation memory manager.

        Args:
            conversations_dir: Directory for storing conversation files
            ttl_hours: Time-to-live for threads in hours
            max_messages: Maximum messages per thread
        """
        self.conversations_dir = conversations_dir
        self.ttl_hours = ttl_hours
        self.max_messages = max_messages

        # Ensure conversations directory exists
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ConversationMemory initialized: dir={conversations_dir}, "
            f"ttl={ttl_hours}h, max_messages={max_messages}"
        )

    # ========================================================================
    # Thread Creation and Management (task-1-3-1)
    # ========================================================================

    def create_thread(
        self,
        workflow_name: str,
        initial_context: dict[str, Any] | None = None,
        parent_thread_id: str | None = None,
    ) -> str:
        """
        Create new conversation thread.

        Generates UUID-based thread ID and initializes thread context.
        Thread is immediately persisted to disk.

        Args:
            workflow_name: Name of workflow creating this thread
            initial_context: Optional initial request parameters
            parent_thread_id: Optional parent thread for conversation chains

        Returns:
            Thread ID (UUID string) for use as continuation_id

        Example:
            >>> memory = ConversationMemory()
            >>> thread_id = memory.create_thread("consensus", {"prompt": "Analyze this"})
            >>> print(thread_id)
            '550e8400-e29b-41d4-a716-446655440000'
        """
        # Generate UUID for thread
        thread_id = str(uuid.uuid4())

        # Create thread context
        now = datetime.now(UTC).isoformat()
        thread = ConversationThread(
            thread_id=thread_id,
            parent_thread_id=parent_thread_id,
            created_at=now,
            last_updated_at=now,
            workflow_name=workflow_name,
            messages=[],
            state={},
            initial_context=initial_context or {},
            status="active",
        )

        # Persist to disk
        self._save_thread(thread)

        logger.info(f"Created thread {thread_id} for workflow '{workflow_name}'")
        return thread_id

    def get_thread(self, thread_id: str) -> ConversationThread | None:
        """
        Retrieve conversation thread by ID.

        Checks TTL and returns None if thread expired or doesn't exist.
        Expired threads are automatically deleted.

        Args:
            thread_id: Thread ID to retrieve

        Returns:
            ConversationThread if found and not expired, None otherwise

        Example:
            >>> thread = memory.get_thread(thread_id)
            >>> if thread:
            ...     print(f"Thread has {len(thread.messages)} messages")
        """
        thread_file = self.conversations_dir / f"{thread_id}.json"

        if not thread_file.exists():
            logger.debug(f"Thread {thread_id} not found")
            return None

        # Check TTL
        file_age = datetime.now(UTC) - datetime.fromtimestamp(
            thread_file.stat().st_mtime, UTC
        )
        if file_age > timedelta(hours=self.ttl_hours):
            logger.info(f"Thread {thread_id} expired (age: {file_age})")
            self._delete_thread(thread_id)
            return None

        # Load and parse
        lock = filelock.FileLock(f"{thread_file}.lock", timeout=5)
        try:
            with lock:
                with open(thread_file) as f:
                    data = json.load(f)
                    thread = ConversationThread(**data)
                    logger.debug(
                        f"Retrieved thread {thread_id} with {len(thread.messages)} messages"
                    )
                    return thread
        except filelock.Timeout:
            logger.error(f"Timeout acquiring lock for thread {thread_id}")
            return None
        except Exception as e:
            logger.error(f"Error loading thread {thread_id}: {e}")
            return None

    def get_thread_chain(
        self, thread_id: str, max_depth: int = 20
    ) -> list[ConversationThread]:
        """
        Retrieve thread and all parent threads in chronological order.

        Traverses parent_thread_id links up to max_depth to prevent
        circular references and infinite loops.

        Args:
            thread_id: Starting thread ID
            max_depth: Maximum parent chain depth

        Returns:
            List of threads in chronological order (oldest first)

        Example:
            >>> chain = memory.get_thread_chain(thread_id)
            >>> for thread in chain:
            ...     print(f"Thread {thread.thread_id}: {len(thread.messages)} messages")
        """
        chain: list[ConversationThread] = []
        current_id: str | None = thread_id
        depth = 0

        while current_id and depth < max_depth:
            thread = self.get_thread(current_id)
            if not thread:
                break

            chain.insert(0, thread)  # Add to beginning for chronological order
            current_id = thread.parent_thread_id
            depth += 1

        if depth >= max_depth:
            logger.warning(
                f"Thread chain depth limit reached ({max_depth}) for {thread_id}"
            )

        return chain

    # ========================================================================
    # Message Operations (task-1-3-2)
    # ========================================================================

    def add_message(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        files: list[str] | None = None,
        workflow_name: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add message to conversation thread.

        Thread-safe operation with file locking. Enforces max_messages limit
        by truncating oldest messages if exceeded. Updates thread's last_updated_at.

        Args:
            thread_id: Thread to add message to
            role: Message role ('user' or 'assistant')
            content: Message content
            files: Optional list of file paths referenced
            workflow_name: Optional workflow that generated message
            model_provider: Optional provider type used
            model_name: Optional model identifier
            metadata: Optional additional metadata

        Returns:
            True if successful, False if thread not found

        Example:
            >>> success = memory.add_message(
            ...     thread_id,
            ...     "user",
            ...     "Analyze this code",
            ...     files=["src/main.py"]
            ... )
        """
        thread = self.get_thread(thread_id)
        if not thread:
            # If thread doesn't exist, create it
            now = datetime.now(UTC).isoformat()
            thread = ConversationThread(
                thread_id=thread_id,
                created_at=now,
                last_updated_at=now,
                workflow_name=workflow_name or "unknown",
                messages=[],
                state={},
                initial_context={},
                status="active",
            )
            logger.info(f"Created new thread {thread_id} implicitly from add_message")

        # Create message
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(UTC).isoformat(),
            files=files,
            workflow_name=workflow_name,
            model_provider=model_provider,
            model_name=model_name,
            metadata=metadata or {},
        )

        # Add to thread
        thread.messages.append(message)

        # Enforce max messages limit (keep most recent)
        if len(thread.messages) > self.max_messages:
            removed_count = len(thread.messages) - self.max_messages
            thread.messages = thread.messages[-self.max_messages :]
            logger.warning(
                f"Thread {thread_id} exceeded max messages, "
                f"removed {removed_count} oldest messages"
            )

        # Update timestamp
        thread.last_updated_at = datetime.now(UTC).isoformat()

        # Persist
        self._save_thread(thread)

        logger.debug(
            f"Added {role} message to thread {thread_id} ({len(thread.messages)} total)"
        )
        return True

    def get_messages(
        self, thread_id: str, limit: int | None = None, role: str | None = None
    ) -> list[ConversationMessage]:
        """
        Retrieve messages from thread with optional filtering.

        Args:
            thread_id: Thread to get messages from
            limit: Optional limit on number of messages (most recent)
            role: Optional filter by role ('user' or 'assistant')

        Returns:
            List of messages in chronological order

        Example:
            >>> messages = memory.get_messages(thread_id, limit=10, role="user")
            >>> for msg in messages:
            ...     print(f"{msg.timestamp}: {msg.content[:50]}...")
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return []

        messages = thread.messages

        # Filter by role if specified
        if role:
            messages = [msg for msg in messages if msg.role == role]

        # Apply limit if specified (keep most recent)
        if limit and len(messages) > limit:
            messages = messages[-limit:]

        return messages

    # ========================================================================
    # Context Window Management (task-1-3-3)
    # ========================================================================

    def build_conversation_history(
        self,
        thread_id: str,
        max_messages: int | None = None,
        include_files: bool = True,
    ) -> tuple[str, int]:
        """
        Build formatted conversation history for context injection.

        Constructs human-readable conversation history with file context,
        using newest-first prioritization for both files and messages.
        Adapted from Zen MCP's sophisticated build strategy.

        Args:
            thread_id: Thread to build history from
            max_messages: Optional limit on messages to include
            include_files: Whether to embed file contents

        Returns:
            Tuple of (formatted_history, message_count)

        Format:
            === CONVERSATION HISTORY (CONTINUATION) ===
            Thread: {thread_id}
            Workflow: {workflow_name}
            Turn {current_turn}/{total_turns}

            [Files section if include_files=True]

            Previous conversation:

            --- Turn 1 (user) ---
            <content>

            --- Turn 2 (assistant via claude-3-opus) ---
            <content>

            === END CONVERSATION HISTORY ===

        Example:
            >>> history, count = memory.build_conversation_history(thread_id, max_messages=10)
            >>> enhanced_prompt = f"{history}\\n\\n{new_user_input}"
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return "", 0

        messages = thread.messages

        # Limit messages if specified (keep most recent)
        if max_messages and len(messages) > max_messages:
            messages = messages[-max_messages:]
            truncated = len(thread.messages) - len(messages)
        else:
            truncated = 0

        # Build header
        lines = [
            "=== CONVERSATION HISTORY (CONTINUATION) ===",
            f"Thread: {thread_id}",
            f"Workflow: {thread.workflow_name}",
            f"Messages: {len(messages)}/{len(thread.messages)}",
            "",
        ]

        if truncated > 0:
            lines.append(f"[NOTE: {truncated} earlier messages omitted]")
            lines.append("")

        # Build files section if requested
        if include_files:
            # Collect unique files (newest-first deduplication)
            file_map = {}  # path -> latest message index
            for idx, msg in enumerate(reversed(messages)):
                if msg.files:
                    for file_path in msg.files:
                        if file_path not in file_map:
                            file_map[file_path] = len(messages) - 1 - idx

            if file_map:
                lines.append("=== FILES REFERENCED ===")
                for file_path in sorted(file_map.keys()):
                    lines.append(
                        f"- {file_path} (referenced in turn {file_map[file_path] + 1})"
                    )
                lines.append("=== END FILES ===")
                lines.append("")

        # Build conversation turns
        lines.append("Previous conversation:")
        lines.append("")

        for idx, msg in enumerate(messages, 1):
            # Turn header
            model_info = ""
            if msg.model_name:
                provider = msg.model_provider or "unknown"
                model_info = f" via {msg.model_name} ({provider})"

            lines.append(f"--- Turn {idx} ({msg.role}{model_info}) ---")

            if msg.files:
                lines.append(f"Files: {', '.join(msg.files)}")

            lines.append("")
            lines.append(msg.content)
            lines.append("")

        lines.append("=== END CONVERSATION HISTORY ===")
        lines.append("")
        lines.append(
            "IMPORTANT: You are continuing an existing conversation thread. "
            f"Use the history above for context. This is message {len(messages) + 1} "
            "in the conversation."
        )

        history = "\n".join(lines)
        return history, len(messages)

    def get_context_summary(self, thread_id: str) -> dict[str, Any]:
        """
        Get summary statistics for thread context.

        Useful for context window management and debugging.

        Args:
            thread_id: Thread to summarize

        Returns:
            Dict with summary statistics

        Example:
            >>> summary = memory.get_context_summary(thread_id)
            >>> print(f"Total messages: {summary['message_count']}")
            >>> print(f"Files referenced: {summary['unique_files']}")
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return {
                "found": False,
                "message_count": 0,
                "unique_files": 0,
                "user_messages": 0,
                "assistant_messages": 0,
            }

        # Collect statistics
        unique_files = set()
        user_count = 0
        assistant_count = 0

        for msg in thread.messages:
            if msg.role == "user":
                user_count += 1
            elif msg.role == "assistant":
                assistant_count += 1

            if msg.files:
                unique_files.update(msg.files)

        return {
            "found": True,
            "thread_id": thread_id,
            "workflow_name": thread.workflow_name,
            "status": thread.status,
            "message_count": len(thread.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "unique_files": len(unique_files),
            "created_at": thread.created_at,
            "last_updated_at": thread.last_updated_at,
            "has_parent": thread.parent_thread_id is not None,
        }

    # ========================================================================
    # Thread Lifecycle Management (task-1-3-4)
    # ========================================================================

    def complete_thread(self, thread_id: str) -> bool:
        """
        Mark thread as completed.

        Completed threads remain accessible but are marked for archival.

        Args:
            thread_id: Thread to mark as completed

        Returns:
            True if successful, False if thread not found

        Example:
            >>> memory.complete_thread(thread_id)
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return False

        thread.status = "completed"
        thread.last_updated_at = datetime.now(UTC).isoformat()
        self._save_thread(thread)

        logger.info(f"Thread {thread_id} marked as completed")
        return True

    def archive_thread(self, thread_id: str) -> bool:
        """
        Mark thread as archived.

        Archived threads are eligible for cleanup.

        Args:
            thread_id: Thread to archive

        Returns:
            True if successful, False if thread not found

        Example:
            >>> memory.archive_thread(thread_id)
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return False

        thread.status = "archived"
        thread.last_updated_at = datetime.now(UTC).isoformat()
        self._save_thread(thread)

        logger.info(f"Thread {thread_id} archived")
        return True

    def cleanup_expired_threads(self) -> int:
        """
        Remove threads older than TTL.

        Checks all thread files and deletes those past their TTL.
        Safe to call periodically for maintenance.

        Returns:
            Number of threads deleted

        Example:
            >>> deleted = memory.cleanup_expired_threads()
            >>> print(f"Cleaned up {deleted} expired threads")
        """
        now = datetime.now(UTC)
        ttl = timedelta(hours=self.ttl_hours)
        deleted = 0

        for thread_file in self.conversations_dir.glob("*.json"):
            try:
                file_age = now - datetime.fromtimestamp(
                    thread_file.stat().st_mtime, UTC
                )
                if file_age > ttl:
                    thread_id = thread_file.stem
                    self._delete_thread(thread_id)
                    deleted += 1
            except Exception as e:
                logger.error(f"Error checking thread {thread_file}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired threads")

        return deleted

    def cleanup_archived_threads(self) -> int:
        """
        Remove archived threads regardless of TTL.

        Removes all threads with status='archived'.
        Useful for explicit cleanup after workflows complete.

        Returns:
            Number of threads deleted

        Example:
            >>> deleted = memory.cleanup_archived_threads()
            >>> print(f"Cleaned up {deleted} archived threads")
        """
        deleted = 0

        for thread_file in self.conversations_dir.glob("*.json"):
            try:
                thread_id = thread_file.stem
                thread = self.get_thread(thread_id)

                if thread and thread.status == "archived":
                    self._delete_thread(thread_id)
                    deleted += 1
            except Exception as e:
                logger.error(f"Error checking thread {thread_file}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} archived threads")

        return deleted

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _save_thread(self, thread: ConversationThread) -> None:
        """
        Save thread to disk with file locking.

        Args:
            thread: Thread to persist
        """
        thread_file = self.conversations_dir / f"{thread.thread_id}.json"
        lock = filelock.FileLock(f"{thread_file}.lock", timeout=5)

        try:
            with lock:
                with open(thread_file, "w") as f:
                    json.dump(thread.model_dump(), f, indent=2)
        except filelock.Timeout:
            logger.error(f"Timeout acquiring lock for thread {thread.thread_id}")
            raise
        except Exception as e:
            logger.error(f"Error saving thread {thread.thread_id}: {e}")
            raise

    def _delete_thread(self, thread_id: str) -> None:
        """
        Delete thread file and lock file.

        Args:
            thread_id: Thread to delete
        """
        thread_file = self.conversations_dir / f"{thread_id}.json"
        lock_file = Path(f"{thread_file}.lock")

        try:
            if thread_file.exists():
                thread_file.unlink()
            if lock_file.exists():
                lock_file.unlink()
            logger.debug(f"Deleted thread {thread_id}")
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
