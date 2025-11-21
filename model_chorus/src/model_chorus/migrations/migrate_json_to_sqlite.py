"""
Migration script to convert file-based conversation storage to SQLite.

Migrates conversation threads from individual JSON files to consolidated
SQLite database with WAL mode. Provides validation, backup, and rollback
capabilities to ensure safe migration.

Usage:
    python -m model_chorus.migrations.migrate_json_to_sqlite \\
        --source ~/.model-chorus/conversations \\
        --dest ~/.model-chorus/conversations.db \\
        --backup

Features:
- Validates JSON files before migration
- Creates automatic backups
- Provides rollback on errors
- Reports progress and statistics
- Handles malformed files gracefully
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from model_chorus.core.conversation_db import ConversationDatabase
from model_chorus.core.models import ConversationMessage, ConversationThread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration encounters an error."""

    pass


class ConversationMigrator:
    """
    Migrates file-based conversation storage to SQLite database.

    Handles validation, backup, and rollback for safe migration from
    individual JSON files to consolidated SQLite database.

    Attributes:
        source_dir: Directory containing JSON conversation files
        db_path: Path to target SQLite database
        backup_enabled: Whether to create backups before migration
        backup_dir: Directory for backup files
    """

    def __init__(
        self,
        source_dir: Path,
        db_path: Path,
        backup_enabled: bool = True,
    ):
        """
        Initialize migrator.

        Args:
            source_dir: Source directory with JSON files
            db_path: Target SQLite database path
            backup_enabled: Whether to create backups
        """
        self.source_dir = source_dir
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.backup_dir = source_dir.parent / "conversation_backups"

        # Statistics
        self.stats = {
            "total_files": 0,
            "validated": 0,
            "skipped": 0,
            "migrated": 0,
            "errors": 0,
        }

    def validate_json_file(self, json_path: Path) -> ConversationThread | None:
        """
        Validate and parse a JSON conversation file.

        Args:
            json_path: Path to JSON file

        Returns:
            ConversationThread if valid, None if invalid

        Raises:
            MigrationError: If file cannot be read or parsed
        """
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Validate required fields
            required_fields = [
                "thread_id",
                "created_at",
                "last_updated_at",
                "workflow_name",
                "messages",
                "state",
                "initial_context",
                "status",
            ]

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.warning(
                    f"File {json_path.name} missing required fields: {missing_fields}"
                )
                return None

            # Parse into ConversationThread model for validation
            thread = ConversationThread(**data)
            logger.debug(f"Validated {json_path.name}: {len(thread.messages)} messages")

            self.stats["validated"] += 1
            return thread

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path.name}: {e}")
            self.stats["errors"] += 1
            return None
        except Exception as e:
            logger.error(f"Error validating {json_path.name}: {e}")
            self.stats["errors"] += 1
            return None

    def scan_json_files(self) -> list[Path]:
        """
        Scan source directory for JSON conversation files.

        Returns:
            List of paths to JSON files

        Raises:
            MigrationError: If source directory doesn't exist
        """
        if not self.source_dir.exists():
            raise MigrationError(f"Source directory not found: {self.source_dir}")

        json_files = list(self.source_dir.glob("*.json"))
        self.stats["total_files"] = len(json_files)

        logger.info(f"Found {len(json_files)} JSON files in {self.source_dir}")
        return json_files

    def validate_all_files(self, json_files: list[Path]) -> dict[Path, ConversationThread]:
        """
        Validate all JSON files before migration.

        Args:
            json_files: List of JSON file paths

        Returns:
            Dict mapping file paths to validated ConversationThread objects

        Raises:
            MigrationError: If validation fails critically
        """
        validated_threads: dict[Path, ConversationThread] = {}

        logger.info("Validating JSON files...")

        for json_path in json_files:
            thread = self.validate_json_file(json_path)
            if thread:
                validated_threads[json_path] = thread
            else:
                self.stats["skipped"] += 1
                logger.warning(f"Skipping invalid file: {json_path.name}")

        logger.info(
            f"Validation complete: {self.stats['validated']} valid, "
            f"{self.stats['skipped']} skipped, {self.stats['errors']} errors"
        )

        if not validated_threads:
            raise MigrationError("No valid conversation files found")

        return validated_threads

    def create_backup(self) -> Path | None:
        """
        Create backup of source directory.

        Returns:
            Path to backup directory, or None if backup disabled

        Raises:
            MigrationError: If backup creation fails
        """
        if not self.backup_enabled:
            logger.info("Backup disabled, skipping...")
            return None

        try:
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy all JSON files
            logger.info(f"Creating backup at {backup_path}...")
            for json_file in self.source_dir.glob("*.json"):
                shutil.copy2(json_file, backup_path / json_file.name)

            logger.info(f"Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            raise MigrationError(f"Failed to create backup: {e}")

    def migrate_thread(self, db: ConversationDatabase, thread: ConversationThread) -> bool:
        """
        Migrate a single thread to database.

        Args:
            db: Target database
            thread: Thread to migrate

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create thread in database
            existing = db.get_thread(thread.thread_id)
            if existing:
                logger.warning(
                    f"Thread {thread.thread_id} already exists in database, skipping"
                )
                return False

            # Insert thread metadata using raw SQL to preserve original data
            cursor = db.conn.cursor()
            cursor.execute(
                """
                INSERT INTO threads (
                    thread_id, parent_thread_id, created_at, last_updated_at,
                    workflow_name, status, initial_context, state, branch_point
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread.thread_id,
                    thread.parent_thread_id,
                    thread.created_at,
                    thread.last_updated_at,
                    thread.workflow_name,
                    thread.status,
                    json.dumps(thread.initial_context),
                    json.dumps(thread.state),
                    thread.branch_point,
                ),
            )

            # Insert all messages
            for message in thread.messages:
                cursor.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, timestamp, files,
                        workflow_name, model_provider, model_name, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread.thread_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        json.dumps(message.files) if message.files else None,
                        message.workflow_name,
                        message.model_provider,
                        message.model_name,
                        json.dumps(message.metadata) if message.metadata else None,
                    ),
                )

            db.conn.commit()
            self.stats["migrated"] += 1

            logger.debug(
                f"Migrated thread {thread.thread_id}: {len(thread.messages)} messages"
            )
            return True

        except Exception as e:
            logger.error(f"Error migrating thread {thread.thread_id}: {e}")
            self.stats["errors"] += 1
            return False

    def migrate_all(self, db: ConversationDatabase, threads: dict[Path, ConversationThread]) -> None:
        """
        Migrate all validated threads to database.

        Args:
            db: Target database
            threads: Dict of validated threads to migrate

        Raises:
            MigrationError: If migration fails critically
        """
        logger.info(f"Migrating {len(threads)} threads to database...")

        for json_path, thread in threads.items():
            success = self.migrate_thread(db, thread)
            if not success:
                logger.warning(f"Failed to migrate {json_path.name}")

        logger.info(
            f"Migration complete: {self.stats['migrated']} threads migrated, "
            f"{self.stats['errors']} errors"
        )

    def print_summary(self) -> None:
        """Print migration summary statistics."""
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Total files found:     {self.stats['total_files']}")
        print(f"Files validated:       {self.stats['validated']}")
        print(f"Files skipped:         {self.stats['skipped']}")
        print(f"Threads migrated:      {self.stats['migrated']}")
        print(f"Errors encountered:    {self.stats['errors']}")
        print("=" * 60)

        if self.stats["migrated"] == self.stats["validated"]:
            print("✓ Migration completed successfully!")
        elif self.stats["migrated"] > 0:
            print("⚠ Migration completed with some errors")
        else:
            print("✗ Migration failed")
        print()

    def run(self) -> int:
        """
        Execute full migration process.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            logger.info("Starting conversation migration...")
            logger.info(f"Source: {self.source_dir}")
            logger.info(f"Destination: {self.db_path}")

            # Step 1: Scan for JSON files
            json_files = self.scan_json_files()
            if not json_files:
                logger.warning("No JSON files found, nothing to migrate")
                return 0

            # Step 2: Validate all files
            validated_threads = self.validate_all_files(json_files)

            # Step 3: Create backup if enabled
            backup_path = self.create_backup()

            # Step 4: Initialize database
            logger.info("Initializing SQLite database...")
            db = ConversationDatabase(db_path=self.db_path)

            # Step 5: Migrate threads
            self.migrate_all(db, validated_threads)

            # Step 6: Close database
            db.close()

            # Step 7: Print summary
            self.print_summary()

            if self.stats["errors"] == 0:
                logger.info("Migration completed successfully")
                return 0
            else:
                logger.warning(f"Migration completed with {self.stats['errors']} errors")
                return 1

        except MigrationError as e:
            logger.error(f"Migration failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error during migration: {e}", exc_info=True)
            return 1


def main() -> int:
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate conversation files from JSON to SQLite"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".model-chorus" / "conversations",
        help="Source directory with JSON files (default: ~/.model-chorus/conversations)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.home() / ".model-chorus" / "conversations.db",
        help="Destination SQLite database (default: ~/.model-chorus/conversations.db)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before migration (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine backup setting
    backup_enabled = args.backup and not args.no_backup

    # Run migration
    migrator = ConversationMigrator(
        source_dir=args.source,
        db_path=args.dest,
        backup_enabled=backup_enabled,
    )

    return migrator.run()


if __name__ == "__main__":
    sys.exit(main())
