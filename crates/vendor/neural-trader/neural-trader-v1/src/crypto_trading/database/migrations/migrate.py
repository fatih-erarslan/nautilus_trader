"""Database migration runner

Handles applying and rolling back database migrations in order.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationRunner:
    """Runs database migrations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations_dir = Path(__file__).parent
        
    def get_migration_files(self) -> List[Tuple[str, Path]]:
        """Get all migration files sorted by version"""
        migrations = []
        
        for file in self.migrations_dir.glob("*.py"):
            if file.name.startswith(("001", "002", "003")):  # Migration files
                version = file.name.split("_")[0]
                migrations.append((version, file))
        
        # Sort by version number
        migrations.sort(key=lambda x: x[0])
        return migrations
    
    def load_migration(self, migration_path: Path):
        """Dynamically load a migration module"""
        spec = importlib.util.spec_from_file_location(
            migration_path.stem,
            migration_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def run_migrations(self, target_version: str = None):
        """
        Run all pending migrations up to target version
        
        Args:
            target_version: Version to migrate to (None for latest)
        """
        migrations = self.get_migration_files()
        
        if not migrations:
            logger.info("No migrations found")
            return
        
        applied_count = 0
        
        for version, migration_path in migrations:
            # Stop if we've reached target version
            if target_version and version > target_version:
                break
            
            # Load migration module
            migration = self.load_migration(migration_path)
            
            # Check if already applied
            if hasattr(migration, 'is_applied') and migration.is_applied(self.db_path):
                logger.info(f"Migration {version} already applied, skipping")
                continue
            
            # Apply migration
            try:
                logger.info(f"Applying migration {version}: {migration_path.name}")
                migration.upgrade(self.db_path)
                applied_count += 1
            except Exception as e:
                logger.error(f"Failed to apply migration {version}: {e}")
                raise
        
        logger.info(f"Applied {applied_count} migrations")
    
    def rollback_migration(self, target_version: str):
        """
        Rollback to a specific version
        
        Args:
            target_version: Version to rollback to
        """
        migrations = self.get_migration_files()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version, migration_path in reversed(migrations):
            if version > target_version:
                to_rollback.append((version, migration_path))
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return
        
        for version, migration_path in to_rollback:
            migration = self.load_migration(migration_path)
            
            # Check if applied before rolling back
            if hasattr(migration, 'is_applied') and not migration.is_applied(self.db_path):
                logger.info(f"Migration {version} not applied, skipping rollback")
                continue
            
            try:
                logger.info(f"Rolling back migration {version}: {migration_path.name}")
                migration.downgrade(self.db_path)
            except Exception as e:
                logger.error(f"Failed to rollback migration {version}: {e}")
                raise
    
    def get_current_version(self) -> str:
        """Get the current migration version"""
        migrations = self.get_migration_files()
        
        for version, migration_path in reversed(migrations):
            migration = self.load_migration(migration_path)
            if hasattr(migration, 'is_applied') and migration.is_applied(self.db_path):
                return version
        
        return "000"  # No migrations applied


def main():
    """CLI for running migrations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "command",
        choices=["migrate", "rollback", "status"],
        help="Command to run"
    )
    parser.add_argument(
        "--db-path",
        default="data/crypto_trading/trading.db",
        help="Path to database file"
    )
    parser.add_argument(
        "--version",
        help="Target version for migrate/rollback"
    )
    
    args = parser.parse_args()
    
    # Ensure database directory exists
    db_dir = os.path.dirname(args.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    runner = MigrationRunner(args.db_path)
    
    if args.command == "migrate":
        runner.run_migrations(args.version)
    elif args.command == "rollback":
        if not args.version:
            print("--version required for rollback")
            sys.exit(1)
        runner.rollback_migration(args.version)
    elif args.command == "status":
        current = runner.get_current_version()
        print(f"Current migration version: {current}")
        
        # Show pending migrations
        migrations = runner.get_migration_files()
        pending = []
        for version, path in migrations:
            migration = runner.load_migration(path)
            if hasattr(migration, 'is_applied') and not migration.is_applied(args.db_path):
                pending.append(version)
        
        if pending:
            print(f"Pending migrations: {', '.join(pending)}")
        else:
            print("All migrations applied")


if __name__ == "__main__":
    main()