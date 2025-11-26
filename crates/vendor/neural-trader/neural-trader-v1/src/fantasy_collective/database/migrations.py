"""
Database Migration Manager for Fantasy Collective System

Handles database schema migrations and version management.
"""

import os
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import hashlib

from .connection import DatabaseConnection, get_db

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of migration content."""
        content = f"{self.version}{self.name}{self.up_sql}{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def __str__(self):
        return f"Migration {self.version}: {self.name}"


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_db()
        self.migrations: List[Migration] = []
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER
        )
        """
        self.db.execute_modify(create_table_sql)
        logger.info("Migration tracking table initialized")
    
    def add_migration(self, migration: Migration):
        """Add a migration to the manager."""
        # Check for version conflicts
        existing = next((m for m in self.migrations if m.version == migration.version), None)
        if existing:
            raise ValueError(f"Migration version {migration.version} already exists")
        
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
        logger.debug(f"Added migration: {migration}")
    
    def load_migrations_from_directory(self, migrations_dir: str):
        """Load migrations from SQL files in a directory."""
        migrations_path = Path(migrations_dir)
        if not migrations_path.exists():
            logger.warning(f"Migrations directory does not exist: {migrations_dir}")
            return
        
        sql_files = sorted(migrations_path.glob("*.sql"))
        for sql_file in sql_files:
            self._load_migration_from_file(sql_file)
    
    def _load_migration_from_file(self, file_path: Path):
        """Load a single migration from a SQL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse version and name from filename (format: V001__migration_name.sql)
            filename = file_path.stem
            parts = filename.split('__', 1)
            
            if len(parts) != 2:
                logger.warning(f"Skipping file with invalid format: {filename}")
                return
            
            version = parts[0]
            name = parts[1].replace('_', ' ').title()
            
            # Split up and down migrations if present
            up_sql = content
            down_sql = ""
            
            if "-- DOWN MIGRATION" in content:
                parts = content.split("-- DOWN MIGRATION", 1)
                up_sql = parts[0].strip()
                down_sql = parts[1].strip()
            
            migration = Migration(version, name, up_sql, down_sql)
            self.add_migration(migration)
            
            logger.debug(f"Loaded migration from file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load migration from {file_path}: {e}")
    
    def get_applied_migrations(self) -> List[Dict]:
        """Get list of applied migrations from database."""
        query = """
        SELECT version, name, checksum, applied_at, execution_time_ms 
        FROM schema_migrations 
        ORDER BY version
        """
        results = self.db.execute_query(query)
        return [dict(row) for row in results]
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied."""
        applied = {row['version']: row for row in self.get_applied_migrations()}
        pending = []
        
        for migration in self.migrations:
            if migration.version not in applied:
                pending.append(migration)
            else:
                # Check if checksum has changed
                if applied[migration.version]['checksum'] != migration.checksum:
                    logger.warning(f"Migration {migration.version} checksum mismatch - may have been modified")
        
        return pending
    
    def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """
        Apply pending migrations up to target version.
        
        Args:
            target_version: Optional target version. If None, applies all pending migrations.
            
        Returns:
            List of applied migration versions.
        """
        pending = self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations to apply")
            return []
        
        applied_versions = []
        
        for migration in pending:
            try:
                logger.info(f"Applying migration: {migration}")
                start_time = datetime.now()
                
                with self.db.transaction():
                    # Execute migration SQL
                    self._execute_migration_sql(migration.up_sql)
                    
                    # Record migration as applied
                    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    self._record_migration(migration, execution_time)
                
                applied_versions.append(migration.version)
                logger.info(f"Successfully applied migration {migration.version}")
                
            except Exception as e:
                logger.error(f"Failed to apply migration {migration.version}: {e}")
                raise
        
        logger.info(f"Applied {len(applied_versions)} migrations")
        return applied_versions
    
    def rollback(self, target_version: str) -> List[str]:
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Version to rollback to.
            
        Returns:
            List of rolled back migration versions.
        """
        applied = self.get_applied_migrations()
        to_rollback = [m for m in applied if m['version'] > target_version]
        to_rollback.reverse()  # Rollback in reverse order
        
        if not to_rollback:
            logger.info(f"Already at or below target version {target_version}")
            return []
        
        rolled_back_versions = []
        
        for migration_info in to_rollback:
            migration = next(
                (m for m in self.migrations if m.version == migration_info['version']), 
                None
            )
            
            if not migration:
                logger.error(f"Migration {migration_info['version']} not found in loaded migrations")
                continue
            
            if not migration.down_sql:
                logger.error(f"No rollback SQL for migration {migration.version}")
                continue
            
            try:
                logger.info(f"Rolling back migration: {migration}")
                
                with self.db.transaction():
                    # Execute rollback SQL
                    self._execute_migration_sql(migration.down_sql)
                    
                    # Remove migration record
                    self.db.execute_modify(
                        "DELETE FROM schema_migrations WHERE version = ?",
                        (migration.version,)
                    )
                
                rolled_back_versions.append(migration.version)
                logger.info(f"Successfully rolled back migration {migration.version}")
                
            except Exception as e:
                logger.error(f"Failed to rollback migration {migration.version}: {e}")
                raise
        
        logger.info(f"Rolled back {len(rolled_back_versions)} migrations")
        return rolled_back_versions
    
    def _execute_migration_sql(self, sql: str):
        """Execute migration SQL, handling multiple statements."""
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement:
                self.db.execute_modify(statement)
    
    def _record_migration(self, migration: Migration, execution_time_ms: int):
        """Record migration as applied in the database."""
        self.db.execute_modify(
            """
            INSERT INTO schema_migrations (version, name, checksum, execution_time_ms)
            VALUES (?, ?, ?, ?)
            """,
            (migration.version, migration.name, migration.checksum, execution_time_ms)
        )
    
    def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        result = self.db.execute_single(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        )
        return result['version'] if result else None
    
    def create_migration_file(self, name: str, up_sql: str, down_sql: str = "", 
                             migrations_dir: str = None) -> str:
        """
        Create a new migration file.
        
        Args:
            name: Migration name
            up_sql: SQL for applying the migration
            down_sql: SQL for rolling back the migration
            migrations_dir: Directory to create the file in
            
        Returns:
            Path to created migration file
        """
        if migrations_dir is None:
            migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        
        # Ensure migrations directory exists
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Generate version number based on timestamp
        version = datetime.now().strftime("V%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{version}__{name.replace(' ', '_').lower()}.sql"
        file_path = os.path.join(migrations_dir, filename)
        
        # Create migration content
        content = f"-- Migration: {name}\n"
        content += f"-- Created: {datetime.now().isoformat()}\n\n"
        content += "-- UP MIGRATION\n"
        content += up_sql
        
        if down_sql:
            content += "\n\n-- DOWN MIGRATION\n"
            content += down_sql
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created migration file: {file_path}")
        return file_path
    
    def status(self) -> Dict:
        """Get migration status information."""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        current_version = self.get_current_version()
        
        return {
            'current_version': current_version,
            'total_migrations': len(self.migrations),
            'applied_count': len(applied),
            'pending_count': len(pending),
            'applied_migrations': applied,
            'pending_migrations': [
                {
                    'version': m.version,
                    'name': m.name,
                    'checksum': m.checksum
                }
                for m in pending
            ]
        }
    
    def validate_migrations(self) -> Dict:
        """Validate migration consistency."""
        issues = []
        applied = {row['version']: row for row in self.get_applied_migrations()}
        
        # Check for missing migrations
        for version, migration_info in applied.items():
            migration = next((m for m in self.migrations if m.version == version), None)
            if not migration:
                issues.append(f"Applied migration {version} not found in current migrations")
                continue
            
            # Check checksum
            if migration.checksum != migration_info['checksum']:
                issues.append(f"Migration {version} checksum mismatch - may have been modified")
        
        # Check for gaps in version sequence
        versions = sorted([m.version for m in self.migrations])
        applied_versions = sorted(applied.keys())
        
        for i, version in enumerate(applied_versions):
            if version not in versions:
                continue
            
            # Check if there are unapplied migrations before this one
            version_index = versions.index(version)
            for j in range(version_index):
                if versions[j] not in applied_versions:
                    issues.append(f"Unapplied migration {versions[j]} before applied migration {version}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


def create_initial_migration(migrations_dir: str = None) -> str:
    """Create initial migration with the complete schema."""
    if migrations_dir is None:
        migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
    
    # Read the schema.sql file
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    # Create migration manager and create initial migration
    manager = MigrationManager()
    
    down_sql = """
    -- Drop all tables in reverse dependency order
    DROP TABLE IF EXISTS audit_logs;
    DROP TABLE IF EXISTS system_notifications;
    DROP TABLE IF EXISTS user_notification_preferences;
    DROP TABLE IF EXISTS system_config;
    DROP TABLE IF EXISTS maintenance_log;
    
    DROP TABLE IF EXISTS user_badges;
    DROP TABLE IF EXISTS rewards;
    DROP TABLE IF EXISTS user_achievements;
    DROP TABLE IF EXISTS achievements;
    
    DROP TABLE IF EXISTS global_rankings;
    DROP TABLE IF EXISTS league_rankings;
    DROP TABLE IF EXISTS scoring_events;
    DROP TABLE IF EXISTS scoring_periods;
    
    DROP TABLE IF EXISTS prediction_group_members;
    DROP TABLE IF EXISTS prediction_groups;
    DROP TABLE IF EXISTS predictions;
    DROP TABLE IF EXISTS prediction_markets;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS event_categories;
    
    DROP TABLE IF EXISTS team_transactions;
    DROP TABLE IF EXISTS team_rosters;
    DROP TABLE IF EXISTS league_participants;
    DROP TABLE IF EXISTS league_seasons;
    DROP TABLE IF EXISTS league_templates;
    DROP TABLE IF EXISTS leagues;
    
    DROP TABLE IF EXISTS payment_methods;
    DROP TABLE IF EXISTS withdrawal_requests;
    DROP TABLE IF EXISTS transactions;
    DROP TABLE IF EXISTS user_wallets;
    
    DROP TABLE IF EXISTS user_verifications;
    DROP TABLE IF EXISTS user_preferences;
    DROP TABLE IF EXISTS user_sessions;
    DROP TABLE IF EXISTS users;
    
    DROP TABLE IF EXISTS schema_migrations;
    """
    
    return manager.create_migration_file(
        "Initial Schema", 
        schema_sql, 
        down_sql,
        migrations_dir
    )