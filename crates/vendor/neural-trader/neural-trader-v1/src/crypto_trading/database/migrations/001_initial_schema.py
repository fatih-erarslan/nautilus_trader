"""Initial database schema migration

Creates all base tables for the crypto trading system.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATION_VERSION = "001"
MIGRATION_NAME = "initial_schema"


def upgrade(db_path: str):
    """Apply the migration"""
    logger.info(f"Applying migration {MIGRATION_VERSION}: {MIGRATION_NAME}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Read and execute schema
        schema_path = Path(__file__).parent.parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        cursor.executescript(schema_sql)
        
        # Create migration tracking table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Record migration
        cursor.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (MIGRATION_VERSION, MIGRATION_NAME)
        )
        
        conn.commit()
        logger.info(f"Migration {MIGRATION_VERSION} applied successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration {MIGRATION_VERSION} failed: {e}")
        raise
    finally:
        conn.close()


def downgrade(db_path: str):
    """Rollback the migration"""
    logger.info(f"Rolling back migration {MIGRATION_VERSION}: {MIGRATION_NAME}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Drop all tables
        tables = [
            'vault_positions',
            'yield_history',
            'crypto_transactions',
            'portfolio_summary'
        ]
        
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Drop views
        cursor.execute("DROP VIEW IF EXISTS active_positions_summary")
        cursor.execute("DROP VIEW IF EXISTS daily_portfolio_performance")
        
        # Remove migration record
        cursor.execute(
            "DELETE FROM schema_migrations WHERE version = ?",
            (MIGRATION_VERSION,)
        )
        
        conn.commit()
        logger.info(f"Migration {MIGRATION_VERSION} rolled back successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Rollback of migration {MIGRATION_VERSION} failed: {e}")
        raise
    finally:
        conn.close()


def is_applied(db_path: str) -> bool:
    """Check if migration has been applied"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 1 FROM schema_migrations 
            WHERE version = ?
        """, (MIGRATION_VERSION,))
        
        return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        # Migration table doesn't exist yet
        return False
    finally:
        conn.close()