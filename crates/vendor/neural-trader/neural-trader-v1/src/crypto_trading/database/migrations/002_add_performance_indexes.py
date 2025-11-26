"""Add performance optimization indexes

Adds additional indexes for improved query performance.
"""

import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MIGRATION_VERSION = "002"
MIGRATION_NAME = "add_performance_indexes"


def upgrade(db_path: str):
    """Apply the migration"""
    logger.info(f"Applying migration {MIGRATION_VERSION}: {MIGRATION_NAME}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Additional composite indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vault_positions_status_chain 
            ON vault_positions(status, chain)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_yield_history_composite 
            ON yield_history(position_id, recorded_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_crypto_transactions_composite 
            ON crypto_transactions(status, created_at)
        """)
        
        # Index for calculating portfolio value by chain
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vault_positions_value 
            ON vault_positions(status, current_value)
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
        # Drop the new indexes
        indexes = [
            'idx_vault_positions_status_chain',
            'idx_yield_history_composite',
            'idx_crypto_transactions_composite',
            'idx_vault_positions_value'
        ]
        
        for index in indexes:
            cursor.execute(f"DROP INDEX IF EXISTS {index}")
        
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
    except:
        return False
    finally:
        conn.close()