"""
Database Connection Manager for Fantasy Collective System

Provides connection pooling, transaction management, and database utilities.
"""

import sqlite3
import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import os

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Thread-safe SQLite database connection manager with connection pooling.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[str] = None):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection manager."""
        if hasattr(self, 'initialized'):
            return
            
        self.db_path = db_path or self._get_default_db_path()
        self.connections = {}
        self.connection_lock = threading.Lock()
        self.initialized = True
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        logger.info(f"Database connection manager initialized: {self.db_path}")
    
    def _get_default_db_path(self) -> str:
        """Get default database path based on environment."""
        env = os.getenv('ENVIRONMENT', 'development')
        base_dir = Path(__file__).parent.parent.parent.parent
        
        if env == 'production':
            return str(base_dir / 'data' / 'production' / 'fantasy_collective.db')
        elif env == 'test':
            return str(base_dir / 'data' / 'test' / 'fantasy_collective_test.db')
        else:
            return str(base_dir / 'data' / 'development' / 'fantasy_collective_dev.db')
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        thread_id = threading.get_ident()
        
        with self.connection_lock:
            if thread_id not in self.connections:
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0,
                    isolation_level=None  # Use autocommit mode by default
                )
                
                # Configure connection
                conn.row_factory = sqlite3.Row  # Enable column access by name
                
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Set WAL mode for better concurrent access
                conn.execute("PRAGMA journal_mode = WAL")
                
                # Optimize performance
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                conn.execute("PRAGMA temp_store = memory")
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map
                
                self.connections[thread_id] = conn
                logger.debug(f"Created new database connection for thread {thread_id}")
        
        return self.connections[thread_id]
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """Context manager for database cursor with automatic commit/rollback."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    @contextmanager
    def transaction(self):
        """Context manager for explicit database transactions."""
        conn = self.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        
        try:
            yield conn
            conn.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_single(self, query: str, params: Optional[Tuple] = None) -> Optional[sqlite3.Row]:
        """Execute a SELECT query and return single result."""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
    
    def execute_modify(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def get_last_insert_id(self) -> int:
        """Get the last inserted row ID."""
        conn = self.get_connection()
        return conn.lastrowid
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
        """
        result = self.execute_single(query, (table_name,))
        return result is not None
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get information about table columns."""
        with self.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            return [dict(col) for col in columns]
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key information for a table."""
        with self.get_cursor() as cursor:
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            return [dict(fk) for fk in fks]
    
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        with self.get_cursor() as cursor:
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            return [dict(idx) for idx in indexes]
    
    def vacuum(self):
        """Vacuum the database to reclaim space and optimize."""
        conn = self.get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuum completed")
    
    def analyze(self):
        """Analyze the database to update query planner statistics."""
        conn = self.get_connection()
        conn.execute("ANALYZE")
        logger.info("Database analysis completed")
    
    def backup(self, backup_path: str):
        """Create a backup of the database."""
        conn = self.get_connection()
        backup_conn = sqlite3.connect(backup_path)
        
        try:
            conn.backup(backup_conn)
            logger.info(f"Database backup created: {backup_path}")
        finally:
            backup_conn.close()
    
    def get_database_size(self) -> Dict[str, int]:
        """Get database size statistics."""
        query = """
        SELECT 
            page_count * page_size as total_size,
            (page_count - freelist_count) * page_size as used_size,
            freelist_count * page_size as free_size
        FROM pragma_page_count(), pragma_page_size(), pragma_freelist_count()
        """
        result = self.execute_single(query)
        
        if result:
            return {
                'total_bytes': result['total_size'],
                'used_bytes': result['used_size'],
                'free_bytes': result['free_size'],
                'total_mb': round(result['total_size'] / (1024 * 1024), 2),
                'used_mb': round(result['used_size'] / (1024 * 1024), 2),
                'free_mb': round(result['free_size'] / (1024 * 1024), 2)
            }
        return {}
    
    def get_table_sizes(self) -> Dict[str, Dict[str, Any]]:
        """Get size information for all tables."""
        tables_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        tables = self.execute_query(tables_query)
        
        table_sizes = {}
        for table in tables:
            table_name = table['name']
            
            # Get row count
            count_result = self.execute_single(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = count_result['count'] if count_result else 0
            
            # Estimate size (rough calculation)
            size_query = f"""
            SELECT 
                SUM(LENGTH(quote(col_data))) as estimated_size
            FROM (
                SELECT * FROM {table_name} LIMIT 1000
            ) t,
            json_each('["' || replace(quote(t.*), '"', '""') || '"]') j(col_data)
            """
            
            try:
                size_result = self.execute_single(size_query)
                estimated_size = size_result['estimated_size'] if size_result else 0
                avg_row_size = estimated_size / 1000 if estimated_size else 0
                total_estimated_size = avg_row_size * row_count
            except:
                avg_row_size = 0
                total_estimated_size = 0
            
            table_sizes[table_name] = {
                'row_count': row_count,
                'avg_row_size_bytes': int(avg_row_size),
                'estimated_total_bytes': int(total_estimated_size),
                'estimated_total_mb': round(total_estimated_size / (1024 * 1024), 2)
            }
        
        return table_sizes
    
    def optimize_database(self):
        """Run optimization operations on the database."""
        logger.info("Starting database optimization...")
        
        # Update query planner statistics
        self.analyze()
        
        # Optimize the database
        conn = self.get_connection()
        conn.execute("PRAGMA optimize")
        
        # Get database size before and after
        size_info = self.get_database_size()
        logger.info(f"Database size: {size_info.get('total_mb', 0)} MB")
        
        logger.info("Database optimization completed")
    
    def close_connection(self, thread_id: Optional[int] = None):
        """Close database connection for specific thread or current thread."""
        target_thread_id = thread_id or threading.get_ident()
        
        with self.connection_lock:
            if target_thread_id in self.connections:
                self.connections[target_thread_id].close()
                del self.connections[target_thread_id]
                logger.debug(f"Closed database connection for thread {target_thread_id}")
    
    def close_all_connections(self):
        """Close all database connections."""
        with self.connection_lock:
            for thread_id, conn in self.connections.items():
                try:
                    conn.close()
                    logger.debug(f"Closed database connection for thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")
            self.connections.clear()
            logger.info("All database connections closed")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            # Test basic connectivity
            result = self.execute_single("SELECT 1 as test")
            connectivity = result['test'] == 1 if result else False
            
            # Check foreign key constraints
            fk_check = self.execute_single("PRAGMA foreign_key_check")
            foreign_keys_ok = fk_check is None
            
            # Get database info
            size_info = self.get_database_size()
            
            # Count active connections
            active_connections = len(self.connections)
            
            return {
                'status': 'healthy' if connectivity and foreign_keys_ok else 'unhealthy',
                'connectivity': connectivity,
                'foreign_keys_ok': foreign_keys_ok,
                'database_path': self.db_path,
                'database_size_mb': size_info.get('total_mb', 0),
                'active_connections': active_connections,
                'wal_mode': True,
                'foreign_keys_enabled': True
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_path': self.db_path
            }


# Global database instance
db = DatabaseConnection()


def get_db() -> DatabaseConnection:
    """Get the global database instance."""
    return db


def init_db(db_path: Optional[str] = None) -> DatabaseConnection:
    """Initialize database with optional custom path."""
    global db
    if db_path:
        db = DatabaseConnection(db_path)
    return db