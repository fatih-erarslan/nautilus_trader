"""Unified Database Connection Pool Manager

Optimized connection pooling for high-performance database operations.
"""

import threading
import queue
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    max_overflow: int = 10
    recycle_time: float = 3600.0  # 1 hour
    health_check_interval: float = 60.0
    
    # SQLite optimizations
    cache_size: int = -64000  # 64MB
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456  # 256MB


class PooledConnection:
    """Wrapper for database connection with pooling metadata."""
    
    def __init__(self, connection: sqlite3.Connection, pool: 'ConnectionPool'):
        self.connection = connection
        self.pool = pool
        self.created_at = time.time()
        self.last_used = time.time()
        self.in_use = False
        self.thread_id = None
        self.query_count = 0
        
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        now = time.time()
        return (now - self.created_at) > self.pool.config.recycle_time
    
    def is_idle_timeout(self) -> bool:
        """Check if connection has been idle too long."""
        now = time.time()
        return (now - self.last_used) > self.pool.config.idle_timeout
    
    def ping(self) -> bool:
        """Test connection health."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False


class ConnectionPool:
    """Thread-safe database connection pool with advanced features."""
    
    def __init__(self, db_path: str, config: Optional[PoolConfig] = None):
        self.db_path = db_path
        self.config = config or PoolConfig()
        self._pool = queue.Queue(maxsize=self.config.max_connections)
        self._overflow = []
        self._lock = threading.Lock()
        self._stats = {
            'connections_created': 0,
            'connections_recycled': 0,
            'connections_failed': 0,
            'queries_executed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'wait_time_total': 0.0,
            'active_connections': 0
        }
        
        # Initialize minimum connections
        self._initialize_pool()
        
        # Start health check thread
        self._start_health_check()
    
    def _initialize_pool(self):
        """Create initial pool connections."""
        for _ in range(self.config.min_connections):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
    
    def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new database connection with optimizations."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=self.config.connection_timeout,
                isolation_level=None
            )
            
            # Apply SQLite optimizations
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Performance optimizations
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            cursor.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            cursor.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            cursor.execute(f"PRAGMA temp_store = {self.config.temp_store}")
            cursor.execute(f"PRAGMA mmap_size = {self.config.mmap_size}")
            
            # Additional optimizations
            cursor.execute("PRAGMA optimize")
            cursor.close()
            
            pooled_conn = PooledConnection(conn, self)
            self._stats['connections_created'] += 1
            
            logger.debug(f"Created new database connection (total: {self._stats['connections_created']})")
            return pooled_conn
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self._stats['connections_failed'] += 1
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        start_time = time.time()
        conn = None
        
        try:
            # Try to get from pool
            try:
                conn = self._pool.get_nowait()
                self._stats['pool_hits'] += 1
            except queue.Empty:
                self._stats['pool_misses'] += 1
                
                # Check if we can create overflow connection
                with self._lock:
                    if len(self._overflow) < self.config.max_overflow:
                        conn = self._create_connection()
                        if conn:
                            self._overflow.append(conn)
                
                # If still no connection, wait for one
                if not conn:
                    conn = self._pool.get(timeout=self.config.connection_timeout)
            
            if not conn:
                raise RuntimeError("Unable to acquire database connection")
            
            # Update connection metadata
            conn.in_use = True
            conn.thread_id = threading.get_ident()
            conn.last_used = time.time()
            
            # Update stats
            wait_time = time.time() - start_time
            self._stats['wait_time_total'] += wait_time
            self._stats['active_connections'] += 1
            
            yield conn.connection
            
        finally:
            if conn:
                # Update usage stats
                conn.query_count += 1
                self._stats['queries_executed'] += 1
                conn.in_use = False
                conn.thread_id = None
                self._stats['active_connections'] -= 1
                
                # Check if connection should be recycled
                if conn.is_expired() or not conn.ping():
                    self._recycle_connection(conn)
                else:
                    # Return to pool
                    if conn in self._overflow:
                        with self._lock:
                            self._overflow.remove(conn)
                    
                    try:
                        self._pool.put_nowait(conn)
                    except queue.Full:
                        # Pool is full, close overflow connection
                        conn.connection.close()
                        logger.debug("Closed overflow connection")
    
    def _recycle_connection(self, conn: PooledConnection):
        """Recycle an old or broken connection."""
        try:
            conn.connection.close()
        except:
            pass
        
        self._stats['connections_recycled'] += 1
        
        # Create replacement connection
        new_conn = self._create_connection()
        if new_conn:
            try:
                self._pool.put_nowait(new_conn)
            except queue.Full:
                new_conn.connection.close()
    
    def _start_health_check(self):
        """Start background thread for connection health checks."""
        def health_check_worker():
            while True:
                time.sleep(self.config.health_check_interval)
                self._perform_health_check()
        
        thread = threading.Thread(target=health_check_worker, daemon=True)
        thread.start()
    
    def _perform_health_check(self):
        """Check health of idle connections."""
        connections_to_check = []
        
        # Get all connections from pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                connections_to_check.append(conn)
            except queue.Empty:
                break
        
        # Check each connection
        for conn in connections_to_check:
            if conn.is_idle_timeout() or conn.is_expired() or not conn.ping():
                self._recycle_connection(conn)
            else:
                self._pool.put(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        stats = self._stats.copy()
        stats['pool_size'] = self._pool.qsize()
        stats['overflow_size'] = len(self._overflow)
        stats['avg_wait_time'] = (
            stats['wait_time_total'] / max(1, stats['pool_hits'] + stats['pool_misses'])
        )
        return stats
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.connection.close()
            except:
                pass
        
        for conn in self._overflow:
            try:
                conn.connection.close()
            except:
                pass
        
        self._overflow.clear()
        logger.info("All connections closed")


# Global pool instance
_pool_instance: Optional[ConnectionPool] = None


def get_pool(db_path: str, config: Optional[PoolConfig] = None) -> ConnectionPool:
    """Get or create global connection pool."""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = ConnectionPool(db_path, config)
    return _pool_instance


@contextmanager
def get_db_connection(db_path: str):
    """Get a database connection from the pool."""
    pool = get_pool(db_path)
    with pool.get_connection() as conn:
        yield conn