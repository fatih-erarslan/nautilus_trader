"""Database Connection Manager for Crypto Trading

Handles SQLite connections with proper pooling, session management, and error handling.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.engine import Engine
from .models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections with connection pooling and session handling"""
    
    def __init__(self, db_path: Optional[str] = None, echo: bool = False):
        """
        Initialize database connection manager
        
        Args:
            db_path: Path to SQLite database file
            echo: Whether to log SQL statements
        """
        self.db_path = db_path or self._get_default_db_path()
        self.echo = echo
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        base_dir = Path(__file__).parent.parent.parent.parent  # Project root
        db_dir = base_dir / 'data' / 'crypto_trading'
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / 'trading.db')
    
    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine with connection pooling"""
        if self._engine is None:
            # Create engine with connection pooling
            self._engine = create_engine(
                f'sqlite:///{self.db_path}',
                echo=self.echo,
                # SQLite-specific pool configuration
                poolclass=pool.StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 15,
                    'isolation_level': 'DEFERRED'
                }
            )
            
            # Configure SQLite for better performance
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
            
            logger.info(f"Database engine created for: {self.db_path}")
        
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    @property
    def scoped_session(self) -> scoped_session:
        """Get thread-local scoped session"""
        if self._scoped_session is None:
            self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations
        
        Usage:
            with db.session_scope() as session:
                session.add(model)
                session.commit()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.session_factory()
    
    def close(self):
        """Close database connections and cleanup"""
        if self._scoped_session:
            self._scoped_session.remove()
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
    
    def execute_sql(self, sql: str, params: Optional[dict] = None) -> list:
        """
        Execute raw SQL query
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            List of result rows
        """
        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return result.fetchall()
    
    def backup_database(self, backup_path: str):
        """
        Create a backup of the database
        
        Args:
            backup_path: Path where backup should be saved
        """
        try:
            source = sqlite3.connect(self.db_path)
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            backup.close()
            source.close()
            logger.info(f"Database backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def optimize_database(self):
        """Run VACUUM and ANALYZE to optimize database"""
        from sqlalchemy import text
        try:
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM"))
                conn.execute(text("ANALYZE"))
            logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def get_table_stats(self) -> dict:
        """Get statistics about database tables"""
        from sqlalchemy import text
        stats = {}
        with self.engine.connect() as conn:
            # Get table names
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            )
            tables = [row[0] for row in result]
            
            # Get row counts for each table
            for table in tables:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = count_result.scalar()
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Global database instance
_db_instance: Optional[DatabaseConnection] = None


def get_database(db_path: Optional[str] = None) -> DatabaseConnection:
    """
    Get or create global database instance
    
    Args:
        db_path: Optional database path
        
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection(db_path)
        _db_instance.create_tables()
    return _db_instance


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session using the global instance
    
    Usage:
        with get_db_session() as session:
            session.query(Model).all()
    """
    db = get_database()
    with db.session_scope() as session:
        yield session


def init_database(db_path: Optional[str] = None):
    """Initialize database with tables"""
    db = get_database(db_path)
    db.create_tables()
    logger.info("Database initialized successfully")
    return db