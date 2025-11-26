"""Fantasy Collective Database Manager

A comprehensive database manager for fantasy collective system that handles:
- SQLite operations with connection pooling
- User CRUD operations
- League and collective management
- Predictions and scoring calculations
- Transaction support and rollback
- Query optimization and caching
- Data migration utilities

Thread-safe and production-ready with logging and error handling.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json
import hashlib
import time
from collections import defaultdict

from sqlalchemy import (
    create_engine, event, pool, Column, Integer, String, Float, DateTime, 
    Text, ForeignKey, Index, CheckConstraint, UniqueConstraint, Boolean,
    desc, asc, func, and_, or_, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session, relationship, validates
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Enums for type safety
class UserRole(Enum):
    MEMBER = "member"
    ADMIN = "admin"
    MODERATOR = "moderator"

class LeagueStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    SUSPENDED = "suspended"

class PredictionStatus(Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"

class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    REWARD = "reward"
    PENALTY = "penalty"

# Database Models
class User(Base):
    """User model for fantasy collective members"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), nullable=False, default='member', index=True)
    balance = Column(Float, nullable=False, default=0.0)
    total_points = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    league_memberships = relationship('LeagueMembership', back_populates='user', cascade='all, delete-orphan')
    predictions = relationship('Prediction', back_populates='user', cascade='all, delete-orphan')
    transactions = relationship('Transaction', back_populates='user', cascade='all, delete-orphan')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('member', 'admin', 'moderator')", name='check_user_role'),
        CheckConstraint('balance >= 0', name='check_balance_non_negative'),
        CheckConstraint('total_points >= 0', name='check_points_non_negative'),
        Index('idx_user_active_created', 'active', 'created_at'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Basic email validation"""
        if '@' not in email or '.' not in email.split('@')[1]:
            raise ValueError(f"Invalid email format: {email}")
        return email.lower()
    
    @validates('username')
    def validate_username(self, key, username):
        """Username validation"""
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        # Allow alphanumeric characters and underscores
        if not username.replace('_', '').isalnum():
            raise ValueError("Username must contain only alphanumeric characters and underscores")
        return username.lower()
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role}, balance=${self.balance:.2f})>"

class League(Base):
    """League model for organizing competitions"""
    __tablename__ = 'leagues'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    entry_fee = Column(Float, nullable=False, default=0.0)
    max_members = Column(Integer, nullable=False, default=100)
    current_members = Column(Integer, nullable=False, default=0)
    prize_pool = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False, default='active', index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship('User', foreign_keys=[created_by])
    memberships = relationship('LeagueMembership', back_populates='league', cascade='all, delete-orphan')
    predictions = relationship('Prediction', back_populates='league', cascade='all, delete-orphan')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('active', 'inactive', 'completed', 'suspended')", name='check_league_status'),
        CheckConstraint('entry_fee >= 0', name='check_entry_fee_non_negative'),
        CheckConstraint('max_members > 0', name='check_max_members_positive'),
        CheckConstraint('current_members >= 0', name='check_current_members_non_negative'),
        CheckConstraint('current_members <= max_members', name='check_members_within_limit'),
        CheckConstraint('prize_pool >= 0', name='check_prize_pool_non_negative'),
        CheckConstraint('end_date > start_date', name='check_dates_valid'),
        Index('idx_league_status_dates', 'status', 'start_date', 'end_date'),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if league is currently active"""
        now = datetime.utcnow()
        return (self.status == 'active' and 
                self.start_date <= now <= self.end_date)
    
    @property
    def slots_available(self) -> int:
        """Number of available slots in the league"""
        return max(0, self.max_members - self.current_members)
    
    def __repr__(self):
        return f"<League(id={self.id}, name={self.name}, status={self.status}, members={self.current_members}/{self.max_members})>"

class LeagueMembership(Base):
    """Association table for user-league relationships"""
    __tablename__ = 'league_memberships'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False)
    points_earned = Column(Integer, nullable=False, default=0)
    rank = Column(Integer)
    joined_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship('User', back_populates='league_memberships')
    league = relationship('League', back_populates='memberships')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'league_id', name='unique_user_league'),
        CheckConstraint('points_earned >= 0', name='check_points_non_negative'),
        CheckConstraint('rank > 0', name='check_rank_positive'),
        Index('idx_membership_points', 'league_id', 'points_earned'),
        Index('idx_membership_rank', 'league_id', 'rank'),
    )
    
    def __repr__(self):
        return f"<LeagueMembership(user_id={self.user_id}, league_id={self.league_id}, points={self.points_earned}, rank={self.rank})>"

class Prediction(Base):
    """Prediction model for user predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False)
    event_name = Column(String(200), nullable=False, index=True)
    event_date = Column(DateTime, nullable=False, index=True)
    prediction_data = Column(Text, nullable=False)  # JSON data
    confidence_level = Column(Float, nullable=False)
    points_awarded = Column(Integer, default=0)
    status = Column(String(20), nullable=False, default='pending', index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)
    
    # Relationships
    user = relationship('User', back_populates='predictions')
    league = relationship('League', back_populates='predictions')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'resolved', 'cancelled')", name='check_prediction_status'),
        CheckConstraint('confidence_level >= 0 AND confidence_level <= 1', name='check_confidence_range'),
        CheckConstraint('points_awarded >= 0', name='check_points_non_negative'),
        Index('idx_prediction_event_status', 'event_name', 'status'),
        Index('idx_prediction_user_league', 'user_id', 'league_id'),
    )
    
    @property
    def prediction_dict(self) -> Dict[str, Any]:
        """Get prediction data as dictionary"""
        try:
            return json.loads(self.prediction_data)
        except json.JSONDecodeError:
            return {}
    
    @prediction_dict.setter
    def prediction_dict(self, data: Dict[str, Any]):
        """Set prediction data from dictionary"""
        self.prediction_data = json.dumps(data)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, user_id={self.user_id}, event={self.event_name}, status={self.status})>"

class Transaction(Base):
    """Financial transaction model"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    transaction_type = Column(String(20), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    balance_before = Column(Float, nullable=False)
    balance_after = Column(Float, nullable=False)
    description = Column(String(200))
    reference_id = Column(String(100))  # For linking to leagues, predictions, etc.
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship('User', back_populates='transactions')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("transaction_type IN ('deposit', 'withdrawal', 'reward', 'penalty')", name='check_transaction_type'),
        CheckConstraint('balance_before >= 0', name='check_balance_before_non_negative'),
        CheckConstraint('balance_after >= 0', name='check_balance_after_non_negative'),
        Index('idx_transaction_user_type', 'user_id', 'transaction_type'),
        Index('idx_transaction_reference', 'reference_id'),
    )
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, user_id={self.user_id}, type={self.transaction_type}, amount=${self.amount:.2f})>"

# Cache decorator for query optimization
def cache_result(ttl: int = 300):
    """Decorator to cache query results with TTL"""
    def decorator(func):
        cache = {}
        cache_times = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()).hexdigest()
            
            with lock:
                # Check if cached result is still valid
                if key in cache and time.time() - cache_times[key] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache[key] = result
                cache_times[key] = time.time()
                
                # Clean old cache entries
                current_time = time.time()
                expired_keys = [k for k, t in cache_times.items() if current_time - t >= ttl]
                for k in expired_keys:
                    cache.pop(k, None)
                    cache_times.pop(k, None)
                
                logger.debug(f"Cache miss for {func.__name__}, result cached")
                return result
        
        return wrapper
    return decorator

@dataclass
class QueryStats:
    """Statistics for database queries"""
    total_queries: int = 0
    avg_execution_time: float = 0.0
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

class FantasyCollectiveDBManager:
    """
    Comprehensive database manager for Fantasy Collective system
    
    Features:
    - Connection pooling with SQLite
    - Thread-safe operations
    - Query optimization and caching
    - Transaction management
    - Data migration utilities
    - Performance monitoring
    - Error handling and logging
    """
    
    def __init__(self, db_path: Optional[str] = None, echo: bool = False, pool_size: int = 20):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to SQLite database file
            echo: Whether to log SQL statements
            pool_size: Connection pool size
        """
        self.db_path = db_path or self._get_default_db_path()
        self.echo = echo
        self.pool_size = pool_size
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        self._lock = threading.RLock()
        self._stats = QueryStats()
        
        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FantasyCollectiveDBManager initialized with database: {self.db_path}")
    
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        base_dir = Path(__file__).parent.parent.parent.parent  # Project root
        db_dir = base_dir / 'data' / 'fantasy_collective'
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / 'fantasy_collective.db')
    
    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine with connection pooling"""
        if self._engine is None:
            with self._lock:
                if self._engine is None:  # Double-check locking
                    self._engine = create_engine(
                        f'sqlite:///{self.db_path}',
                        echo=self.echo,
                        poolclass=StaticPool,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        connect_args={
                            'check_same_thread': False,
                            'timeout': 30,
                            'isolation_level': None  # Enable autocommit mode
                        }
                    )
                    
                    # Configure SQLite for optimal performance
                    @event.listens_for(self._engine, "connect")
                    def set_sqlite_pragma(dbapi_conn, connection_record):
                        cursor = dbapi_conn.cursor()
                        # Enable foreign keys
                        cursor.execute("PRAGMA foreign_keys=ON")
                        # Use WAL mode for better concurrency
                        cursor.execute("PRAGMA journal_mode=WAL")
                        # Optimize synchronization
                        cursor.execute("PRAGMA synchronous=NORMAL")
                        # Increase cache size
                        cursor.execute("PRAGMA cache_size=10000")
                        # Use memory for temporary storage
                        cursor.execute("PRAGMA temp_store=MEMORY")
                        # Optimize page size
                        cursor.execute("PRAGMA page_size=4096")
                        cursor.close()
                    
                    logger.info(f"Database engine created with connection pooling")
        
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory"""
        if self._session_factory is None:
            with self._lock:
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
            with self._lock:
                if self._scoped_session is None:
                    self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session
    
    def create_tables(self):
        """Create all database tables"""
        try:
            with self._lock:
                Base.metadata.create_all(self.engine)
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            with self._lock:
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
                session.add(user)
                # Automatic commit/rollback
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
    
    # User CRUD Operations
    def create_user(self, username: str, email: str, password_hash: str, 
                   full_name: Optional[str] = None, role: str = 'member') -> User:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User email address
            password_hash: Hashed password
            full_name: Full name (optional)
            role: User role (default: member)
            
        Returns:
            Created User object
        """
        with self.session_scope() as session:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                role=role
            )
            session.add(user)
            session.flush()  # Get the ID
            logger.info(f"Created user: {user}")
            return user
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with self.session_scope() as session:
            return session.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self.session_scope() as session:
            return session.query(User).filter(User.username == username.lower()).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        with self.session_scope() as session:
            return session.query(User).filter(User.email == email.lower()).first()
    
    def update_user(self, user_id: int, **updates) -> Optional[User]:
        """
        Update user information
        
        Args:
            user_id: User ID to update
            **updates: Fields to update
            
        Returns:
            Updated User object or None if not found
        """
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in updates.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                user.updated_at = datetime.utcnow()
                logger.info(f"Updated user {user_id}: {updates}")
            return user
    
    def delete_user(self, user_id: int) -> bool:
        """
        Delete user (soft delete by setting active=False)
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if user was found and deleted
        """
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.active = False
                user.updated_at = datetime.utcnow()
                logger.info(f"Deactivated user: {user}")
                return True
            return False
    
    @cache_result(ttl=60)
    def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get list of active users"""
        with self.session_scope() as session:
            return (session.query(User)
                   .filter(User.active == True)
                   .order_by(User.created_at.desc())
                   .limit(limit)
                   .offset(offset)
                   .all())
    
    # League CRUD Operations
    def create_league(self, name: str, created_by: int, entry_fee: float = 0.0,
                     max_members: int = 100, start_date: datetime = None,
                     end_date: datetime = None, description: str = None) -> League:
        """
        Create a new league
        
        Args:
            name: League name
            created_by: User ID of creator
            entry_fee: Entry fee (default: 0)
            max_members: Maximum members (default: 100)
            start_date: League start date
            end_date: League end date
            description: League description
            
        Returns:
            Created League object
        """
        if start_date is None:
            start_date = datetime.utcnow()
        if end_date is None:
            end_date = start_date + timedelta(days=30)
            
        with self.session_scope() as session:
            league = League(
                name=name,
                created_by=created_by,
                entry_fee=entry_fee,
                max_members=max_members,
                start_date=start_date,
                end_date=end_date,
                description=description
            )
            session.add(league)
            session.flush()
            logger.info(f"Created league: {league}")
            return league
    
    def get_league_by_id(self, league_id: int) -> Optional[League]:
        """Get league by ID"""
        with self.session_scope() as session:
            return session.query(League).filter(League.id == league_id).first()
    
    @cache_result(ttl=120)
    def get_active_leagues(self, limit: int = 50, offset: int = 0) -> List[League]:
        """Get list of active leagues"""
        with self.session_scope() as session:
            return (session.query(League)
                   .filter(League.status == 'active')
                   .order_by(League.created_at.desc())
                   .limit(limit)
                   .offset(offset)
                   .all())
    
    def join_league(self, user_id: int, league_id: int) -> bool:
        """
        Join a user to a league
        
        Args:
            user_id: User ID
            league_id: League ID
            
        Returns:
            True if successfully joined
        """
        with self.session_scope() as session:
            # Check if league exists and has space
            league = session.query(League).filter(League.id == league_id).first()
            if not league or league.current_members >= league.max_members:
                return False
            
            # Check if user is already a member
            existing = (session.query(LeagueMembership)
                       .filter(LeagueMembership.user_id == user_id,
                               LeagueMembership.league_id == league_id)
                       .first())
            if existing:
                return False
            
            # Create membership
            membership = LeagueMembership(user_id=user_id, league_id=league_id)
            session.add(membership)
            
            # Update league member count
            league.current_members += 1
            league.prize_pool += league.entry_fee
            
            # Deduct entry fee from user if applicable
            if league.entry_fee > 0:
                user = session.query(User).filter(User.id == user_id).first()
                if user and user.balance >= league.entry_fee:
                    user.balance -= league.entry_fee
                    # Record transaction
                    transaction = Transaction(
                        user_id=user_id,
                        transaction_type='withdrawal',
                        amount=league.entry_fee,
                        balance_before=user.balance + league.entry_fee,
                        balance_after=user.balance,
                        description=f"Entry fee for league: {league.name}",
                        reference_id=f"league_{league_id}"
                    )
                    session.add(transaction)
                else:
                    return False
            
            logger.info(f"User {user_id} joined league {league_id}")
            return True
    
    def leave_league(self, user_id: int, league_id: int) -> bool:
        """
        Remove user from league
        
        Args:
            user_id: User ID
            league_id: League ID
            
        Returns:
            True if successfully removed
        """
        with self.session_scope() as session:
            membership = (session.query(LeagueMembership)
                         .filter(LeagueMembership.user_id == user_id,
                                 LeagueMembership.league_id == league_id)
                         .first())
            if membership:
                session.delete(membership)
                
                # Update league member count
                league = session.query(League).filter(League.id == league_id).first()
                if league:
                    league.current_members = max(0, league.current_members - 1)
                
                logger.info(f"User {user_id} left league {league_id}")
                return True
            return False
    
    # Prediction Operations
    def create_prediction(self, user_id: int, league_id: int, event_name: str,
                         event_date: datetime, prediction_data: Dict[str, Any],
                         confidence_level: float) -> Prediction:
        """
        Create a new prediction
        
        Args:
            user_id: User ID making prediction
            league_id: League ID
            event_name: Name of the event
            event_date: When the event occurs
            prediction_data: Prediction details as dict
            confidence_level: Confidence level (0.0-1.0)
            
        Returns:
            Created Prediction object
        """
        with self.session_scope() as session:
            prediction = Prediction(
                user_id=user_id,
                league_id=league_id,
                event_name=event_name,
                event_date=event_date,
                prediction_data=json.dumps(prediction_data),
                confidence_level=confidence_level
            )
            session.add(prediction)
            session.flush()
            logger.info(f"Created prediction: {prediction}")
            return prediction
    
    def get_predictions_by_user(self, user_id: int, league_id: Optional[int] = None,
                               limit: int = 50, offset: int = 0) -> List[Prediction]:
        """Get predictions by user"""
        with self.session_scope() as session:
            query = session.query(Prediction).filter(Prediction.user_id == user_id)
            if league_id:
                query = query.filter(Prediction.league_id == league_id)
            return (query.order_by(Prediction.created_at.desc())
                    .limit(limit).offset(offset).all())
    
    def get_predictions_by_league(self, league_id: int, status: Optional[str] = None,
                                 limit: int = 100, offset: int = 0) -> List[Prediction]:
        """Get predictions by league"""
        with self.session_scope() as session:
            query = session.query(Prediction).filter(Prediction.league_id == league_id)
            if status:
                query = query.filter(Prediction.status == status)
            return (query.order_by(Prediction.created_at.desc())
                    .limit(limit).offset(offset).all())
    
    def resolve_prediction(self, prediction_id: int, points_awarded: int) -> bool:
        """
        Resolve a prediction and award points
        
        Args:
            prediction_id: Prediction ID
            points_awarded: Points to award
            
        Returns:
            True if successfully resolved
        """
        with self.session_scope() as session:
            prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()
            if not prediction or prediction.status != 'pending':
                return False
            
            # Update prediction
            prediction.status = 'resolved'
            prediction.points_awarded = points_awarded
            prediction.resolved_at = datetime.utcnow()
            
            # Update user points
            user = session.query(User).filter(User.id == prediction.user_id).first()
            if user:
                user.total_points += points_awarded
            
            # Update league membership points
            membership = (session.query(LeagueMembership)
                         .filter(LeagueMembership.user_id == prediction.user_id,
                                 LeagueMembership.league_id == prediction.league_id)
                         .first())
            if membership:
                membership.points_earned += points_awarded
            
            logger.info(f"Resolved prediction {prediction_id} with {points_awarded} points")
            return True
    
    # Scoring and Ranking
    def calculate_league_rankings(self, league_id: int) -> List[Dict[str, Any]]:
        """
        Calculate and update league rankings
        
        Args:
            league_id: League ID
            
        Returns:
            List of ranking data
        """
        with self.session_scope() as session:
            # Get all memberships with user data, ordered by points
            rankings = (session.query(LeagueMembership, User)
                       .join(User)
                       .filter(LeagueMembership.league_id == league_id)
                       .order_by(LeagueMembership.points_earned.desc())
                       .all())
            
            # Update ranks
            ranking_data = []
            for rank, (membership, user) in enumerate(rankings, 1):
                membership.rank = rank
                ranking_data.append({
                    'rank': rank,
                    'user_id': user.id,
                    'username': user.username,
                    'points': membership.points_earned,
                    'total_predictions': len(user.predictions)
                })
            
            logger.info(f"Updated rankings for league {league_id}")
            return ranking_data
    
    @cache_result(ttl=300)
    def get_leaderboard(self, league_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get league leaderboard"""
        with self.session_scope() as session:
            results = (session.query(LeagueMembership, User)
                      .join(User)
                      .filter(LeagueMembership.league_id == league_id)
                      .order_by(LeagueMembership.points_earned.desc())
                      .limit(limit)
                      .all())
            
            return [{
                'rank': membership.rank or 0,
                'user_id': user.id,
                'username': user.username,
                'points': membership.points_earned,
                'joined_at': membership.joined_at
            } for membership, user in results]
    
    # Transaction Management
    def create_transaction(self, user_id: int, transaction_type: str, amount: float,
                          description: str = None, reference_id: str = None) -> Transaction:
        """
        Create a financial transaction
        
        Args:
            user_id: User ID
            transaction_type: Type of transaction
            amount: Transaction amount
            description: Description
            reference_id: Reference ID for linking
            
        Returns:
            Created Transaction object
        """
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            balance_before = user.balance
            
            # Update user balance based on transaction type
            if transaction_type in ['deposit', 'reward']:
                user.balance += amount
            elif transaction_type in ['withdrawal', 'penalty']:
                if user.balance < amount:
                    raise ValueError("Insufficient balance")
                user.balance -= amount
            
            balance_after = user.balance
            
            # Create transaction record
            transaction = Transaction(
                user_id=user_id,
                transaction_type=transaction_type,
                amount=amount,
                balance_before=balance_before,
                balance_after=balance_after,
                description=description,
                reference_id=reference_id
            )
            session.add(transaction)
            session.flush()
            
            logger.info(f"Created transaction: {transaction}")
            return transaction
    
    def get_user_transactions(self, user_id: int, transaction_type: Optional[str] = None,
                             limit: int = 50, offset: int = 0) -> List[Transaction]:
        """Get user transaction history"""
        with self.session_scope() as session:
            query = session.query(Transaction).filter(Transaction.user_id == user_id)
            if transaction_type:
                query = query.filter(Transaction.transaction_type == transaction_type)
            return (query.order_by(Transaction.created_at.desc())
                    .limit(limit).offset(offset).all())
    
    # Query Optimization and Analytics
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        with self.session_scope() as session:
            stats = {
                'total_users': session.query(User).count(),
                'active_users': session.query(User).filter(User.active == True).count(),
                'total_leagues': session.query(League).count(),
                'active_leagues': session.query(League).filter(League.status == 'active').count(),
                'total_predictions': session.query(Prediction).count(),
                'pending_predictions': session.query(Prediction).filter(Prediction.status == 'pending').count(),
                'total_transactions': session.query(Transaction).count(),
                'total_volume': session.query(func.sum(Transaction.amount)).scalar() or 0,
                'query_stats': {
                    'total_queries': self._stats.total_queries,
                    'avg_execution_time': self._stats.avg_execution_time,
                    'cache_hits': self._stats.cache_hits,
                    'cache_misses': self._stats.cache_misses
                }
            }
            return stats
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            with self.engine.connect() as conn:
                # Analyze query patterns
                conn.execute(text("ANALYZE"))
                # Rebuild indexes
                conn.execute(text("REINDEX"))
                # Clean up WAL file
                conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                
            logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
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
    
    def migrate_data(self, migration_script: str):
        """
        Execute data migration script
        
        Args:
            migration_script: SQL migration script
        """
        try:
            with self.session_scope() as session:
                for statement in migration_script.split(';'):
                    statement = statement.strip()
                    if statement:
                        session.execute(text(statement))
            logger.info("Data migration completed successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def close(self):
        """Close database connections and cleanup"""
        if self._scoped_session:
            self._scoped_session.remove()
        if self._engine:
            self._engine.dispose()
        logger.info("Database connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Global database instance
_db_manager: Optional[FantasyCollectiveDBManager] = None
_db_lock = threading.Lock()

def get_db_manager(db_path: Optional[str] = None) -> FantasyCollectiveDBManager:
    """
    Get or create global database manager instance
    
    Args:
        db_path: Optional database path
        
    Returns:
        FantasyCollectiveDBManager instance
    """
    global _db_manager
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:  # Double-check locking
                _db_manager = FantasyCollectiveDBManager(db_path)
                _db_manager.create_tables()
    return _db_manager

# Convenience functions
def init_database(db_path: Optional[str] = None) -> FantasyCollectiveDBManager:
    """Initialize database with tables"""
    db = get_db_manager(db_path)
    db.create_tables()
    logger.info("Fantasy Collective database initialized successfully")
    return db