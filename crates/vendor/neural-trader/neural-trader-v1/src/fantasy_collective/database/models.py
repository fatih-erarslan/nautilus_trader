"""
Database Models for Fantasy Collective System

Base classes and utilities for data access layer.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, asdict
from decimal import Decimal

from .connection import DatabaseConnection, get_db

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all database models."""
    
    table_name: str = ""
    primary_key: str = "id"
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_db()
    
    @classmethod
    def get_table_name(cls) -> str:
        """Get the database table name for this model."""
        return cls.table_name or cls.__name__.lower() + 's'
    
    @classmethod
    def get_primary_key(cls) -> str:
        """Get the primary key column name."""
        return cls.primary_key
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for database storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        return value
    
    def _deserialize_value(self, value: Any, field_type: Type = None) -> Any:
        """Deserialize a value from database."""
        if value is None:
            return None
        
        if field_type == dict or field_type == list:
            try:
                return json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                return value
        elif field_type == Decimal:
            return Decimal(str(value))
        elif field_type == datetime:
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return value
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'db':
                data[key] = value
        return data
    
    def from_dict(self, data: Dict[str, Any]) -> 'BaseModel':
        """Populate model from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class QueryBuilder:
    """SQL query builder for common database operations."""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.reset()
    
    def reset(self):
        """Reset the query builder."""
        self._select_fields = []
        self._where_conditions = []
        self._where_params = []
        self._joins = []
        self._order_by = []
        self._group_by = []
        self._having_conditions = []
        self._limit_value = None
        self._offset_value = None
        return self
    
    def select(self, *fields: str):
        """Add SELECT fields."""
        self._select_fields.extend(fields)
        return self
    
    def where(self, condition: str, *params):
        """Add WHERE condition."""
        self._where_conditions.append(condition)
        self._where_params.extend(params)
        return self
    
    def where_in(self, field: str, values: List[Any]):
        """Add WHERE IN condition."""
        placeholders = ','.join(['?' for _ in values])
        self._where_conditions.append(f"{field} IN ({placeholders})")
        self._where_params.extend(values)
        return self
    
    def where_between(self, field: str, start: Any, end: Any):
        """Add WHERE BETWEEN condition."""
        self._where_conditions.append(f"{field} BETWEEN ? AND ?")
        self._where_params.extend([start, end])
        return self
    
    def join(self, table: str, condition: str, join_type: str = "INNER"):
        """Add JOIN clause."""
        self._joins.append(f"{join_type} JOIN {table} ON {condition}")
        return self
    
    def left_join(self, table: str, condition: str):
        """Add LEFT JOIN clause."""
        return self.join(table, condition, "LEFT")
    
    def order_by(self, field: str, direction: str = "ASC"):
        """Add ORDER BY clause."""
        self._order_by.append(f"{field} {direction}")
        return self
    
    def group_by(self, *fields: str):
        """Add GROUP BY clause."""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition: str, *params):
        """Add HAVING condition."""
        self._having_conditions.append(condition)
        self._where_params.extend(params)
        return self
    
    def limit(self, count: int):
        """Add LIMIT clause."""
        self._limit_value = count
        return self
    
    def offset(self, count: int):
        """Add OFFSET clause."""
        self._offset_value = count
        return self
    
    def build_select(self) -> tuple[str, list]:
        """Build SELECT query."""
        # SELECT clause
        fields = ', '.join(self._select_fields) if self._select_fields else '*'
        query = f"SELECT {fields} FROM {self.table_name}"
        
        # JOIN clauses
        if self._joins:
            query += " " + " ".join(self._joins)
        
        # WHERE clause
        if self._where_conditions:
            query += " WHERE " + " AND ".join(self._where_conditions)
        
        # GROUP BY clause
        if self._group_by:
            query += " GROUP BY " + ", ".join(self._group_by)
        
        # HAVING clause
        if self._having_conditions:
            query += " HAVING " + " AND ".join(self._having_conditions)
        
        # ORDER BY clause
        if self._order_by:
            query += " ORDER BY " + ", ".join(self._order_by)
        
        # LIMIT clause
        if self._limit_value:
            query += f" LIMIT {self._limit_value}"
        
        # OFFSET clause
        if self._offset_value:
            query += f" OFFSET {self._offset_value}"
        
        return query, self._where_params
    
    def build_count(self) -> tuple[str, list]:
        """Build COUNT query."""
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        
        # JOIN clauses
        if self._joins:
            query += " " + " ".join(self._joins)
        
        # WHERE clause
        if self._where_conditions:
            query += " WHERE " + " AND ".join(self._where_conditions)
        
        # GROUP BY clause (for COUNT with GROUP BY, we need to count the groups)
        if self._group_by:
            # Wrap in subquery for proper count
            base_query = f"SELECT 1 FROM {self.table_name}"
            if self._joins:
                base_query += " " + " ".join(self._joins)
            if self._where_conditions:
                base_query += " WHERE " + " AND ".join(self._where_conditions)
            base_query += " GROUP BY " + ", ".join(self._group_by)
            query = f"SELECT COUNT(*) as count FROM ({base_query})"
        
        return query, self._where_params


class Repository(ABC):
    """Base repository class for data access."""
    
    def __init__(self, model_class: Type[BaseModel], db: Optional[DatabaseConnection] = None):
        self.model_class = model_class
        self.table_name = model_class.get_table_name()
        self.primary_key = model_class.get_primary_key()
        self.db = db or get_db()
    
    def query(self) -> QueryBuilder:
        """Create a new query builder."""
        return QueryBuilder(self.table_name)
    
    def find_by_id(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """Find a record by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE {self.primary_key} = ?"
        result = self.db.execute_single(query, (record_id,))
        return dict(result) if result else None
    
    def find_by(self, **conditions) -> Optional[Dict[str, Any]]:
        """Find a single record by conditions."""
        query_builder = self.query()
        
        for field, value in conditions.items():
            query_builder.where(f"{field} = ?", value)
        
        query, params = query_builder.build_select()
        result = self.db.execute_single(query, tuple(params))
        return dict(result) if result else None
    
    def find_all_by(self, **conditions) -> List[Dict[str, Any]]:
        """Find all records by conditions."""
        query_builder = self.query()
        
        for field, value in conditions.items():
            if isinstance(value, (list, tuple)):
                query_builder.where_in(field, value)
            else:
                query_builder.where(f"{field} = ?", value)
        
        query, params = query_builder.build_select()
        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]
    
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find all records with optional pagination."""
        query_builder = self.query()
        
        if limit:
            query_builder.limit(limit)
        if offset:
            query_builder.offset(offset)
        
        query, params = query_builder.build_select()
        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]
    
    def count(self, **conditions) -> int:
        """Count records matching conditions."""
        query_builder = self.query()
        
        for field, value in conditions.items():
            if isinstance(value, (list, tuple)):
                query_builder.where_in(field, value)
            else:
                query_builder.where(f"{field} = ?", value)
        
        query, params = query_builder.build_count()
        result = self.db.execute_single(query, tuple(params))
        return result['count'] if result else 0
    
    def exists(self, **conditions) -> bool:
        """Check if records exist matching conditions."""
        return self.count(**conditions) > 0
    
    def create(self, data: Dict[str, Any]) -> int:
        """Create a new record."""
        # Remove None values and serialize complex types
        clean_data = {}
        for key, value in data.items():
            if value is not None:
                clean_data[key] = self._serialize_value(value)
        
        if not clean_data:
            raise ValueError("No data provided for creation")
        
        fields = list(clean_data.keys())
        placeholders = ['?' for _ in fields]
        values = list(clean_data.values())
        
        query = f"""
        INSERT INTO {self.table_name} ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """
        
        self.db.execute_modify(query, tuple(values))
        return self.db.get_last_insert_id()
    
    def update(self, record_id: Any, data: Dict[str, Any]) -> int:
        """Update a record by ID."""
        # Remove None values and serialize complex types
        clean_data = {}
        for key, value in data.items():
            if value is not None and key != self.primary_key:
                clean_data[key] = self._serialize_value(value)
        
        if not clean_data:
            raise ValueError("No data provided for update")
        
        set_clauses = [f"{field} = ?" for field in clean_data.keys()]
        values = list(clean_data.values())
        values.append(record_id)  # Add ID for WHERE clause
        
        query = f"""
        UPDATE {self.table_name} 
        SET {', '.join(set_clauses)}
        WHERE {self.primary_key} = ?
        """
        
        return self.db.execute_modify(query, tuple(values))
    
    def update_by(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> int:
        """Update records matching conditions."""
        # Remove None values and serialize complex types
        clean_data = {}
        for key, value in data.items():
            if value is not None:
                clean_data[key] = self._serialize_value(value)
        
        if not clean_data:
            raise ValueError("No data provided for update")
        
        set_clauses = [f"{field} = ?" for field in clean_data.keys()]
        where_clauses = [f"{field} = ?" for field in conditions.keys()]
        
        values = list(clean_data.values())
        values.extend(conditions.values())
        
        query = f"""
        UPDATE {self.table_name} 
        SET {', '.join(set_clauses)}
        WHERE {' AND '.join(where_clauses)}
        """
        
        return self.db.execute_modify(query, tuple(values))
    
    def delete(self, record_id: Any) -> int:
        """Delete a record by ID."""
        query = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = ?"
        return self.db.execute_modify(query, (record_id,))
    
    def delete_by(self, **conditions) -> int:
        """Delete records matching conditions."""
        if not conditions:
            raise ValueError("Conditions required for delete operation")
        
        where_clauses = [f"{field} = ?" for field in conditions.keys()]
        values = list(conditions.values())
        
        query = f"""
        DELETE FROM {self.table_name}
        WHERE {' AND '.join(where_clauses)}
        """
        
        return self.db.execute_modify(query, tuple(values))
    
    def execute_raw(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        results = self.db.execute_query(query, params)
        return [dict(row) for row in results]
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for database storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        return value
    
    def _deserialize_value(self, value: Any, field_type: Type = None) -> Any:
        """Deserialize a value from database."""
        if value is None:
            return None
        
        if field_type == dict or field_type == list:
            try:
                return json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                return value
        elif field_type == Decimal:
            return Decimal(str(value))
        elif field_type == datetime:
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return value
        
        return value


# Model data classes for type safety
@dataclass
class User:
    user_id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    salt: str = ""
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    timezone: str = "UTC"
    account_status: str = "active"
    email_verified: bool = False
    phone_verified: bool = False
    kyc_verified: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    registration_ip: Optional[str] = None
    user_agent: Optional[str] = None
    referral_code: Optional[str] = None
    referred_by: Optional[int] = None


@dataclass
class League:
    league_id: Optional[int] = None
    league_name: str = ""
    league_type: str = "fantasy_sports"
    category: Optional[str] = None
    max_participants: int = 10
    min_participants: int = 2
    entry_fee: Decimal = Decimal('0.00')
    prize_pool: Decimal = Decimal('0.00')
    currency: str = "USD"
    scoring_system: Dict = None
    league_rules: Dict = None
    prediction_categories: Optional[Dict] = None
    time_zone: str = "UTC"
    status: str = "draft"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    registration_deadline: Optional[datetime] = None
    creator_id: int = 0
    is_public: bool = True
    requires_approval: bool = False
    invite_code: Optional[str] = None
    description: Optional[str] = None
    league_logo_url: Optional[str] = None
    tags: Optional[Dict] = None
    custom_fields: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.scoring_system is None:
            self.scoring_system = {}
        if self.league_rules is None:
            self.league_rules = {}


@dataclass
class Prediction:
    prediction_id: Optional[int] = None
    user_id: int = 0
    market_id: int = 0
    league_id: Optional[int] = None
    participant_id: Optional[int] = None
    predicted_outcome: str = ""
    confidence_level: Decimal = Decimal('0.5000')
    stake_amount: Decimal = Decimal('0.00')
    potential_payout: Optional[Decimal] = None
    odds_when_placed: Decimal = Decimal('1.0000')
    implied_probability: Optional[Decimal] = None
    expected_value: Optional[Decimal] = None
    bet_type: str = "straight"
    strategy_notes: Optional[str] = None
    status: str = "pending"
    is_live_bet: bool = False
    actual_outcome: Optional[str] = None
    payout_amount: Decimal = Decimal('0.00')
    profit_loss: Decimal = Decimal('0.00')
    settled_at: Optional[datetime] = None
    prediction_reasoning: Optional[str] = None
    external_reference: Optional[str] = None
    tags: Optional[Dict] = None
    placed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Repository classes for specific models
class UserRepository(Repository):
    """Repository for User model."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        super().__init__(User, db)
        self.table_name = "users"
        self.primary_key = "user_id"
    
    def find_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user by username."""
        return self.find_by(username=username)
    
    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user by email."""
        return self.find_by(email=email)
    
    def find_by_referral_code(self, referral_code: str) -> Optional[Dict[str, Any]]:
        """Find user by referral code."""
        return self.find_by(referral_code=referral_code)


class LeagueRepository(Repository):
    """Repository for League model."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        super().__init__(League, db)
        self.table_name = "leagues"
        self.primary_key = "league_id"
    
    def find_by_creator(self, creator_id: int) -> List[Dict[str, Any]]:
        """Find leagues created by user."""
        return self.find_all_by(creator_id=creator_id)
    
    def find_public_leagues(self, league_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find public leagues."""
        conditions = {"is_public": True, "status": "active"}
        if league_type:
            conditions["league_type"] = league_type
        return self.find_all_by(**conditions)
    
    def find_by_invite_code(self, invite_code: str) -> Optional[Dict[str, Any]]:
        """Find league by invite code."""
        return self.find_by(invite_code=invite_code)


class PredictionRepository(Repository):
    """Repository for Prediction model."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        super().__init__(Prediction, db)
        self.table_name = "predictions"
        self.primary_key = "prediction_id"
    
    def find_by_user(self, user_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find predictions by user."""
        conditions = {"user_id": user_id}
        if status:
            conditions["status"] = status
        return self.find_all_by(**conditions)
    
    def find_by_league(self, league_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find predictions by league."""
        conditions = {"league_id": league_id}
        if status:
            conditions["status"] = status
        return self.find_all_by(**conditions)
    
    def find_by_market(self, market_id: int) -> List[Dict[str, Any]]:
        """Find predictions by market."""
        return self.find_all_by(market_id=market_id)
    
    def get_user_performance(self, user_id: int, league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get user performance statistics."""
        query_builder = self.query()
        query_builder.where("user_id = ?", user_id)
        query_builder.where("status IN ('won', 'lost', 'pushed')")
        
        if league_id:
            query_builder.where("league_id = ?", league_id)
        
        query_builder.select(
            "COUNT(*) as total_predictions",
            "COUNT(CASE WHEN status = 'won' THEN 1 END) as correct_predictions",
            "SUM(profit_loss) as total_profit_loss",
            "AVG(confidence_level) as avg_confidence",
            "SUM(stake_amount) as total_staked"
        )
        
        query, params = query_builder.build_select()
        result = self.db.execute_single(query, tuple(params))
        
        if result:
            data = dict(result)
            # Calculate accuracy rate
            if data['total_predictions'] > 0:
                data['accuracy_rate'] = data['correct_predictions'] / data['total_predictions']
            else:
                data['accuracy_rate'] = 0.0
            
            return data
        
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'total_profit_loss': 0.0,
            'avg_confidence': 0.0,
            'total_staked': 0.0,
            'accuracy_rate': 0.0
        }