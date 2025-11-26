"""Common data models and types for Polymarket API."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional
import json


class TimeFrame(Enum):
    """Time frame enum for charts and data aggregation."""
    
    MINUTE = "1m"
    HOUR = "1h" 
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"
    
    def to_seconds(self) -> int:
        """Convert timeframe to seconds."""
        mapping = {
            "1m": 60,
            "1h": 3600,
            "1d": 86400,
            "1w": 604800,
            "1M": 2592000  # 30 days approximation
        }
        return mapping[self.value]
    
    def __str__(self) -> str:
        return self.value


@dataclass
class TokenInfo:
    """Information about a token used in the platform."""
    
    symbol: str
    name: str
    decimals: int
    address: str
    chain_id: Optional[int] = None
    
    def to_human_readable(self, raw_amount: int) -> Decimal:
        """Convert raw token amount to human readable decimal."""
        return Decimal(raw_amount) / Decimal(10 ** self.decimals)
    
    def to_raw_amount(self, human_amount: Decimal) -> int:
        """Convert human readable amount to raw token amount."""
        return int(human_amount * Decimal(10 ** self.decimals))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "decimals": self.decimals,
            "address": self.address,
            "chain_id": self.chain_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenInfo":
        """Create instance from dictionary."""
        return cls(
            symbol=data["symbol"],
            name=data["name"],
            decimals=data["decimals"],
            address=data["address"],
            chain_id=data.get("chain_id")
        )


@dataclass
class Resolution:
    """Market resolution information."""
    
    outcome_id: str
    resolved_at: datetime
    resolution_source: str
    disputed: bool = False
    dispute_deadline: Optional[datetime] = None
    final_outcome: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "outcome_id": self.outcome_id,
            "resolved_at": self.resolved_at.isoformat(),
            "resolution_source": self.resolution_source,
            "disputed": self.disputed,
            "dispute_deadline": self.dispute_deadline.isoformat() if self.dispute_deadline else None,
            "final_outcome": self.final_outcome
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Resolution":
        """Create instance from dictionary."""
        return cls(
            outcome_id=data["outcome_id"],
            resolved_at=datetime.fromisoformat(data["resolved_at"].replace("Z", "+00:00")),
            resolution_source=data["resolution_source"],
            disputed=data.get("disputed", False),
            dispute_deadline=datetime.fromisoformat(data["dispute_deadline"].replace("Z", "+00:00")) 
                           if data.get("dispute_deadline") else None,
            final_outcome=data.get("final_outcome")
        )


@dataclass
class ApiResponse:
    """Standard API response wrapper."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiResponse":
        """Create instance from dictionary."""
        return cls(
            success=data["success"],
            data=data.get("data"),
            error=data.get("error"),
            error_code=data.get("error_code"),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                     if data.get("timestamp") else None
        )


@dataclass 
class RateLimit:
    """Rate limit information from API responses."""
    
    remaining: int
    reset_at: datetime
    limit: int = 100
    
    @property
    def seconds_until_reset(self) -> int:
        """Get seconds until rate limit resets."""
        return max(0, int((self.reset_at - datetime.now()).total_seconds()))
    
    @property
    def is_exhausted(self) -> bool:
        """Check if rate limit is exhausted."""
        return self.remaining <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "remaining": self.remaining,
            "reset_at": self.reset_at.isoformat(),
            "limit": self.limit
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["RateLimit"]:
        """Create from HTTP response headers."""
        try:
            remaining = int(headers.get("X-RateLimit-Remaining", 0))
            reset_timestamp = int(headers.get("X-RateLimit-Reset", 0))
            limit = int(headers.get("X-RateLimit-Limit", 100))
            
            reset_at = datetime.fromtimestamp(reset_timestamp)
            
            return cls(
                remaining=remaining,
                reset_at=reset_at,
                limit=limit
            )
        except (ValueError, TypeError):
            return None


@dataclass
class PaginationInfo:
    """Pagination information for paginated responses."""
    
    page: int
    per_page: int
    total: int
    total_pages: int
    has_next: bool = False
    has_prev: bool = False
    
    @property
    def offset(self) -> int:
        """Calculate offset for current page."""
        return (self.page - 1) * self.per_page
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "page": self.page,
            "per_page": self.per_page,
            "total": self.total,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaginationInfo":
        """Create instance from dictionary."""
        page = data["page"]
        per_page = data["per_page"]
        total = data["total"]
        total_pages = data["total_pages"]
        
        return cls(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class BaseModel:
    """Base class for all data models with common functionality."""
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                attrs.append(f"{key}={repr(value)}")
        return f"{class_name}({', '.join(attrs)})"