"""
Event data models for Polymarket events

This module defines data structures for events that contain multiple
prediction markets grouped by topic, category or theme.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from enum import Enum


class EventCategory(Enum):
    """Event category enumeration"""
    POLITICS = "Politics"
    SPORTS = "Sports"
    CRYPTO = "Crypto"
    ECONOMICS = "Economics"
    ENTERTAINMENT = "Entertainment"
    TECHNOLOGY = "Technology"
    SCIENCE = "Science"
    OTHER = "Other"


class EventStatus(Enum):
    """Event status enumeration"""
    ACTIVE = "active"
    UPCOMING = "upcoming"
    CLOSED = "closed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass
class EventStatistics:
    """Statistics for an event"""
    market_count: int
    total_volume: Decimal
    total_liquidity: Decimal
    participant_count: int
    avg_market_volume: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate derived statistics"""
        # Convert to Decimal types
        if not isinstance(self.total_volume, Decimal):
            self.total_volume = Decimal(str(self.total_volume))
        if not isinstance(self.total_liquidity, Decimal):
            self.total_liquidity = Decimal(str(self.total_liquidity))
        
        # Calculate average market volume
        if self.market_count > 0:
            self.avg_market_volume = self.total_volume / self.market_count
        else:
            self.avg_market_volume = Decimal('0')


@dataclass
class Event:
    """
    Polymarket event containing multiple related markets
    
    Represents a topic or theme that groups multiple prediction markets together.
    """
    id: str
    title: str
    description: str
    category: EventCategory
    status: EventStatus
    created_at: datetime
    end_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    image_url: Optional[str] = None
    statistics: Optional[EventStatistics] = None
    market_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate event data after initialization"""
        if not self.id:
            raise ValueError("Event ID is required")
        if not self.title:
            raise ValueError("Event title is required")
        if not self.description:
            raise ValueError("Event description is required")
        
        # Validate end date if provided
        if self.end_date and self.status == EventStatus.ACTIVE:
            if self.end_date <= datetime.now():
                raise ValueError("Active event end date must be in the future")
    
    @property
    def is_active(self) -> bool:
        """Check if event is currently active"""
        return (
            self.status == EventStatus.ACTIVE and
            (self.end_date is None or self.end_date > datetime.now())
        )
    
    @property
    def time_to_close(self) -> Optional[int]:
        """Get seconds until event closes"""
        if self.end_date and self.end_date > datetime.now():
            return int((self.end_date - datetime.now()).total_seconds())
        return None
    
    @property
    def market_count(self) -> int:
        """Get number of markets in this event"""
        return len(self.market_ids)
    
    @property
    def total_volume(self) -> Decimal:
        """Get total volume across all markets in event"""
        if self.statistics:
            return self.statistics.total_volume
        return Decimal('0')
    
    @property
    def total_liquidity(self) -> Decimal:
        """Get total liquidity across all markets in event"""
        if self.statistics:
            return self.statistics.total_liquidity
        return Decimal('0')
    
    def add_market(self, market_id: str):
        """Add a market to this event"""
        if market_id not in self.market_ids:
            self.market_ids.append(market_id)
            self.updated_at = datetime.now()
    
    def remove_market(self, market_id: str):
        """Remove a market from this event"""
        if market_id in self.market_ids:
            self.market_ids.remove(market_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        result = {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'image_url': self.image_url,
            'market_ids': self.market_ids,
            'metadata': self.metadata,
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'time_to_close': self.time_to_close,
            'market_count': self.market_count,
        }
        
        if self.end_date:
            result['end_date'] = self.end_date.isoformat()
        
        if self.statistics:
            result['statistics'] = {
                'market_count': self.statistics.market_count,
                'total_volume': float(self.statistics.total_volume),
                'total_liquidity': float(self.statistics.total_liquidity),
                'participant_count': self.statistics.participant_count,
                'avg_market_volume': float(self.statistics.avg_market_volume),
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        # Parse datetime fields
        created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        updated_at = datetime.fromisoformat(
            data.get('updated_at', datetime.now().isoformat()).replace('Z', '+00:00')
        )
        
        end_date = None
        if data.get('end_date'):
            end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00'))
        
        # Parse enums
        category = EventCategory(data['category'])
        status = EventStatus(data['status'])
        
        # Parse statistics if present
        statistics = None
        if 'statistics' in data:
            stats_data = data['statistics']
            statistics = EventStatistics(
                market_count=stats_data['market_count'],
                total_volume=Decimal(str(stats_data['total_volume'])),
                total_liquidity=Decimal(str(stats_data['total_liquidity'])),
                participant_count=stats_data['participant_count']
            )
        
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            category=category,
            status=status,
            created_at=created_at,
            end_date=end_date,
            tags=data.get('tags', []),
            image_url=data.get('image_url'),
            statistics=statistics,
            market_ids=data.get('market_ids', []),
            metadata=data.get('metadata', {}),
            updated_at=updated_at,
        )


@dataclass 
class Outcome:
    """Market outcome representation."""
    
    id: str
    market_id: str
    title: str
    price: Decimal
    probability: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate probability from price."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        self.probability = self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "title": self.title,
            "price": float(self.price),
            "probability": float(self.probability)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Outcome":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            title=data["title"],
            price=Decimal(str(data["price"]))
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