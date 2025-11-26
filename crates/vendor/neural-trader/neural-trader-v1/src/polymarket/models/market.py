"""
Market data models for Polymarket prediction markets

This module defines data structures for markets, order books, pricing,
and market metadata following Polymarket's data schema.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from enum import Enum


class MarketStatus(Enum):
    """Market status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass
class Outcome:
    """Market outcome with pricing and volume data"""
    id: str
    name: str
    price: Decimal
    volume: Decimal = Decimal('0')
    liquidity: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Convert to Decimal types and validate"""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.volume, Decimal):
            self.volume = Decimal(str(self.volume))
        if not isinstance(self.liquidity, Decimal):
            self.liquidity = Decimal(str(self.liquidity))
        
        # Validate price range
        if self.price < 0 or self.price > 1:
            raise ValueError("Outcome price must be between 0 and 1")
        if self.volume < 0:
            raise ValueError("Outcome volume must be non-negative")
        if self.liquidity < 0:
            raise ValueError("Outcome liquidity must be non-negative")


@dataclass
class PricePoint:
    """Price point in order book"""
    price: Decimal
    size: Decimal
    timestamp: datetime
    
    def __post_init__(self):
        """Validate price point data"""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.size, Decimal):
            self.size = Decimal(str(self.size))
            
        if self.price < 0 or self.price > 1:
            raise ValueError("Price must be between 0 and 1")
        if self.size < 0:
            raise ValueError("Size must be non-negative")


@dataclass
class OrderBook:
    """Order book for a specific market outcome"""
    market_id: str
    outcome_id: str
    bids: List[PricePoint] = field(default_factory=list)
    asks: List[PricePoint] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and sort order book"""
        # Sort bids descending (highest first)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        # Sort asks ascending (lowest first)
        self.asks.sort(key=lambda x: x.price)
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def midpoint(self) -> Optional[Decimal]:
        """Get midpoint price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a market"""
    total_volume: Decimal
    volume_24h: Decimal
    total_liquidity: Decimal
    available_liquidity: Decimal
    bid_liquidity: Decimal
    ask_liquidity: Decimal
    turnover_rate: Decimal
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Convert to Decimal types"""
        for field_name in ['total_volume', 'volume_24h', 'total_liquidity', 
                          'available_liquidity', 'bid_liquidity', 'ask_liquidity', 'turnover_rate']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))


@dataclass
class MarketMetadata:
    """Extended market metadata"""
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""
    rules: str = ""
    resolution_source: str = ""
    created_by: Optional[str] = None
    fee_rate: Decimal = Decimal('0.02')  # 2% default fee
    minimum_order_size: Decimal = Decimal('0.01')
    maximum_order_size: Optional[Decimal] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metadata"""
        if not isinstance(self.fee_rate, Decimal):
            self.fee_rate = Decimal(str(self.fee_rate))
        if not isinstance(self.minimum_order_size, Decimal):
            self.minimum_order_size = Decimal(str(self.minimum_order_size))
        if self.maximum_order_size and not isinstance(self.maximum_order_size, Decimal):
            self.maximum_order_size = Decimal(str(self.maximum_order_size))


@dataclass
class Market:
    """
    Polymarket prediction market
    
    Represents a prediction market with multiple outcomes that users can trade on.
    """
    id: str
    question: str
    outcomes: List[str]
    end_date: datetime
    status: MarketStatus
    current_prices: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Optional[MarketMetadata] = None
    liquidity: Optional[LiquidityMetrics] = None
    order_books: Dict[str, OrderBook] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    volume: Decimal = Decimal('0')
    volume_24h: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate market data after initialization"""
        if not self.id:
            raise ValueError("Market ID is required")
        if not self.question:
            raise ValueError("Market question is required")
        if len(self.outcomes) < 2:
            raise ValueError("Market must have at least 2 outcomes")
        
        # Validate end date is in the future for active markets
        if self.status == MarketStatus.ACTIVE and self.end_date <= datetime.now():
            raise ValueError("Active market end date must be in the future")
        
        # Convert prices to Decimal
        for outcome, price in self.current_prices.items():
            if not isinstance(price, Decimal):
                self.current_prices[outcome] = Decimal(str(price))
        
        # Convert volume fields to Decimal
        if not isinstance(self.volume, Decimal):
            self.volume = Decimal(str(self.volume))
        if not isinstance(self.volume_24h, Decimal):
            self.volume_24h = Decimal(str(self.volume_24h))
        
        # Validate that current_prices keys match outcomes
        if self.current_prices:
            for outcome in self.current_prices.keys():
                if outcome not in self.outcomes:
                    raise ValueError(f"Price outcome '{outcome}' not in market outcomes")
    
    @property
    def is_active(self) -> bool:
        """Check if market is currently active for trading"""
        return (
            self.status == MarketStatus.ACTIVE and 
            self.end_date > datetime.now()
        )
    
    @property
    def time_to_close(self) -> Optional[int]:
        """Get seconds until market closes"""
        if self.end_date > datetime.now():
            return int((self.end_date - datetime.now()).total_seconds())
        return None
    
    @property
    def total_liquidity(self) -> Decimal:
        """Get total market liquidity"""
        if self.liquidity:
            return self.liquidity.total_liquidity
        return Decimal('0')
    
    @property
    def volume_24h_property(self) -> Decimal:
        """Get 24-hour trading volume"""
        if self.liquidity:
            return self.liquidity.volume_24h
        return self.volume_24h
    
    def get_outcome_price(self, outcome: str) -> Optional[Decimal]:
        """Get current price for a specific outcome"""
        return self.current_prices.get(outcome)
    
    def get_order_book(self, outcome: str) -> Optional[OrderBook]:
        """Get order book for a specific outcome"""
        return self.order_books.get(outcome)
    
    def update_price(self, outcome: str, price: Decimal):
        """Update price for an outcome"""
        if outcome not in self.outcomes:
            raise ValueError(f"Outcome '{outcome}' not found in market")
        
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
            
        if price < 0 or price > 1:
            raise ValueError("Price must be between 0 and 1")
        
        self.current_prices[outcome] = price
        self.updated_at = datetime.now()
    
    def add_order_book(self, outcome: str, order_book: OrderBook):
        """Add or update order book for an outcome"""
        if outcome not in self.outcomes:
            raise ValueError(f"Outcome '{outcome}' not found in market")
        
        self.order_books[outcome] = order_book
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market to dictionary for serialization"""
        return {
            'id': self.id,
            'question': self.question,
            'outcomes': self.outcomes,
            'end_date': self.end_date.isoformat(),
            'status': self.status.value,
            'current_prices': {k: float(v) for k, v in self.current_prices.items()},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'time_to_close': self.time_to_close,
            'total_liquidity': float(self.total_liquidity),
            'volume': float(self.volume),
            'volume_24h': float(self.volume_24h),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Market':
        """Create market from dictionary"""
        # Parse datetime fields
        end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00'))
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()).replace('Z', '+00:00'))
        updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()).replace('Z', '+00:00'))
        
        # Parse status
        status = MarketStatus(data['status'])
        
        # Parse prices
        current_prices = {k: Decimal(str(v)) for k, v in data.get('current_prices', {}).items()}
        
        return cls(
            id=data['id'],
            question=data['question'],
            outcomes=data['outcomes'],
            end_date=end_date,
            status=status,
            current_prices=current_prices,
            created_at=created_at,
            updated_at=updated_at,
            volume=Decimal(str(data.get('volume', '0'))),
            volume_24h=Decimal(str(data.get('volume_24h', '0'))),
        )