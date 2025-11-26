"""Order data models for Polymarket API."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Any, List
from decimal import Decimal

from .common import BaseModel


class OrderType(Enum):
    """Order type enumeration."""
    
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    
    @property
    def requires_price(self) -> bool:
        """Check if order type requires a price."""
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)
    
    @property
    def requires_stop_price(self) -> bool:
        """Check if order type requires a stop price."""
        return self in (OrderType.STOP, OrderType.STOP_LIMIT)


class OrderSide(Enum):
    """Order side enumeration."""
    
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self in (OrderStatus.OPEN, OrderStatus.PARTIAL)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
    
    @property
    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        return self in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL)


class TimeInForce(Enum):
    """Time in force enumeration."""
    
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    DAY = "day"  # Good for Day


@dataclass
class OrderFill(BaseModel):
    """Order fill/execution information."""
    
    id: str
    order_id: str
    price: float
    size: float
    side: OrderSide
    timestamp: datetime
    fee: float = 0.0
    fee_currency: str = "USDC"
    
    @property
    def value(self) -> float:
        """Calculate fill value."""
        return self.price * self.size
    
    @property
    def net_value(self) -> float:
        """Calculate net value after fees."""
        return self.value - self.fee
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "fee": self.fee,
            "fee_currency": self.fee_currency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderFill":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            order_id=data["order_id"],
            price=float(data["price"]),
            size=float(data["size"]),
            side=OrderSide(data["side"]),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            fee=float(data.get("fee", 0.0)),
            fee_currency=data.get("fee_currency", "USDC")
        )


@dataclass
class Order(BaseModel):
    """Polymarket order representation."""
    
    id: str
    market_id: str
    outcome_id: str
    side: OrderSide
    type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    fills: List[OrderFill] = field(default_factory=list)
    fee_rate: float = 0.02
    client_order_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate order data after initialization."""
        # Validate required fields based on order type
        if self.type.requires_price and self.price is None:
            raise ValueError(f"Order type {self.type.value} requires a price")
        
        if self.type.requires_stop_price and self.stop_price is None:
            raise ValueError(f"Order type {self.type.value} requires a stop price")
        
        # Validate sizes
        if self.size <= 0:
            raise ValueError(f"Order size must be positive, got {self.size}")
        
        if self.filled < 0:
            raise ValueError(f"Filled amount cannot be negative, got {self.filled}")
        
        # Set remaining if not provided
        if self.remaining == 0.0 and self.filled < self.size:
            self.remaining = self.size - self.filled
        
        # Validate price bounds
        if self.price is not None and not (0.0 < self.price <= 1.0):
            raise ValueError(f"Price must be between 0 and 1, got {self.price}")
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.size == 0:
            return 0.0
        return (self.filled / self.size) * 100
    
    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status.is_active
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status.is_complete
    
    @property
    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        return self.status.can_cancel
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side == OrderSide.SELL
    
    @property
    def total_fees(self) -> float:
        """Calculate total fees from all fills."""
        return sum(fill.fee for fill in self.fills)
    
    @property
    def average_fill_price(self) -> Optional[float]:
        """Calculate average fill price from all fills."""
        if not self.fills or self.filled == 0:
            return None
        
        total_value = sum(fill.price * fill.size for fill in self.fills)
        return total_value / self.filled
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the order."""
        if self.price is None:
            return 0.0
        return self.size * self.price
    
    @property
    def remaining_value(self) -> float:
        """Calculate remaining notional value."""
        if self.price is None:
            return 0.0
        return self.remaining * self.price
    
    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to this order."""
        if fill.order_id != self.id:
            raise ValueError(f"Fill order_id {fill.order_id} doesn't match order id {self.id}")
        
        self.fills.append(fill)
        self.filled += fill.size
        self.remaining = max(0, self.size - self.filled)
        
        # Update status based on fill
        if self.remaining == 0:
            self.status = OrderStatus.FILLED
        elif self.filled > 0:
            self.status = OrderStatus.PARTIAL
        
        self.updated_at = datetime.now(timezone.utc)
    
    def get_fills_by_side(self, side: OrderSide) -> List[OrderFill]:
        """Get fills filtered by side."""
        return [fill for fill in self.fills if fill.side == side]
    
    def get_time_since_creation(self) -> Optional[int]:
        """Get seconds since order creation."""
        if not self.created_at:
            return None
        
        now = datetime.now(timezone.utc)
        return int((now - self.created_at).total_seconds())
    
    def is_expired(self) -> bool:
        """Check if order has expired."""
        if not self.expires_at:
            return False
        
        return datetime.now(timezone.utc) >= self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "side": self.side.value,
            "type": self.type.value,
            "size": self.size,
            "price": self.price,
            "stop_price": self.stop_price,
            "filled": self.filled,
            "remaining": self.remaining,
            "status": self.status.value,
            "time_in_force": self.time_in_force.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "fills": [fill.to_dict() for fill in self.fills],
            "fee_rate": self.fee_rate,
            "client_order_id": self.client_order_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create instance from dictionary."""
        # Parse dates
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        
        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        
        # Parse fills
        fills = []
        for fill_data in data.get("fills", []):
            fills.append(OrderFill.from_dict(fill_data))
        
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            side=OrderSide(data["side"]),
            type=OrderType(data["type"]),
            size=float(data["size"]),
            price=float(data["price"]) if data.get("price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            filled=float(data.get("filled", 0.0)),
            remaining=float(data.get("remaining", 0.0)),
            status=OrderStatus(data.get("status", "pending")),
            time_in_force=TimeInForce(data.get("time_in_force", "gtc")),
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            fills=fills,
            fee_rate=float(data.get("fee_rate", 0.02)),
            client_order_id=data.get("client_order_id")
        )


@dataclass
class OrderBook(BaseModel):
    """Order book representation."""
    
    market_id: str
    outcome_id: str
    bids: List[Dict[str, float]] = field(default_factory=list)  # [{"price": 0.6, "size": 100}]
    asks: List[Dict[str, float]] = field(default_factory=list)  # [{"price": 0.65, "size": 150}]
    timestamp: Optional[datetime] = None
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bids:
            return None
        return max(bid["price"] for bid in self.bids)
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.asks:
            return None
        return min(ask["price"] for ask in self.asks)
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2
    
    @property
    def total_bid_size(self) -> float:
        """Calculate total bid size."""
        return sum(bid["size"] for bid in self.bids)
    
    @property
    def total_ask_size(self) -> float:
        """Calculate total ask size."""
        return sum(ask["size"] for ask in self.asks)
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Dict[str, float]]]:
        """Get order book depth for specified levels."""
        # Sort bids (highest price first) and asks (lowest price first)
        sorted_bids = sorted(self.bids, key=lambda x: x["price"], reverse=True)
        sorted_asks = sorted(self.asks, key=lambda x: x["price"])
        
        return {
            "bids": sorted_bids[:levels],
            "asks": sorted_asks[:levels]
        }
    
    def get_price_impact(self, side: OrderSide, size: float) -> Optional[float]:
        """Calculate price impact for a market order of given size."""
        levels = self.asks if side == OrderSide.BUY else self.bids
        
        if not levels:
            return None
        
        remaining_size = size
        total_cost = 0.0
        
        sorted_levels = sorted(levels, key=lambda x: x["price"], 
                             reverse=(side == OrderSide.SELL))
        
        for level in sorted_levels:
            if remaining_size <= 0:
                break
            
            level_size = min(remaining_size, level["size"])
            total_cost += level_size * level["price"]
            remaining_size -= level_size
        
        if remaining_size > 0:
            # Not enough liquidity
            return None
        
        average_price = total_cost / size
        best_price = self.best_ask if side == OrderSide.BUY else self.best_bid
        
        if best_price is None:
            return None
        
        return abs(average_price - best_price) / best_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "bids": self.bids,
            "asks": self.asks,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBook":
        """Create instance from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        
        return cls(
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            timestamp=timestamp
        )


class TradeStatus(Enum):
    """Trade status enumeration."""
    
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Trade(BaseModel):
    """Trade execution representation."""
    
    id: str
    market_id: str
    outcome_id: str
    order_id: Optional[str]
    side: OrderSide
    price: float
    size: float
    timestamp: datetime
    status: TradeStatus = TradeStatus.EXECUTED
    fee: float = 0.0
    fee_currency: str = "USDC"
    
    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.price * self.size
    
    @property
    def net_value(self) -> float:
        """Calculate net value after fees."""
        return self.value - self.fee
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "order_id": self.order_id,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "fee": self.fee,
            "fee_currency": self.fee_currency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            order_id=data.get("order_id"),
            side=OrderSide(data["side"]),
            price=float(data["price"]),
            size=float(data["size"]),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            status=TradeStatus(data.get("status", "executed")),
            fee=float(data.get("fee", 0.0)),
            fee_currency=data.get("fee_currency", "USDC")
        )