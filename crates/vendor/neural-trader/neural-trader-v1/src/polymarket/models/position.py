"""
Position and Portfolio data models for Polymarket

This module defines data structures for tracking user positions,
portfolio performance, and P&L metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any


class PositionStatus(Enum):
    """Position status enumeration."""
    
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class PnLMetrics:
    """Profit and Loss metrics."""
    
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal = field(init=False)
    roi_percentage: Decimal = field(init=False)
    cost_basis: Decimal = Decimal('0')
    current_value: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if not isinstance(self.realized_pnl, Decimal):
            self.realized_pnl = Decimal(str(self.realized_pnl))
        if not isinstance(self.unrealized_pnl, Decimal):
            self.unrealized_pnl = Decimal(str(self.unrealized_pnl))
        if not isinstance(self.cost_basis, Decimal):
            self.cost_basis = Decimal(str(self.cost_basis))
        if not isinstance(self.current_value, Decimal):
            self.current_value = Decimal(str(self.current_value))
        
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        if self.cost_basis > 0:
            self.roi_percentage = (self.total_pnl / self.cost_basis) * 100
        else:
            self.roi_percentage = Decimal('0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.total_pnl),
            "roi_percentage": float(self.roi_percentage),
            "cost_basis": float(self.cost_basis),
            "current_value": float(self.current_value)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PnLMetrics":
        """Create instance from dictionary."""
        return cls(
            realized_pnl=Decimal(str(data["realized_pnl"])),
            unrealized_pnl=Decimal(str(data["unrealized_pnl"])),
            cost_basis=Decimal(str(data.get("cost_basis", 0))),
            current_value=Decimal(str(data.get("current_value", 0)))
        )


@dataclass
class Position:
    """User position in a specific market outcome."""
    
    id: str
    market_id: str
    outcome_id: str
    user_id: str
    shares: Decimal
    average_price: Decimal
    status: PositionStatus
    pnl: Optional[PnLMetrics] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and convert types."""
        if not isinstance(self.shares, Decimal):
            self.shares = Decimal(str(self.shares))
        if not isinstance(self.average_price, Decimal):
            self.average_price = Decimal(str(self.average_price))
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis of position."""
        return self.shares * self.average_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long (positive shares)."""
        return self.shares > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short (negative shares)."""
        return self.shares < 0
    
    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.status == PositionStatus.CLOSED or self.shares == 0
    
    def calculate_pnl(self, current_price: Decimal) -> PnLMetrics:
        """Calculate P&L based on current market price."""
        current_value = self.shares * current_price
        unrealized_pnl = current_value - self.cost_basis
        
        # For closed positions, all P&L is realized
        if self.is_closed:
            realized_pnl = unrealized_pnl
            unrealized_pnl = Decimal('0')
        else:
            realized_pnl = Decimal('0')
        
        return PnLMetrics(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            cost_basis=self.cost_basis,
            current_value=current_value
        )
    
    def update_pnl(self, current_price: Decimal) -> None:
        """Update position P&L with current market price."""
        self.pnl = self.calculate_pnl(current_price)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "user_id": self.user_id,
            "shares": float(self.shares),
            "average_price": float(self.average_price),
            "status": self.status.value,
            "cost_basis": float(self.cost_basis),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
        if self.pnl:
            result["pnl"] = self.pnl.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create instance from dictionary."""
        position = cls(
            id=data["id"],
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            user_id=data["user_id"],
            shares=Decimal(str(data["shares"])),
            average_price=Decimal(str(data["average_price"])),
            status=PositionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        )
        
        if "pnl" in data:
            position.pnl = PnLMetrics.from_dict(data["pnl"])
        
        return position


@dataclass
class Portfolio:
    """User portfolio containing multiple positions."""
    
    user_id: str
    positions: List[Position] = field(default_factory=list)
    cash_balance: Decimal = Decimal('0')
    total_value: Decimal = field(init=False)
    total_pnl: Decimal = field(init=False)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize calculated fields."""
        if not isinstance(self.cash_balance, Decimal):
            self.cash_balance = Decimal(str(self.cash_balance))
        self._calculate_totals()
    
    def _calculate_totals(self):
        """Calculate total portfolio value and P&L."""
        position_value = sum(
            pos.pnl.current_value if pos.pnl else pos.cost_basis 
            for pos in self.positions
        )
        self.total_value = self.cash_balance + position_value
        
        total_pnl = sum(
            pos.pnl.total_pnl if pos.pnl else Decimal('0')
            for pos in self.positions
        )
        self.total_pnl = total_pnl
    
    def add_position(self, position: Position) -> None:
        """Add position to portfolio."""
        # Check if position already exists and merge
        existing = next(
            (p for p in self.positions 
             if p.market_id == position.market_id and p.outcome_id == position.outcome_id),
            None
        )
        
        if existing:
            # Merge positions (weighted average price)
            total_shares = existing.shares + position.shares
            if total_shares != 0:
                existing.average_price = (
                    (existing.shares * existing.average_price + 
                     position.shares * position.average_price) / total_shares
                )
                existing.shares = total_shares
                existing.updated_at = datetime.now()
        else:
            self.positions.append(position)
        
        self._calculate_totals()
        self.updated_at = datetime.now()
    
    def remove_position(self, position_id: str) -> bool:
        """Remove position from portfolio."""
        position = next((p for p in self.positions if p.id == position_id), None)
        if position:
            self.positions.remove(position)
            self._calculate_totals()
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_position(self, market_id: str, outcome_id: str) -> Optional[Position]:
        """Get position for specific market and outcome."""
        return next(
            (p for p in self.positions 
             if p.market_id == market_id and p.outcome_id == outcome_id),
            None
        )
    
    def get_positions_by_market(self, market_id: str) -> List[Position]:
        """Get all positions for a specific market."""
        return [p for p in self.positions if p.market_id == market_id]
    
    def update_all_pnl(self, market_prices: Dict[str, Dict[str, Decimal]]) -> None:
        """Update P&L for all positions with current market prices."""
        for position in self.positions:
            market_data = market_prices.get(position.market_id, {})
            current_price = market_data.get(position.outcome_id)
            
            if current_price:
                position.update_pnl(current_price)
        
        self._calculate_totals()
        self.updated_at = datetime.now()
    
    @property
    def open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if not p.is_closed]
    
    @property
    def closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return [p for p in self.positions if p.is_closed]
    
    @property
    def position_count(self) -> int:
        """Get total number of positions."""
        return len(self.positions)
    
    @property
    def open_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.open_positions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "positions": [pos.to_dict() for pos in self.positions],
            "cash_balance": float(self.cash_balance),
            "total_value": float(self.total_value),
            "total_pnl": float(self.total_pnl),
            "position_count": self.position_count,
            "open_position_count": self.open_position_count,
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        """Create instance from dictionary."""
        positions = [Position.from_dict(pos_data) for pos_data in data.get("positions", [])]
        
        portfolio = cls(
            user_id=data["user_id"],
            positions=positions,
            cash_balance=Decimal(str(data.get("cash_balance", 0))),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        )
        
        return portfolio