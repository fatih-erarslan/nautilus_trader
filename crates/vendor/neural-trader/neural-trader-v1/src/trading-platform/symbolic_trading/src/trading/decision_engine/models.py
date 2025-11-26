"""
Trading Decision Engine Models
Defines core data structures for trading signals and portfolio management
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SHORT = "SHORT"
    COVER = "COVER"


class RiskLevel(Enum):
    """Risk levels for trading positions"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradingStrategy(Enum):
    """Trading strategy types"""
    SWING = "SWING"  # 3-10 day holds
    MOMENTUM = "MOMENTUM"  # Follow trend strength
    MIRROR = "MIRROR"  # Copy institutional trades
    DAY_TRADE = "DAY_TRADE"  # Intraday only
    POSITION = "POSITION"  # Long-term hold


class AssetType(Enum):
    """Asset class types"""
    EQUITY = "EQUITY"
    BOND = "BOND"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"
    FOREX = "FOREX"


@dataclass
class TradingSignal:
    """
    Represents a trading signal with all necessary information
    for execution and risk management
    """
    # Required fields
    id: str
    timestamp: datetime
    asset: str
    asset_type: AssetType
    signal_type: SignalType
    strategy: TradingStrategy
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    risk_level: RiskLevel
    position_size: float  # Fraction of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period: str  # Expected holding time
    source_events: List[str]
    reasoning: str
    
    # Optional fields
    technical_indicators: Optional[Dict[str, Any]] = None
    mirror_source: Optional[str] = None  # For mirror trades
    momentum_score: Optional[float] = None  # For momentum trades
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate field values after initialization"""
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not 0 <= self.position_size <= 1:
            raise ValueError("position_size must be between 0 and 1")


@dataclass
class PortfolioPosition:
    """Represents a current position in the portfolio"""
    asset: str
    asset_type: AssetType
    size: float  # Fraction of portfolio
    entry_price: float
    current_price: float
    unrealized_pnl: float  # Percentage gain/loss
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: TradingStrategy
    
    # Optional fields
    leverage: float = 1.0
    margin_used: float = 0.0
    metadata: Optional[Dict[str, Any]] = None