"""Data models for Trading Decision Engine."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SHORT = "SHORT"
    COVER = "COVER"


class RiskLevel(Enum):
    """Risk levels for trading signals."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradingStrategy(Enum):
    """Trading strategy types."""
    SWING = "SWING"  # 3-10 day holds
    MOMENTUM = "MOMENTUM"  # Follow trend strength
    MIRROR = "MIRROR"  # Copy institutional trades
    DAY_TRADE = "DAY_TRADE"  # Intraday only
    POSITION = "POSITION"  # Long-term hold


class AssetType(Enum):
    """Asset types supported by the engine."""
    EQUITY = "EQUITY"
    BOND = "BOND"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"
    FOREX = "FOREX"


@dataclass
class TradingSignal:
    """Trading signal with complete execution details."""
    
    # Core identifiers
    id: str
    timestamp: datetime
    asset: str
    asset_type: AssetType
    
    # Signal details
    signal_type: SignalType
    strategy: TradingStrategy
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    risk_level: RiskLevel
    
    # Position parameters
    position_size: float  # Fraction of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period: str  # Expected holding time
    
    # Source and reasoning
    source_events: List[str]
    reasoning: str
    
    # Optional fields
    technical_indicators: Optional[Dict[str, Any]] = field(default_factory=dict)
    mirror_source: Optional[str] = None  # For mirror trades
    momentum_score: Optional[float] = None  # For momentum trades
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal parameters."""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Signal strength must be between 0 and 1, got {self.strength}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not 0 < self.position_size <= 1:
            raise ValueError(f"Position size must be between 0 and 1, got {self.position_size}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "asset_type": self.asset_type.value,
            "signal_type": self.signal_type.value,
            "strategy": self.strategy.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "holding_period": self.holding_period,
            "source_events": self.source_events,
            "reasoning": self.reasoning,
            "technical_indicators": self.technical_indicators,
            "mirror_source": self.mirror_source,
            "momentum_score": self.momentum_score,
            "metadata": self.metadata
        }
        
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0
        
    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.signal_type in [SignalType.BUY, SignalType.HOLD]
        
    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.signal_type in [SignalType.SHORT, SignalType.SELL]