"""Performance tracking data models."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class TradeResult:
    """Represents a completed or in-progress trade."""
    
    trade_id: str
    signal_id: str
    asset: str
    entry_time: datetime
    entry_price: float
    position_size: float
    status: TradeStatus = TradeStatus.OPEN
    
    # Optional fields for completed trades
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    
    # News attribution
    news_events: List[str] = field(default_factory=list)
    sentiment_scores: List[float] = field(default_factory=list)
    
    # Additional metadata
    fees: float = 0.0
    slippage: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the trade result."""
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
            
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError("Exit price must be positive")
    
    def calculate_pnl(self) -> None:
        """Calculate P&L if trade is closed."""
        if self.status == TradeStatus.CLOSED and self.exit_price:
            # Calculate gross P&L
            gross_pnl = (self.exit_price - self.entry_price) * self.position_size
            
            # Subtract fees
            self.pnl = gross_pnl - self.fees - self.slippage
            
            # Calculate percentage
            investment = self.entry_price * self.position_size
            self.pnl_percentage = (self.pnl / investment) * 100 if investment > 0 else 0


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    
    # Additional metrics
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    average_trade_duration: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    def __post_init__(self):
        """Validate metrics."""
        if not 0 <= self.win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
            
        if self.total_trades != self.winning_trades + self.losing_trades:
            raise ValueError("Total trades must equal winning + losing trades")
    
    @classmethod
    def from_trades(cls, trades: List[TradeResult]) -> "PerformanceMetrics":
        """Calculate metrics from a list of trades."""
        if not trades:
            return cls(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_pnl=0.0,
            )
        
        # Calculate basic metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        average_win = (
            sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        
        average_loss = (
            sum(t.pnl for t in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl_percentage for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in sorted(trades, key=lambda x: x.exit_time or x.entry_time):
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
        
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            max_drawdown = 0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (value - peak) / peak if peak > 0 else 0
                max_drawdown = min(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        return cls(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_pnl=sum(t.pnl for t in trades),
        )


@dataclass
class Attribution:
    """Trade attribution to news sources."""
    
    source_contributions: Dict[str, float]  # Source -> contribution percentage
    primary_catalyst: Optional[str]  # Primary news event ID
    news_weights: Dict[str, float]  # News ID -> weight
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Validate attribution."""
        # Ensure contributions sum to approximately 1
        total = sum(self.source_contributions.values())
        if total > 0 and not 0.99 <= total <= 1.01:
            # Normalize
            for source in self.source_contributions:
                self.source_contributions[source] /= total


@dataclass
class SourceMetrics:
    """Performance metrics for a news source."""
    
    source_name: str
    total_signals: int
    profitable_signals: int
    accuracy_rate: float
    average_pnl_per_signal: float
    weighted_pnl: float
    signal_quality_score: float
    
    # Time-based metrics
    signals_last_24h: int = 0
    signals_last_7d: int = 0
    signals_last_30d: int = 0
    
    # Sentiment accuracy
    sentiment_mae: float = 0.0
    sentiment_directional_accuracy: float = 0.0


@dataclass
class ModelMetrics:
    """Performance metrics for ML models."""
    
    model_name: str
    prediction_count: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    confidence_calibration: float  # How well confidence matches accuracy
    average_confidence: float
    directional_accuracy: float  # Percentage of correct direction predictions
    
    # Additional metrics
    r_squared: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)