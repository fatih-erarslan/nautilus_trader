"""
Real-time performance metrics tracking for trading systems.
Includes P&L, Sharpe ratio, win rate, and drawdown calculations.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    fees: float
    slippage: float
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def net_pnl(self) -> float:
        """Net P&L after fees and slippage."""
        return self.pnl - self.fees - abs(self.slippage)
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.side == 'long':
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        return self.exit_time - self.entry_time


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    side: str
    current_price: float = 0.0
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.current_price == 0:
            return 0.0
        
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def value(self) -> float:
        """Current position value."""
        return self.current_price * self.quantity


class PerformanceMetrics:
    """
    Comprehensive performance tracking for trading systems.
    """
    
    def __init__(self, initial_capital: float = 100000.0, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics tracker.
        
        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Trade history
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        
        # Performance data
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.daily_returns: deque = deque(maxlen=252)  # 1 year of trading days
        self.high_water_mark = initial_capital
        
        # Real-time metrics
        self._metrics_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 1.0  # seconds
        
        self._lock = asyncio.Lock()
    
    async def add_trade(self, trade: Trade):
        """Add a completed trade."""
        async with self._lock:
            self.trades.append(trade)
            self.current_capital += trade.net_pnl
            self.equity_curve.append((trade.exit_time, self.current_capital))
            
            # Update high water mark
            if self.current_capital > self.high_water_mark:
                self.high_water_mark = self.current_capital
            
            # Update daily returns if it's a new day
            if self.equity_curve[-2][0].date() != trade.exit_time.date():
                prev_equity = self.equity_curve[-2][1]
                daily_return = (self.current_capital - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
            
            # Invalidate cache
            self._cache_timestamp = None
    
    async def update_position(self, symbol: str, position: Optional[Position]):
        """Update or remove a position."""
        async with self._lock:
            if position:
                self.positions[symbol] = position
            elif symbol in self.positions:
                del self.positions[symbol]
    
    async def update_prices(self, prices: Dict[str, float]):
        """Update current prices for positions."""
        async with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = price
    
    def calculate_total_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(trade.net_pnl for trade in self.trades)
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.net_pnl > 0)
        return (winning_trades / len(self.trades)) * 100
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        gross_profits = sum(trade.net_pnl for trade in self.trades if trade.net_pnl > 0)
        gross_losses = abs(sum(trade.net_pnl for trade in self.trades if trade.net_pnl < 0))
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0
        
        return gross_profits / gross_losses
    
    def calculate_sharpe_ratio(self, periods: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            periods: Number of periods per year (252 for daily)
        """
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (self.risk_free_rate / periods)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(periods) * (np.mean(excess_returns) / np.std(excess_returns))
    
    def calculate_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_value)
        """
        if not self.equity_curve:
            return 0.0, 0.0
        
        equity_values = [eq[1] for eq in self.equity_curve]
        peak = equity_values[0]
        max_dd_pct = 0.0
        max_dd_value = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            dd_value = peak - value
            dd_pct = (dd_value / peak) * 100
            
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_value = dd_value
        
        return max_dd_pct, max_dd_value
    
    def calculate_avg_trade_metrics(self) -> Dict[str, float]:
        """Calculate average trade metrics."""
        if not self.trades:
            return {
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'avg_duration_minutes': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        durations = [t.duration.total_seconds() / 60 for t in self.trades]
        
        return {
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'avg_trade': np.mean([t.net_pnl for t in self.trades]),
            'avg_duration_minutes': np.mean(durations),
            'largest_win': max(wins) if wins else 0.0,
            'largest_loss': min(losses) if losses else 0.0
        }
    
    def calculate_slippage_stats(self) -> Dict[str, float]:
        """Calculate slippage statistics."""
        if not self.trades:
            return {
                'total_slippage': 0.0,
                'avg_slippage': 0.0,
                'max_slippage': 0.0,
                'slippage_pct': 0.0
            }
        
        slippages = [abs(t.slippage) for t in self.trades]
        total_volume = sum(t.quantity * t.entry_price for t in self.trades)
        
        return {
            'total_slippage': sum(slippages),
            'avg_slippage': np.mean(slippages),
            'max_slippage': max(slippages),
            'slippage_pct': (sum(slippages) / total_volume * 100) if total_volume > 0 else 0.0
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        max_dd_pct, max_dd_value = self.calculate_max_drawdown()
        
        # Calculate Value at Risk (VaR) at 95% confidence
        if len(self.daily_returns) > 0:
            var_95 = np.percentile(self.daily_returns, 5) * self.current_capital
        else:
            var_95 = 0.0
        
        return {
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_value': max_dd_value,
            'current_drawdown_pct': ((self.high_water_mark - self.current_capital) / self.high_water_mark * 100),
            'var_95_daily': var_95,
            'risk_reward_ratio': self.calculate_risk_reward_ratio()
        }
    
    def calculate_risk_reward_ratio(self) -> float:
        """Calculate average risk/reward ratio."""
        if not self.trades:
            return 0.0
        
        avg_win = np.mean([t.net_pnl for t in self.trades if t.net_pnl > 0]) or 0
        avg_loss = abs(np.mean([t.net_pnl for t in self.trades if t.net_pnl < 0])) or 1
        
        return avg_win / avg_loss
    
    async def get_real_time_metrics(self) -> Dict[str, any]:
        """
        Get all real-time performance metrics with caching.
        """
        async with self._lock:
            # Check cache
            if (self._cache_timestamp and 
                (datetime.now().timestamp() - self._cache_timestamp) < self._cache_ttl):
                return self._metrics_cache
            
            # Calculate all metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'total_pnl': self.calculate_total_pnl(),
                    'unrealized_pnl': self.calculate_unrealized_pnl(),
                    'total_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
                },
                'trade_stats': {
                    'total_trades': len(self.trades),
                    'open_positions': len(self.positions),
                    'win_rate': self.calculate_win_rate(),
                    'profit_factor': self.calculate_profit_factor(),
                    **self.calculate_avg_trade_metrics()
                },
                'risk_metrics': {
                    'sharpe_ratio': self.calculate_sharpe_ratio(),
                    **self.calculate_risk_metrics()
                },
                'slippage': self.calculate_slippage_stats(),
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'value': pos.value
                    } for symbol, pos in self.positions.items()
                }
            }
            
            # Update cache
            self._metrics_cache = metrics
            self._cache_timestamp = datetime.now().timestamp()
            
            return metrics
    
    def export_equity_curve(self) -> pd.DataFrame:
        """Export equity curve as pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['equity'].pct_change()
        df['cumulative_returns'] = ((df['equity'] / self.initial_capital) - 1) * 100
        
        return df
    
    def export_trade_history(self) -> pd.DataFrame:
        """Export trade history as pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'duration_minutes': trade.duration.total_seconds() / 60,
                'side': trade.side,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'fees': trade.fees,
                'slippage': trade.slippage,
                'net_pnl': trade.net_pnl,
                'return_pct': trade.return_pct
            })
        
        return pd.DataFrame(data)
    
    def to_json(self) -> str:
        """Export metrics as JSON."""
        loop = asyncio.get_event_loop()
        metrics = loop.run_until_complete(self.get_real_time_metrics())
        return json.dumps(metrics, indent=2)