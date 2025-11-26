"""
Base Strategy for Alpaca WebSocket Trading

Provides abstract base class for real-time trading strategies
with event handlers, signal generation, and position tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    quantity: int
    confidence: float  # 0.0 to 1.0
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Track position state"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, price: float):
        """Update current price and unrealized PnL"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        
    def close(self, exit_price: float) -> float:
        """Close position and return realized PnL"""
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        return self.realized_pnl


class BaseStreamStrategy(ABC):
    """
    Abstract base class for WebSocket trading strategies
    
    Handles:
    - Event routing (trades, quotes, bars)
    - Position tracking
    - Signal generation framework
    - Risk management
    """
    
    def __init__(self, 
                 symbols: List[str],
                 max_positions: int = 5,
                 position_size: float = 10000,  # USD per position
                 risk_per_trade: float = 0.02,   # 2% risk per trade
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base strategy
        
        Args:
            symbols: List of symbols to trade
            max_positions: Maximum concurrent positions
            position_size: Dollar amount per position
            risk_per_trade: Risk percentage per trade
            logger: Optional logger instance
        """
        self.symbols = symbols
        self.max_positions = max_positions
        self.position_size = position_size
        self.risk_per_trade = risk_per_trade
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, TradingSignal] = {}
        
        # Market data storage
        self.latest_quotes: Dict[str, Dict] = {}
        self.latest_trades: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[float]] = {s: [] for s in symbols}
        self.volume_history: Dict[str, List[float]] = {s: [] for s in symbols}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info(f"Initialized {self.__class__.__name__} for symbols: {symbols}")
    
    def on_trade(self, trade: Dict[str, Any]):
        """
        Handle incoming trade data
        
        Args:
            trade: Trade data dict with keys:
                - symbol: str
                - price: float
                - size: int
                - timestamp: datetime
                - conditions: List[str]
        """
        symbol = trade['symbol']
        self.latest_trades[symbol] = trade
        
        # Update price history
        self.price_history[symbol].append(trade['price'])
        if len(self.price_history[symbol]) > 1000:  # Keep last 1000 prices
            self.price_history[symbol].pop(0)
            
        # Update volume history
        self.volume_history[symbol].append(trade['size'])
        if len(self.volume_history[symbol]) > 1000:
            self.volume_history[symbol].pop(0)
        
        # Update position prices
        if symbol in self.positions:
            self.positions[symbol].update_price(trade['price'])
        
        # Call strategy-specific handler
        self._on_trade(trade)
        
        # Check for signals
        signal = self.generate_signal(symbol)
        if signal:
            self.execute_signal(signal)
    
    def on_quote(self, quote: Dict[str, Any]):
        """
        Handle incoming quote data
        
        Args:
            quote: Quote data dict with keys:
                - symbol: str
                - bid_price: float
                - bid_size: int
                - ask_price: float
                - ask_size: int
                - timestamp: datetime
        """
        symbol = quote['symbol']
        self.latest_quotes[symbol] = quote
        
        # Call strategy-specific handler
        self._on_quote(quote)
        
        # Check for signals
        signal = self.generate_signal(symbol)
        if signal:
            self.execute_signal(signal)
    
    def on_bar(self, bar: Dict[str, Any]):
        """
        Handle incoming bar data
        
        Args:
            bar: Bar data dict with keys:
                - symbol: str
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int
                - timestamp: datetime
        """
        symbol = bar['symbol']
        
        # Call strategy-specific handler
        self._on_bar(bar)
        
        # Check for signals
        signal = self.generate_signal(symbol)
        if signal:
            self.execute_signal(signal)
    
    def execute_signal(self, signal: TradingSignal):
        """Execute trading signal with risk management"""
        symbol = signal.symbol
        
        # Risk checks
        if not self.validate_signal(signal):
            return
        
        # Position management
        if signal.signal_type == SignalType.BUY:
            if len(self.positions) >= self.max_positions:
                self.logger.warning(f"Max positions reached, skipping {symbol}")
                return
                
            if symbol not in self.positions:
                self.open_position(signal)
                
        elif signal.signal_type == SignalType.SELL or signal.signal_type == SignalType.CLOSE:
            if symbol in self.positions:
                self.close_position(symbol, signal)
    
    def open_position(self, signal: TradingSignal):
        """Open a new position"""
        symbol = signal.symbol
        quantity = signal.quantity or self.calculate_position_size(signal)
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            current_price=signal.price,
            metadata={
                'signal_confidence': signal.confidence,
                'entry_reason': signal.reason
            }
        )
        
        self.positions[symbol] = position
        self.total_trades += 1
        
        self.logger.info(
            f"OPENED {symbol} - Qty: {quantity}, "
            f"Price: ${signal.price:.2f}, Reason: {signal.reason}"
        )
    
    def close_position(self, symbol: str, signal: TradingSignal):
        """Close an existing position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        pnl = position.close(signal.price)
        
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
            
        del self.positions[symbol]
        
        self.logger.info(
            f"CLOSED {symbol} - PnL: ${pnl:.2f}, "
            f"Total PnL: ${self.total_pnl:.2f}, "
            f"Win Rate: {self.get_win_rate():.1%}"
        )
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """Calculate position size based on risk management"""
        # Simple fixed position sizing
        # Override in subclasses for more sophisticated sizing
        shares = int(self.position_size / signal.price)
        return max(1, shares)
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal before execution"""
        # Basic validation
        if signal.confidence < 0.5:
            self.logger.debug(f"Signal confidence too low: {signal.confidence}")
            return False
            
        if signal.price <= 0:
            self.logger.error(f"Invalid price: {signal.price}")
            return False
            
        return True
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * (np.mean(returns) / np.std(returns))
    
    def get_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)
    
    @abstractmethod
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal for symbol
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def _on_trade(self, trade: Dict[str, Any]):
        """Strategy-specific trade handler"""
        pass
    
    @abstractmethod
    def _on_quote(self, quote: Dict[str, Any]):
        """Strategy-specific quote handler"""
        pass
    
    @abstractmethod
    def _on_bar(self, bar: Dict[str, Any]):
        """Strategy-specific bar handler"""
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.total_pnl,
            'open_positions': len(self.positions),
            'position_details': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            }
        }