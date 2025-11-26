"""
Momentum Stream Strategy for Alpaca WebSocket Trading

Real-time momentum detection with volume-weighted calculations
and dynamic threshold adjustment.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import deque
from .base_strategy import BaseStreamStrategy, TradingSignal, SignalType


class MomentumStreamStrategy(BaseStreamStrategy):
    """
    Real-time momentum trading strategy
    
    Features:
    - Volume-weighted momentum calculations
    - Dynamic threshold adjustment based on volatility
    - Multiple timeframe analysis
    - Adaptive entry/exit signals
    """
    
    def __init__(self,
                 symbols: List[str],
                 momentum_window: int = 20,
                 volume_window: int = 10,
                 entry_threshold: float = 2.0,  # Standard deviations
                 exit_threshold: float = 0.5,
                 use_volume_weight: bool = True,
                 adaptive_thresholds: bool = True,
                 **kwargs):
        """
        Initialize momentum strategy
        
        Args:
            momentum_window: Lookback period for momentum calculation
            volume_window: Window for volume analysis
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            use_volume_weight: Weight momentum by volume
            adaptive_thresholds: Adjust thresholds based on market conditions
        """
        super().__init__(symbols, **kwargs)
        
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_volume_weight = use_volume_weight
        self.adaptive_thresholds = adaptive_thresholds
        
        # Data storage for each symbol
        self.price_windows = {s: deque(maxlen=momentum_window) for s in symbols}
        self.volume_windows = {s: deque(maxlen=volume_window) for s in symbols}
        self.momentum_scores = {s: deque(maxlen=100) for s in symbols}
        
        # Volatility tracking for adaptive thresholds
        self.volatility = {s: 0.0 for s in symbols}
        self.volatility_window = 50
        
        # Performance tracking
        self.momentum_signals = {s: [] for s in symbols}
        
    def _on_trade(self, trade: Dict[str, Any]):
        """Process trade data for momentum calculations"""
        symbol = trade['symbol']
        price = trade['price']
        volume = trade['size']
        
        # Update price and volume windows
        self.price_windows[symbol].append(price)
        self.volume_windows[symbol].append(volume)
        
        # Update volatility estimate
        if len(self.price_windows[symbol]) >= 2:
            returns = np.diff(list(self.price_windows[symbol])) / list(self.price_windows[symbol])[:-1]
            self.volatility[symbol] = np.std(returns) if len(returns) > 0 else 0.0
    
    def _on_quote(self, quote: Dict[str, Any]):
        """Process quote data"""
        # Momentum primarily uses trade data
        # Quotes used for spread analysis
        pass
    
    def _on_bar(self, bar: Dict[str, Any]):
        """Process bar data for additional momentum confirmation"""
        symbol = bar['symbol']
        
        # Can use bar data for longer-term momentum confirmation
        # For now, focusing on tick-level momentum
        pass
    
    def calculate_momentum(self, symbol: str) -> float:
        """
        Calculate momentum score with volume weighting
        
        Returns:
            Momentum z-score
        """
        prices = list(self.price_windows[symbol])
        volumes = list(self.volume_windows[symbol])
        
        if len(prices) < self.momentum_window:
            return 0.0
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        if self.use_volume_weight and len(volumes) >= len(returns):
            # Volume-weighted momentum
            recent_volumes = volumes[-len(returns):]
            weights = np.array(recent_volumes) / np.sum(recent_volumes)
            weighted_return = np.sum(returns * weights)
            
            # Calculate volume-weighted standard deviation
            mean_return = np.average(returns, weights=weights)
            variance = np.average((returns - mean_return)**2, weights=weights)
            std_return = np.sqrt(variance)
        else:
            # Simple momentum
            weighted_return = np.mean(returns[-5:])  # Recent returns
            std_return = np.std(returns)
        
        # Calculate z-score
        if std_return > 0:
            z_score = weighted_return / std_return
        else:
            z_score = 0.0
            
        # Store momentum score
        self.momentum_scores[symbol].append(z_score)
        
        return z_score
    
    def calculate_volume_surge(self, symbol: str) -> float:
        """Calculate volume surge indicator"""
        volumes = list(self.volume_windows[symbol])
        
        if len(volumes) < self.volume_window:
            return 1.0
            
        recent_volume = np.mean(volumes[-3:])
        avg_volume = np.mean(volumes)
        
        if avg_volume > 0:
            return recent_volume / avg_volume
        return 1.0
    
    def get_dynamic_threshold(self, symbol: str, is_entry: bool) -> float:
        """
        Calculate dynamic threshold based on market conditions
        
        Args:
            symbol: Trading symbol
            is_entry: True for entry threshold, False for exit
            
        Returns:
            Adjusted threshold
        """
        if not self.adaptive_thresholds:
            return self.entry_threshold if is_entry else self.exit_threshold
            
        base_threshold = self.entry_threshold if is_entry else self.exit_threshold
        
        # Adjust based on volatility
        vol_adjustment = 1.0
        if self.volatility[symbol] > 0:
            # Higher volatility = higher thresholds
            avg_volatility = 0.02  # 2% typical daily volatility
            vol_ratio = self.volatility[symbol] / avg_volatility
            vol_adjustment = max(0.5, min(2.0, vol_ratio))
        
        # Adjust based on recent momentum accuracy
        if len(self.momentum_signals[symbol]) >= 10:
            recent_signals = self.momentum_signals[symbol][-10:]
            accuracy = sum(1 for s in recent_signals if s['profitable']) / len(recent_signals)
            
            # Lower threshold if high accuracy, raise if low
            if accuracy > 0.7:
                vol_adjustment *= 0.9
            elif accuracy < 0.3:
                vol_adjustment *= 1.1
                
        return base_threshold * vol_adjustment
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate momentum-based trading signal"""
        # Need enough data
        if len(self.price_windows[symbol]) < self.momentum_window:
            return None
            
        # Calculate momentum
        momentum_score = self.calculate_momentum(symbol)
        volume_surge = self.calculate_volume_surge(symbol)
        
        # Get current price
        current_price = self.price_windows[symbol][-1]
        
        # Dynamic thresholds
        entry_threshold = self.get_dynamic_threshold(symbol, True)
        exit_threshold = self.get_dynamic_threshold(symbol, False)
        
        # Check if we have a position
        has_position = symbol in self.positions
        
        # Generate signals
        signal = None
        
        if not has_position:
            # Entry signals
            if momentum_score > entry_threshold and volume_surge > 1.2:
                # Strong positive momentum with volume
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    price=current_price,
                    quantity=None,  # Will be calculated
                    confidence=min(0.95, momentum_score / 3.0),
                    reason=f"Momentum surge: {momentum_score:.2f}σ, Volume: {volume_surge:.1f}x",
                    metadata={
                        'momentum_score': momentum_score,
                        'volume_surge': volume_surge,
                        'volatility': self.volatility[symbol]
                    }
                )
            elif momentum_score < -entry_threshold and volume_surge > 1.2:
                # Short opportunity (if enabled)
                # For now, skip shorts
                pass
                
        else:
            # Exit signals for existing position
            position = self.positions[symbol]
            
            # Exit conditions
            exit_signal = False
            reason = ""
            
            # Momentum reversal
            if momentum_score < -exit_threshold:
                exit_signal = True
                reason = f"Momentum reversal: {momentum_score:.2f}σ"
                
            # Take profit - momentum exhaustion
            elif momentum_score < exit_threshold and position.unrealized_pnl > 0:
                momentum_trend = list(self.momentum_scores[symbol])[-5:]
                if len(momentum_trend) >= 5 and all(m < momentum_trend[0] for m in momentum_trend[1:]):
                    exit_signal = True
                    reason = "Momentum exhaustion - taking profit"
                    
            # Stop loss - sharp reversal
            elif position.unrealized_pnl < -self.position_size * self.risk_per_trade:
                exit_signal = True
                reason = "Stop loss triggered"
                
            if exit_signal:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    timestamp=datetime.now(),
                    price=current_price,
                    quantity=position.quantity,
                    confidence=0.8,
                    reason=reason,
                    metadata={
                        'momentum_score': momentum_score,
                        'position_pnl': position.unrealized_pnl
                    }
                )
        
        # Track signal for performance analysis
        if signal:
            self.momentum_signals[symbol].append({
                'timestamp': signal.timestamp,
                'type': signal.signal_type,
                'momentum': momentum_score,
                'profitable': None  # Updated later
            })
            
        return signal
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """
        Calculate position size based on momentum strength and volatility
        """
        base_size = super().calculate_position_size(signal)
        
        # Adjust based on signal confidence and volatility
        symbol = signal.symbol
        volatility = self.volatility[symbol]
        
        # Lower size for higher volatility
        if volatility > 0:
            vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
        else:
            vol_adjustment = 1.0
            
        # Adjust based on momentum strength
        momentum_adjustment = min(1.5, signal.confidence + 0.5)
        
        adjusted_size = int(base_size * vol_adjustment * momentum_adjustment)
        
        return max(1, adjusted_size)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get momentum-specific metrics"""
        base_metrics = self.get_performance_summary()
        
        # Add momentum-specific metrics
        momentum_metrics = {
            'avg_momentum_score': {},
            'volatility': self.volatility,
            'signal_counts': {}
        }
        
        for symbol in self.symbols:
            if self.momentum_scores[symbol]:
                momentum_metrics['avg_momentum_score'][symbol] = np.mean(list(self.momentum_scores[symbol]))
            
            momentum_metrics['signal_counts'][symbol] = len(self.momentum_signals[symbol])
        
        base_metrics['momentum_metrics'] = momentum_metrics
        
        return base_metrics