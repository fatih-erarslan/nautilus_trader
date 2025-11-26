"""
Arbitrage Stream Strategy for Alpaca WebSocket Trading

Real-time cross-asset arbitrage detection with spread calculation,
correlation tracking, and execution triggers.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
from scipy import stats
from .base_strategy import BaseStreamStrategy, TradingSignal, SignalType


class ArbitrageStreamStrategy(BaseStreamStrategy):
    """
    Real-time arbitrage trading strategy
    
    Features:
    - Statistical arbitrage between correlated assets
    - Real-time spread calculation and z-score tracking
    - Dynamic correlation updates
    - Cointegration testing
    - Optimal hedge ratio calculation
    """
    
    def __init__(self,
                 symbols: List[str],
                 pairs: Optional[List[Tuple[str, str]]] = None,
                 correlation_window: int = 100,
                 spread_window: int = 50,
                 entry_z_score: float = 2.0,
                 exit_z_score: float = 0.5,
                 min_correlation: float = 0.7,
                 use_cointegration: bool = True,
                 **kwargs):
        """
        Initialize arbitrage strategy
        
        Args:
            pairs: List of symbol pairs to trade (if None, finds pairs automatically)
            correlation_window: Window for correlation calculation
            spread_window: Window for spread z-score calculation
            entry_z_score: Z-score threshold for entering positions
            exit_z_score: Z-score threshold for exiting positions
            min_correlation: Minimum correlation to consider pair
            use_cointegration: Use cointegration test for pair validation
        """
        super().__init__(symbols, **kwargs)
        
        self.correlation_window = correlation_window
        self.spread_window = spread_window
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.min_correlation = min_correlation
        self.use_cointegration = use_cointegration
        
        # Pair management
        self.trading_pairs = pairs or []
        self.pair_stats: Dict[Tuple[str, str], Dict] = {}
        
        # Price series for correlation
        self.price_series = {s: deque(maxlen=correlation_window) for s in symbols}
        self.return_series = {s: deque(maxlen=correlation_window) for s in symbols}
        
        # Spread tracking
        self.spreads: Dict[Tuple[str, str], deque] = {}
        self.spread_means: Dict[Tuple[str, str], float] = {}
        self.spread_stds: Dict[Tuple[str, str], float] = {}
        
        # Execution tracking
        self.arbitrage_positions: Dict[Tuple[str, str], Dict] = {}
        
        # Performance metrics
        self.arbitrage_trades = []
        
        # Initialize pairs if not provided
        if not self.trading_pairs:
            self._find_trading_pairs()
    
    def _find_trading_pairs(self):
        """Automatically find correlated pairs for trading"""
        # Will be populated as data comes in
        self.logger.info("Auto-discovery of trading pairs enabled")
    
    def _on_trade(self, trade: Dict[str, Any]):
        """Process trade data for arbitrage calculations"""
        symbol = trade['symbol']
        price = trade['price']
        
        # Update price series
        self.price_series[symbol].append(price)
        
        # Calculate return if we have previous price
        if len(self.price_series[symbol]) >= 2:
            prev_price = self.price_series[symbol][-2]
            log_return = np.log(price / prev_price)
            self.return_series[symbol].append(log_return)
        
        # Update correlations and spreads
        self._update_pair_statistics()
        
        # Check for pair discovery if needed
        if not self.trading_pairs and len(self.price_series[symbol]) >= self.correlation_window:
            self._discover_new_pairs()
    
    def _on_quote(self, quote: Dict[str, Any]):
        """Process quote data for spread analysis"""
        # Can use bid-ask spreads for execution cost analysis
        pass
    
    def _on_bar(self, bar: Dict[str, Any]):
        """Process bar data"""
        # Can use for longer-term correlation confirmation
        pass
    
    def _update_pair_statistics(self):
        """Update correlation and spread statistics for all pairs"""
        for pair in self.trading_pairs:
            sym1, sym2 = pair
            
            # Need enough data
            if (len(self.return_series[sym1]) < self.spread_window or
                len(self.return_series[sym2]) < self.spread_window):
                continue
                
            # Calculate correlation
            returns1 = np.array(list(self.return_series[sym1]))
            returns2 = np.array(list(self.return_series[sym2]))
            
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Calculate optimal hedge ratio
            hedge_ratio = self._calculate_hedge_ratio(sym1, sym2)
            
            # Calculate spread
            prices1 = np.array(list(self.price_series[sym1]))
            prices2 = np.array(list(self.price_series[sym2]))
            
            spread = prices1 - hedge_ratio * prices2
            
            # Initialize spread tracking if needed
            if pair not in self.spreads:
                self.spreads[pair] = deque(maxlen=self.spread_window)
                
            self.spreads[pair].append(spread[-1])
            
            # Update spread statistics
            if len(self.spreads[pair]) >= 20:
                self.spread_means[pair] = np.mean(list(self.spreads[pair]))
                self.spread_stds[pair] = np.std(list(self.spreads[pair]))
            
            # Store pair statistics
            self.pair_stats[pair] = {
                'correlation': correlation,
                'hedge_ratio': hedge_ratio,
                'spread_mean': self.spread_means.get(pair, 0),
                'spread_std': self.spread_stds.get(pair, 1),
                'last_update': datetime.now()
            }
    
    def _calculate_hedge_ratio(self, sym1: str, sym2: str) -> float:
        """
        Calculate optimal hedge ratio using OLS regression
        """
        if (len(self.price_series[sym1]) < 20 or 
            len(self.price_series[sym2]) < 20):
            return 1.0
            
        prices1 = np.array(list(self.price_series[sym1]))
        prices2 = np.array(list(self.price_series[sym2]))
        
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(prices2, prices1)
        
        return slope
    
    def _test_cointegration(self, sym1: str, sym2: str) -> bool:
        """
        Test for cointegration between two series
        Simplified version - in production would use Johansen or ADF tests
        """
        if not self.use_cointegration:
            return True
            
        # Need sufficient data
        if (len(self.price_series[sym1]) < self.correlation_window or
            len(self.price_series[sym2]) < self.correlation_window):
            return False
            
        # For now, use high correlation as proxy
        # In production, implement proper cointegration test
        if (sym1, sym2) in self.pair_stats:
            return abs(self.pair_stats[(sym1, sym2)]['correlation']) > self.min_correlation
            
        return False
    
    def _discover_new_pairs(self):
        """Discover new trading pairs based on correlation"""
        # Only run periodically to avoid excessive computation
        correlations = []
        
        for i, sym1 in enumerate(self.symbols):
            for j, sym2 in enumerate(self.symbols[i+1:], i+1):
                if (len(self.return_series[sym1]) >= self.correlation_window and
                    len(self.return_series[sym2]) >= self.correlation_window):
                    
                    returns1 = np.array(list(self.return_series[sym1]))
                    returns2 = np.array(list(self.return_series[sym2]))
                    
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    
                    if abs(corr) >= self.min_correlation:
                        correlations.append(((sym1, sym2), abs(corr)))
        
        # Sort by correlation and take top pairs
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Add top correlated pairs
        for pair, corr in correlations[:5]:  # Top 5 pairs
            if pair not in self.trading_pairs:
                self.trading_pairs.append(pair)
                self.logger.info(f"Discovered arbitrage pair: {pair[0]}-{pair[1]}, correlation: {corr:.3f}")
    
    def calculate_spread_zscore(self, pair: Tuple[str, str]) -> float:
        """Calculate current z-score of spread"""
        if pair not in self.spreads or len(self.spreads[pair]) < 20:
            return 0.0
            
        current_spread = self.spreads[pair][-1]
        mean_spread = self.spread_means.get(pair, 0)
        std_spread = self.spread_stds.get(pair, 1)
        
        if std_spread > 0:
            return (current_spread - mean_spread) / std_spread
        return 0.0
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate arbitrage trading signals"""
        signals = []
        
        # Check all pairs involving this symbol
        for pair in self.trading_pairs:
            if symbol not in pair:
                continue
                
            sym1, sym2 = pair
            
            # Need sufficient data
            if pair not in self.pair_stats or pair not in self.spreads:
                continue
                
            # Get current z-score
            z_score = self.calculate_spread_zscore(pair)
            
            # Get current prices
            if (not self.price_series[sym1] or not self.price_series[sym2]):
                continue
                
            price1 = self.price_series[sym1][-1]
            price2 = self.price_series[sym2][-1]
            hedge_ratio = self.pair_stats[pair]['hedge_ratio']
            
            # Check if we have a position in this pair
            has_position = pair in self.arbitrage_positions
            
            if not has_position:
                # Entry signals
                if abs(z_score) > self.entry_z_score:
                    # Determine direction
                    if z_score > self.entry_z_score:
                        # Spread too high - sell sym1, buy sym2
                        if symbol == sym1:
                            signal_type = SignalType.SELL
                            quantity = self.calculate_arbitrage_position_size(pair, sym1)
                        else:
                            signal_type = SignalType.BUY
                            quantity = int(self.calculate_arbitrage_position_size(pair, sym1) * hedge_ratio)
                    else:
                        # Spread too low - buy sym1, sell sym2
                        if symbol == sym1:
                            signal_type = SignalType.BUY
                            quantity = self.calculate_arbitrage_position_size(pair, sym1)
                        else:
                            signal_type = SignalType.SELL
                            quantity = int(self.calculate_arbitrage_position_size(pair, sym1) * hedge_ratio)
                    
                    # Create signal
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        timestamp=datetime.now(),
                        price=price1 if symbol == sym1 else price2,
                        quantity=quantity,
                        confidence=min(0.9, abs(z_score) / 3.0),
                        reason=f"Arbitrage: {sym1}-{sym2}, z-score: {z_score:.2f}",
                        metadata={
                            'pair': pair,
                            'z_score': z_score,
                            'hedge_ratio': hedge_ratio,
                            'correlation': self.pair_stats[pair]['correlation']
                        }
                    )
                    
                    # Track position
                    if signal_type in [SignalType.BUY, SignalType.SELL]:
                        self.arbitrage_positions[pair] = {
                            'entry_z_score': z_score,
                            'entry_time': datetime.now(),
                            'sym1_position': 'long' if z_score < 0 else 'short',
                            'sym2_position': 'short' if z_score < 0 else 'long'
                        }
                    
                    signals.append(signal)
                    
            else:
                # Exit signals
                position = self.arbitrage_positions[pair]
                
                # Mean reversion - z-score returned to normal
                if abs(z_score) <= self.exit_z_score:
                    # Close position
                    if symbol == sym1:
                        signal_type = SignalType.CLOSE
                    else:
                        signal_type = SignalType.CLOSE
                        
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        timestamp=datetime.now(),
                        price=price1 if symbol == sym1 else price2,
                        quantity=None,  # Close full position
                        confidence=0.8,
                        reason=f"Arbitrage exit: z-score normalized to {z_score:.2f}",
                        metadata={
                            'pair': pair,
                            'z_score': z_score,
                            'entry_z_score': position['entry_z_score']
                        }
                    )
                    
                    # Remove position tracking
                    if signal_type == SignalType.CLOSE:
                        del self.arbitrage_positions[pair]
                        
                    signals.append(signal)
                
                # Stop loss - z-score moved against us
                elif (position['entry_z_score'] > 0 and z_score > position['entry_z_score'] + 1.0) or \
                     (position['entry_z_score'] < 0 and z_score < position['entry_z_score'] - 1.0):
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE,
                        timestamp=datetime.now(),
                        price=price1 if symbol == sym1 else price2,
                        quantity=None,
                        confidence=0.6,
                        reason=f"Arbitrage stop: z-score diverged to {z_score:.2f}",
                        metadata={
                            'pair': pair,
                            'z_score': z_score
                        }
                    )
                    
                    del self.arbitrage_positions[pair]
                    signals.append(signal)
        
        # Return first signal (if multiple, they'll be processed in sequence)
        return signals[0] if signals else None
    
    def calculate_arbitrage_position_size(self, pair: Tuple[str, str], base_symbol: str) -> int:
        """Calculate position size for arbitrage trade"""
        # Risk allocation per pair
        pair_allocation = self.position_size / max(1, len(self.trading_pairs))
        
        # Adjust for correlation strength
        correlation = abs(self.pair_stats[pair]['correlation'])
        correlation_adjustment = correlation  # Higher correlation = larger position
        
        # Calculate shares
        base_price = self.price_series[base_symbol][-1]
        position_value = pair_allocation * correlation_adjustment
        shares = int(position_value / base_price)
        
        return max(1, shares)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get arbitrage-specific metrics"""
        base_metrics = self.get_performance_summary()
        
        # Add arbitrage-specific metrics
        arbitrage_metrics = {
            'active_pairs': len(self.trading_pairs),
            'active_positions': len(self.arbitrage_positions),
            'pair_statistics': {}
        }
        
        for pair, stats in self.pair_stats.items():
            arbitrage_metrics['pair_statistics'][f"{pair[0]}-{pair[1]}"] = {
                'correlation': stats['correlation'],
                'hedge_ratio': stats['hedge_ratio'],
                'current_z_score': self.calculate_spread_zscore(pair),
                'has_position': pair in self.arbitrage_positions
            }
        
        base_metrics['arbitrage_metrics'] = arbitrage_metrics
        
        return base_metrics