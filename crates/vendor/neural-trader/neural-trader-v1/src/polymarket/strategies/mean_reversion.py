"""
Mean Reversion Trading Strategy

Bets against extreme probabilities in prediction markets based on the principle
that prices tend to revert to their mean over time.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionSignal:
    """Represents a mean reversion trading signal"""
    market_id: str
    current_price: float
    mean_price: float
    std_dev: float
    z_score: float
    reversion_probability: float
    expected_profit: float
    position_size: float
    signal_type: str  # 'overbought', 'oversold', 'neutral'
    confidence: float


class MeanReversionStrategy:
    """
    Trading strategy that bets against extreme probabilities.
    
    Key principles:
    1. Extreme probabilities (near 0 or 1) often overreact
    2. Markets tend to revert to historical means
    3. Higher volatility creates more opportunities
    4. Entry/exit based on statistical deviations
    """
    
    def __init__(self,
                 lookback_period: int = 30,
                 z_score_entry: float = 2.0,
                 z_score_exit: float = 0.5,
                 min_volatility: float = 0.05,
                 max_position_pct: float = 0.1,
                 extreme_threshold: float = 0.1):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Days to calculate mean and std dev
            z_score_entry: Z-score threshold for entry
            z_score_exit: Z-score threshold for exit
            min_volatility: Minimum volatility to trade
            max_position_pct: Maximum position as % of portfolio
            extreme_threshold: Price threshold for extreme levels (0.1 = 10% and 90%)
        """
        self.lookback_period = lookback_period
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.min_volatility = min_volatility
        self.max_position_pct = max_position_pct
        self.extreme_threshold = extreme_threshold
        self.price_history = {}
        
    def update_price_history(self, market_id: str, price: float):
        """Update price history for a market."""
        if market_id not in self.price_history:
            self.price_history[market_id] = deque(maxlen=self.lookback_period)
        self.price_history[market_id].append(price)
        
    def calculate_statistics(self, market_id: str) -> Optional[Dict]:
        """
        Calculate statistical measures for a market.
        
        Returns:
            Dict with mean, std_dev, current_z_score
        """
        if market_id not in self.price_history:
            return None
            
        prices = list(self.price_history[market_id])
        
        if len(prices) < 10:  # Need minimum history
            return None
            
        mean_price = np.mean(prices)
        std_dev = np.std(prices)
        
        if std_dev < self.min_volatility:
            return None
            
        current_price = prices[-1]
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
        
        return {
            'mean_price': mean_price,
            'std_dev': std_dev,
            'z_score': z_score,
            'current_price': current_price,
            'sample_size': len(prices)
        }
    
    def generate_signal(self, 
                       market_id: str,
                       current_price: float,
                       portfolio_value: float) -> Optional[MeanReversionSignal]:
        """
        Generate mean reversion signal for a market.
        
        Args:
            market_id: Market identifier
            current_price: Current market price
            portfolio_value: Total portfolio value
            
        Returns:
            MeanReversionSignal if actionable, None otherwise
        """
        # Update history
        self.update_price_history(market_id, current_price)
        
        # Calculate statistics
        stats = self.calculate_statistics(market_id)
        if not stats:
            return None
            
        z_score = stats['z_score']
        mean_price = stats['mean_price']
        std_dev = stats['std_dev']
        
        # Check for extreme prices
        is_extreme_high = current_price > (1 - self.extreme_threshold)
        is_extreme_low = current_price < self.extreme_threshold
        
        # Determine signal type
        if abs(z_score) < self.z_score_entry and not (is_extreme_high or is_extreme_low):
            return None  # No signal
            
        if z_score > self.z_score_entry or is_extreme_high:
            signal_type = 'overbought'
            position_direction = -1  # Sell/short
        elif z_score < -self.z_score_entry or is_extreme_low:
            signal_type = 'oversold'
            position_direction = 1  # Buy
        else:
            signal_type = 'neutral'
            position_direction = 0
            
        if position_direction == 0:
            return None
            
        # Calculate reversion probability
        reversion_prob = self._calculate_reversion_probability(
            z_score, is_extreme_high, is_extreme_low
        )
        
        # Calculate expected profit
        expected_move = (mean_price - current_price) * position_direction
        expected_profit = expected_move * reversion_prob
        
        # Calculate position size
        position_size = self._calculate_position_size(
            expected_profit,
            reversion_prob,
            portfolio_value,
            std_dev
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            z_score, std_dev, stats['sample_size'], reversion_prob
        )
        
        return MeanReversionSignal(
            market_id=market_id,
            current_price=current_price,
            mean_price=mean_price,
            std_dev=std_dev,
            z_score=z_score,
            reversion_probability=reversion_prob,
            expected_profit=expected_profit,
            position_size=position_size * position_direction,
            signal_type=signal_type,
            confidence=confidence
        )
    
    def _calculate_reversion_probability(self,
                                       z_score: float,
                                       is_extreme_high: bool,
                                       is_extreme_low: bool) -> float:
        """
        Calculate probability of mean reversion.
        
        Based on historical reversion rates and current deviation.
        """
        # Base probability from z-score
        abs_z = abs(z_score)
        if abs_z > 3:
            base_prob = 0.85
        elif abs_z > 2.5:
            base_prob = 0.75
        elif abs_z > 2:
            base_prob = 0.65
        else:
            base_prob = 0.5
            
        # Boost for extreme prices
        if is_extreme_high or is_extreme_low:
            # Extreme prices have higher reversion probability
            extreme_boost = 0.15
            base_prob = min(base_prob + extreme_boost, 0.95)
            
        return base_prob
    
    def _calculate_position_size(self,
                               expected_profit: float,
                               probability: float,
                               portfolio_value: float,
                               volatility: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion with adjustments.
        """
        # Kelly fraction
        if expected_profit <= 0:
            return 0
            
        kelly_fraction = (probability * expected_profit - (1 - probability) * 0.1) / expected_profit
        kelly_fraction = max(0, kelly_fraction)
        
        # Adjust for volatility (higher vol = smaller position)
        vol_adjustment = np.exp(-volatility * 10)  # Exponential decay
        
        # Conservative Kelly (1/4 Kelly)
        conservative_fraction = kelly_fraction * 0.25 * vol_adjustment
        
        # Apply maximum position limit
        max_position = portfolio_value * self.max_position_pct
        position_size = min(conservative_fraction * portfolio_value, max_position)
        
        return position_size
    
    def _calculate_confidence(self,
                            z_score: float,
                            volatility: float,
                            sample_size: int,
                            reversion_prob: float) -> float:
        """Calculate confidence in the signal."""
        # Z-score contribution
        z_confidence = min(abs(z_score) / 4, 1.0)  # Max at z=4
        
        # Sample size contribution
        sample_confidence = min(sample_size / self.lookback_period, 1.0)
        
        # Volatility contribution (moderate volatility is best)
        if volatility < 0.05:
            vol_confidence = 0.5
        elif volatility > 0.2:
            vol_confidence = 0.7
        else:
            vol_confidence = 1.0
            
        # Combine factors
        confidence = (z_confidence * 0.3 + 
                     sample_confidence * 0.2 + 
                     vol_confidence * 0.2 + 
                     reversion_prob * 0.3)
        
        return min(confidence, 0.95)
    
    def should_exit_position(self,
                           market_id: str,
                           current_price: float,
                           entry_price: float,
                           position_direction: int) -> Tuple[bool, str]:
        """
        Determine if position should be exited.
        
        Args:
            market_id: Market identifier
            current_price: Current market price
            entry_price: Price when position was entered
            position_direction: 1 for long, -1 for short
            
        Returns:
            Tuple of (should_exit, reason)
        """
        stats = self.calculate_statistics(market_id)
        if not stats:
            return True, "insufficient_data"
            
        z_score = stats['z_score']
        
        # Exit if z-score crosses zero (mean reversion complete)
        if position_direction > 0 and z_score > 0:
            return True, "mean_crossed"
        elif position_direction < 0 and z_score < 0:
            return True, "mean_crossed"
            
        # Exit if z-score reaches exit threshold
        if abs(z_score) <= self.z_score_exit:
            return True, "z_score_exit"
            
        # Stop loss at 2 standard deviations adverse move
        adverse_move = (current_price - entry_price) * position_direction * -1
        if adverse_move > 2 * stats['std_dev']:
            return True, "stop_loss"
            
        # Take profit at 3 standard deviations favorable move
        favorable_move = (current_price - entry_price) * position_direction
        if favorable_move > 3 * stats['std_dev']:
            return True, "take_profit"
            
        return False, "hold"
    
    def scan_markets(self,
                    markets: List[Dict],
                    portfolio_value: float) -> List[MeanReversionSignal]:
        """
        Scan multiple markets for mean reversion opportunities.
        
        Args:
            markets: List of market data with 'id' and 'price'
            portfolio_value: Total portfolio value
            
        Returns:
            List of signals sorted by expected profit
        """
        signals = []
        
        for market in markets:
            signal = self.generate_signal(
                market['id'],
                market['price'],
                portfolio_value
            )
            
            if signal and abs(signal.expected_profit) > 0.01:  # Min 1% expected profit
                signals.append(signal)
                
        # Sort by expected absolute profit
        signals.sort(key=lambda s: abs(s.expected_profit), reverse=True)
        
        return signals
    
    def calculate_extreme_probability_edge(self, price: float) -> float:
        """
        Calculate the edge from betting against extreme probabilities.
        
        Theory: Extreme probabilities are often overconfident due to:
        1. Psychological biases (certainty effect)
        2. Limited information
        3. Black swan blindness
        """
        if price < self.extreme_threshold:
            # Betting on unlikely event
            # True probability likely higher than market price
            estimated_true_prob = price + (self.extreme_threshold - price) * 0.5
            edge = estimated_true_prob - price
            
        elif price > (1 - self.extreme_threshold):
            # Betting against "certain" event
            # True probability likely lower than market price
            estimated_true_prob = price - (price - (1 - self.extreme_threshold)) * 0.5
            edge = price - estimated_true_prob
            
        else:
            edge = 0
            
        return edge
    
    def backtest(self,
                historical_data: List[Dict],
                initial_capital: float = 10000) -> Dict:
        """
        Backtest mean reversion strategy on historical data.
        
        Args:
            historical_data: List of daily market snapshots
            initial_capital: Starting capital
            
        Returns:
            Backtest results with performance metrics
        """
        capital = initial_capital
        positions = {}  # market_id -> (size, entry_price, direction)
        trades = []
        equity_curve = []
        
        for day_data in historical_data:
            timestamp = day_data['timestamp']
            markets = day_data['markets']
            
            # Check existing positions for exits
            for market_id, (size, entry_price, direction) in list(positions.items()):
                market = next((m for m in markets if m['id'] == market_id), None)
                if market:
                    should_exit, reason = self.should_exit_position(
                        market_id,
                        market['price'],
                        entry_price,
                        direction
                    )
                    
                    if should_exit:
                        # Close position
                        pnl = size * (market['price'] - entry_price) * direction
                        capital += pnl
                        
                        trades.append({
                            'timestamp': timestamp,
                            'market_id': market_id,
                            'action': 'close',
                            'price': market['price'],
                            'size': size,
                            'pnl': pnl,
                            'reason': reason
                        })
                        
                        del positions[market_id]
            
            # Scan for new opportunities
            available_capital = capital - sum(pos[0] for pos in positions.values())
            signals = self.scan_markets(markets, available_capital)
            
            # Enter new positions (limit to top 5)
            for signal in signals[:5]:
                if signal.market_id not in positions and signal.position_size > 50:
                    # Enter position
                    positions[signal.market_id] = (
                        abs(signal.position_size),
                        signal.current_price,
                        np.sign(signal.position_size)
                    )
                    
                    trades.append({
                        'timestamp': timestamp,
                        'market_id': signal.market_id,
                        'action': 'open',
                        'price': signal.current_price,
                        'size': signal.position_size,
                        'z_score': signal.z_score,
                        'signal_type': signal.signal_type
                    })
            
            # Calculate portfolio value
            portfolio_value = capital
            for market_id, (size, entry_price, direction) in positions.items():
                market = next((m for m in markets if m['id'] == market_id), None)
                if market:
                    portfolio_value += size * market['price'] * direction
                    
            equity_curve.append((timestamp, portfolio_value))
        
        # Calculate metrics
        returns = np.diff([e[1] for e in equity_curve]) / [e[1] for e in equity_curve[:-1]]
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        return {
            'total_return': (equity_curve[-1][1] - initial_capital) / initial_capital,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'total_trades': len([t for t in trades if t['action'] == 'open']),
            'win_rate': len(winning_trades) / len([t for t in trades if t['action'] == 'close']) if trades else 0,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[Tuple]) -> float:
        """Calculate maximum drawdown from equity curve."""
        values = [e[1] for e in equity_curve]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd