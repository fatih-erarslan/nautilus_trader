#!/usr/bin/env python3
"""
Advanced Stop Loss Manager for Neural Trading
Implements multiple stop loss strategies with real-time adaptation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopLossConfig:
    """Configuration for stop loss strategies"""
    strategy: str = 'adaptive'
    base_stop_percent: float = 0.06
    trailing_enabled: bool = True
    trailing_activation: float = 0.03
    trailing_distance: float = 0.05
    use_atr: bool = True
    atr_multiplier: float = 2.0
    regime_adaptive: bool = True
    multi_layer_enabled: bool = False
    time_decay_enabled: bool = True
    max_hold_days: int = 30


class StopLossManager:
    """
    Comprehensive stop loss management system
    Combines multiple strategies for optimal protection
    """

    def __init__(self, config: Optional[StopLossConfig] = None):
        self.config = config or StopLossConfig()
        self.positions = {}
        self.market_regime = 'NEUTRAL'
        self.vix_level = 16.0

    def calculate_stop(self,
                      symbol: str,
                      entry_price: float,
                      current_price: float,
                      position_data: Dict) -> Dict:
        """
        Calculate optimal stop loss for a position
        Returns dictionary with stop price and strategy used
        """

        # Select best strategy based on conditions
        strategy = self._select_strategy(symbol, position_data)

        # Calculate stop based on selected strategy
        if strategy == 'fixed_percentage':
            stop_price = self._fixed_percentage_stop(entry_price)
        elif strategy == 'trailing':
            stop_price = self._trailing_stop(entry_price, current_price, position_data)
        elif strategy == 'atr_based':
            stop_price = self._atr_stop(entry_price, position_data.get('atr', 2.0))
        elif strategy == 'support_level':
            stop_price = self._support_stop(entry_price, position_data.get('support', entry_price * 0.94))
        elif strategy == 'regime_adaptive':
            stop_price = self._regime_stop(entry_price)
        elif strategy == 'multi_layer':
            stop_price = self._multi_layer_stop(entry_price, current_price)
        elif strategy == 'time_decay':
            stop_price = self._time_decay_stop(entry_price, position_data.get('days_held', 0))
        else:
            stop_price = self._adaptive_stop(entry_price, current_price, position_data)

        return {
            'stop_price': round(stop_price, 2),
            'strategy': strategy,
            'distance_percent': round((entry_price - stop_price) / entry_price * 100, 2),
            'distance_dollars': round(entry_price - stop_price, 2),
            'risk_reward': self._calculate_risk_reward(entry_price, stop_price, position_data.get('target', entry_price * 1.25))
        }

    def _select_strategy(self, symbol: str, position_data: Dict) -> str:
        """Select optimal strategy based on position characteristics"""

        volatility = position_data.get('volatility', 0.02)
        trend = position_data.get('trend_strength', 20)
        days_held = position_data.get('days_held', 0)
        position_size = position_data.get('position_size', 10000)
        profit_percent = position_data.get('profit_percent', 0)

        # Multi-layer for large positions
        if position_size > 20000 and self.config.multi_layer_enabled:
            return 'multi_layer'

        # Trailing for profitable trending positions
        if profit_percent > self.config.trailing_activation and trend > 30:
            return 'trailing'

        # Time decay for stale positions
        if days_held > 10 and profit_percent < 0.02 and self.config.time_decay_enabled:
            return 'time_decay'

        # ATR for high volatility
        if volatility > 0.025 and self.config.use_atr:
            return 'atr_based'

        # Regime adaptive if enabled
        if self.config.regime_adaptive:
            return 'regime_adaptive'

        # Support level for ranging markets
        if trend < 20:
            return 'support_level'

        # Default to adaptive
        return 'adaptive'

    def _fixed_percentage_stop(self, entry_price: float) -> float:
        """Simple percentage-based stop"""
        return entry_price * (1 - self.config.base_stop_percent)

    def _trailing_stop(self, entry_price: float, current_price: float, position_data: Dict) -> float:
        """Trailing stop that follows price upward"""
        highest = position_data.get('highest_price', current_price)

        if current_price > highest:
            highest = current_price

        trailing_stop = highest * (1 - self.config.trailing_distance)
        initial_stop = entry_price * (1 - self.config.base_stop_percent)

        # Never trail below initial stop
        return max(trailing_stop, initial_stop)

    def _atr_stop(self, entry_price: float, atr: float) -> float:
        """ATR-based stop for volatility adjustment"""
        stop_distance = atr * self.config.atr_multiplier
        return entry_price - stop_distance

    def _support_stop(self, entry_price: float, support_level: float) -> float:
        """Stop based on technical support levels"""
        buffer = 0.01  # 1% below support
        support_stop = support_level * (1 - buffer)
        max_stop = entry_price * (1 - 0.10)  # Never more than 10% stop

        return max(support_stop, max_stop)

    def _regime_stop(self, entry_price: float) -> float:
        """Adjust stop based on market regime"""
        regime_multipliers = {
            'BULLISH': 1.2,    # Wider stop
            'BEARISH': 0.67,   # Tighter stop
            'NEUTRAL': 1.0,    # Normal stop
            'VOLATILE': 1.5    # Very wide stop
        }

        # VIX adjustment
        vix_adj = 1.0
        if self.vix_level > 20:
            vix_adj = 1.3
        elif self.vix_level < 15:
            vix_adj = 0.9

        multiplier = regime_multipliers.get(self.market_regime, 1.0)
        stop_percent = self.config.base_stop_percent * multiplier * vix_adj

        return entry_price * (1 - stop_percent)

    def _multi_layer_stop(self, entry_price: float, current_price: float) -> Tuple[float, Dict]:
        """Multiple stop levels with position scaling"""
        layers = {
            'tight': {'percent': 0.03, 'exit_size': 0.33},
            'medium': {'percent': 0.06, 'exit_size': 0.33},
            'wide': {'percent': 0.09, 'exit_size': 0.34}
        }

        # Find current applicable stop
        for name, layer in layers.items():
            stop_price = entry_price * (1 - layer['percent'])
            if current_price > stop_price:
                return stop_price

        return entry_price * 0.91  # Maximum stop

    def _time_decay_stop(self, entry_price: float, days_held: int) -> float:
        """Tighten stop over time for non-performing positions"""
        decay_rate = 0.005  # 0.5% per day

        # Start with base stop
        initial_stop = self.config.base_stop_percent

        # Reduce stop distance over time
        current_stop = initial_stop - (days_held * decay_rate)

        # Minimum 2% stop
        final_stop = max(current_stop, 0.02)

        return entry_price * (1 - final_stop)

    def _adaptive_stop(self, entry_price: float, current_price: float, position_data: Dict) -> float:
        """
        Adaptive stop that combines multiple factors
        This is the most sophisticated strategy
        """

        # Base calculation
        base_stop = entry_price * (1 - self.config.base_stop_percent)

        # Adjustments
        adjustments = []

        # Volatility adjustment
        if 'volatility' in position_data:
            vol_adj = 1 + (position_data['volatility'] - 0.02) * 10
            adjustments.append(vol_adj)

        # Trend adjustment
        if 'trend_strength' in position_data:
            trend_adj = 1 + (position_data['trend_strength'] - 20) / 100
            adjustments.append(trend_adj)

        # Time adjustment
        if 'days_held' in position_data:
            time_adj = 1 - (position_data['days_held'] / 100)
            adjustments.append(max(time_adj, 0.5))

        # Calculate final adjustment
        if adjustments:
            total_adj = np.mean(adjustments)
            stop_distance = (entry_price - base_stop) * total_adj
            return entry_price - stop_distance

        return base_stop

    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk/reward ratio"""
        risk = entry - stop
        reward = target - entry

        if risk > 0:
            return round(reward / risk, 2)
        return 0

    def update_market_conditions(self, regime: str, vix: float):
        """Update market conditions for adaptive stops"""
        self.market_regime = regime
        self.vix_level = vix
        logger.info(f"Market conditions updated: {regime}, VIX: {vix}")

    def get_portfolio_stops(self, positions: Dict) -> Dict:
        """Calculate stops for entire portfolio"""
        stops = {}

        for symbol, data in positions.items():
            stop_info = self.calculate_stop(
                symbol,
                data['entry_price'],
                data['current_price'],
                data
            )
            stops[symbol] = stop_info

        return stops

    def check_stops_hit(self, positions: Dict, current_prices: Dict) -> list:
        """Check if any stops have been hit"""
        triggered = []

        for symbol, position in positions.items():
            if symbol in current_prices:
                stop_info = self.calculate_stop(
                    symbol,
                    position['entry_price'],
                    current_prices[symbol],
                    position
                )

                if current_prices[symbol] <= stop_info['stop_price']:
                    triggered.append({
                        'symbol': symbol,
                        'stop_price': stop_info['stop_price'],
                        'current_price': current_prices[symbol],
                        'action': 'SELL',
                        'reason': f"Stop hit using {stop_info['strategy']} strategy"
                    })

        return triggered


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = StopLossManager()

    # Update market conditions
    manager.update_market_conditions('BULLISH', 16.1)

    # Example position
    position_data = {
        'volatility': 0.025,
        'trend_strength': 35,
        'days_held': 5,
        'position_size': 15000,
        'profit_percent': 0.04,
        'atr': 2.5,
        'support': 165,
        'highest_price': 172,
        'target': 185
    }

    # Calculate stop
    stop_info = manager.calculate_stop(
        symbol='NVDA',
        entry_price=170,
        current_price=172,
        position_data=position_data
    )

    print(f"Stop Loss Analysis for NVDA:")
    print(f"  Strategy: {stop_info['strategy']}")
    print(f"  Stop Price: ${stop_info['stop_price']}")
    print(f"  Risk: {stop_info['distance_percent']}% (${stop_info['distance_dollars']})")
    print(f"  Risk/Reward: {stop_info['risk_reward']}:1")