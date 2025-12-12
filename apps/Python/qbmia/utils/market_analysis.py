"""
Market data processing utilities for QBMIA.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from scipy import stats, signal
from collections import deque
import logging

logger = logging.getLogger(__name__)

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_returns_numba(prices: np.ndarray, log_returns: bool = True) -> np.ndarray:
    """
    Calculate returns from price series using Numba.

    Args:
        prices: Price series
        log_returns: Whether to calculate log returns

    Returns:
        Returns array
    """
    n = len(prices)
    if n < 2:
        return np.empty(0, dtype=np.float64)

    returns = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        if log_returns:
            if prices[i-1] > 0 and prices[i] > 0:
                returns[i-1] = np.log(prices[i] / prices[i-1])
            else:
                returns[i-1] = 0.0
        else:
            if prices[i-1] != 0:
                returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                returns[i-1] = 0.0

    return returns

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_volatility_numba(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility using Numba.

    Args:
        returns: Returns series
        window: Rolling window size

    Returns:
        Volatility series
    """
    n = len(returns)
    volatility = np.empty(n, dtype=np.float64)

    for i in range(n):
        if i < window - 1:
            # Use expanding window for early values
            volatility[i] = np.std(returns[:i+1])
        else:
            # Use rolling window
            volatility[i] = np.std(returns[i-window+1:i+1])

    return volatility * np.sqrt(252)  # Annualized

@nb.jit(nopython=True, fastmath=True, cache=True)
def detect_price_levels_numba(prices: np.ndarray, window: int = 50,
                            min_touches: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect support and resistance levels using Numba.

    Args:
        prices: Price series
        window: Window for level detection
        min_touches: Minimum touches to confirm level

    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    n = len(prices)
    support_levels = []
    resistance_levels = []

    for i in range(window, n - window):
        # Check for local minimum (support)
        is_min = True
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j != i and prices[j] < prices[i]:
                is_min = False
                break

        if is_min:
            # Count touches
            touches = 0
            level = prices[i]
            tolerance = level * 0.02  # 2% tolerance

            for k in range(n):
                if abs(prices[k] - level) <= tolerance:
                    touches += 1

            if touches >= min_touches:
                support_levels.append(level)

        # Check for local maximum (resistance)
        is_max = True
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j != i and prices[j] > prices[i]:
                is_max = False
                break

        if is_max:
            # Count touches
            touches = 0
            level = prices[i]
            tolerance = level * 0.02  # 2% tolerance

            for k in range(n):
                if abs(prices[k] - level) <= tolerance:
                    touches += 1

            if touches >= min_touches:
                resistance_levels.append(level)

    return np.array(support_levels), np.array(resistance_levels)

class MarketAnalyzer:
    """
    Comprehensive market analysis utilities for QBMIA.
    """

    def __init__(self):
        """Initialize market analyzer."""
        self.indicators_cache = {}
        self.microstructure_cache = {}

    def analyze_price_action(self, prices: np.ndarray, volumes: np.ndarray,
                           timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive price action analysis.

        Args:
            prices: Price series
            volumes: Volume series
            timestamps: Timestamp series

        Returns:
            Price action analysis
        """
        # Calculate returns
        returns = calculate_returns_numba(prices)

        # Volatility analysis
        volatility = calculate_volatility_numba(returns)
        current_vol = volatility[-1] if len(volatility) > 0 else 0.0

        # Trend analysis
        trend_analysis = self._analyze_trend(prices)

        # Support/Resistance levels
        support, resistance = detect_price_levels_numba(prices)

        # Volume analysis
        volume_analysis = self._analyze_volume(volumes, prices)

        # Momentum indicators
        momentum = self._calculate_momentum_indicators(prices, volumes)

        # Market regime
        regime = self._identify_market_regime(returns, volatility)

        return {
            'current_price': prices[-1] if len(prices) > 0 else 0,
            'returns': {
                'last': returns[-1] if len(returns) > 0 else 0,
                'mean': np.mean(returns) if len(returns) > 0 else 0,
                'std': np.std(returns) if len(returns) > 0 else 0
            },
            'volatility': {
                'current': current_vol,
                'mean': np.mean(volatility) if len(volatility) > 0 else 0,
                'trend': 'increasing' if len(volatility) > 20 and volatility[-1] > np.mean(volatility[-20:]) else 'decreasing'
            },
            'trend': trend_analysis,
            'support_resistance': {
                'support_levels': support.tolist(),
                'resistance_levels': resistance.tolist(),
                'nearest_support': self._find_nearest_level(prices[-1], support, 'below') if len(support) > 0 else None,
                'nearest_resistance': self._find_nearest_level(prices[-1], resistance, 'above') if len(resistance) > 0 else None
            },
            'volume': volume_analysis,
            'momentum': momentum,
            'regime': regime
        }

    def _analyze_trend(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze price trend."""
        if len(prices) < 20:
            return {'direction': 'neutral', 'strength': 0.0}

        # Simple moving averages
        sma_short = np.mean(prices[-20:])
        sma_long = np.mean(prices[-50:]) if len(prices) >= 50 else sma_short

        # Trend direction
        current_price = prices[-1]
        if current_price > sma_short > sma_long:
            direction = 'bullish'
        elif current_price < sma_short < sma_long:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Trend strength (based on angle)
        if len(prices) >= 20:
            x = np.arange(20)
            y = prices[-20:]
            slope, _, r_value, _, _ = stats.linregress(x, y)
            strength = abs(r_value)  # R-squared as strength measure
        else:
            strength = 0.0

        return {
            'direction': direction,
            'strength': strength,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'slope': slope if 'slope' in locals() else 0.0
        }

    def _analyze_volume(self, volumes: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if len(volumes) < 20 or len(prices) < 20:
            return {'profile': 'low_data', 'vwap': 0.0}

        # Volume profile
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # VWAP calculation
        vwap = np.sum(prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:])

        # Volume trend
        volume_trend = 'increasing' if current_volume > avg_volume else 'decreasing'

        # Price-volume divergence
        price_change = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] != 0 else 0
        volume_change = (volumes[-1] - volumes[-20]) / volumes[-20] if volumes[-20] != 0 else 0

        divergence = 'bullish' if price_change < 0 and volume_change > 0 else \
                    'bearish' if price_change > 0 and volume_change < 0 else 'neutral'

        return {
            'current': current_volume,
            'average': avg_volume,
            'ratio': volume_ratio,
            'trend': volume_trend,
            'vwap': vwap,
            'divergence': divergence,
            'profile': 'high' if volume_ratio > 1.5 else 'normal' if volume_ratio > 0.5 else 'low'
        }

    def _calculate_momentum_indicators(self, prices: np.ndarray,
                                     volumes: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators."""
        if len(prices) < 20:
            return {'rsi': 50.0, 'macd': 0.0, 'momentum': 0.0}

        # RSI calculation
        returns = calculate_returns_numba(prices, log_returns=False)
        rsi = self._calculate_rsi(returns)

        # MACD calculation
        macd_line, signal_line, histogram = self._calculate_macd(prices)

        # Rate of change
        roc = ((prices[-1] - prices[-20]) / prices[-20] * 100) if prices[-20] != 0 else 0

        return {
            'rsi': rsi,
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'rate_of_change': roc,
            'momentum': roc / 20  # Normalized momentum
        }

    def _calculate_rsi(self, returns: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(returns) < period:
            return 50.0

        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray,
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        # Exponential moving averages
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        if len(prices) >= slow + signal:
            macd_values = []
            for i in range(slow, len(prices) + 1):
                fast_ema = self._calculate_ema(prices[:i], fast)
                slow_ema = self._calculate_ema(prices[:i], slow)
                macd_values.append(fast_ema - slow_ema)

            signal_line = self._calculate_ema(np.array(macd_values), signal)
        else:
            signal_line = macd_line

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)

        alpha = 2 / (period + 1)
        ema = data[0]

        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema

        return ema

    def _identify_market_regime(self, returns: np.ndarray,
                              volatility: np.ndarray) -> Dict[str, Any]:
        """Identify current market regime."""
        if len(returns) < 100:
            return {'type': 'insufficient_data', 'confidence': 0.0}

        # Statistical properties
        mean_return = np.mean(returns)
        current_vol = volatility[-1] if len(volatility) > 0 else 0
        vol_percentile = stats.percentileofscore(volatility, current_vol)

        # Regime classification
        if vol_percentile > 80:
            if mean_return < -0.001:
                regime_type = 'crisis'
            else:
                regime_type = 'volatile_bull'
        elif vol_percentile < 20:
            regime_type = 'low_volatility'
        else:
            if abs(mean_return) < 0.0001:
                regime_type = 'ranging'
            elif mean_return > 0:
                regime_type = 'trending_bull'
            else:
                regime_type = 'trending_bear'

        # Regime stability (how long in current regime)
        regime_changes = self._detect_regime_changes(returns, volatility)
        if regime_changes:
            bars_in_regime = len(returns) - regime_changes[-1]
            stability = min(1.0, bars_in_regime / 100)
        else:
            stability = 1.0

        return {
            'type': regime_type,
            'volatility_percentile': vol_percentile,
            'stability': stability,
            'confidence': stability * 0.8
        }

    def _detect_regime_changes(self, returns: np.ndarray,
                             volatility: np.ndarray) -> List[int]:
        """Detect points where market regime changed."""
        if len(returns) < 50:
            return []

        # Use rolling statistics to detect changes
        window = 20
        changes = []

        for i in range(window * 2, len(returns)):
            # Compare statistics of two adjacent windows
            window1_returns = returns[i-window*2:i-window]
            window2_returns = returns[i-window:i]

            window1_vol = np.std(window1_returns)
            window2_vol = np.std(window2_returns)

            # Significant change in volatility
            vol_change = abs(window2_vol - window1_vol) / (window1_vol + 1e-8)

            if vol_change > 0.5:  # 50% change in volatility
                changes.append(i)

        return changes

    def _find_nearest_level(self, current_price: float, levels: np.ndarray,
                          direction: str) -> Optional[float]:
        """Find nearest support or resistance level."""
        if len(levels) == 0:
            return None

        if direction == 'below':
            below_levels = levels[levels < current_price]
            if len(below_levels) > 0:
                return float(np.max(below_levels))
        else:  # above
            above_levels = levels[levels > current_price]
            if len(above_levels) > 0:
                return float(np.min(above_levels))

        return None

    def analyze_microstructure(self, order_book: Dict[str, Any],
                             trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market microstructure.

        Args:
            order_book: Current order book snapshot
            trades: Recent trades

        Returns:
            Microstructure analysis
        """
        # Order book imbalance
        bid_volume = sum(order['size'] for order in order_book.get('bids', []))
        ask_volume = sum(order['size'] for order in order_book.get('asks', []))

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

        # Spread analysis
        if order_book.get('bids') and order_book.get('asks'):
            best_bid = order_book['bids'][0]['price']
            best_ask = order_book['asks'][0]['price']
            spread = best_ask - best_bid
            spread_bps = (spread / best_bid) * 10000 if best_bid > 0 else 0
        else:
            spread = 0
            spread_bps = 0

        # Trade flow analysis
        buy_volume = sum(t['size'] for t in trades if t.get('side') == 'buy')
        sell_volume = sum(t['size'] for t in trades if t.get('side') == 'sell')

        trade_flow = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0

        # Price impact estimation
        if trades:
            avg_trade_size = np.mean([t['size'] for t in trades])
            price_impact = self._estimate_price_impact(avg_trade_size, bid_volume + ask_volume)
        else:
            price_impact = 0

        return {
            'order_book_imbalance': imbalance,
            'spread': spread,
            'spread_bps': spread_bps,
            'trade_flow': trade_flow,
            'buy_pressure': buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5,
            'price_impact_estimate': price_impact,
            'liquidity_score': self._calculate_liquidity_score(order_book, spread_bps)
        }

    def _estimate_price_impact(self, trade_size: float, total_liquidity: float) -> float:
        """Estimate price impact of a trade."""
        if total_liquidity == 0:
            return 0.01  # 1% default impact

        # Kyle's lambda model simplified
        impact = np.sqrt(trade_size / total_liquidity) * 0.01

        return min(0.05, impact)  # Cap at 5%

    def _calculate_liquidity_score(self, order_book: Dict[str, Any],
                                 spread_bps: float) -> float:
        """Calculate overall liquidity score."""
        # Factors: tight spread, deep book, balanced book

        # Spread score (tighter is better)
        spread_score = max(0, 1 - spread_bps / 100)  # 100 bps = 0 score

        # Depth score
        total_bid_depth = sum(order['size'] for order in order_book.get('bids', [])[:10])
        total_ask_depth = sum(order['size'] for order in order_book.get('asks', [])[:10])

        depth_score = min(1.0, (total_bid_depth + total_ask_depth) / 10000)  # Normalize by 10k

        # Balance score
        if total_bid_depth + total_ask_depth > 0:
            balance = min(total_bid_depth, total_ask_depth) / max(total_bid_depth, total_ask_depth)
        else:
            balance = 0

        # Combined score
        liquidity_score = spread_score * 0.4 + depth_score * 0.4 + balance * 0.2

        return liquidity_score

    def detect_anomalies(self, prices: np.ndarray, volumes: np.ndarray,
                        sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in price and volume data.

        Args:
            prices: Price series
            volumes: Volume series
            sensitivity: Anomaly detection sensitivity (std deviations)

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Price anomalies
        returns = calculate_returns_numba(prices)
        if len(returns) > 20:
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            for i in range(20, len(returns)):
                if abs(returns[i] - mean_return) > sensitivity * std_return:
                    anomalies.append({
                        'index': i,
                        'type': 'price_spike',
                        'magnitude': abs(returns[i] - mean_return) / std_return,
                        'direction': 'up' if returns[i] > mean_return else 'down',
                        'value': returns[i]
                    })

        # Volume anomalies
        if len(volumes) > 20:
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)

            for i in range(20, len(volumes)):
                if abs(volumes[i] - mean_volume) > sensitivity * std_volume:
                    anomalies.append({
                        'index': i,
                        'type': 'volume_spike',
                        'magnitude': abs(volumes[i] - mean_volume) / std_volume,
                        'value': volumes[i]
                    })

        # Price-volume anomalies (divergence)
        if len(prices) == len(volumes) and len(prices) > 20:
            for i in range(20, len(prices)):
                price_change = abs(returns[i-1]) if i > 0 else 0
                volume_ratio = volumes[i] / mean_volume if mean_volume > 0 else 1

                # High price change with low volume
                if price_change > 0.02 and volume_ratio < 0.5:
                    anomalies.append({
                        'index': i,
                        'type': 'price_volume_divergence',
                        'subtype': 'high_price_low_volume',
                        'price_change': price_change,
                        'volume_ratio': volume_ratio
                    })

        return anomalies
