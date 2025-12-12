"""
Machiavellian strategic framework for market manipulation detection and strategic deception.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import deque
import asyncio
from scipy import signal, stats

logger = logging.getLogger(__name__)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _detect_spoofing_pattern_numba(order_sizes: np.ndarray, order_times: np.ndarray,
                                  cancel_mask: np.ndarray, window_size: int = 100) -> float:
    """
    Numba-accelerated spoofing detection.

    Args:
        order_sizes: Array of order sizes
        order_times: Array of order timestamps
        cancel_mask: Boolean mask of cancelled orders
        window_size: Rolling window size

    Returns:
        Spoofing probability score
    """
    n = len(order_sizes)
    if n < window_size:
        return 0.0

    spoofing_score = 0.0
    windows_analyzed = 0

    for i in range(window_size, n):
        window_start = i - window_size

        # Extract window data
        window_sizes = order_sizes[window_start:i]
        window_cancels = cancel_mask[window_start:i]

        # Calculate metrics
        large_order_threshold = np.percentile(window_sizes, 90)
        large_orders = window_sizes > large_order_threshold

        # Spoofing indicator: large orders that are quickly cancelled
        large_cancelled = large_orders & window_cancels

        if np.sum(large_orders) > 0:
            cancel_rate = np.sum(large_cancelled) / np.sum(large_orders)

            # Time-based analysis
            if cancel_rate > 0.7:  # High cancellation rate
                spoofing_score += cancel_rate
                windows_analyzed += 1

    return spoofing_score / max(1, windows_analyzed)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _detect_layering_numba(price_levels: np.ndarray, order_sizes: np.ndarray,
                         timestamps: np.ndarray, threshold: float = 0.8) -> float:
    """
    Numba-accelerated layering detection.

    Layering involves placing multiple orders at different price levels
    to create false impression of market depth.
    """
    n = len(price_levels)
    if n < 10:
        return 0.0

    layering_score = 0.0

    # Group orders by time windows
    time_window = 60.0  # 60 seconds
    current_time = timestamps[0]
    window_orders = []

    for i in range(n):
        if timestamps[i] - current_time <= time_window:
            window_orders.append(i)
        else:
            # Analyze window
            if len(window_orders) >= 5:
                # Check for multiple price levels
                window_prices = price_levels[window_orders[0]:window_orders[-1]+1]
                unique_prices = np.unique(window_prices)

                if len(unique_prices) >= 3:
                    # Check for similar sizes (indicating algorithmic placement)
                    window_sizes = order_sizes[window_orders[0]:window_orders[-1]+1]
                    size_variance = np.var(window_sizes) / (np.mean(window_sizes) ** 2 + 1e-8)

                    if size_variance < 0.1:  # Low variance suggests algorithmic
                        layering_score += 1.0

            # Start new window
            current_time = timestamps[i]
            window_orders = [i]

    return min(1.0, layering_score / 10.0)  # Normalize

class MachiavellianFramework:
    """
    Framework for detecting market manipulation and executing strategic deception.
    """

    def __init__(self, hw_optimizer: Any = None, sensitivity: float = 0.7):
        """
        Initialize Machiavellian framework.

        Args:
            hw_optimizer: Hardware optimizer for acceleration
            sensitivity: Detection sensitivity threshold
        """
        self.hw_optimizer = hw_optimizer
        self.sensitivity = sensitivity

        # Detection models
        self.manipulation_patterns = {
            'spoofing': 0.3,      # Weight
            'layering': 0.25,
            'wash_trading': 0.2,
            'pump_dump': 0.15,
            'front_running': 0.1
        }

        # Historical data buffers
        self.order_history = deque(maxlen=10000)
        self.price_history = deque(maxlen=10000)
        self.detection_history = deque(maxlen=1000)

        # Strategic deception parameters
        self.deception_strategies = {
            'noise_injection': {'enabled': True, 'intensity': 0.1},
            'pattern_masking': {'enabled': True, 'complexity': 0.5},
            'false_signals': {'enabled': False, 'frequency': 0.05}
        }

        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'detection_latency': []
        }

        logger.info("Machiavellian framework initialized")

    async def detect_manipulation(self, order_flow: List[Dict[str, Any]],
                                price_history: List[float]) -> Dict[str, Any]:
        """
        Detect market manipulation patterns.

        Args:
            order_flow: List of order events
            price_history: Recent price history

        Returns:
            Manipulation detection results
        """
        start_time = asyncio.get_event_loop().time()

        # Convert to numpy arrays for acceleration
        order_data = self._preprocess_order_flow(order_flow)
        prices = np.array(price_history, dtype=np.float64)

        # Run detection algorithms in parallel
        detection_tasks = [
            self._detect_spoofing(order_data),
            self._detect_layering(order_data),
            self._detect_wash_trading(order_data, prices),
            self._detect_pump_dump(prices),
            self._detect_front_running(order_data, prices)
        ]

        results = await asyncio.gather(*detection_tasks)

        # Aggregate results
        manipulation_scores = dict(zip(self.manipulation_patterns.keys(), results))

        # Calculate weighted overall score
        overall_score = sum(
            score * self.manipulation_patterns[pattern]
            for pattern, score in manipulation_scores.items()
        )

        # Determine if manipulation detected
        detected = overall_score > self.sensitivity

        # Update statistics
        self.detection_stats['total_detections'] += 1
        if detected:
            self.detection_history.append({
                'timestamp': asyncio.get_event_loop().time(),
                'scores': manipulation_scores.copy(),
                'overall': overall_score
            })

        execution_time = asyncio.get_event_loop().time() - start_time
        self.detection_stats['detection_latency'].append(execution_time)

        return {
            'detected': detected,
            'confidence': float(overall_score),
            'manipulation_scores': manipulation_scores,
            'primary_pattern': max(manipulation_scores, key=manipulation_scores.get),
            'execution_time': execution_time,
            'recommended_action': self._recommend_action(manipulation_scores, overall_score)
        }

    def _preprocess_order_flow(self, order_flow: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Preprocess order flow data for analysis."""
        if not order_flow:
            return {
                'sizes': np.array([]),
                'prices': np.array([]),
                'timestamps': np.array([]),
                'sides': np.array([]),
                'cancelled': np.array([])
            }

        # Extract arrays
        sizes = np.array([order.get('size', 0) for order in order_flow])
        prices = np.array([order.get('price', 0) for order in order_flow])
        timestamps = np.array([order.get('timestamp', 0) for order in order_flow])
        sides = np.array([1 if order.get('side') == 'buy' else -1 for order in order_flow])
        cancelled = np.array([order.get('cancelled', False) for order in order_flow])

        return {
            'sizes': sizes,
            'prices': prices,
            'timestamps': timestamps,
            'sides': sides,
            'cancelled': cancelled
        }

    async def _detect_spoofing(self, order_data: Dict[str, np.ndarray]) -> float:
        """Detect spoofing patterns."""
        if len(order_data['sizes']) < 100:
            return 0.0

        # Use Numba-accelerated function
        score = _detect_spoofing_pattern_numba(
            order_data['sizes'],
            order_data['timestamps'],
            order_data['cancelled']
        )

        return float(score)

    async def _detect_layering(self, order_data: Dict[str, np.ndarray]) -> float:
        """Detect layering patterns."""
        if len(order_data['prices']) < 10:
            return 0.0

        # Use Numba-accelerated function
        score = _detect_layering_numba(
            order_data['prices'],
            order_data['sizes'],
            order_data['timestamps']
        )

        return float(score)

    async def _detect_wash_trading(self, order_data: Dict[str, np.ndarray],
                                  prices: np.ndarray) -> float:
        """Detect wash trading patterns."""
        if len(prices) < 100:
            return 0.0

        # Wash trading characteristics:
        # 1. High volume with minimal price movement
        # 2. Repeated buy/sell patterns from same entity

        # Calculate price volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)

        # Calculate volume intensity
        if len(order_data['sizes']) > 0:
            volume_intensity = np.sum(order_data['sizes']) / len(order_data['sizes'])

            # Normalize by typical values
            typical_volatility = 0.02
            typical_volume = np.mean(order_data['sizes'])

            # Low volatility + high volume = potential wash trading
            if volatility < typical_volatility * 0.5 and volume_intensity > typical_volume * 2:
                score = min(1.0, (volume_intensity / typical_volume) * (typical_volatility / (volatility + 1e-8)))
                return float(score)

        return 0.0

    async def _detect_pump_dump(self, prices: np.ndarray) -> float:
        """Detect pump and dump patterns."""
        if len(prices) < 200:
            return 0.0

        # Pump and dump characteristics:
        # 1. Sharp price increase (pump)
        # 2. Followed by sharp decrease (dump)

        # Calculate rolling returns
        window = 20
        returns = np.diff(prices) / prices[:-1]

        if len(returns) < window * 2:
            return 0.0

        # Find potential pump periods
        pump_threshold = np.percentile(returns, 95)
        dump_threshold = np.percentile(returns, 5)

        score = 0.0

        for i in range(window, len(returns) - window):
            # Check for pump
            recent_returns = returns[i-window:i]
            if np.mean(recent_returns) > pump_threshold:
                # Check for subsequent dump
                future_returns = returns[i:i+window]
                if np.mean(future_returns) < dump_threshold:
                    # Pump-dump pattern detected
                    pump_magnitude = np.mean(recent_returns)
                    dump_magnitude = abs(np.mean(future_returns))
                    pattern_strength = (pump_magnitude + dump_magnitude) / 2
                    score = max(score, min(1.0, pattern_strength * 10))

        return float(score)

    async def _detect_front_running(self, order_data: Dict[str, np.ndarray],
                                  prices: np.ndarray) -> float:
        """Detect front-running patterns."""
        if len(order_data['timestamps']) < 50 or len(prices) < 50:
            return 0.0

        # Front-running characteristics:
        # Orders placed just before large market moves

        # Identify large price moves
        price_changes = np.abs(np.diff(prices))
        large_move_threshold = np.percentile(price_changes, 90)
        large_moves = np.where(price_changes > large_move_threshold)[0]

        if len(large_moves) == 0:
            return 0.0

        score = 0.0

        # Check for suspicious order patterns before large moves
        for move_idx in large_moves:
            if move_idx < 10:
                continue

            # Look at orders in the few seconds before the move
            move_time = order_data['timestamps'][min(move_idx, len(order_data['timestamps'])-1)]
            pre_move_mask = (order_data['timestamps'] < move_time) & \
                           (order_data['timestamps'] > move_time - 5.0)  # 5 second window

            if np.sum(pre_move_mask) > 0:
                # Check for unusual order characteristics
                pre_move_sizes = order_data['sizes'][pre_move_mask]
                if len(pre_move_sizes) > 0:
                    size_anomaly = np.max(pre_move_sizes) / (np.mean(order_data['sizes']) + 1e-8)
                    if size_anomaly > 3.0:  # Order 3x larger than average
                        score = max(score, min(1.0, size_anomaly / 10.0))

        return float(score)

    def _recommend_action(self, manipulation_scores: Dict[str, float],
                        overall_score: float) -> str:
        """Recommend action based on manipulation detection."""
        if overall_score < 0.3:
            return "CONTINUE_NORMAL"
        elif overall_score < 0.5:
            return "INCREASE_VIGILANCE"
        elif overall_score < 0.7:
            return "ADJUST_STRATEGY"
        else:
            # High manipulation detected
            primary_pattern = max(manipulation_scores, key=manipulation_scores.get)

            if primary_pattern == 'spoofing':
                return "IGNORE_LARGE_ORDERS"
            elif primary_pattern == 'layering':
                return "FOCUS_ON_EXECUTED_ORDERS"
            elif primary_pattern == 'wash_trading':
                return "REDUCE_POSITION_SIZE"
            elif primary_pattern == 'pump_dump':
                return "AVOID_MOMENTUM_TRADING"
            elif primary_pattern == 'front_running':
                return "RANDOMIZE_EXECUTION_TIMING"
            else:
                return "DEFENSIVE_TRADING"

    async def generate_strategy(self, manipulation_detected: Dict[str, Any],
                              competitors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategic response based on detected manipulation.

        Args:
            manipulation_detected: Results from manipulation detection
            competitors: Information about market competitors

        Returns:
            Strategic recommendations
        """
        strategy = {
            'recommended_action': 2,  # Default to hold
            'confidence': 0.5,
            'tactics': []
        }

        # Adjust strategy based on manipulation
        if manipulation_detected['detected']:
            confidence_penalty = manipulation_detected['confidence'] * 0.3
            strategy['confidence'] = max(0.1, 0.5 - confidence_penalty)

            primary_pattern = manipulation_detected.get('primary_pattern')

            if primary_pattern == 'pump_dump':
                strategy['recommended_action'] = 1  # Sell
                strategy['tactics'].append('EXIT_BEFORE_DUMP')
            elif primary_pattern == 'spoofing':
                strategy['tactics'].append('IGNORE_FAKE_ORDERS')
            elif primary_pattern == 'front_running':
                strategy['tactics'].append('DELAYED_EXECUTION')

        # Consider competitive landscape
        if competitors:
            # Simplified competitive analysis
            num_competitors = len(competitors)
            if num_competitors > 5:
                strategy['tactics'].append('INCREASE_DECEPTION')

                # Enable strategic deception
                if self.deception_strategies['noise_injection']['enabled']:
                    strategy['tactics'].append('INJECT_NOISE')

        return strategy

    def enable_deception_mode(self, mode: str, params: Dict[str, Any]):
        """
        Enable strategic deception mode.

        Args:
            mode: Deception mode ('noise_injection', 'pattern_masking', etc.)
            params: Mode-specific parameters
        """
        if mode in self.deception_strategies:
            self.deception_strategies[mode].update(params)
            logger.info(f"Enabled deception mode: {mode}")

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'sensitivity': self.sensitivity,
            'manipulation_patterns': self.manipulation_patterns.copy(),
            'deception_strategies': self.deception_strategies.copy(),
            'detection_stats': self.detection_stats.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.sensitivity = state.get('sensitivity', self.sensitivity)
        self.manipulation_patterns = state.get('manipulation_patterns', self.manipulation_patterns).copy()
        self.deception_strategies = state.get('deception_strategies', self.deception_strategies).copy()
        self.detection_stats = state.get('detection_stats', self.detection_stats).copy()
