"""Support and resistance level detection."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SupportResistanceDetector:
    """Detects support and resistance levels in price data."""
    
    def __init__(self, tolerance: float = 0.02):
        """Initialize the detector.
        
        Args:
            tolerance: Price tolerance for grouping levels (2% default)
        """
        self.tolerance = tolerance
        
    def detect_levels(self, prices: List[float]) -> Dict[str, any]:
        """Detect support and resistance levels.
        
        Args:
            prices: List of prices
            
        Returns:
            Dictionary with support/resistance levels and strengths
        """
        if len(prices) < 5:
            return {"support": [], "resistance": [], "strength": {}}
        
        # Find local minima and maxima
        local_mins = []
        local_maxs = []
        
        for i in range(1, len(prices) - 1):
            # Local minimum
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                local_mins.append(prices[i])
            # Local maximum
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                local_maxs.append(prices[i])
        
        # Group similar levels
        support_levels = self._group_levels(local_mins)
        resistance_levels = self._group_levels(local_maxs)
        
        # Calculate strength of levels
        strengths = {}
        
        for level, count in support_levels.items():
            strengths[level] = min(1.0, count / 3)  # Max strength at 3+ touches
            
        for level, count in resistance_levels.items():
            strengths[level] = min(1.0, count / 3)
        
        return {
            "support": list(support_levels.keys()),
            "resistance": list(resistance_levels.keys()),
            "strength": strengths,
        }
    
    def _group_levels(self, levels: List[float]) -> Dict[float, int]:
        """Group similar price levels together.
        
        Args:
            levels: List of price levels
            
        Returns:
            Dictionary of grouped levels with counts
        """
        if not levels:
            return {}
        
        grouped = defaultdict(int)
        
        for level in levels:
            # Find if this level is close to an existing group
            matched = False
            for group_level in list(grouped.keys()):
                if abs(level - group_level) / group_level < self.tolerance:
                    grouped[group_level] += 1
                    matched = True
                    break
            
            if not matched:
                grouped[level] = 1
        
        return dict(grouped)
    
    def find_nearest_levels(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find nearest support and resistance levels.
        
        Args:
            current_price: Current price
            support_levels: List of support levels
            resistance_levels: List of resistance levels
            
        Returns:
            Tuple of (nearest support, nearest resistance)
        """
        nearest_support = None
        nearest_resistance = None
        
        # Find nearest support below current price
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            nearest_support = max(supports_below)
        
        # Find nearest resistance above current price
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            nearest_resistance = min(resistances_above)
        
        return nearest_support, nearest_resistance
    
    def calculate_price_position(
        self,
        current_price: float,
        support: float,
        resistance: float
    ) -> float:
        """Calculate price position between support and resistance.
        
        Args:
            current_price: Current price
            support: Support level
            resistance: Resistance level
            
        Returns:
            Position as percentage (0 = at support, 1 = at resistance)
        """
        if resistance <= support:
            return 0.5
        
        position = (current_price - support) / (resistance - support)
        return max(0, min(1, position))
    
    def identify_chart_patterns(self, prices: List[float]) -> List[Dict[str, any]]:
        """Identify basic chart patterns.
        
        Args:
            prices: List of prices
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Double bottom
        double_bottom = self._check_double_bottom(prices)
        if double_bottom:
            patterns.append(double_bottom)
        
        # Double top
        double_top = self._check_double_top(prices)
        if double_top:
            patterns.append(double_top)
        
        # Triangle patterns
        triangle = self._check_triangle(prices)
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def _check_double_bottom(self, prices: List[float]) -> Optional[Dict[str, any]]:
        """Check for double bottom pattern.
        
        Args:
            prices: List of prices
            
        Returns:
            Pattern details if found
        """
        if len(prices) < 10:
            return None
        
        # Look for two similar lows with a peak in between
        min_price = min(prices[-10:])
        min_indices = [i for i, p in enumerate(prices[-10:]) if abs(p - min_price) / min_price < 0.02]
        
        if len(min_indices) >= 2:
            # Check for peak between lows
            first_low = min_indices[0]
            last_low = min_indices[-1]
            
            if last_low - first_low >= 3:  # At least 3 bars between
                peak = max(prices[-10 + first_low:-10 + last_low + 1])
                if peak > min_price * 1.03:  # At least 3% bounce
                    return {
                        "type": "double_bottom",
                        "support": min_price,
                        "neckline": peak,
                        "confidence": 0.7,
                    }
        
        return None
    
    def _check_double_top(self, prices: List[float]) -> Optional[Dict[str, any]]:
        """Check for double top pattern.
        
        Args:
            prices: List of prices
            
        Returns:
            Pattern details if found
        """
        if len(prices) < 10:
            return None
        
        # Look for two similar highs with a trough in between
        max_price = max(prices[-10:])
        max_indices = [i for i, p in enumerate(prices[-10:]) if abs(p - max_price) / max_price < 0.02]
        
        if len(max_indices) >= 2:
            # Check for trough between highs
            first_high = max_indices[0]
            last_high = max_indices[-1]
            
            if last_high - first_high >= 3:
                trough = min(prices[-10 + first_high:-10 + last_high + 1])
                if trough < max_price * 0.97:  # At least 3% pullback
                    return {
                        "type": "double_top",
                        "resistance": max_price,
                        "neckline": trough,
                        "confidence": 0.7,
                    }
        
        return None
    
    def _check_triangle(self, prices: List[float]) -> Optional[Dict[str, any]]:
        """Check for triangle patterns.
        
        Args:
            prices: List of prices
            
        Returns:
            Pattern details if found
        """
        if len(prices) < 15:
            return None
        
        # Get recent highs and lows
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append((i, prices[i]))
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Check for converging lines
            high_trend = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0])
            low_trend = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0])
            
            if high_trend < 0 and low_trend > 0:
                return {
                    "type": "symmetrical_triangle",
                    "apex": len(prices) + 5,  # Estimated apex
                    "confidence": 0.6,
                }
            elif high_trend < 0 and abs(low_trend) < abs(high_trend) * 0.3:
                return {
                    "type": "descending_triangle",
                    "support": np.mean([l[1] for l in lows]),
                    "confidence": 0.65,
                }
            elif low_trend > 0 and abs(high_trend) < abs(low_trend) * 0.3:
                return {
                    "type": "ascending_triangle",
                    "resistance": np.mean([h[1] for h in highs]),
                    "confidence": 0.65,
                }
        
        return None