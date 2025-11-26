"""Support and resistance level detection for stocks"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class SupportResistanceDetector:
    """Detect support and resistance levels in price data"""
    
    def __init__(self, min_touches: int = 2, price_threshold: float = 0.5):
        """
        Initialize the detector
        
        Args:
            min_touches: Minimum number of touches to confirm a level
            price_threshold: Absolute price threshold for grouping levels
        """
        self.min_touches = min_touches
        self.price_threshold = price_threshold  # Keep as absolute value
        
    def detect_levels(self, prices: List[float]) -> Dict[str, any]:
        """
        Detect support and resistance levels from price data
        
        Args:
            prices: List of prices
            
        Returns:
            Dictionary with support/resistance levels and their strength
        """
        if len(prices) < 10:
            return {"support": [], "resistance": [], "strength": {}}
            
        prices_array = np.array(prices)
        
        # Find local minima (potential support) and maxima (potential resistance)
        # Use smaller order for more sensitivity with short data
        order = min(1, len(prices) // 8)
        local_min_idx = argrelextrema(prices_array, np.less, order=max(1, order))[0]
        local_max_idx = argrelextrema(prices_array, np.greater, order=max(1, order))[0]
        
        # Extract price levels
        support_candidates = [prices_array[i] for i in local_min_idx]
        resistance_candidates = [prices_array[i] for i in local_max_idx]
        
        # Add additional candidates by finding repeated or near-repeated values
        # This helps with flat support/resistance levels
        unique_prices = list(set(prices))
        for price in unique_prices:
            count = sum(1 for p in prices if abs(p - price) < self.price_threshold)
            if count >= self.min_touches:
                # Determine if it's more likely support or resistance
                if price <= np.median(prices):
                    support_candidates.append(price)
                else:
                    resistance_candidates.append(price)
        
        # Cluster similar levels
        support_levels = self._cluster_levels(support_candidates)
        resistance_levels = self._cluster_levels(resistance_candidates)
        
        # Calculate strength for each level
        strength = {}
        for level in support_levels + resistance_levels:
            strength[level] = self._calculate_level_strength(prices, level)
            
        return {
            "support": sorted(support_levels),
            "resistance": sorted(resistance_levels),
            "strength": strength
        }
    
    def detect_levels_with_volume(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect levels using price and volume data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume-weighted support/resistance levels
        """
        # Calculate typical price
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Find turning points
        prices = typical_price.values
        volumes = data['Volume'].values
        
        # Find local extrema with more sensitivity
        order = max(1, len(prices) // 8)
        local_min_idx = argrelextrema(prices, np.less, order=order)[0]
        local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
        
        # Weight levels by volume (lower threshold for inclusion)
        support_levels = []
        resistance_levels = []
        avg_volume = np.mean(volumes)
        
        for idx in local_min_idx:
            if idx < len(volumes):
                # High volume at support is stronger
                weight = volumes[idx] / avg_volume
                if weight > 0.8:  # More lenient threshold
                    support_levels.append(prices[idx])
                    
        for idx in local_max_idx:
            if idx < len(volumes):
                weight = volumes[idx] / avg_volume
                if weight > 0.8:  # More lenient threshold
                    resistance_levels.append(prices[idx])
                    
        # Also add repeated levels with high volume like the basic method
        unique_prices = list(set(prices))
        for price in unique_prices:
            # Find indices where price occurs
            indices = [i for i, p in enumerate(prices) if abs(p - price) < self.price_threshold]
            if len(indices) >= self.min_touches:
                # Check if any have above average volume
                max_volume_weight = max(volumes[i] / avg_volume for i in indices if i < len(volumes))
                if max_volume_weight > 0.8:
                    if price <= np.median(prices):
                        support_levels.append(price)
                    else:
                        resistance_levels.append(price)
                    
        # Cluster and clean levels
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)
        
        # Calculate volume-weighted strength
        strength = {}
        for level in support_levels + resistance_levels:
            strength[level] = self._calculate_volume_strength(data, level)
            
        return {
            "support": sorted(support_levels),
            "resistance": sorted(resistance_levels),
            "strength": strength
        }
    
    def detect_trend_lines(self, prices: List[float]) -> Dict[str, any]:
        """
        Detect trend lines as dynamic support/resistance
        
        Args:
            prices: List of prices
            
        Returns:
            Dictionary with trend line parameters
        """
        if len(prices) < 5:
            return {"support_line": None, "resistance_line": None}
            
        x = np.arange(len(prices))
        prices_array = np.array(prices)
        
        # Find local minima for support line
        order = max(1, len(prices) // 10)
        local_mins = argrelextrema(prices_array, np.less, order=order)[0]
        if len(local_mins) >= 2:
            # Fit line through minima
            support_slope, support_intercept = np.polyfit(
                local_mins, prices_array[local_mins], 1
            )
        elif len(local_mins) == 1:
            # Use overall trend as fallback
            overall_slope = (prices[-1] - prices[0]) / (len(prices) - 1)
            support_slope = overall_slope
            support_intercept = prices_array[local_mins[0]] - support_slope * local_mins[0]
        else:
            # Use overall trend of lower envelope
            support_slope = (prices[-1] - prices[0]) / (len(prices) - 1)
            support_intercept = min(prices)
            
        # Find local maxima for resistance line
        local_maxs = argrelextrema(prices_array, np.greater, order=order)[0]
        if len(local_maxs) >= 2:
            # Fit line through maxima
            resistance_slope, resistance_intercept = np.polyfit(
                local_maxs, prices_array[local_maxs], 1
            )
        elif len(local_maxs) == 1:
            # Use overall trend as fallback
            overall_slope = (prices[-1] - prices[0]) / (len(prices) - 1)
            resistance_slope = overall_slope
            resistance_intercept = prices_array[local_maxs[0]] - resistance_slope * local_maxs[0]
        else:
            # Use overall trend of upper envelope
            resistance_slope = (prices[-1] - prices[0]) / (len(prices) - 1)
            resistance_intercept = max(prices)
            
        return {
            "support_line": {
                "slope": support_slope,
                "intercept": support_intercept,
                "current_level": support_slope * (len(prices) - 1) + support_intercept
            },
            "resistance_line": {
                "slope": resistance_slope,
                "intercept": resistance_intercept,
                "current_level": resistance_slope * (len(prices) - 1) + resistance_intercept
            }
        }
    
    def analyze_breakout(self, prices: List[float], timeframe: int = 10) -> Dict[str, any]:
        """
        Analyze potential breakouts from support/resistance
        
        Args:
            prices: List of prices
            timeframe: Number of periods to analyze
            
        Returns:
            Dictionary with breakout analysis
        """
        if len(prices) < timeframe + 5:
            return {"breakout_detected": False}
            
        # Get levels from recent data
        recent_prices = prices[-timeframe-5:-5]
        levels = self.detect_levels(recent_prices)
        
        current_price = prices[-1]
        recent_high = max(prices[-5:])
        recent_low = min(prices[-5:])
        
        # Check for resistance breakout
        if levels["resistance"]:
            highest_resistance = max(levels["resistance"])
            if current_price > highest_resistance * 1.01:  # 1% above resistance
                # Calculate strength as percentage of recent range
                recent_range = recent_high - recent_low
                breakout_amount = current_price - highest_resistance
                strength = breakout_amount / recent_range if recent_range > 0 else 0
                return {
                    "breakout_detected": True,
                    "breakout_type": "resistance",
                    "breakout_level": highest_resistance,
                    "breakout_strength": strength
                }
                
        # Check for support breakdown
        if levels["support"]:
            lowest_support = min(levels["support"])
            if current_price < lowest_support * 0.99:  # 1% below support
                # Calculate strength as percentage of recent range
                recent_range = recent_high - recent_low
                breakdown_amount = lowest_support - current_price
                strength = breakdown_amount / recent_range if recent_range > 0 else 0
                return {
                    "breakout_detected": True,
                    "breakout_type": "support",
                    "breakout_level": lowest_support,
                    "breakout_strength": strength
                }
                
        return {"breakout_detected": False}
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate pivot points for intraday support/resistance
        
        Args:
            high: Previous period high
            low: Previous period low  
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        # Standard pivot calculation
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = r1 + (high - low)
        
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = s1 - (high - low)
        
        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3
        }
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, any]:
        """
        Calculate volume profile to identify high volume nodes
        
        Args:
            data: DataFrame with price and volume
            bins: Number of price bins
            
        Returns:
            Dictionary with volume profile analysis
        """
        # Create price bins
        price_min = data['price'].min()
        price_max = data['price'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Calculate volume at each price level
        volume_profile = []
        
        for i in range(len(price_bins) - 1):
            bin_mask = (data['price'] >= price_bins[i]) & (data['price'] < price_bins[i + 1])
            bin_volume = data.loc[bin_mask, 'volume'].sum()
            bin_center = (price_bins[i] + price_bins[i + 1]) / 2
            
            volume_profile.append({
                'price': bin_center,
                'volume': bin_volume,
                'range': (price_bins[i], price_bins[i + 1])
            })
            
        # Sort by volume to find high volume nodes
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        
        # Identify high volume nodes (top 20%)
        high_volume_threshold = volume_profile[int(len(volume_profile) * 0.2)]['volume']
        high_volume_nodes = [vp for vp in volume_profile if vp['volume'] >= high_volume_threshold]
        
        return {
            "volume_profile": volume_profile,
            "high_volume_nodes": high_volume_nodes,
            "poc": volume_profile[0]['price']  # Point of Control (highest volume)
        }
    
    def combine_timeframe_levels(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Combine support/resistance levels from multiple timeframes
        
        Args:
            timeframe_data: Dictionary mapping timeframe to price data
            
        Returns:
            Combined levels with timeframe weighting
        """
        all_levels = []
        
        # Weight by timeframe (longer = stronger)
        timeframe_weights = {
            '1H': 0.3,
            '4H': 0.5,
            '1D': 0.8,
            '1W': 1.0
        }
        
        for timeframe, data in timeframe_data.items():
            # Get levels for this timeframe
            if 'Close' in data.columns:
                prices = data['Close'].tolist()
            else:
                prices = data['Close'].tolist() if hasattr(data, 'Close') else []
                
            if prices:
                levels = self.detect_levels(prices)
                weight = timeframe_weights.get(timeframe, 0.5)
                
                # Add levels with timeframe info
                for level in levels["support"]:
                    all_levels.append({
                        "price": level,
                        "type": "support",
                        "timeframe": timeframe,
                        "strength": levels["strength"].get(level, 0.5) * weight
                    })
                    
                for level in levels["resistance"]:
                    all_levels.append({
                        "price": level,
                        "type": "resistance",
                        "timeframe": timeframe,
                        "strength": levels["strength"].get(level, 0.5) * weight
                    })
                    
        # Cluster nearby levels
        clustered_levels = self._cluster_multi_timeframe_levels(all_levels)
        
        return {
            "levels": clustered_levels,
            "summary": self._summarize_levels(clustered_levels)
        }
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            # Check if close to previous level (use absolute threshold)
            if abs(levels[i] - current_cluster[-1]) < self.price_threshold:
                current_cluster.append(levels[i])
            else:
                # Start new cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [levels[i]]
                
        # Don't forget last cluster
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _calculate_level_strength(self, prices: List[float], level: float) -> float:
        """Calculate strength of a support/resistance level"""
        touches = 0
        for price in prices:
            if abs(price - level) < self.price_threshold:
                touches += 1
                
        # Normalize strength between 0 and 1
        strength = min(touches / self.min_touches, 1.0)
        return strength
    
    def _calculate_volume_strength(self, data: pd.DataFrame, level: float) -> float:
        """Calculate volume-weighted strength of a level"""
        total_volume = 0
        touches = 0
        
        for _, row in data.iterrows():
            typical_price = (row['High'] + row['Low'] + row['Close']) / 3
            if abs(typical_price - level) < self.price_threshold:
                total_volume += row['Volume']
                touches += 1
                
        if touches == 0:
            return 0.0
            
        # Compare to average volume
        avg_volume = data['Volume'].mean()
        volume_ratio = (total_volume / touches) / avg_volume
        
        # Combine touch count and volume ratio
        touch_strength = min(touches / self.min_touches, 1.0)
        volume_strength = min(volume_ratio / 2, 1.0)  # Cap at 2x average
        
        return (touch_strength + volume_strength) / 2
    
    def _cluster_multi_timeframe_levels(self, levels: List[Dict]) -> List[Dict]:
        """Cluster levels from multiple timeframes"""
        if not levels:
            return []
            
        # Sort by price
        levels.sort(key=lambda x: x['price'])
        
        clustered = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            # Check if close to previous level (use absolute threshold)
            if abs(levels[i]['price'] - current_cluster[-1]['price']) < self.price_threshold:
                current_cluster.append(levels[i])
            else:
                # Merge cluster
                merged = self._merge_level_cluster(current_cluster)
                clustered.append(merged)
                current_cluster = [levels[i]]
                
        # Don't forget last cluster
        if current_cluster:
            clustered.append(self._merge_level_cluster(current_cluster))
            
        return clustered
    
    def _merge_level_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge a cluster of levels into a single level"""
        price = np.mean([l['price'] for l in cluster])
        strength = max([l['strength'] for l in cluster])
        timeframes = list(set([l['timeframe'] for l in cluster]))
        
        # Determine type (support/resistance)
        types = [l['type'] for l in cluster]
        level_type = max(set(types), key=types.count)
        
        return {
            "price": price,
            "type": level_type,
            "timeframe": timeframes[0] if len(timeframes) == 1 else "Multiple",
            "strength": strength,
            "occurrences": len(cluster)
        }
    
    def _summarize_levels(self, levels: List[Dict]) -> Dict:
        """Summarize key levels"""
        if not levels:
            return {"strongest_support": None, "strongest_resistance": None}
            
        supports = [l for l in levels if l['type'] == 'support']
        resistances = [l for l in levels if l['type'] == 'resistance']
        
        strongest_support = max(supports, key=lambda x: x['strength']) if supports else None
        strongest_resistance = max(resistances, key=lambda x: x['strength']) if resistances else None
        
        return {
            "strongest_support": strongest_support,
            "strongest_resistance": strongest_resistance,
            "total_supports": len(supports),
            "total_resistances": len(resistances)
        }