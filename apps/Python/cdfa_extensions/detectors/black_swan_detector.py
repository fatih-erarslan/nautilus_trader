#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:42:16 2025

@author: ashina
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from scipy import stats
from dataclasses import dataclass
import hashlib
from numba import njit, float64, int64, boolean, prange

# Indicator defaults for distribution analysis
DEFAULT_TAIL_INDEX = 3.0
DEFAULT_WINDOW_SIZE = 100
DEFAULT_VAR = 0.02
DEFAULT_ES = 0.03

# Optional Catalyst integration
try:
    from catalyst import qjit
    CATALYST_AVAILABLE = True
except ImportError:
    def qjit(func): return func
    CATALYST_AVAILABLE = False

@njit(cache=True)
def _calculate_rolling_stats(values, window):
    """JIT-optimized rolling mean and std calculation"""
    n = len(values)
    means = np.zeros(n)
    stds = np.zeros(n)
    
    for i in range(window, n):
        window_slice = values[i-window:i]
        means[i] = np.mean(window_slice)
        stds[i] = np.std(window_slice)
    
    return means, stds

@njit(cache=True)
def _vectorized_sigmoid(x, threshold=0.5, steepness=5):
    """JIT-optimized sigmoid function"""
    return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))

@njit(cache=True)
def _safe_div(a, b, default=0.0):
    """Safe division with default for division by zero"""
    if abs(b) > 1e-10:
        return a / b
    return default

@njit(cache=True)
def _clip(x, min_val, max_val):
    """Clip values to range"""
    return max(min(x, max_val), min_val)
@dataclass
class BlackSwanParameters:
    extreme_z_score: float = 3.0
    extreme_event_threshold: float = 0.55
    min_tail_data_points: int = 20
    tail_weight: float = 0.30
    jump_weight: float = 0.20
    cluster_weight: float = 0.20
    extreme_weight: float = 0.30
    max_tail_index: float = 5.0
    min_tail_index: float = 1.5

# Numba optimized helper functions
@njit(cache=True)
def _extremes_detection_impl(returns, rolling_mean, rolling_std, threshold):
    n = len(returns)
    z_scores = np.zeros(n)
    extremes = np.zeros(n, dtype=boolean)
    for i in range(n):
        if not np.isfinite(returns[i]) or not np.isfinite(rolling_mean[i]) or not np.isfinite(rolling_std[i]): continue
        if rolling_std[i] > 1e-9:
            z_scores[i] = (returns[i] - rolling_mean[i]) / rolling_std[i]
            if np.isfinite(z_scores[i]) and np.abs(z_scores[i]) > threshold:
                extremes[i] = True
    return extremes

@njit(cache=True)
def _calculate_hill_estimator(sorted_data):
    n = len(sorted_data)
    if n <= 1: return DEFAULT_TAIL_INDEX
    positive_data = sorted_data[sorted_data > 1e-12]
    n_pos = len(positive_data)
    if n_pos <= 1: return DEFAULT_TAIL_INDEX
    log_data = np.log(positive_data)
    log_threshold = log_data[-1]
    mean_excess = 0.0
    for i in range(n_pos): mean_excess += log_data[i] - log_threshold
    mean_excess /= n_pos
    if mean_excess <= 1e-9: return DEFAULT_TAIL_INDEX
    return 1.0 / mean_excess

@njit(cache=True)
def _calculate_volatility_clustering(squared_returns):
    n = len(squared_returns)
    if n <= 1: return 0.1
    finite_sq_returns = squared_returns[np.isfinite(squared_returns)]
    n_finite = len(finite_sq_returns)
    if n_finite <= 1: return 0.1
    mean_sq = np.mean(finite_sq_returns)
    numerator = 0.0
    variance_sum_sq = np.sum((finite_sq_returns - mean_sq)**2)
    for i in range(1, n_finite):
        diff_i = finite_sq_returns[i] - mean_sq
        diff_i_1 = finite_sq_returns[i-1] - mean_sq
        numerator += diff_i * diff_i_1
    if variance_sum_sq <= 1e-9: return 0.1
    autocorr = numerator / variance_sum_sq
    return min(0.9, abs(autocorr))

# Array hashing function
def array_hash(arr, max_items=20):
    if not isinstance(arr, np.ndarray) or len(arr) == 0: return "empty_array"
    arr_finite = arr[np.isfinite(arr)]
    if len(arr_finite) == 0: return f"all_nonfinite_{len(arr)}"
    if len(arr_finite) > max_items:
        sample = arr_finite[:max_items]
        stats = f"{len(arr)}_{np.mean(sample):.6f}_{np.std(sample):.6f}"
        hash_input = f"{sample.tobytes()}{stats}".encode('utf-8', errors='ignore')
    else:
        hash_input = arr_finite.tobytes()
    return hashlib.md5(hash_input).hexdigest()


    
@njit(cache=True)
def _swan_prepare_features_jit(open_vals, high_vals, low_vals, close_vals, volume_vals, row_index):
    """JIT-compiled implementation of black swan feature preparation"""
    # Pre-allocate result array
    features = np.zeros(12, dtype=np.float64)
    feature_idx = 0
    
    # Basic price and volume
    features[feature_idx] = close_vals[row_index]; feature_idx += 1
    features[feature_idx] = volume_vals[row_index]; feature_idx += 1
    
    # Returns calculation
    if row_index > 0 and close_vals[row_index-1] > 1e-10:
        features[feature_idx] = close_vals[row_index] / close_vals[row_index-1] - 1.0
    else:
        features[feature_idx] = 0.0
    feature_idx += 1
    
    # Recent volatility (index 3)
    if row_index > 20:
        returns = np.zeros(20)
        count = 0
        for i in range(row_index - 20, row_index):
            if i > 0 and close_vals[i-1] > 1e-10:
                returns[count] = close_vals[i] / close_vals[i-1] - 1.0
                count += 1
        
        if count > 1:
            # Calculate mean
            ret_mean = 0.0
            for i in range(count):
                ret_mean += returns[i]
            ret_mean = ret_mean / count
            
            # Calculate std
            var_sum = 0.0
            for i in range(count):
                var_sum += (returns[i] - ret_mean) ** 2
            
            std = np.sqrt(var_sum / (count - 1))
            features[feature_idx] = std * np.sqrt(252.0)  # Annualized
        else:
            features[feature_idx] = 0.2  # Default volatility
        feature_idx += 1
    else:
        features[feature_idx] = 0.2; feature_idx += 1  # Default volatility
    
    # Volume spike (index 4)
    if row_index > 10:
        vol_sum = 0.0
        vol_count = 0
        for i in range(row_index - 10, row_index):
            vol_sum += volume_vals[i]
            vol_count += 1
        
        if vol_count > 0 and vol_sum > 1e-10:
            vol_ma = vol_sum / vol_count
            features[feature_idx] = volume_vals[row_index] / vol_ma
        else:
            features[feature_idx] = 1.0
        feature_idx += 1
    else:
        features[feature_idx] = 1.0; feature_idx += 1
    
    # High-low range (index 5)
    if close_vals[row_index] > 1e-10:
        features[feature_idx] = (high_vals[row_index] - low_vals[row_index]) / close_vals[row_index]
    else:
        features[feature_idx] = 0.0
    feature_idx += 1
    
    # Additional black swan specific features
    
    # Price acceleration (index 6) - 2nd derivative of price
    if row_index > 2:
        ret1 = 0.0
        if close_vals[row_index-2] > 1e-10:
            ret1 = close_vals[row_index-1] / close_vals[row_index-2] - 1.0
        
        ret2 = 0.0
        if close_vals[row_index-1] > 1e-10:
            ret2 = close_vals[row_index] / close_vals[row_index-1] - 1.0
        
        features[feature_idx] = ret2 - ret1  # Acceleration
        feature_idx += 1
    else:
        features[feature_idx] = 0.0; feature_idx += 1
    
    # Tail event indicator (index 7)
    if row_index > 30:
        # Calculate mean and std of returns
        ret_sum = 0.0
        count = 0
        for i in range(row_index - 30, row_index):
            if i > 0 and close_vals[i-1] > 1e-10:
                ret_sum += close_vals[i] / close_vals[i-1] - 1.0
                count += 1
        
        ret_mean = 0.0
        if count > 0:
            ret_mean = ret_sum / count
        
        var_sum = 0.0
        for i in range(row_index - 30, row_index):
            if i > 0 and close_vals[i-1] > 1e-10:
                ret = close_vals[i] / close_vals[i-1] - 1.0
                var_sum += (ret - ret_mean) ** 2
        
        ret_std = 0.001  # Default small value
        if count > 1:
            ret_std = np.sqrt(var_sum / (count - 1))
        
        # Current return
        curr_ret = 0.0
        if row_index > 0 and close_vals[row_index-1] > 1e-10:
            curr_ret = close_vals[row_index] / close_vals[row_index-1] - 1.0
        
        # Z-score
        if ret_std > 1e-10:
            z_score = (curr_ret - ret_mean) / ret_std
            features[feature_idx] = min(abs(z_score) / 4.0, 1.0)  # Normalize to 0-1
        else:
            features[feature_idx] = 0.0
        feature_idx += 1
    else:
        features[feature_idx] = 0.0; feature_idx += 1
    
    # Volume-price divergence (index 8)
    if row_index > 5:
        # Calculate price direction
        price_direction = 0.0
        if close_vals[row_index-5] > 1e-10:
            price_change = close_vals[row_index] / close_vals[row_index-5] - 1.0
            price_direction = 1.0 if price_change > 0.0 else -1.0
        
        # Calculate volume direction
        vol_direction = 0.0
        if volume_vals[row_index-5] > 1e-10:
            vol_change = volume_vals[row_index] / volume_vals[row_index-5] - 1.0
            vol_direction = 1.0 if vol_change > 0.0 else -1.0
        
        # Divergence is when directions are opposite
        divergence = 0.0
        if abs(price_direction) > 0.0 and abs(vol_direction) > 0.0:
            divergence = 1.0 if price_direction * vol_direction < 0.0 else 0.0
        
        features[feature_idx] = divergence
        feature_idx += 1
    else:
        features[feature_idx] = 0.0; feature_idx += 1
    
    return features

@njit(cache=True)
def _calculate_fat_tail_probability_impl(returns, period):
    """JIT-compiled implementation of fat tail probability calculation
    
    Args:
        returns: numpy array of price returns
        period: lookback period
        
    Returns:
        numpy array of fat tail probabilities
    """
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)
    
    # Need at least 2*period data points
    if n < 2*period:
        return result
    
    # Calculate rolling mean and std
    for i in range(period, n):
        # Calculate mean
        mean_sum = 0.0
        count = 0
        for j in range(i-period, i):
            if np.isfinite(returns[j]):
                mean_sum += returns[j]
                count += 1
        
        mean = 0.0
        if count > 0:
            mean = mean_sum / count
        
        # Calculate std
        std_sum = 0.0
        count = 0
        for j in range(i-period, i):
            if np.isfinite(returns[j]):
                std_sum += (returns[j] - mean) ** 2
                count += 1
        
        std = 0.001  # Default small value
        if count > 1:
            std = np.sqrt(std_sum / (count - 1))
        
        # Calculate z-scores
        abs_z_scores = np.zeros(period)
        count = 0
        for j in range(i-period, i):
            if np.isfinite(returns[j]) and std > 1e-10:
                abs_z_scores[count] = abs((returns[j] - mean) / std)
                count += 1
        
        # Sort z-scores to get quantiles
        if count > 5:  # Need at least a few data points
            # Simple bubble sort (for small arrays)
            for j in range(count):
                for k in range(j+1, count):
                    if abs_z_scores[j] > abs_z_scores[k]:
                        # Swap
                        temp = abs_z_scores[j]
                        abs_z_scores[j] = abs_z_scores[k]
                        abs_z_scores[k] = temp
            
            # Calculate 95th and 99th percentiles
            idx_95 = int(0.95 * count)
            idx_99 = int(0.99 * count)
            
            if idx_95 < count and idx_99 < count:
                threshold_95 = abs_z_scores[idx_95]
                threshold_99 = abs_z_scores[idx_99]
                
                # Normal thresholds should be ~1.96 for 95% and ~2.58 for 99%
                fat_tail_factor_95 = 1.96 / max(threshold_95, 0.01)
                fat_tail_factor_99 = 2.58 / max(threshold_99, 0.01)
                
                # Fat tail score (0-1 range)
                fat_tail_score = 1.0 - min((fat_tail_factor_95 + fat_tail_factor_99) / 2.0, 1.0)
                
                # Calculate extreme observation clustering
                extreme_count = 0
                for j in range(max(0, i-10), i):
                    if j < n and np.isfinite(returns[j]) and std > 1e-10:
                        z_score = abs((returns[j] - mean) / std)
                        if z_score > threshold_95:
                            extreme_count += 1
                
                clustering = min(extreme_count / 10.0, 1.0)
                
                # Combine fat tail score with clustering
                result[i] = 0.7 * fat_tail_score + 0.3 * clustering
    
    return result

@njit(cache=True)
def _calculate_liquidity_crisis_probability_impl(volume, high, low, close, period):
    """JIT-compiled implementation of liquidity crisis probability calculation
    
    Args:
        volume: numpy array of volume data
        high, low, close: numpy arrays of price data
        period: lookback period
        
    Returns:
        numpy array of liquidity crisis probabilities
    """
    n = len(volume)
    result = np.zeros(n, dtype=np.float64)
    
    # Need at least period data points
    if n < period:
        return result
    
    # Calculate rolling volume mean
    vol_ma = np.zeros(n)
    for i in range(period, n):
        vol_sum = 0.0
        count = 0
        for j in range(i-period, i):
            if volume[j] > 0:
                vol_sum += volume[j]
                count += 1
        
        if count > 0:
            vol_ma[i] = vol_sum / count
    
    # Calculate volume ratio
    for i in range(period, n):
        # Avoid division by zero
        safe_ma = max(vol_ma[i], 1.0)
        
        # Low liquidity when volume falls below threshold
        volume_ratio = volume[i] / safe_ma
        low_liquidity = 1.0 if volume_ratio < 0.5 else 0.0
        
        # Calculate price ranges if high/low available
        large_ranges = 0.0
        if np.all(np.isfinite([high[i], low[i], close[i]])) and close[i] > 1e-10:
            # Calculate day range
            day_range = (high[i] - low[i]) / close[i]
            
            # Calculate average range
            avg_range = 0.0
            count = 0
            for j in range(i-period, i):
                if j >= 0 and np.all(np.isfinite([high[j], low[j], close[j]])) and close[j] > 1e-10:
                    avg_range += (high[j] - low[j]) / close[j]
                    count += 1
            
            if count > 0:
                avg_range = avg_range / count
                
                # Large ranges indicate poor liquidity
                safe_avg = max(avg_range, 0.001)
                range_ratio = day_range / safe_avg
                large_ranges = 1.0 if range_ratio > 2.0 else 0.0
        
        # Combine indicators
        result[i] = 0.6 * low_liquidity + 0.4 * large_ranges
    
    return result

class BlackSwanDetector:
    """
    Black Swan event detector for predicting rare, high-impact market events.
    
    This detector implements algorithms to identify potential black swan events
    before they fully manifest, based on market microstructure anomalies,
    volatility patterns, and other early warning signals.
    """
    
    def __init__(self, iqad=None, qerc=None, nqo=None, hardware_manager=None, use_classical: bool = False, 
                 params: Optional[BlackSwanParameters] = None, cache_size: int = 100, use_jit: bool = True, 
                 log_level: str = "INFO", optimization_interval_hours: int = 24):
        """
        Initialize the Black Swan Detector.
        
        Args:
            iqad: Immune-Inspired Quantum Anomaly Detector instance
            qerc: Quantum-Enhanced Reservoir Computing instance
            hardware_manager: Hardware manager instance
            use_classical (bool): Force classical implementation
            params: BlackSwanParameters instance for fine-tuning detection
            cache_size: Size of the calculation cache
            log_level: Logging level
        """
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
        else:
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Quantum components (from original)
        self.iqad = iqad
        self.qerc = qerc
        self.nqo = nqo 
        self.hardware_manager = hardware_manager
        self.use_classical = use_classical
        
        # Detection parameters (from original)
        self.sensitivity = 0.75  # Detection sensitivity (0-1)
        self.tail_quantile = 0.95  # Tail event threshold (95th percentile)
        self.vol_lookback = 60  # Volatility lookback period
        self.corr_lookback = 30  # Correlation lookback period
        self.history_size = 100  # Maximum size of historical data
        
        # Optimization settings
        self.optimization_interval_hours = optimization_interval_hours
        self.last_optimization_time = None
        self.optimized_params = None
        
        # New parameters from optimized version
        self.params = params or BlackSwanParameters()
        self.cache_size = cache_size
        self.window_size = self.history_size  # Map history_size to window_size for compatibility
        
        # Internal state (from original)
        self.historical_probabilities = []
        self.last_detection_timestamp = None
        self.historical_volatility = []
        self.regime_transitions = {}
        
        # Cache for optimization (from new version)
        self._calculation_cache = {}
        self._direct_tail_risk = self._calculate_tail_risk_direct
        
        self.logger.info(f"Black Swan Detector initialized (IQAD: {iqad is not None}, QERC: {qerc is not None}, Use Classical: {use_classical})")
    
    def _clear_cache(self):
        """Clear calculation cache"""
        self._calculation_cache.clear()
        self.logger.debug("BlackSwanDetector cache cleared.")

    def _create_cache_key(self, dataframe, lookback=None):
        """Create a cache key for the dataframe"""
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty: return "empty_dataframe"
        lookback = lookback or min(len(dataframe), self.window_size * 2)
        try:
            cols_for_key = ['open', 'high', 'low', 'close', 'volume']
            relevant_cols = [col for col in cols_for_key if col in dataframe.columns]
            if not relevant_cols: return f"no_ohlcv_{len(dataframe)}"
            df_subset = dataframe[relevant_cols].iloc[-lookback:]
            data_bytes = df_subset.to_numpy(dtype=np.float32, na_value=0.0).tobytes()
            timestamp_str = ""
            try:
                last_timestamp = dataframe.index[-1]
                if isinstance(last_timestamp, pd.Timestamp): timestamp_str = str(last_timestamp.value)
                elif isinstance(last_timestamp, (int, float, np.number)): timestamp_str = str(int(last_timestamp))
            except Exception: pass
            hash_input = data_bytes + timestamp_str.encode('utf-8')
            return f"bsprob_{hashlib.md5(hash_input).hexdigest()}"
        except Exception as e:
            self.logger.warning(f"Error creating cache key: {e}. Using fallback.")
            return f"bsprob_fallback_{len(dataframe)}_{dataframe.index[-1]}"

    def _prepare_iqad_features(self, dataframe: pd.DataFrame, row_index: int) -> Dict[str, Any]:
        """
        Non-JIT version that calls the JIT-compiled implementation.
        """
        try:
            # Extract numpy arrays for JIT
            arrays = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in dataframe.columns:
                    arrays[col] = dataframe[col].values
                else:
                    arrays[col] = np.ones(len(dataframe))
            
            # Call the JIT-compiled implementation
            feature_array = _swan_prepare_features_jit(
                arrays['open'], arrays['high'], arrays['low'], 
                arrays['close'], arrays['volume'], row_index
            )
            
            # Convert to dictionary with named keys
            feature_keys = ['close', 'volume', 'return', 'volatility', 'volume_ratio', 
                           'range', 'price_acceleration', 'tail_event']
            features = {}
            for i, key in enumerate(feature_keys):
                if i < len(feature_array):
                    features[key] = float(feature_array[i])
            
            # Add non-JIT compatible fields
            if 'panarchy_phase' in dataframe.columns:
                features['market_phase'] = str(dataframe['panarchy_phase'].iloc[row_index])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing IQAD features: {e}")
            return {'close': 1.0, 'volume': 1.0, 'return': 0.0, 'volatility': 0.2}


    @njit(cache=True)
    def _calculate_fat_tail_probability_impl(returns, period):
        """JIT-compiled implementation of fat tail probability calculation
        
        Args:
            returns: numpy array of price returns
            period: lookback period
            
        Returns:
            numpy array of fat tail probabilities
        """
        n = len(returns)
        result = np.zeros(n, dtype=np.float64)
        
        # Need at least 2*period data points
        if n < 2*period:
            return result
        
        # Calculate rolling mean and std
        for i in range(period, n):
            # Calculate mean
            mean_sum = 0.0
            count = 0
            for j in range(i-period, i):
                if np.isfinite(returns[j]):
                    mean_sum += returns[j]
                    count += 1
            
            mean = 0.0
            if count > 0:
                mean = mean_sum / count
            
            # Calculate std
            std_sum = 0.0
            count = 0
            for j in range(i-period, i):
                if np.isfinite(returns[j]):
                    std_sum += (returns[j] - mean) ** 2
                    count += 1
            
            std = 0.001  # Default small value
            if count > 1:
                std = np.sqrt(std_sum / (count - 1))
            
            # Calculate z-scores
            abs_z_scores = np.zeros(period)
            count = 0
            for j in range(i-period, i):
                if np.isfinite(returns[j]) and std > 1e-10:
                    abs_z_scores[count] = abs((returns[j] - mean) / std)
                    count += 1
            
            # Sort z-scores to get quantiles
            if count > 5:  # Need at least a few data points
                # Simple bubble sort (for small arrays)
                for j in range(count):
                    for k in range(j+1, count):
                        if abs_z_scores[j] > abs_z_scores[k]:
                            # Swap
                            temp = abs_z_scores[j]
                            abs_z_scores[j] = abs_z_scores[k]
                            abs_z_scores[k] = temp
                
                # Calculate 95th and 99th percentiles
                idx_95 = int(0.95 * count)
                idx_99 = int(0.99 * count)
                
                if idx_95 < count and idx_99 < count:
                    threshold_95 = abs_z_scores[idx_95]
                    threshold_99 = abs_z_scores[idx_99]
                    
                    # Normal thresholds should be ~1.96 for 95% and ~2.58 for 99%
                    fat_tail_factor_95 = 1.96 / max(threshold_95, 0.01)
                    fat_tail_factor_99 = 2.58 / max(threshold_99, 0.01)
                    
                    # Fat tail score (0-1 range)
                    fat_tail_score = 1.0 - min((fat_tail_factor_95 + fat_tail_factor_99) / 2.0, 1.0)
                    
                    # Calculate extreme observation clustering
                    extreme_count = 0
                    for j in range(max(0, i-10), i):
                        if j < n and np.isfinite(returns[j]) and std > 1e-10:
                            z_score = abs((returns[j] - mean) / std)
                            if z_score > threshold_95:
                                extreme_count += 1
                    
                    clustering = min(extreme_count / 10.0, 1.0)
                    
                    # Combine fat tail score with clustering
                    result[i] = 0.7 * fat_tail_score + 0.3 * clustering
        
        return result
    
    @njit(cache=True)
    def _calculate_liquidity_crisis_probability_impl(volume, high, low, close, period):
        """JIT-compiled implementation of liquidity crisis probability calculation
        
        Args:
            volume: numpy array of volume data
            high, low, close: numpy arrays of price data
            period: lookback period
            
        Returns:
            numpy array of liquidity crisis probabilities
        """
        n = len(volume)
        result = np.zeros(n, dtype=np.float64)
        
        # Need at least period data points
        if n < period:
            return result
        
        # Calculate rolling volume mean
        vol_ma = np.zeros(n)
        for i in range(period, n):
            vol_sum = 0.0
            count = 0
            for j in range(i-period, i):
                if volume[j] > 0:
                    vol_sum += volume[j]
                    count += 1
            
            if count > 0:
                vol_ma[i] = vol_sum / count
        
        # Calculate volume ratio
        for i in range(period, n):
            # Avoid division by zero
            safe_ma = max(vol_ma[i], 1.0)
            
            # Low liquidity when volume falls below threshold
            volume_ratio = volume[i] / safe_ma
            low_liquidity = 1.0 if volume_ratio < 0.5 else 0.0
            
            # Calculate price ranges if high/low available
            large_ranges = 0.0
            if np.all(np.isfinite([high[i], low[i], close[i]])) and close[i] > 1e-10:
                # Calculate day range
                day_range = (high[i] - low[i]) / close[i]
                
                # Calculate average range
                avg_range = 0.0
                count = 0
                for j in range(i-period, i):
                    if j >= 0 and np.all(np.isfinite([high[j], low[j], close[j]])) and close[j] > 1e-10:
                        avg_range += (high[j] - low[j]) / close[j]
                        count += 1
                
                if count > 0:
                    avg_range = avg_range / count
                    
                    # Large ranges indicate poor liquidity
                    safe_avg = max(avg_range, 0.001)
                    range_ratio = day_range / safe_avg
                    large_ranges = 1.0 if range_ratio > 2.0 else 0.0
            
            # Combine indicators
            result[i] = 0.6 * low_liquidity + 0.4 * large_ranges
        
        return result
    
    def _calculate_fat_tail_probability(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate probability based on fat-tail return distribution.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Fat tail probability
        """
        try:
            # Calculate returns
            returns = dataframe['close'].pct_change().dropna()
            
            # Skip if not enough data
            if len(returns) < 20:
                return pd.Series(0.0, index=dataframe.index)
            
            # Calculate rolling window statistics
            rolling_mean = returns.rolling(window=20).mean()
            rolling_std = returns.rolling(window=20).std()
            
            # Calculate z-scores (standard deviations from mean)
            z_scores = (returns - rolling_mean) / rolling_std.replace(0, 1e-8)  # Avoid division by zero
            
            # Convert to absolutes for tail analysis
            abs_z_scores = z_scores.abs()
            
            # Calculate tail thresholds (95th and 99th percentiles)
            threshold_95 = abs_z_scores.quantile(0.95)
            threshold_99 = abs_z_scores.quantile(0.99)
            
            # Calculate probability of exceeding thresholds
            # Theoretical threshold for normal distribution at 95%: ~1.96
            # Theoretical threshold for normal distribution at 99%: ~2.58
            
            # If actual thresholds are lower than theoretical, distribution has fat tails
            fat_tail_factor_95 = 1.96 / threshold_95 if threshold_95 > 0 else 1.0
            fat_tail_factor_99 = 2.58 / threshold_99 if threshold_99 > 0 else 1.0
            
            # Calculate fat tail score (normalized to 0-1)
            fat_tail_score = 1 - np.clip((fat_tail_factor_95 + fat_tail_factor_99) / 2, 0, 1)
            
            # Create result series
            fat_tail_prob = pd.Series(0.0, index=dataframe.index)
            
            # Find extreme observations in recent data
            extreme_obs = (abs_z_scores > threshold_95).astype(float)
            
            # Calculate clustering of extreme observations
            clustering = extreme_obs.rolling(window=10).sum() / 10
            
            # Combine fat tail score with clustering
            for i in range(len(dataframe)):
                if i < len(clustering):
                    fat_tail_prob.iloc[i] = 0.7 * fat_tail_score + 0.3 * clustering.iloc[i]
                else:
                    fat_tail_prob.iloc[i] = fat_tail_score
            
            return fat_tail_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating fat tail probability: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_liquidity_crisis_probability(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate probability based on liquidity indicators.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Liquidity crisis probability
        """
        try:
            # Initialize result series
            liquidity_prob = pd.Series(0.0, index=dataframe.index)
            
            # Calculate volume-based liquidity indicator
            volume = dataframe['volume']
            volume_ma = volume.rolling(window=30).mean()
            
            # Avoid division by zero
            volume_ma_safe = volume_ma.replace(0, 1)
            
            # Volume ratio (current volume / average volume)
            volume_ratio = volume / volume_ma_safe
            
            # Liquidity dries up when volume ratio falls below threshold
            low_liquidity = (volume_ratio < 0.5).astype(float)
            
            # Calculate price gaps if high-low available
            if 'high' in dataframe.columns and 'low' in dataframe.columns:
                # Calculate price ranges
                day_range = (dataframe['high'] - dataframe['low']) / dataframe['low']
                avg_range = day_range.rolling(window=30).mean()
                
                # Avoid division by zero
                avg_range_safe = avg_range.replace(0, 1)
                
                # Range ratio (current range / average range)
                range_ratio = day_range / avg_range_safe
                
                # Large price ranges indicate poor liquidity
                large_ranges = (range_ratio > 2.0).astype(float)
                
                # Combine volume and range indicators
                for i in range(len(dataframe)):
                    liquidity_prob.iloc[i] = 0.6 * low_liquidity.iloc[i] + 0.4 * large_ranges.iloc[i]
            else:
                # Use only volume indicator if range data not available
                liquidity_prob = low_liquidity
            
            return liquidity_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity crisis probability: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_correlation_breakdown_probability(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate probability based on correlation breakdown.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Correlation breakdown probability
        """            
        try:
            # Initialize correlation_prob Series
            correlation_prob = pd.Series(0.0, index=dataframe.index)
            if len(dataframe) < self.corr_lookback * 2:
                return correlation_prob

            correlation_breakdown = []
            for i in range(self.corr_lookback, len(dataframe)):
                # First window: [i-30 : i] (length 30)
                window1 = dataframe.iloc[i-self.corr_lookback:i]
                # Previous window: [i-60 : i-30] (length 30)
                window2 = dataframe.iloc[i-2*self.corr_lookback:i-self.corr_lookback]

                if len(window1) == self.corr_lookback and len(window2) == self.corr_lookback:
                    # Calculate returns for window 1
                    returns1_close = window1['close'].pct_change().dropna()
                    returns1_vol = window1['volume'].pct_change().dropna()
                    # Calculate returns for window 2
                    returns2_close = window2['close'].pct_change().dropna()
                    returns2_vol = window2['volume'].pct_change().dropna()

                    # Add Robust Checks Before corrcoef
                    corr1 = 0.0  # Default
                    if len(returns1_close) == len(returns1_vol) and len(returns1_close) > 1:
                        # Check for NaNs/Infs after dropna
                        if np.all(np.isfinite(returns1_close)) and np.all(np.isfinite(returns1_vol)):
                            # Check for variance
                            if np.std(returns1_close) > 1e-9 and np.std(returns1_vol) > 1e-9:
                                try:
                                    corr1 = np.corrcoef(returns1_close, returns1_vol)[0, 1]
                                    if np.isnan(corr1): corr1 = 0.0  # Handle NaN result
                                except Exception as e_corr1:
                                    self.logger.warning(f"Error in np.corrcoef window 1 (idx {i}): {e_corr1}")
                                    corr1 = 0.0
                            else: 
                                self.logger.debug(f"Skipping corr1 (idx {i}): Zero variance.")
                        else: 
                            self.logger.debug(f"Skipping corr1 (idx {i}): Non-finite values found.")
                    else: 
                        self.logger.debug(f"Skipping corr1 (idx {i}): Length mismatch or insufficient data ({len(returns1_close)} vs {len(returns1_vol)}).")

                    corr2 = 0.0  # Default
                    if len(returns2_close) == len(returns2_vol) and len(returns2_close) > 1:
                        if np.all(np.isfinite(returns2_close)) and np.all(np.isfinite(returns2_vol)):
                            if np.std(returns2_close) > 1e-9 and np.std(returns2_vol) > 1e-9:
                                try:
                                    corr2 = np.corrcoef(returns2_close, returns2_vol)[0, 1]
                                    if np.isnan(corr2): corr2 = 0.0
                                except Exception as e_corr2:
                                    self.logger.warning(f"Error in np.corrcoef window 2 (idx {i}): {e_corr2}")
                                    corr2 = 0.0
                            else: 
                                self.logger.debug(f"Skipping corr2 (idx {i}): Zero variance.")
                        else: 
                            self.logger.debug(f"Skipping corr2 (idx {i}): Non-finite values found.")
                    else: 
                        self.logger.debug(f"Skipping corr2 (idx {i}): Length mismatch or insufficient data ({len(returns2_close)} vs {len(returns2_vol)}).")

                    # Calculate correlation change
                    corr_change = abs(corr1 - corr2)

                    # Normalize to 0-1 range
                    corr_breakdown = min(corr_change / 1.0, 1.0)
                    correlation_breakdown.append(corr_breakdown)
                else:
                    correlation_breakdown.append(0.0)  # Append default if windows aren't full length
            
            # Pad beginning with zeros
            pad_length = len(dataframe) - len(correlation_breakdown)
            correlation_breakdown = [0.0] * pad_length + correlation_breakdown
            
            # Create series
            correlation_prob = pd.Series(correlation_breakdown, index=dataframe.index)
            
            return correlation_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation breakdown probability: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_anomaly_score(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate anomaly score using IQAD.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Anomaly score
        """
        try:
            # Initialize result series
            anomaly_score = pd.Series(0.0, index=dataframe.index)
            
            # Process most recent data with IQAD
            last_idx = dataframe.index[-1]
            last_row = dataframe.iloc[-1]
            
            # Prepare features for IQAD
            features = {}
            
            # Essential features
            features['close'] = float(last_row['close'])
            features['volume'] = float(last_row['volume'])
            
            # Optional features if available
            for feature in ['high', 'low', 'rsi_14', 'adx', 'atr_14']:
                if feature in last_row:
                    features[feature] = float(last_row[feature])
            
            # Advanced features if available
            for feature in ['volatility_regime', 'antifragility', 'soc_equilibrium', 
                           'soc_fragility', 'panarchy_phase']:
                if feature in last_row:
                    if isinstance(last_row[feature], (int, float)):
                        features[feature] = float(last_row[feature])
                    else:
                        features[feature] = str(last_row[feature])
            
            # Add temporal features if present in the optimized version
            window = 20
            start_idx = max(0, len(dataframe) - window)
            if 'close' in dataframe.columns and start_idx < len(dataframe):
                close_window = dataframe['close'].iloc[start_idx:]
                if len(close_window) > 1:
                    first_close = close_window.iloc[0]; last_close = close_window.iloc[-1]
                    trend = (last_close / first_close - 1) if first_close > 1e-9 else 0.0
                    pct_changes = close_window.pct_change()
                    momentum = np.nanmean(pct_changes) if not pct_changes.isnull().all() else 0.0
                    volatility = np.nanstd(pct_changes) if not pct_changes.isnull().all() else 0.0
                    
                    features['temporal_features'] = {
                        'trend': np.nan_to_num(trend, nan=0.0),
                        'momentum': np.nan_to_num(momentum, nan=0.0),
                        'volatility': np.nan_to_num(volatility, nan=0.0),
                        'regime': 0.5  # Default
                    }
                    
                    # Add regime if available
                    phase = last_row.get('panarchy_phase', 'unknown')
                    regime_values = {'growth': 0.25, 'conservation': 0.5, 'release': 0.75, 'reorganization': 0.9, 'unknown': 0.5}
                    features['temporal_features']['regime'] = regime_values.get(phase, 0.5)
            
            # Detect anomalies using IQAD
            result = self.iqad.detect_anomalies(features)
            
            # Extract anomaly score
            score = result.get('score', 0.0)
            
            # Set score for most recent candle
            anomaly_score.iloc[-1] = score
            
            # Fill previous candles with 0 for visualization
            # This creates a spike only at detection point
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"Error calculating IQAD anomaly score: {str(e)}", exc_info=True)
            return self._calculate_anomaly_score_classical(dataframe)
    
    def _calculate_anomaly_score_classical(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Classical approach for anomaly detection.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Anomaly score
        """
        try:
            # Initialize result series
            anomaly_score = pd.Series(0.0, index=dataframe.index)
            
            # Calculate Z-scores for multiple features
            features = []
            
            # Price returns
            if 'close' in dataframe.columns:
                returns = dataframe['close'].pct_change()
                returns_mean = returns.rolling(window=20).mean()
                returns_std = returns.rolling(window=20).std().replace(0, 1e-8)  # Avoid division by zero
                returns_z = (returns - returns_mean) / returns_std
                features.append(returns_z.abs())
            
            # Volume changes
            if 'volume' in dataframe.columns:
                volume = dataframe['volume'].pct_change()
                volume_mean = volume.rolling(window=20).mean()
                volume_std = volume.rolling(window=20).std().replace(0, 1e-8)
                volume_z = (volume - volume_mean) / volume_std
                features.append(volume_z.abs())
            
            # Volatility changes
            if 'high' in dataframe.columns and 'low' in dataframe.columns:
                volatility = (dataframe['high'] - dataframe['low']) / dataframe['low']
                vol_mean = volatility.rolling(window=20).mean()
                vol_std = volatility.rolling(window=20).std().replace(0, 1e-8)
                vol_z = (volatility - vol_mean) / vol_std
                features.append(vol_z.abs())
            
            # Combine features if available
            if features:
                # Take maximum Z-score across features
                combined_z = pd.concat(features, axis=1).max(axis=1)
                
                # Calculate anomaly score based on extreme z-scores
                # Z-score of 3 (3 standard deviations) = anomaly score of ~0.5
                # Z-score of 6 (6 standard deviations) = anomaly score of ~1.0
                anomaly_score = 1 - 1 / (1 + np.exp(combined_z - 3))
            
            return anomaly_score.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating classical anomaly score: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def calculate_black_swan_probability(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate probability of a black swan event.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Black swan probability (0-1)
        """
        try:
            # Check cache first for performance
            cache_key = self._create_cache_key(dataframe)
            if cache_key in self._calculation_cache:
                cached_data = self._calculation_cache[cache_key]
                if 'probability' in cached_data:
                    self.logger.debug("Using cached Black Swan probability")
                    if 'direction' in cached_data and isinstance(cached_data['direction'], pd.Series):
                        dataframe['swan_direction'] = cached_data['direction']
                    return cached_data['probability']
            
            # Verify sufficient data
            if len(dataframe) < self.vol_lookback:
                return pd.Series(0.0, index=dataframe.index)
            
            # Check if optimization is needed
            current_time = pd.Timestamp.now()
            if (self.last_optimization_time is None or 
                (current_time - self.last_optimization_time).total_seconds() / 3600 >= self.optimization_interval_hours):
                if self.nqo is not None and not self.use_classical:
                    self._run_parameter_optimization(dataframe)
            
            # Initialize result series
            probability = pd.Series(0.0, index=dataframe.index)
            
            # Extract arrays for JIT operations
            arrays = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in dataframe.columns:
                    arrays[col] = dataframe[col].values
                else:
                    arrays[col] = np.ones(len(dataframe))
            
            # Use IQAD for real-time black swan detection
            if self.iqad is not None and not self.use_classical:
                self.logger.debug("Using IQAD for black swan detection")
                
                # Process data with IQAD
                detection_window = min(50, len(dataframe))
                start_idx = max(0, len(dataframe) - detection_window)
                
                for i in range(start_idx, len(dataframe)):
                    # Prepare features using JIT function
                    feature_array = _swan_prepare_features_jit(
                        arrays['open'], arrays['high'], arrays['low'], 
                        arrays['close'], arrays['volume'], i
                    )
                    
                    # Convert to dictionary
                    feature_dict = self._array_to_feature_dict(feature_array)
                    
                    # Add non-JIT compatible fields
                    if 'panarchy_phase' in dataframe.columns:
                        feature_dict['market_phase'] = str(dataframe['panarchy_phase'].iloc[i])
                    
                    # Get anomaly score from IQAD
                    result = self.iqad.detect_anomalies(feature_dict)
                    
                    # Calculate black swan probability from anomaly score
                    raw_score = result['score']
                    probability.iloc[i] = raw_score ** 1.5  # Power function increases rarity
                
                # Fill earlier values
                if start_idx > 0:
                    probability.iloc[:start_idx] = probability.iloc[start_idx]
            else:
                # Fallback to component-based approach
                fat_tail_prob = self._calculate_fat_tail_probability(dataframe)
                liquidity_prob = self._calculate_liquidity_crisis_probability(dataframe)
                correlation_prob = self._calculate_correlation_breakdown_probability(dataframe)
                
                if self.iqad is not None and not self.use_classical:
                    anomaly_score = self._calculate_anomaly_score(dataframe)
                else:
                    anomaly_score = self._calculate_anomaly_score_classical(dataframe)
                
                if self.qerc is not None and not self.use_classical:
                    regime_prob = self._calculate_regime_transition_probability(dataframe)
                else:
                    regime_prob = self._calculate_regime_transition_classical(dataframe)
                
                # Use optimized weights if available
                if self.optimized_params is not None:
                    weights = self.optimized_params
                    fat_tail_weight = weights.get('fat_tail_weight', 0.3)
                    liquidity_weight = weights.get('liquidity_weight', 0.2)
                    correlation_weight = weights.get('correlation_weight', 0.15)
                    anomaly_weight = weights.get('anomaly_weight', 0.25)
                    regime_weight = weights.get('regime_weight', 0.1)
                else:
                    fat_tail_weight = 0.3
                    liquidity_weight = 0.2
                    correlation_weight = 0.15
                    anomaly_weight = 0.25
                    regime_weight = 0.1
                
                # Combine probabilities with weights
                combined_probability = (
                    fat_tail_weight * fat_tail_prob +
                    liquidity_weight * liquidity_prob +
                    correlation_weight * correlation_prob +
                    anomaly_weight * anomaly_score +
                    regime_weight * regime_prob
                )
                
                # Apply sigmoid to normalize to 0-1 range
                probability = 1 / (1 + np.exp(-6 * (combined_probability - 0.5)))
            
            # Store historical data
            self.historical_probabilities.append(probability.iloc[-1])
            if len(self.historical_probabilities) > self.history_size:
                self.historical_probabilities = self.historical_probabilities[-self.history_size:]
            
            self.last_detection_timestamp = pd.Timestamp.now()
            
            # Calculate direction
            direction = self.get_black_swan_direction(dataframe)
            
            # Store in cache
            self._calculation_cache[cache_key] = {
                'probability': probability.copy(),
                'direction': direction.copy()
            }
            
            # Limit cache size
            if len(self._calculation_cache) > self.cache_size * 1.2:
                keys_to_remove = list(self._calculation_cache.keys())[:-self.cache_size]
                for k in keys_to_remove: self._calculation_cache.pop(k, None)
            
            # Store internal value for strategy use
            dataframe['black_swan_internal'] = probability.copy()
            
            # Return inverted probability for display
            display_probability = 1.0 - probability
            return display_probability
            
        except Exception as e:
            self.logger.error(f"Error calculating black swan probability: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
        
    def _array_to_feature_dict(self, feature_array) -> Dict[str, Any]:
        """Convert feature array from JIT function to dictionary for IQAD"""
        feature_keys = ['close', 'volume', 'return', 'volatility', 'volume_ratio', 
                        'range', 'price_acceleration', 'tail_event', 'volume_price_divergence']
        
        features = {}
        for i, key in enumerate(feature_keys):
            if i < len(feature_array) and not np.isnan(feature_array[i]):
                features[key] = float(feature_array[i])
        
        return features

    def _bs_prepare_iqad_features_impl(self, open_vals, high_vals, low_vals, close_vals, volume_vals, row_index):
        """
        Implementation of black swan feature preparation.
        
        Args:
            open_vals, high_vals, low_vals, close_vals, volume_vals: numpy arrays of price/volume data
            row_index: current candle index to analyze
            
        Returns:
            numpy array of computed features
        """
        # Pre-allocate result array
        features = np.zeros(12, dtype=np.float64)
        feature_idx = 0
        
        # Basic price and volume
        features[feature_idx] = close_vals[row_index]; feature_idx += 1
        features[feature_idx] = volume_vals[row_index]; feature_idx += 1
        
        # Returns calculation
        if row_index > 0 and close_vals[row_index-1] > 1e-10:
            features[feature_idx] = close_vals[row_index] / close_vals[row_index-1] - 1.0
        else:
            features[feature_idx] = 0.0
        feature_idx += 1
        
        # Recent volatility
        if row_index > 20:
            returns = np.zeros(20)
            count = 0
            for i in range(row_index - 20, row_index):
                if i > 0 and close_vals[i-1] > 1e-10:
                    returns[count] = close_vals[i] / close_vals[i-1] - 1.0
                    count += 1
            
            if count > 1:
                # Calculate std
                ret_mean = 0.0
                for i in range(count):
                    ret_mean += returns[i]
                ret_mean = ret_mean / count
                
                var_sum = 0.0
                for i in range(count):
                    var_sum += (returns[i] - ret_mean) ** 2
                
                std = np.sqrt(var_sum / (count - 1))
                features[feature_idx] = std * np.sqrt(252.0)  # Annualized
            else:
                features[feature_idx] = 0.2  # Default volatility
            feature_idx += 1
        else:
            features[feature_idx] = 0.2; feature_idx += 1  # Default volatility
        
        # Volume spike
        if row_index > 10:
            vol_sum = 0.0
            vol_count = 0
            for i in range(row_index - 10, row_index):
                vol_sum += volume_vals[i]
                vol_count += 1
            
            if vol_count > 0 and vol_sum > 1e-10:
                vol_ma = vol_sum / vol_count
                features[feature_idx] = volume_vals[row_index] / vol_ma
            else:
                features[feature_idx] = 1.0
            feature_idx += 1
        else:
            features[feature_idx] = 1.0; feature_idx += 1
        
        # High-low range
        if close_vals[row_index] > 1e-10:
            features[feature_idx] = (high_vals[row_index] - low_vals[row_index]) / close_vals[row_index]
        else:
            features[feature_idx] = 0.0
        feature_idx += 1
        
        # Return the features array
        return features
    
    def _train_iqad_on_historical_data(self, dataframe: pd.DataFrame) -> None:
        """
        Train IQAD on historical data to establish baseline for normal patterns.
        Optimized with JIT-compiled feature extraction.
        
        Args:
            dataframe: Historical market data
        """
        if self.iqad is None or len(dataframe) < 60:
            return
        
        try:
            self.logger.info(f"Training IQAD for black swan detection")
            
            # Extract numpy arrays for JIT operations
            arrays = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in dataframe.columns:
                    arrays[col] = dataframe[col].values
                else:
                    arrays[col] = np.ones(len(dataframe))
            
            # Calculate returns
            returns = np.zeros(len(dataframe))
            for i in range(1, len(dataframe)):
                if arrays['close'][i-1] > 1e-10:
                    returns[i] = arrays['close'][i] / arrays['close'][i-1] - 1.0
            
            # For black swan detection, use stable periods with normal returns
            abs_returns = np.abs(returns)
            valid_indices = np.where(np.isfinite(abs_returns))[0]
            
            if len(valid_indices) > 0:
                # Get threshold for 80th percentile (lower 80% considered normal)
                valid_abs_returns = abs_returns[valid_indices]
                threshold = np.percentile(valid_abs_returns, 80)
                
                # Find normal indices
                normal_indices = []
                for i in valid_indices:
                    if abs_returns[i] <= threshold:
                        normal_indices.append(i)
                
                # Limit training set size
                max_samples = 200
                if len(normal_indices) > max_samples:
                    step = len(normal_indices) // max_samples
                    normal_indices = normal_indices[::step][:max_samples]
                
                # Prepare normal patterns
                normal_patterns = []
                for idx in normal_indices:
                    if idx < len(dataframe):
                        # Use JIT-compiled feature extraction
                        feature_array = _bs_prepare_iqad_features_impl(
                            arrays['open'], arrays['high'], arrays['low'], 
                            arrays['close'], arrays['volume'], idx
                        )
                        
                        # Convert to dictionary
                        feature_dict = self._array_to_feature_dict(feature_array)
                        
                        # Add non-JIT compatible fields
                        if 'panarchy_phase' in dataframe.columns:
                            feature_dict['market_phase'] = str(dataframe['panarchy_phase'].iloc[idx])
                        
                        normal_patterns.append(feature_dict)
                
                # Train IQAD
                if normal_patterns:
                    self.iqad.train_on_normal_data(normal_patterns)
                    self.logger.info(f"IQAD trained on {len(normal_patterns)} normal patterns for black swan detection")
            
        except Exception as e:
            self.logger.error(f"Error training IQAD for black swan detection: {e}", exc_info=True)
    
    def _calculate_regime_transition_classical(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Classical approach for regime transition detection.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Regime transition probability
        """
        try:
            # Initialize result series
            regime_prob = pd.Series(0.0, index=dataframe.index)
            
            # Need sufficient data
            if len(dataframe) < 20:
                return regime_prob
            
            # Calculate volatility regime
            returns = dataframe['close'].pct_change().dropna()
            
            # Calculate rolling volatility (standard deviation of returns)
            vol = returns.rolling(window=20).std()
            
            # Calculate volatility of volatility (vol-of-vol)
            vol_of_vol = vol.rolling(window=20).std()
            
            # Detect regime changes based on vol-of-vol spikes
            vol_of_vol_mean = vol_of_vol.rolling(window=40).mean()
            vol_of_vol_ratio = vol_of_vol / vol_of_vol_mean.replace(0, 1e-8)
            
            # Detect regime transitions
            transitions = vol_of_vol_ratio > 1.5
            
            # Calculate transition probability
            for i in range(20, len(dataframe)):
                # Look for transitions in recent past
                # window = transitions.iloc[i-10:i] # OLD - window is a Series slice
                # Check if window slice is valid before proceeding
                start_idx = i - 10
                end_idx = i
                if start_idx < 0 or end_idx > len(transitions):
                     continue # Skip if window indices are out of bounds
            
                window_slice = transitions.iloc[start_idx:end_idx]
            
                if window_slice.any(): # Correct check for Series
                    # --- FIX: Convert to list before reversed iteration ---
                    window_list = window_slice.tolist()
                    for j, has_transition in enumerate(reversed(window_list)):
                    # --- End Fix ---
                        if has_transition:
                            # More recent transitions have higher impact
                            recency_factor = 1 - (j / 10) # 1.0 for most recent, 0.1 for oldest
                            # --- FIX: Use .iloc for assignment ---
                            prob_index = i # Index we are calculating for
                            if prob_index < len(regime_prob): # Bounds check
                                regime_prob.iloc[prob_index] = max(regime_prob.iloc[prob_index], recency_factor)
            
            return regime_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating classical regime probability: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def get_black_swan_direction(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Predict the likely direction of impact if a black swan occurs.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Direction probability (-1 to 1, negative=downside, positive=upside)
        """
        try:
            # Check if swan_direction already exists in dataframe
            if 'swan_direction' in dataframe.columns and not dataframe['swan_direction'].isnull().all():
                return dataframe['swan_direction'].fillna(0).astype(int)
                
            # Initialize result series
            direction = pd.Series(0, index=dataframe.index, dtype=int)
            
            # Check for sufficient data
            if len(dataframe) < 20:
                dataframe['swan_direction'] = direction
                return direction
                
            # Calculate skewness of returns
            returns = dataframe['close'].pct_change().dropna()
            
            # Rolling skewness window
            window_size = min(50, len(returns) - 1)
            
            for i in range(window_size, len(dataframe)):
                # Calculate skewness of returns in window
                window_returns = returns.iloc[i-window_size:i].dropna()
                
                if len(window_returns) >= 20:  # Need sufficient data for reliable skewness
                    skew = stats.skew(window_returns)
                    
                    # Negative skew indicates higher probability of negative shock
                    # Positive skew indicates higher probability of positive shock
                    
                    # Normalize skew to -1 to 1 range
                    normalized_skew = np.clip(-skew, -1, 1)  # Invert skew for intuitive direction
                    
                    direction.iloc[i] = int(np.sign(normalized_skew))
            
            # Smooth direction signal
            direction = direction.rolling(window=5).mean().fillna(0)
            direction = direction.astype(int)  # Convert back to integers
            
            # Store in dataframe for future use
            dataframe['swan_direction'] = direction
            
            return direction
            
        except Exception as e:
            self.logger.error(f"Error predicting black swan direction: {str(e)}", exc_info=True)
            # Default direction
            direction = pd.Series(0, index=dataframe.index, dtype=int)
            dataframe['swan_direction'] = direction
            return direction
    
    def _calculate_tail_risk_direct(self, returns_data: np.ndarray, q: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate tail risk metrics directly from return data
        
        Args:
            returns_data (np.ndarray): Array of returns
            q (float): Quantile threshold for tail (default: 0.95)
            
        Returns:
            Tuple[float, float, float]: (tail_index, value_at_risk, expected_shortfall)
        """
        min_points = self.params.min_tail_data_points
        if not isinstance(returns_data, np.ndarray) or returns_data.ndim != 1:
            self.logger.warning("Invalid input `returns_data` for tail risk calc.")
            return DEFAULT_TAIL_INDEX, DEFAULT_VAR, DEFAULT_ES
            
        finite_returns = returns_data[np.isfinite(returns_data)]
        n_finite = len(finite_returns)
        if n_finite < min_points:
            return DEFAULT_TAIL_INDEX, DEFAULT_VAR, DEFAULT_ES
        
        try:
            abs_returns = np.abs(finite_returns)
            abs_returns = np.sort(abs_returns)[::-1]  # Sort in descending order
            threshold_idx = min(int(n_finite * (1 - q)), n_finite - 1)
            threshold_val = abs_returns[threshold_idx]
            if threshold_val < 1e-12: threshold_val = 1e-12
            
            tail_data = abs_returns[abs_returns >= threshold_val]
            if len(tail_data) < min_points: tail_data = abs_returns[:min_points]
            if len(tail_data) == 0: return DEFAULT_TAIL_INDEX, DEFAULT_VAR, DEFAULT_ES
            
            tail_index = _calculate_hill_estimator(tail_data)
            tail_index = np.clip(tail_index, self.params.min_tail_index, self.params.max_tail_index)
            
            value_at_risk = threshold_val
            expected_shortfall = np.mean(tail_data)
            
            return tail_index, value_at_risk, expected_shortfall
            
        except Exception as e:
            self.logger.error(f"Error in tail risk calculation: {e}", exc_info=True)
            return DEFAULT_TAIL_INDEX, DEFAULT_VAR, DEFAULT_ES
    
    def _extremes_detection(self, returns: np.ndarray, rolling_mean: np.ndarray, rolling_std: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect extreme events in return series
        
        Args:
            returns (np.ndarray): Returns array
            rolling_mean (np.ndarray): Rolling mean of returns
            rolling_std (np.ndarray): Rolling standard deviation of returns
            threshold (float): Z-score threshold for extreme events
            
        Returns:
            np.ndarray: Boolean array indicating extreme events
        """
        n = len(returns)
        if n != len(rolling_mean) or n != len(rolling_std):
            self.logger.warning("Input array lengths mismatch in _extremes_detection.")
            return np.zeros(n, dtype=bool)
            
        returns_nb = returns.astype(np.float64)
        rolling_mean_nb = rolling_mean.astype(np.float64)
        rolling_std_nb = rolling_std.astype(np.float64)
        
        try:
            return _extremes_detection_impl(returns_nb, rolling_mean_nb, rolling_std_nb, threshold)
        except Exception as e:
            self.logger.error(f"Numba extremes detection failed: {e}. Falling back.", exc_info=True)
            valid_std = rolling_std > 1e-9
            z_scores = np.full(n, 0.0)
            z_scores[valid_std] = (returns[valid_std] - rolling_mean[valid_std]) / rolling_std[valid_std]
            extremes = np.abs(z_scores) > threshold
            return extremes
    
    def estimate_black_swan_severity(self, dataframe: pd.DataFrame, 
                                    probability: float, direction: float) -> Dict:
        """
        Estimate potential severity of a black swan event.
        
        Args:
            dataframe (pd.DataFrame): Market data
            probability (float): Black swan probability
            direction (float): Expected direction (-1 to 1)
            
        Returns:
            Dict: Severity estimation
        """
        try:
            # Only estimate if probability is significant
            if probability < 0.5:
                return {
                    'severity': 'low',
                    'estimated_impact': 0.0,
                    'confidence': 0.0
                }
                
            # Calculate returns and volatility
            returns = dataframe['close'].pct_change().dropna()
            
            # Historical volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate VaR and Expected Shortfall
            confidence_level = 0.99
            var = -np.percentile(returns, 100 * (1 - confidence_level))
            
            # Expected shortfall (Conditional VaR)
            extreme_returns = returns[returns <= -var]
            expected_shortfall = extreme_returns.mean() if len(extreme_returns) > 0 else -var
            
            # Adjust based on probability and direction
            impact_multiplier = 2.0 + (probability - 0.5) * 6.0  # 2x to 5x normal ES
            expected_impact = expected_shortfall * impact_multiplier * direction
            
            # Determine severity category
            if abs(expected_impact) < 0.05:
                severity = 'low'
            elif abs(expected_impact) < 0.10:
                severity = 'medium'
            elif abs(expected_impact) < 0.15:
                severity = 'high'
            else:
                severity = 'extreme'
                
            # Calculate confidence in estimate
            confidence = min(0.5 + probability * 0.5, 0.9)  # Max 90% confidence
            
            return {
                'severity': severity,
                'estimated_impact': float(expected_impact),
                'confidence': float(confidence),
                'timeframe': '1-5 days'  # Typical black swan manifestation timeframe
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating black swan severity: {str(e)}", exc_info=True)
            return {
                'severity': 'unknown',
                'estimated_impact': 0.0,
                'confidence': 0.0
            }
        
    def _calculate_evt_probability(self, dataframe: pd.DataFrame, returns: np.ndarray) -> pd.Series:
        """
        Calculate Black Swan probability using Extreme Value Theory.
        
        Args:
            dataframe (pd.DataFrame): Market data
            returns (np.ndarray): Array of returns
            
        Returns:
            pd.Series: Black swan probability using EVT
        """
        default_prob = 0.1
        if len(returns) < max(self.window_size // 4, self.params.min_tail_data_points):
            return pd.Series(default_prob, index=dataframe.index)

        probability = pd.Series(default_prob, index=dataframe.index)

        try:
            finite_returns = returns[np.isfinite(returns)]
            if len(finite_returns) < self.params.min_tail_data_points:
                self.logger.debug("EVT Calc: Not enough finite returns for full calculation.")
                return probability  # Return default

            pos_returns = finite_returns[finite_returns > 0]
            neg_returns = -finite_returns[finite_returns < 0]
            pos_tail_idx, _, _ = self._direct_tail_risk(pos_returns)
            neg_tail_idx, _, _ = self._direct_tail_risk(neg_returns)

            # Calculate Components as Series
            combined_tail_idx = (pos_tail_idx + neg_tail_idx) / 2.0
            tail_component_score = 1.0 - ((np.clip(combined_tail_idx, self.params.min_tail_index, self.params.max_tail_index) - self.params.min_tail_index) / (self.params.max_tail_index - self.params.min_tail_index))
            # Ensure Component is a Series aligned with the input dataframe index
            tail_component = pd.Series(tail_component_score, index=dataframe.index).ffill().fillna(0.5)

            vol_cluster_score = 0.1
            cluster_lookback = max(10, self.window_size // 4)
            if len(finite_returns) >= cluster_lookback:
                squared_returns = finite_returns[-cluster_lookback:]**2
                if len(squared_returns) > 1: vol_cluster_score = _calculate_volatility_clustering(squared_returns)
            # Ensure Component is a Series aligned with the input dataframe index
            cluster_component = pd.Series(np.clip(vol_cluster_score, 0, 1), index=dataframe.index).ffill().fillna(0.1)

            min_periods_z = max(5, self.window_size // 4)
            # Use pandas directly for rolling to ensure index alignment
            returns_series = pd.Series(returns, index=dataframe.index)  # Ensure returns are a Series with correct index
            rolling_mean_s = returns_series.rolling(window=self.window_size, min_periods=min_periods_z).mean()
            rolling_std_s = returns_series.rolling(window=self.window_size, min_periods=min_periods_z).std().replace(0, 1e-8)

            # Pass numpy arrays to numba function, but create Series from result with correct index
            extreme_events_arr = self._extremes_detection(returns, rolling_mean_s.fillna(0).values, rolling_std_s.fillna(1e-8).values, self.params.extreme_z_score)
            extreme_events_series = pd.Series(extreme_events_arr.astype(float), index=dataframe.index)

            recent_extremes_freq = extreme_events_series.rolling(window=max(5, self.window_size//4), min_periods=3).mean().fillna(0)
            # Ensure Component is a Series
            extreme_component = np.clip(recent_extremes_freq * 5, 0, 1)

            jump_component = extreme_component.copy()  # Already a Series

            # Combine Components with Weights
            w_tail = self.params.tail_weight
            w_jump = self.params.jump_weight
            w_cluster = self.params.cluster_weight
            w_extreme = self.params.extreme_weight

            # Ensure adaptive weight calculation results in a Series aligned with others
            w_tail_adj = w_tail * (1 + (1.0 - tail_component) * 0.5)  # w_tail_adj is a Series

            # CORRECTED APPROACH: Perform weighted sum directly using Series * scalar
            weighted_tail = w_tail_adj * tail_component      # Series * Series = Series
            weighted_jump = w_jump * jump_component         # Scalar * Series = Series
            weighted_cluster = w_cluster * cluster_component  # Scalar * Series = Series
            weighted_extreme = w_extreme * extreme_component  # Scalar * Series = Series

            # Sum of weighted components (Series + Series + ... = Series)
            prob_numerator = weighted_tail + weighted_jump + weighted_cluster + weighted_extreme

            # Sum of weights (Series + scalar + ... = Series)
            prob_denominator = w_tail_adj + w_jump + w_cluster + w_extreme  # Series

            # Ensure denominator is safe for division (element-wise)
            prob_denominator_safe = prob_denominator.replace(0, 1e-9)

            # Calculate final probability using element-wise division
            probability = prob_numerator / prob_denominator_safe

            # Apply final smoothing
            probability = probability.ewm(span=max(3, self.window_size//10), min_periods=2).mean()

            return probability.fillna(default_prob).clip(0.01, 0.99)

        except Exception as e:
            # Log the specific error encountered during EVT calculation
            self.logger.error(f"Error within _calculate_evt_probability: {e}", exc_info=True)
            return pd.Series(default_prob, index=dataframe.index)  # Return default Series on error
        
    def detect(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect potential black swan events in market data.
        This method is a wrapper around calculate_black_swan_probability
        to maintain consistency with other detector classes.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Binary indicator of black swan conditions
        """
        # Calculate probability
        probabilities = self.calculate_black_swan_probability(dataframe)
        
        # Convert to binary signal (detect high probability events)
        threshold = 0.5  # Threshold for binary detection
        binary_signal = (probabilities > threshold).astype(int)
        
        return binary_signal
    
    def detect_signals(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect black swan signals from price and volume arrays for CDFA server compatibility.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            
        Returns:
            Dict with signal, confidence, and metadata
        """
        try:
            # Create a minimal DataFrame from the arrays
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes,
                'open': prices,  # Use close as approximation
                'high': prices,  # Use close as approximation  
                'low': prices    # Use close as approximation
            })
            
            # Calculate black swan probabilities
            probabilities = self.calculate_black_swan_probability(df)
            
            if len(probabilities) > 0:
                # Get the latest probability
                latest_prob = float(probabilities.iloc[-1])
                
                # Calculate confidence based on stability of recent probabilities
                if len(probabilities) > 5:
                    recent_probs = probabilities.tail(5)
                    confidence = float(1.0 - np.std(recent_probs))
                else:
                    confidence = 0.5
                
                return {
                    "signal": latest_prob,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "detected": latest_prob > 0.5,
                    "processing_time": 0.0
                }
            else:
                return {
                    "signal": 0.5,
                    "confidence": 0.0,
                    "detected": False,
                    "processing_time": 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error in detect_signals: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "detected": False,
                "error": str(e),
                "processing_time": 0.0
            }

                
    def _run_parameter_optimization(self, dataframe: pd.DataFrame) -> None:
        """
        Run periodic parameter optimization using NQO.
        
        Args:
            dataframe: Market data used for optimization
        """
        if self.nqo is None or self.use_classical:
            return
        
        start_time = time.time()
        self.logger.info("Running periodic black swan parameter optimization with NQO")
        
        try:
            # Define initial parameters
            initial_params = np.array([
                0.3,  # fat_tail_weight
                0.2,  # liquidity_weight
                0.15, # correlation_weight
                0.25, # anomaly_weight
                0.1,  # regime_weight
                self.params.extreme_z_score  # extreme z-score threshold
            ], dtype=float)
            
            # Prepare data slice for optimization
            data_slice_len = min(200, len(dataframe))
            dataframe_slice = dataframe.iloc[-data_slice_len:].copy()
            
            # Define objective function for parameter optimization
            def objective_function(params_array, data=None):
                # Map parameters
                fat_tail_weight = params_array[0]
                liquidity_weight = params_array[1]
                correlation_weight = params_array[2]
                anomaly_weight = params_array[3]
                regime_weight = params_array[4]
                extreme_z_score = params_array[5]
                
                # Temporarily set the extreme z-score threshold
                old_z_score = self.params.extreme_z_score
                self.params.extreme_z_score = extreme_z_score
                
                try:
                    # Calculate component probabilities
                    fat_tail_prob = self._calculate_fat_tail_probability(data)
                    liquidity_prob = self._calculate_liquidity_crisis_probability(data)
                    correlation_prob = self._calculate_correlation_breakdown_probability(data)
                    anomaly_score = self._calculate_anomaly_score_classical(data)
                    regime_prob = self._calculate_regime_transition_classical(data)
                    
                    # Combine with weights
                    combined_prob = (
                        fat_tail_weight * fat_tail_prob +
                        liquidity_weight * liquidity_prob +
                        correlation_weight * correlation_prob +
                        anomaly_weight * anomaly_score +
                        regime_weight * regime_prob
                    )
                    
                    # Calculate objective value based on detection quality
                    # For black swan, we want a low baseline probability with occasional spikes
                    
                    # Get the 95th percentile as the "spike" level
                    spike_level = np.percentile(combined_prob, 95)
                    
                    # Calculate the ratio of the spike to the median (signal-to-noise)
                    baseline = np.median(combined_prob)
                    if baseline > 0:
                        signal_to_noise = spike_level / baseline
                    else:
                        signal_to_noise = 1.0
                    
                    # Calculate the frequency of probability > 0.5 (should be rare)
                    freq_high = np.mean(combined_prob > 0.5)
                    
                    # Ideal case: High signal-to-noise ratio, low frequency of high values
                    objective_value = signal_to_noise * (1.0 - freq_high)
                    
                    # Restore original parameter
                    self.params.extreme_z_score = old_z_score
                    
                    return objective_value
                    
                except Exception as e:
                    self.logger.error(f"Error in objective function: {e}")
                    # Restore original parameter
                    self.params.extreme_z_score = old_z_score
                    return 0.0
            
            # Create partial function for objective
            from functools import partial
            cost_function = partial(objective_function, data=dataframe_slice)
            
            # Run optimization with NQO
            optimized_result = self.nqo.optimize_parameters(
                initial_params=initial_params,
                objective=cost_function,
                iterations=10
            )
            
            # Process optimization results
            if (optimized_result and
                'params' in optimized_result and
                isinstance(optimized_result['params'], np.ndarray) and
                optimized_result.get('error') is None):
                
                optimized_params_array = optimized_result['params']
                
                # Map parameters
                param_map = ['fat_tail_weight', 'liquidity_weight', 'correlation_weight', 
                             'anomaly_weight', 'regime_weight', 'extreme_z_score']
                
                optimized_params_dict = {name: optimized_params_array[i] for i, name in enumerate(param_map)}
                
                # Store optimized parameters
                self.optimized_params = optimized_params_dict
                
                # Update BlackSwanParameters
                old_z_score = self.params.extreme_z_score
                new_z_score = optimized_params_dict.get('extreme_z_score', old_z_score)
                self.params.extreme_z_score = new_z_score
                
                self.logger.info(f"Optimization complete. Updated extreme_z_score from {old_z_score:.2f} to {new_z_score:.2f}")
            else:
                self.logger.warning("Optimization failed or returned invalid result")
            
            # Update optimization timestamp
            self.last_optimization_time = pd.Timestamp.now()
            
            optimization_time = (time.time() - start_time) * 1000  # ms
            self.logger.info(f"Black swan parameter optimization took {optimization_time:.2f} ms")
        
        except Exception as e:
            self.logger.error(f"Error during black swan parameter optimization: {str(e)}", exc_info=True)