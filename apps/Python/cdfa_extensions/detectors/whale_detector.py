#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:41:03 2025

@author: ashina
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from functools import lru_cache
import time
from dataclasses import dataclass
import hashlib
import numba as nb
from numba import njit, float64, int64, boolean, prange

# PennyLane Catalyst JIT support - keeping for backward compatibility
try:
    from catalyst import qjit
    CATALYST_AVAILABLE = True
except ImportError:
    # Fallback decorator when Catalyst isn't available
    def qjit(func):
        return func
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

# Numba-optimized helper functions
@njit(cache=True)
def _volume_zscore_impl(volume, window):
    """Numba implementation of volume Z-score calculation"""
    n = len(volume)
    z_scores = np.zeros(n)
    
    for i in range(window, n):
        window_slice = volume[i-window:i]
        mean_vol = np.mean(window_slice)
        std_vol = np.std(window_slice)
        
        # Avoid division by zero
        if std_vol > 1e-9:
            z_scores[i] = (volume[i] - mean_vol) / std_vol
    
    return z_scores

@njit(cache=True)
def _calculate_gini_impl(volumes):
    """Numba implementation of Gini coefficient calculation"""
    if len(volumes) == 0 or np.sum(volumes) == 0:
        return 0.5
    
    # Sort volumes
    sorted_volumes = np.sort(volumes)
    
    # Calculate cumulative sum
    n = len(sorted_volumes)
    cum_volumes = np.zeros(n)
    cum_volumes[0] = sorted_volumes[0]
    
    for i in range(1, n):
        cum_volumes[i] = cum_volumes[i-1] + sorted_volumes[i]
    
    # Calculate Lorenz curve and Gini coefficient
    total_volume = cum_volumes[-1]
    gini = 0.0
    
    for i in range(1, n):
        x_width = 1.0/n
        y_avg = (cum_volumes[i] + cum_volumes[i-1]) / (2 * total_volume)
        gini += x_width * y_avg
    
    # Gini coefficient = 1 - 2 * area under Lorenz curve
    gini = 1.0 - 2.0 * gini
    
    return gini

@njit(cache=True)
def _prepare_whale_features_jit(open_vals, high_vals, low_vals, close_vals, volume_vals, row_index):
    """JIT-compiled implementation of feature preparation for whale detection"""
    # Pre-allocate result array (12 features max)
    features = np.zeros(12, dtype=np.float64)
    feature_idx = 0
    
    # Basic price and volume
    features[feature_idx] = close_vals[row_index]; feature_idx += 1
    features[feature_idx] = volume_vals[row_index]; feature_idx += 1
    
    # Volume ratios (indices 2-3)
    if row_index > 0:
        # Previous volume change
        prev_vol = volume_vals[row_index-1]
        if prev_vol > 1e-10:
            vol_change = volume_vals[row_index] / prev_vol - 1.0
        else:
            vol_change = 0.0
        features[feature_idx] = vol_change; feature_idx += 1
        
        # Volume ratio to average (30-period lookback)
        lookback = min(30, row_index)
        if lookback > 0:
            avg_vol = 0.0
            for i in range(row_index - lookback, row_index):
                avg_vol += volume_vals[i]
            avg_vol = avg_vol / lookback
            if avg_vol > 1e-10:
                vol_ratio = volume_vals[row_index] / avg_vol
            else:
                vol_ratio = 1.0
            features[feature_idx] = vol_ratio; feature_idx += 1
    
    # Price changes (indices 4-5)
    if row_index > 0 and open_vals[row_index] > 1e-10:
        # Current candle price change
        price_change = (close_vals[row_index] - open_vals[row_index]) / open_vals[row_index]
        features[feature_idx] = price_change; feature_idx += 1
        
        # Price impact (price change per unit volume)
        if volume_vals[row_index] > 1e-10:
            price_impact = abs(price_change) / volume_vals[row_index]
        else:
            price_impact = 0.0
        features[feature_idx] = price_impact; feature_idx += 1
    
    # Trend and momentum (indices 6-8)
    window = min(20, row_index)
    if window > 0:
        # Calculate trend as percent change over window
        start_price = close_vals[row_index - window]
        if start_price > 1e-10:
            trend = close_vals[row_index] / start_price - 1.0
        else:
            trend = 0.0
        features[feature_idx] = trend; feature_idx += 1
        
        # Calculate momentum as average of recent returns
        momentum_sum = 0.0
        momentum_count = 0
        for i in range(row_index - min(5, window), row_index):
            if i > 0 and close_vals[i-1] > 1e-10:
                momentum_sum += close_vals[i] / close_vals[i-1] - 1.0
                momentum_count += 1
        
        if momentum_count > 0:
            momentum = momentum_sum / momentum_count
        else:
            momentum = 0.0
        features[feature_idx] = momentum; feature_idx += 1
        
        # Calculate volatility as standard deviation of returns
        vol_sum = 0.0
        returns = np.zeros(window)
        ret_count = 0
        for i in range(row_index - window, row_index):
            if i > 0 and close_vals[i-1] > 1e-10:
                returns[ret_count] = close_vals[i] / close_vals[i-1] - 1.0
                ret_count += 1
        
        if ret_count > 1:
            # Calculate mean first
            ret_mean = 0.0
            for i in range(ret_count):
                ret_mean += returns[i]
            ret_mean = ret_mean / ret_count
            
            # Calculate std
            vol_sum = 0.0
            for i in range(ret_count):
                vol_sum += (returns[i] - ret_mean) ** 2
            
            volatility = np.sqrt(vol_sum / (ret_count - 1))
        else:
            volatility = 0.0
        features[feature_idx] = volatility; feature_idx += 1
    
    # Range information (index 9)
    if row_index > 0 and close_vals[row_index] > 1e-10:
        range_ratio = (high_vals[row_index] - low_vals[row_index]) / close_vals[row_index]
        features[feature_idx] = range_ratio; feature_idx += 1
    
    return features

@njit(cache=True)
def _detect_volume_anomalies_impl(volume, period):
    """JIT-compiled volume anomaly detection
    
    Args:
        volume: numpy array of volume data
        period: lookback period
        
    Returns:
        numpy array of volume anomaly scores
    """
    n = len(volume)
    result = np.zeros(n, dtype=np.float64)
    
    # Calculate rolling mean and standard deviation
    vol_ma = np.zeros(n)
    vol_std = np.zeros(n)
    
    for i in range(period, n):
        # Calculate mean
        ma_sum = 0.0
        for j in range(i - period, i):
            ma_sum += volume[j]
        vol_ma[i] = ma_sum / period
        
        # Calculate std
        std_sum = 0.0
        for j in range(i - period, i):
            std_sum += (volume[j] - vol_ma[i]) ** 2
        vol_std[i] = np.sqrt(std_sum / period)
    
    # Calculate z-scores and volume ratios
    for i in range(period, n):
        # Avoid division by zero
        safe_std = max(vol_std[i], 1e-10)
        
        # Calculate z-score
        z_score = (volume[i] - vol_ma[i]) / safe_std
        
        # Calculate volume ratio
        safe_ma = max(vol_ma[i], 1e-10)
        vol_ratio = volume[i] / safe_ma
        
        # Detect volume spikes (volume > 1.5 * average)
        vol_spike = 1.0 if vol_ratio > 1.5 else 0.0
        
        # Combine signals (0.6 * spikes + 0.4 * normalized z-score)
        normalized_z = max(min(z_score / 6.0 + 0.5, 1.0), 0.0)
        result[i] = 0.6 * vol_spike + 0.4 * normalized_z
    
    return result

@njit(cache=True)
def _calculate_price_impact_impl(close, volume, period):
    """JIT-compiled price impact calculation
    
    Args:
        close: numpy array of closing prices
        volume: numpy array of volume data
        period: lookback period
        
    Returns:
        numpy array of price impact scores
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    
    # Calculate returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = abs(close[i] / close[i-1] - 1.0)
    
    # Calculate volume ratios
    vol_ma = np.zeros(n)
    vol_ratio = np.zeros(n)
    
    for i in range(period, n):
        # Calculate volume MA
        ma_sum = 0.0
        for j in range(i - period, i):
            ma_sum += volume[j]
        vol_ma[i] = ma_sum / period
        
        # Calculate volume ratio
        safe_ma = max(vol_ma[i], 1e-10)
        vol_ratio[i] = volume[i] / safe_ma
    
    # Calculate price impact (price change * volume ratio)
    price_impact = np.zeros(n)
    for i in range(period, n):
        price_impact[i] = returns[i] * vol_ratio[i]
    
    # Normalize using rolling max
    for i in range(period*2, n):
        # Find max in window
        max_impact = 0.0
        for j in range(i - period*2, i):
            max_impact = max(max_impact, price_impact[j])
        
        # Normalize
        safe_max = max(max_impact, 1e-10)
        normalized_impact = price_impact[i] / safe_max
        
        # Apply threshold for significant impacts (> 0.01)
        significant = 1.0 if normalized_impact > 0.01 else 0.0
        
        # Combine
        result[i] = 0.7 * normalized_impact + 0.3 * significant
    
    return result
from numba import njit

@njit(cache=True)
def _detect_volume_anomalies_jit(volume, period):
    """JIT-compiled volume anomaly detection"""
    n = len(volume)
    result = np.zeros(n, dtype=np.float64)
    
    # Calculate rolling mean and standard deviation
    vol_ma = np.zeros(n)
    vol_std = np.zeros(n)
    
    for i in range(period, n):
        # Calculate mean
        ma_sum = 0.0
        for j in range(i - period, i):
            ma_sum += volume[j]
        vol_ma[i] = ma_sum / period
        
        # Calculate std
        std_sum = 0.0
        for j in range(i - period, i):
            std_sum += (volume[j] - vol_ma[i]) ** 2
        vol_std[i] = np.sqrt(std_sum / period)
    
    # Calculate z-scores and volume ratios
    for i in range(period, n):
        # Avoid division by zero
        safe_std = max(vol_std[i], 1e-10)
        
        # Calculate z-score
        z_score = (volume[i] - vol_ma[i]) / safe_std
        
        # Calculate volume ratio
        safe_ma = max(vol_ma[i], 1e-10)
        vol_ratio = volume[i] / safe_ma
        
        # Detect volume spikes (volume > 1.5 * average)
        vol_spike = 1.0 if vol_ratio > 1.5 else 0.0
        
        # Combine signals (0.6 * spikes + 0.4 * normalized z-score)
        normalized_z = max(min(z_score / 6.0 + 0.5, 1.0), 0.0)
        result[i] = 0.6 * vol_spike + 0.4 * normalized_z
    
    return result

@njit(cache=True)
def _price_impact_calculation(close, volume, period):
    """JIT-optimized implementation of price impact calculation."""
    n = len(close)
    result = np.zeros(n, dtype=np.float64)
    
    # Calculate returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = abs(close[i] / close[i-1] - 1.0)
    
    # Calculate volume ratios
    vol_ma = np.zeros(n)
    vol_ratio = np.zeros(n)
    
    for i in range(period, n):
        # Calculate volume MA
        ma_sum = 0.0
        for j in range(i - period, i):
            ma_sum += volume[j]
        vol_ma[i] = ma_sum / period
        
        # Calculate volume ratio
        safe_ma = max(vol_ma[i], 1e-10)
        vol_ratio[i] = volume[i] / safe_ma
    
    # Calculate price impact (price change * volume ratio)
    price_impact = np.zeros(n)
    for i in range(period, n):
        price_impact[i] = returns[i] * vol_ratio[i]
    
    # Normalize using rolling max
    for i in range(period*2, n):
        # Find max in window
        max_impact = 0.0
        for j in range(i - period*2, i):
            max_impact = max(max_impact, price_impact[j])
        
        # Normalize
        safe_max = max(max_impact, 1e-10)
        normalized_impact = price_impact[i] / safe_max
        
        # Apply threshold for significant impacts (> 0.01)
        significant = 1.0 if normalized_impact > 0.01 else 0.0
        
        # Combine
        result[i] = 0.7 * normalized_impact + 0.3 * significant
    
    return result

@dataclass
class WhaleParameters:
    """Parameters for whale activity detection"""
    # Thresholds
    volume_z_score_threshold: float = 2.5
    price_impact_threshold: float = 0.01
    volume_increase_threshold: float = 2.0  # Min volume multiplier for anomaly
    
    # Weights for detection components
    volume_weight: float = 0.4
    impact_weight: float = 0.4
    gini_weight: float = 0.2
    
    # Time windows
    volume_history: int = 200  # Long-term volume history
    gini_window: int = 50  # Window for volume concentration (Gini)
    
class WhaleDetector:
    """
    Whale Detector for identifying large market participant activity.
    
    This detector implements algorithms to identify unusual volume patterns,
    coordinated transactions, and market movements typically associated with
    "whale" activity (large market participants).
    """
    
    def __init__(self, iqad=None, qerc=None, nqo=None, hardware_manager=None, use_classical: bool = False,
                 volume_threshold: float = 1.5, price_impact: float = 0.01, window: int = 6,
                 cache_size: int = 100, params: Optional[WhaleParameters] = None, use_jit: bool = True,
                 log_level: str = "INFO", optimization_interval_hours: int = 24):
        """
        Initialize the Whale Detector.
        
        Args:
            qerc: Quantum-Enhanced Reservoir Computing instance
            nqo: Neuromorphic Quantum Optimizer instance
            hardware_manager: Hardware manager instance
            use_classical (bool): Force classical implementation
            volume_threshold: Threshold for volume anomaly detection
            price_impact: Expected price impact threshold
            window: Window size for recent analysis
            cache_size: Size of the LRU cache for expensive calculations
            params: Optional custom parameters
            log_level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, log_level))
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(getattr(logging, log_level))
        
        # Quantum components
        self.iqad = iqad 
        self.qerc = qerc
        self.nqo = nqo
        self.hardware_manager = hardware_manager
        self.use_classical = use_classical
        
        # Backward compatibility parameters from the original class
        self.volume_threshold_factor = volume_threshold
        self.price_impact_threshold = price_impact
        self.window = window
        self.sensitivity = 0.75  # Detection sensitivity (0-1)
        self.lookback_periods = [5, 15, 30]  # Multiple timeframes for analysis
        
        # Optimization settings
        self.optimization_interval_hours = optimization_interval_hours
        self.last_optimization_time = None
        self.optimized_params = None
        
        # From optimized version
        self.cache_size = cache_size
        self.params = params or WhaleParameters()
        
        # Internal state tracking
        self.historical_whale_activity = []
        self.last_detection_timestamp = None
        self.whale_confidence_scores = []
        self._price_buffer = []
        self._volume_buffer = []
        
        # Setup result cache
        self._calculation_cache = {}
        
        # Apply caching
        self._setup_cached_methods()
        
        self.logger.info(f"Initialized WhaleDetector (QERC: {qerc is not None}, NQO: {nqo is not None}, JIT: {CATALYST_AVAILABLE})")
    
    def _setup_cached_methods(self):
        """Setup method caching using LRU cache decorator"""
        # Apply caching to expensive calculations
        self._cached_calc_gini = lru_cache(maxsize=self.cache_size)(self._calculate_gini)
        
        # Keep backward compatibility - but we'll use Numba JIT internally
        self._jit_volume_zscore = self._volume_zscore
    
    def _volume_zscore(self, volume: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate volume Z-scores using rolling window.
        
        Args:
            volume: Volume array
            window: Rolling window size
            
        Returns:
            Array of volume Z-scores
        """
        n = len(volume)
        if n < window:
            return np.zeros(n)
        
        try:
            # Call the Numba implementation
            return _volume_zscore_impl(volume, window)
        except Exception as e:
            self.logger.warning(f"Numba implementation failed: {e}, falling back to Python")
            # Fallback to Python implementation for robustness
            z_scores = np.zeros(n)
            
            # Calculate rolling mean and std
            for i in range(window, n):
                window_slice = volume[i-window:i]
                mean_vol = np.mean(window_slice)
                std_vol = np.std(window_slice)
                
                # Avoid division by zero
                if std_vol > 1e-9:
                    z_scores[i] = (volume[i] - mean_vol) / std_vol
            
            return z_scores
    
    def _calculate_gini(self, data_key: str, volumes: np.ndarray, window: int) -> float:
        """
        Calculate Gini coefficient for volume distribution (for LRU cache).
        
        Args:
            data_key: Cache key
            volumes: Volume array
            window: Window size for Gini calculation
            
        Returns:
            Gini coefficient
        """
        if len(volumes) < window or np.sum(volumes) == 0:
            return 0.5
            
        try:
            # Use Numba implementation
            return _calculate_gini_impl(volumes)
        except Exception as e:
            self.logger.warning(f"Numba Gini calculation failed: {e}, falling back to Python")
            # Fall back to Python implementation
            # Sort volumes
            sorted_volumes = np.sort(volumes)
            
            # Calculate cumulative sum
            n = len(sorted_volumes)
            cum_volumes = np.cumsum(sorted_volumes)
            
            # Calculate Lorenz curve coordinates
            lorenz_x = np.arange(n) / (n - 1)
            lorenz_y = cum_volumes / cum_volumes[-1]
            
            # Calculate Gini coefficient (area between diagonal and Lorenz curve)
            gini = 1 - 2 * np.sum((lorenz_x[1:] - lorenz_x[:-1]) * (lorenz_y[1:] + lorenz_y[:-1])) / 2
            
            return gini
    
    def _prepare_iqad_features(self, dataframe: pd.DataFrame, row_index: int) -> Dict[str, Any]:
        """
        Prepare features for IQAD anomaly detection.
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
            feature_array = _prepare_whale_features_jit(
                arrays['open'], arrays['high'], arrays['low'], 
                arrays['close'], arrays['volume'], row_index
            )
            
            # Convert to dictionary with named keys
            feature_keys = ['close', 'volume', 'volume_ratio', 'price_change', 
                           'price_impact', 'trend']
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
            return {'close': 1.0, 'volume': 1.0, 'volume_ratio': 1.0, 'price_change': 0.0}
    

    def _detect_volume_anomalies(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Non-JIT version that calls the JIT-compiled implementation.
        """
        try:
            # Extract volume array
            if 'volume' not in dataframe.columns:
                return pd.Series(0.0, index=dataframe.index)
                
            volume = dataframe['volume'].values
            
            # Call JIT-compiled function
            anomalies = _detect_volume_anomalies_jit(volume, period)
            
            # Convert back to pandas Series
            return pd.Series(anomalies, index=dataframe.index)
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {e}")
            return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_price_impact_impl(self, close, volume, period):
        """Calls the JIT-optimized implementation."""
        return _price_impact_calculation(close, volume, period)

    
    def _create_cache_key(self, dataframe, suffix=""):
        """Create cache key for dataframe"""
        try:
            if dataframe.empty:
                return f"empty_df_{suffix}"
                
            # Use last few rows for the key to avoid recalculation when only new data is added
            key_rows = min(20, len(dataframe))
            
            # Create hash using relevant columns
            cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in cols if c in dataframe.columns]
            
            if not available_cols:
                return f"no_ohlcv_{len(dataframe)}_{suffix}"
                
            # Extract data for hashing
            data = dataframe[available_cols].iloc[-key_rows:].values
            
            # Create hash
            hasher = hashlib.md5(data.tobytes())
            hash_value = hasher.hexdigest()
            
            # Include timestamp if available
            if hasattr(dataframe.index[-1], 'timestamp'):
                hash_value += f"_{int(dataframe.index[-1].timestamp())}"
            
            return f"whale_{hash_value}_{suffix}"
        except Exception as e:
            self.logger.warning(f"Error creating cache key: {e}")
            return f"fallback_key_{len(dataframe)}_{suffix}"
    
    def detect_whale_activity(self, dataframe: pd.DataFrame, period: int = 30) -> pd.Series:
        """
        Detect whale activity in market data using IQAD for real-time detection.
        Optimized with JIT and vectorization.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Lookback period for analysis
            
        Returns:
            pd.Series: Whale activity score (0-1)
        """
        try:
            # Check cache first
            cache_key = self._create_cache_key(dataframe, f"period_{period}")
            if cache_key in self._calculation_cache:
                self.logger.debug("Using cached whale activity calculation")
                return self._calculation_cache[cache_key]
            
            # Verify sufficient data
            if len(dataframe) < period + 5:
                return pd.Series(0.0, index=dataframe.index)
            
            # Check if optimization is needed (first run or time elapsed)
            current_time = pd.Timestamp.now()
            if (self.last_optimization_time is None or 
                (current_time - self.last_optimization_time).total_seconds() / 3600 >= self.optimization_interval_hours):
                self._run_parameter_optimization(dataframe)
            
            # Extract numpy arrays for JIT operations
            arrays = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in dataframe.columns:
                    arrays[col] = dataframe[col].values
                else:
                    # Create default arrays if columns missing
                    arrays[col] = np.ones(len(dataframe))
            
            # Initialize result array
            whale_scores = np.zeros(len(dataframe))
            
            # Train IQAD if not already done
            if self.iqad is not None and not hasattr(self.iqad, '_trained') and not self.use_classical:
                self._train_iqad_on_historical_data(dataframe)
                # Mark as trained to avoid retraining on every call
                self.iqad._trained = True
            
            # Use IQAD for real-time detection
            if self.iqad is not None and not self.use_classical:
                self.logger.debug("Using IQAD for whale activity detection")
                
                # Process recent candles with IQAD
                detection_window = min(50, len(dataframe))
                start_idx = max(0, len(dataframe) - detection_window)
                
                # Process each candle
                for i in range(start_idx, len(dataframe)):
                    # Get features using JIT-compiled function
                    feature_array = self._prepare_iqad_features(
                        arrays['open'], arrays['high'], arrays['low'], 
                        arrays['close'], arrays['volume'], i
                    )
                    
                    # Convert array to dictionary for IQAD (which expects dict)
                    feature_dict = self._array_to_feature_dict(feature_array)
                    
                    # Add non-JIT compatible fields
                    if 'panarchy_phase' in dataframe.columns:
                        feature_dict['market_phase'] = str(dataframe['panarchy_phase'].iloc[i])
                    
                    # Detect anomalies using IQAD
                    result = self.iqad.detect_anomalies(feature_dict)
                    
                    # Store anomaly score
                    whale_scores[i] = result['score']
                
                # Fill in earlier candles
                if start_idx > 0:
                    whale_scores[:start_idx] = whale_scores[start_idx]
            else:
                # Fallback to JIT-optimized component-based approach
                volume_anomalies = self._detect_volume_anomalies(dataframe, period)
                price_impact = self._calculate_price_impact_impl(arrays['close'], arrays['volume'], period)
                
                # Temporal patterns (classical)
                temporal_patterns = np.zeros(len(dataframe))
                if self.qerc is not None and not self.use_classical:
                    # QERC-based temporal patterns (not JIT-compatible)
                    qerc_result = self._detect_temporal_patterns(dataframe)
                    temporal_patterns = qerc_result.values if isinstance(qerc_result, pd.Series) else np.zeros(len(dataframe))
                else:
                    # Classical temporal patterns (could be JIT-compiled too)
                    classical_result = self._detect_temporal_patterns_classical(dataframe, period)
                    temporal_patterns = classical_result.values if isinstance(classical_result, pd.Series) else np.zeros(len(dataframe))
                
                # Orderbook imbalance
                if 'bid_volume' in dataframe.columns and 'ask_volume' in dataframe.columns:
                    orderbook_result = self._detect_orderbook_imbalance(dataframe)
                    orderbook_imbalance = orderbook_result.values if isinstance(orderbook_result, pd.Series) else np.zeros(len(dataframe))
                else:
                    orderbook_imbalance = np.zeros(len(dataframe))
                
                # Use optimized weights if available
                if self.optimized_params is not None:
                    weights = self.optimized_params
                    volume_weight = weights.get('volume_weight', 0.4)
                    price_impact_weight = weights.get('price_impact_weight', 0.3)
                    temporal_weight = weights.get('temporal_weight', 0.2)
                    orderbook_weight = weights.get('orderbook_weight', 0.1)
                else:
                    volume_weight = 0.4
                    price_impact_weight = 0.3
                    temporal_weight = 0.2
                    orderbook_weight = 0.1
                
                # Combine components (vectorized)
                # Make sure all component arrays have the same length as dataframe
                n_rows = len(dataframe)
                # Resize/pad arrays if necessary (shouldn't be needed if calculated correctly)
                if len(volume_anomalies) != n_rows: volume_anomalies = np.pad(volume_anomalies, (n_rows - len(volume_anomalies), 0), 'edge')
                if len(price_impact) != n_rows: price_impact = np.pad(price_impact, (n_rows - len(price_impact), 0), 'edge')
                if len(temporal_patterns) != n_rows: temporal_patterns = np.pad(temporal_patterns, (n_rows - len(temporal_patterns), 0), 'edge')
                if len(orderbook_imbalance) != n_rows: orderbook_imbalance = np.pad(orderbook_imbalance, (n_rows - len(orderbook_imbalance), 0), 'edge')

                for i in range(n_rows):
                    # Ensure index is valid before accessing
                    if i < len(volume_anomalies) and i < len(price_impact) and \
                       i < len(temporal_patterns) and i < len(orderbook_imbalance):
                           whale_scores[i] = (
                               volume_weight * volume_anomalies[i] +        # Use [i]
                               price_impact_weight * price_impact[i] +      # Use [i]
                               temporal_weight * temporal_patterns[i] +     # Use [i]
                               orderbook_weight * orderbook_imbalance[i]    # Use [i]
                           )
                    else:
                         whale_scores[i] = 0.0 # Default if index out of bounds (shouldn't happen with padding)
                            
            # Apply sigmoid activation (vectorized)
            whale_scores = _vectorized_sigmoid(whale_scores, 0.5, 5)
            
            # Convert to pandas Series
            whale_activity = pd.Series(whale_scores, index=dataframe.index)
            
            # Store historical data
            self.historical_whale_activity.append(whale_activity.iloc[-1])
            if len(self.historical_whale_activity) > 100:
                self.historical_whale_activity = self.historical_whale_activity[-100:]
            
            self.last_detection_timestamp = pd.Timestamp.now()
            
            # Cache result
            self._calculation_cache[cache_key] = whale_activity
            
            # Limit cache size
            if len(self._calculation_cache) > self.cache_size:
                keys = list(self._calculation_cache.keys())
                for key in keys[:-self.cache_size]:
                    self._calculation_cache.pop(key, None)
            
            return whale_activity
            
        except Exception as e:
            self.logger.error(f"Error in whale activity detection: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _array_to_feature_dict(self, feature_array) -> Dict[str, Any]:
        """Convert feature array from JIT function to dictionary for IQAD"""
        feature_keys = ['close', 'volume', 'volume_change', 'volume_ratio', 
                       'price_change', 'price_impact', 'trend', 'momentum', 
                       'volatility', 'range_ratio']
        
        features = {}
        for i, key in enumerate(feature_keys):
            if i < len(feature_array) and not np.isnan(feature_array[i]):
                features[key] = float(feature_array[i])
        
        return features
    
    def _train_iqad_on_historical_data(self, dataframe: pd.DataFrame) -> None:
        """
        Train IQAD on historical data to establish baseline for normal patterns.
        Optimized with JIT-compiled feature extraction.
        
        Args:
            dataframe: Historical market data
        """
        if self.iqad is None or len(dataframe) < 50:
            return
        
        try:
            self.logger.info(f"Training IQAD on historical patterns for whale detection")
            
            # Extract numpy arrays for JIT operations
            arrays = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in dataframe.columns:
                    arrays[col] = dataframe[col].values
                else:
                    arrays[col] = np.ones(len(dataframe))
            
            # For whale detection, we use periods with lower volume as "normal"
            if 'volume' in dataframe.columns:
                volumes = arrays['volume']
                volume_threshold = np.percentile(volumes[volumes > 0], 70)  # Lower 70% of volumes
                
                # Find normal indices (non-volatile periods)
                normal_indices = []
                for i in range(len(dataframe)):
                    if volumes[i] <= volume_threshold and volumes[i] > 0:
                        normal_indices.append(i)
                
                # Limit to reasonable sample size
                max_samples = 200
                if len(normal_indices) > max_samples:
                    step = len(normal_indices) // max_samples
                    normal_indices = normal_indices[::step][:max_samples]
                
                # Prepare normal patterns
                normal_patterns = []
                for idx in normal_indices:
                    # Use JIT-compiled feature extraction
                    feature_array = self,_prepare_iqad_features_impl(
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
                    self.logger.info(f"IQAD trained on {len(normal_patterns)} normal patterns for whale detection")
            
        except Exception as e:
            self.logger.error(f"Error training IQAD for whale detection: {e}", exc_info=True)
    
    
    def _detect_price_impact(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Detect price impact from large trades.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Lookback period
            
        Returns:
            pd.Series: Price impact scores
        """
        try:
            # Calculate price changes
            price_change = dataframe['close'].pct_change().abs()
            
            # Calculate volume-weighted price impact
            volume = dataframe['volume']
            volume_ma = volume.rolling(window=period).mean()
            volume_ratio = volume / volume_ma.replace(0, 1)  # Avoid division by zero
            
            # Price impact = price change * volume ratio
            price_impact = price_change * volume_ratio
            
            # Normalize to 0-1 range
            # Use rolling max to create adaptive threshold
            price_impact_max = price_impact.rolling(window=period*2).max()
            normalized_impact = price_impact / price_impact_max.replace(0, 1)
            
            # Apply threshold to focus on significant price impacts
            significant_impact = (normalized_impact > self.price_impact_threshold).astype(float)
            
            # Combine raw and thresholded metrics
            combined_impact = (0.7 * normalized_impact + 0.3 * significant_impact).fillna(0.0)
            
            return combined_impact
            
        except Exception as e:
            self.logger.error(f"Error detecting price impact: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _detect_orderbook_imbalance(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect orderbook imbalance.
        
        Args:
            dataframe (pd.DataFrame): Market data with orderbook info
            
        Returns:
            pd.Series: Orderbook imbalance scores
        """
        try:
            # Calculate bid-ask imbalance if data available
            if 'bid_volume' in dataframe.columns and 'ask_volume' in dataframe.columns:
                bid_volume = dataframe['bid_volume']
                ask_volume = dataframe['ask_volume']
                
                # Avoid division by zero
                total_volume = bid_volume + ask_volume
                total_volume = total_volume.replace(0, 1)
                
                # Calculate imbalance (ranges from -1 to 1)
                imbalance = (bid_volume - ask_volume) / total_volume
                
                # Convert to 0-1 range with 0.5 as neutral
                normalized_imbalance = (imbalance + 1) / 2
                
                return normalized_imbalance
            else:
                # If orderbook data not available, use placeholder
                return pd.Series(0.5, index=dataframe.index)
                
        except Exception as e:
            self.logger.error(f"Error detecting orderbook imbalance: {str(e)}", exc_info=True)
            return pd.Series(0.5, index=dataframe.index)
    
    def _detect_temporal_patterns(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect temporal patterns using QERC.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Temporal pattern scores
        """
        try:
            # Extract features for QERC
            features = np.column_stack([
                dataframe['close'].values,
                dataframe['volume'].values,
                dataframe['high'].values,
                dataframe['low'].values
            ])
            
            # Process with QERC
            qerc_result = self.qerc.process(features)
            
            # Check if results contain error
            if qerc_result.get('error'):
                self.logger.warning(f"QERC error in whale detection: {qerc_result.get('error')}")
                return self._detect_temporal_patterns_classical(dataframe, 30)
            
            # Extract relevant signals from QERC
            trend = qerc_result.get('trend', 0.0)
            momentum = qerc_result.get('momentum', 0.0)
            volatility = qerc_result.get('volatility', 0.0)
            
            # Create pattern score with weighted QERC outputs
            pattern_score = 0.4 * abs(trend) + 0.4 * abs(momentum) + 0.2 * volatility
            
            # Create series with pattern score
            temporal_pattern = pd.Series(0.0, index=dataframe.index)
            temporal_pattern.iloc[-1] = pattern_score
            
            # Forward fill for visualization
            temporal_pattern = temporal_pattern.replace(0, np.nan).ffill().fillna(0)
            
            return temporal_pattern
            
        except Exception as e:
            self.logger.error(f"Error in QERC temporal pattern detection: {str(e)}", exc_info=True)
            return self._detect_temporal_patterns_classical(dataframe, 30)
    
    def _detect_temporal_patterns_classical(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Classical approach for temporal pattern detection.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Lookback period
            
        Returns:
            pd.Series: Temporal pattern scores
        """
        try:
            # Initialize result series
            pattern_scores = pd.Series(0.0, index=dataframe.index)
            
            # Define pattern templates
            templates = {
                'whale_entry': [0.1, 0.2, 0.4, 0.7, 1.0, 0.9, 0.8],  # Volume profile for whale entry
                'whale_exit': [0.1, 0.3, 0.6, 1.0, 0.9, 0.6, 0.3]    # Volume profile for whale exit
            }
            
            # For each template, check for pattern match
            for name, template in templates.items():
                template_len = len(template)
                
                # Ensure enough data points
                if len(dataframe) >= template_len + 5:
                    # For each possible position
                    for i in range(len(dataframe) - template_len + 1):
                        if i + template_len <= len(dataframe):
                            # Extract window
                            window = dataframe['volume'].iloc[i:i+template_len].values
                            
                            # Normalize window to 0-1 range
                            if np.max(window) > np.min(window):
                                window = (window - np.min(window)) / (np.max(window) - np.min(window))
                            else:
                                window = np.zeros_like(window)
                            
                            # Calculate correlation with template
                            correlation = np.corrcoef(window, template)[0, 1]
                            
                            # If correlation is NaN, set to 0
                            if np.isnan(correlation):
                                correlation = 0
                            
                            # If strong correlation, mark as pattern match
                            if correlation > 0.7:
                                # Set pattern score for corresponding candles
                                pattern_scores.iloc[i:i+template_len] = correlation
            
            return pattern_scores
            
        except Exception as e:
            self.logger.error(f"Error in classical temporal pattern detection: {str(e)}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _optimize_detection_parameters(self, dataframe: pd.DataFrame, initial_scores: pd.Series) -> pd.Series:
        """
        Optimize whale detection parameters using NQO.
        
        Args:
            dataframe (pd.DataFrame): Market data
            initial_scores (pd.Series): Initial whale activity scores
            
        Returns:
            pd.Series: Optimized whale activity scores
        """
        self.logger.debug("Entering NQO whale detection parameter optimization...")
        start_time_opt = time.time() # For timing this specific method
    
        # --- Pre-checks ---
        if self.nqo is None:
            self.logger.warning("NQO optimizer not available. Skipping parameter optimization.")
            return initial_scores
        if not hasattr(self, '_whale_detection_objective'):
            self.logger.error("Internal error: _whale_detection_objective method not found.")
            return initial_scores
        if dataframe.empty:
            self.logger.warning("Cannot optimize whale parameters: Input dataframe is empty.")
            return initial_scores
    
        try:
            # --- 1. Define Initial Parameters as a NumPy Array ---
            # The order MUST match the param_map in _whale_detection_objective
            # Use current values as the starting point for optimization.
            initial_optimization_params_array = np.array([
                0.4,                          # initial guess for volume_weight
                0.3,                          # initial guess for price_impact_weight
                0.2,                          # initial guess for temporal_weight
                0.1,                          # initial guess for orderbook_weight
                self.volume_threshold_factor, # current value for threshold_factor
                self.sensitivity              # current value for sensitivity
            ], dtype=float) # Ensure dtype is float
            self.logger.debug(f"Initial parameter array for NQO: {initial_optimization_params_array}")
    
            # --- 2. Prepare Data Slice for Objective Function ---
            # The objective function might only need recent data for evaluation.
            data_slice_len = 50 # Define how much recent data the objective needs
            dataframe_slice = dataframe.iloc[-data_slice_len:].copy() if len(dataframe) > data_slice_len else dataframe.copy()
    
            if dataframe_slice.empty:
                self.logger.warning("Cannot optimize whale params: Data slice is empty after slicing.")
                return initial_scores
            self.logger.debug(f"Using data slice of length {len(dataframe_slice)} for objective function.")
    
            # --- 3. Create Partial Function for Objective ---
            # Fix the 'data' argument, NQO will vary the parameters array.
            from functools import partial
            cost_function = partial(self._whale_detection_objective, data=dataframe_slice)
    
            # --- 4. Call NQO Optimizer ---
            self.logger.debug("Calling NQO optimize_parameters...")
            # NQO will iterate, calling cost_function with different params arrays
            optimized_result = self.nqo.optimize_parameters(
                initial_params=initial_optimization_params_array, # Pass the NumPy ARRAY
                objective=cost_function
                # Add iterations=X if you want to control NQO's internal iterations
            )
            self.logger.debug(f"NQO optimize_parameters result: {optimized_result}")
    
    
            # --- 5. Process Optimization Results ---
            # Check if optimization was successful and returned valid parameters
            if (optimized_result and
                'params' in optimized_result and
                isinstance(optimized_result['params'], np.ndarray) and # Verify it's an array
                optimized_result.get('error') is None):
    
                optimized_params_array = optimized_result['params'] # Get the optimized ARRAY
    
                # --- 5a. Map Optimized Array Back to Dictionary ---
                param_map = ['volume_weight', 'price_impact_weight', 'temporal_weight', 'orderbook_weight', 'threshold_factor', 'sensitivity']
                # Robustness check: Ensure array length matches map
                if len(optimized_params_array) != len(param_map):
                    self.logger.error(f"NQO returned param array of unexpected length {len(optimized_params_array)}. Expected {len(param_map)}. Using initial scores.")
                    return initial_scores
    
                optimized_params_dict = {name: optimized_params_array[i] for i, name in enumerate(param_map)}
                self.logger.info(f"NQO optimized parameters (dict): {optimized_params_dict}")
    
    
                # --- 5b. Update WhaleDetector Attributes ---
                # Update the instance attributes with the optimized values
                self.volume_threshold_factor = optimized_params_dict.get('threshold_factor', self.volume_threshold_factor)
                self.sensitivity = optimized_params_dict.get('sensitivity', self.sensitivity)
                self.logger.info(f"Updated self.volume_threshold_factor to {self.volume_threshold_factor}")
                self.logger.info(f"Updated self.sensitivity to {self.sensitivity}")
    
                # --- 5c. Recalculate Scores with Optimized Parameters ---
                # Use the FULL dataframe here to get scores for all candles
                # Pass the full dataframe to the underlying detection methods
                volume_anomalies = self._detect_volume_anomalies(dataframe, 30)
                price_impact = self._detect_price_impact(dataframe, 30)
    
                # Decide whether to use QERC or classical temporal patterns
                if self.qerc is not None and not self.use_classical:
                    temporal_patterns = self._detect_temporal_patterns(dataframe)
                else:
                    temporal_patterns = self._detect_temporal_patterns_classical(dataframe, 30)
    
                # Handle potential order book data absence
                orderbook_imbalance = self._detect_orderbook_imbalance(dataframe) if 'bid_volume' in dataframe.columns else pd.Series(0.5, index=dataframe.index)
    
                # --- 5d. Combine Scores Using Optimized Weights ---
                # Use the OPTIMIZED weights from optimized_params_dict
                # Fill potential NaNs in intermediate Series before calculation
                optimized_scores = (
                    optimized_params_dict.get('volume_weight', 0.4) * volume_anomalies.fillna(0.0) +
                    optimized_params_dict.get('price_impact_weight', 0.3) * price_impact.fillna(0.0) +
                    optimized_params_dict.get('temporal_weight', 0.2) * temporal_patterns.fillna(0.0) + # temporal_patterns might have NaNs
                    optimized_params_dict.get('orderbook_weight', 0.1) * orderbook_imbalance.fillna(0.5) # Default OB imbalance is 0.5
                )
    
                # Ensure the final combined Series doesn't have NaNs before sigmoid
                optimized_scores = optimized_scores.fillna(0.0)
    
                # Apply sigmoid activation to normalize
                # Use np.clip to avoid overflow in np.exp for very large negative values
                exp_arg = np.clip(-5 * (optimized_scores - 0.5), -700, 700)
                optimized_scores = 1 / (1 + np.exp(exp_arg))
    
                self.logger.info(f"Successfully calculated optimized whale scores. Optimization took {(time.time() - start_time_opt)*1000:.2f} ms.")
                return optimized_scores # Return the newly calculated Series
    
            else:
                # If optimization failed or returned error
                self.logger.warning(f"NQO optimization failed or returned unusable result: {optimized_result}. Using initial scores.")
                return initial_scores # Fallback to the scores calculated before optimization
    
        except Exception as e:
            # Catch any unexpected errors within this method
            self.logger.error(f"Error during whale detection optimization process: {str(e)}", exc_info=True)
            return initial_scores # Fallback to initial scores on error
    
    def _whale_detection_objective(self, params_array: np.ndarray, data: pd.DataFrame) -> float:
        """
        Objective function for whale detection optimization.
        
        Args:
            params (np.ndarray): Detection parameters
            data (pd.DataFrame): Market data
            
        Returns:
            float: Objective value (higher is better)
        """
        try:
            # Define the order matching the array passed during optimization
            param_map = ['volume_weight', 'price_impact_weight', 'temporal_weight', 'orderbook_weight', 'threshold_factor', 'sensitivity']
            # Ensure params_array has the expected length
            if len(params_array) != len(param_map):
                self.logger.error(f"Objective function received param array of unexpected length {len(params_array)}. Expected {len(param_map)}.")
                return 1e6 # Return a large value to penalize
            
            params = {name: params_array[i] for i, name in enumerate(param_map)}
            
            # Extract parameters
            volume_weight = params.get('volume_weight', 0.4)
            price_impact_weight = params.get('price_impact_weight', 0.3)
            temporal_weight = params.get('temporal_weight', 0.2)
            orderbook_weight = params.get('orderbook_weight', 0.1)
            threshold_factor = params.get('threshold_factor', 2.5)
            sensitivity = params.get('sensitivity', 0.75)
            
            # Temporarily set parameters
            old_threshold = self.volume_threshold_factor
            old_sensitivity = self.sensitivity
            
            self.volume_threshold_factor = threshold_factor
            self.sensitivity = sensitivity
            
            # Calculate detection metrics
            volume_anomalies = self._detect_volume_anomalies(data, 30)
            price_impact = self._detect_price_impact(data, 30)
            temporal_patterns = self._detect_temporal_patterns_classical(data, 30)
            
            # If orderbook data available
            if 'bid_volume' in data.columns and 'ask_volume' in data.columns:
                orderbook_imbalance = self._detect_orderbook_imbalance(data)
            else:
                orderbook_imbalance = pd.Series(0.5, index=data.index)
            
            # Combined detection
            whale_activity = (
                volume_weight * volume_anomalies +
                price_impact_weight * price_impact +
                temporal_weight * temporal_patterns +
                orderbook_weight * orderbook_imbalance
            )
            
            # Apply sigmoid activation
            whale_activity = 1 / (1 + np.exp(-5 * (whale_activity - 0.5)))
            
            # Calculate objective value
            # 1. Correlation with volume spikes
            volume_corr = np.corrcoef(whale_activity, volume_anomalies)[0, 1]
            if np.isnan(volume_corr):
                volume_corr = 0
            
            # 2. Correlation with price movements
            price_change = data['close'].pct_change().abs()
            price_corr = np.corrcoef(whale_activity, price_change)[0, 1]
            if np.isnan(price_corr):
                price_corr = 0
            
            # 3. Signal clarity (avoid noisy signals)
            signal_clarity = 1.0 - np.std(np.diff(whale_activity))
            if np.isnan(signal_clarity):
                signal_clarity = 0
            
            # Combine metrics
            objective_value = 0.4 * volume_corr + 0.4 * price_corr + 0.2 * signal_clarity
            
            # Restore original parameters
            self.volume_threshold_factor = old_threshold
            self.sensitivity = old_sensitivity
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Error in whale detection objective function: {str(e)}", exc_info=True)
            return 0.0
    
    def detect_whale_direction(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect direction of whale activity (buy or sell).
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Whale direction (-1 to 1, positive=buy, negative=sell)
        """
        try:
            direction = pd.Series(0, index=dataframe.index)
            
            # Check for sufficient data
            if len(dataframe) < 10:
                return direction
                
            # Look for price direction during volume spikes
            # Get volume ratio
            volume = dataframe['volume']
            volume_ma = volume.rolling(window=20).mean()
            volume_ratio = volume / volume_ma.replace(0, 1)  # Avoid division by zero
            
            # Get price changes
            price_change = dataframe['close'].pct_change()
            
            # Identify candles with volume spikes
            volume_spikes = volume_ratio > self.volume_threshold_factor
            
            # For each spike, determine direction
            for i in range(len(dataframe)):
                if volume_spikes.iloc[i]:
                    # Look at price movement during spike
                    if price_change.iloc[i] > 0:
                        direction.iloc[i] = 1  # Buy direction
                    elif price_change.iloc[i] < 0:
                        direction.iloc[i] = -1  # Sell direction
            
            # Smooth direction signal
            direction = direction.rolling(window=3).mean().fillna(0)
            
            return direction
            
        except Exception as e:
            self.logger.error(f"Error in whale direction detection: {str(e)}", exc_info=True)
            return pd.Series(0, index=dataframe.index)
    
    def get_whale_confidence(self, whale_activity: float) -> float:
        """
        Calculate confidence score for whale activity.
        
        Args:
            whale_activity (float): Whale activity score
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Base confidence on activity level
            base_confidence = whale_activity
            
            # Adjust based on historical detections
            if len(self.historical_whale_activity) > 10:
                # If current activity is higher than historical average
                historical_mean = np.mean(self.historical_whale_activity)
                if whale_activity > historical_mean:
                    # Increase confidence
                    confidence_boost = (whale_activity - historical_mean) / (1 - historical_mean)
                    adjusted_confidence = base_confidence * (1 + 0.5 * confidence_boost)
                else:
                    # Decrease confidence
                    adjusted_confidence = base_confidence * 0.8
                    
                # Ensure confidence is in 0-1 range
                confidence = np.clip(adjusted_confidence, 0, 1)
            else:
                confidence = base_confidence
                
            # Track confidence scores
            self.whale_confidence_scores.append(confidence)
            if len(self.whale_confidence_scores) > 100:
                self.whale_confidence_scores = self.whale_confidence_scores[-100:]
                
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating whale confidence: {str(e)}", exc_info=True)
            return whale_activity  # Return original score on error
        

    def detect(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect whale activity in market data.
        This method is a wrapper around detect_whale_activity
        to maintain consistency with other detector classes.
        
        Args:
            dataframe (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Binary indicator of whale activity
        """
        # Calculate whale activity scores
        activity_scores = self.detect_whale_activity(dataframe)
        
        # Convert to binary signal
        threshold = 0.6  # Threshold for binary detection
        binary_signal = (activity_scores > threshold).astype(int)
        
        return binary_signal
    
    def detect_signals(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect whale signals from price and volume arrays for CDFA server compatibility.
        
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
            
            # Calculate whale activity scores
            activity_scores = self.detect_whale_activity(df)
            
            if len(activity_scores) > 0:
                # Get the latest signal
                latest_score = float(activity_scores.iloc[-1])
                
                # Calculate confidence based on consistency of recent signals
                if len(activity_scores) > 5:
                    recent_scores = activity_scores.tail(5)
                    confidence = float(1.0 - np.std(recent_scores))
                else:
                    confidence = 0.5
                
                return {
                    "signal": latest_score,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "detected": latest_score > 0.6,
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
        self.logger.info("Running periodic parameter optimization with NQO")
        
        try:
            # Define initial parameters as a NumPy array
            initial_optimization_params_array = np.array([
                0.4,                          # volume_weight
                0.3,                          # price_impact_weight
                0.2,                          # temporal_weight
                0.1,                          # orderbook_weight
                self.volume_threshold_factor, # threshold_factor
                self.sensitivity              # sensitivity
            ], dtype=float)
            
            # Prepare data slice for optimization
            data_slice_len = min(200, len(dataframe))
            dataframe_slice = dataframe.iloc[-data_slice_len:].copy()
            
            # Create partial function for objective
            from functools import partial
            cost_function = partial(self._whale_detection_objective, data=dataframe_slice)
            
            # Run optimization with NQO
            optimized_result = self.nqo.optimize_parameters(
                initial_params=initial_optimization_params_array,
                objective=cost_function,
                iterations=10  # Increased iterations for better results
            )
            
            # Process optimization results
            if (optimized_result and
                'params' in optimized_result and
                isinstance(optimized_result['params'], np.ndarray) and
                optimized_result.get('error') is None):
                
                optimized_params_array = optimized_result['params']
                
                # Map parameters
                param_map = ['volume_weight', 'price_impact_weight', 'temporal_weight', 'orderbook_weight', 'threshold_factor', 'sensitivity']
                optimized_params_dict = {name: optimized_params_array[i] for i, name in enumerate(param_map)}
                
                # Store optimized parameters
                self.optimized_params = optimized_params_dict
                
                # Update instance attributes
                self.volume_threshold_factor = optimized_params_dict.get('threshold_factor', self.volume_threshold_factor)
                self.sensitivity = optimized_params_dict.get('sensitivity', self.sensitivity)
                
                self.logger.info(f"Optimization complete. New parameters: volume_threshold={self.volume_threshold_factor:.4f}, sensitivity={self.sensitivity:.4f}")
            else:
                self.logger.warning("Optimization failed or returned invalid result")
            
            # Update optimization timestamp
            self.last_optimization_time = pd.Timestamp.now()
            
            optimization_time = (time.time() - start_time) * 1000  # ms
            self.logger.info(f"Parameter optimization took {optimization_time:.2f} ms")
        
        except Exception as e:
            self.logger.error(f"Error during parameter optimization: {str(e)}", exc_info=True)