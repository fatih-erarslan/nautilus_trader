#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 03:02:59 2025

@author: ashina
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from functools import lru_cache
import time
from dataclasses import dataclass
import numba as nb
from numba import njit, float64, int64, boolean

# PennyLane Catalyst JIT support - keeping for backward compatibility
try:
    from catalyst import qjit
    CATALYST_AVAILABLE = True
except ImportError:
    # Fallback decorator when Catalyst isn't available
    def qjit(func):
        return func
    CATALYST_AVAILABLE = False


@dataclass
class SOCParameters:
    """Parameters for SOC calculations"""
    # Sample Entropy params
    sample_entropy_m: int = 2
    sample_entropy_r: float = 0.2
    sample_entropy_min_points: int = 20
    
    # Entropy Rate params
    entropy_rate_lag: int = 1
    n_bins: int = 10
    
    # SOC regime thresholds
    critical_threshold_complexity: float = 0.7
    critical_threshold_equilibrium: float = 0.3
    critical_threshold_fragility: float = 0.6
    
    stable_threshold_equilibrium: float = 0.7
    stable_threshold_fragility: float = 0.3
    stable_threshold_entropy: float = 0.6
    
    unstable_threshold_equilibrium: float = 0.3
    unstable_threshold_fragility: float = 0.7
    unstable_threshold_entropy: float = 0.7


# Replace the current Numba implementations with the original working ones

@njit(cache=True)
def _sample_entropy_impl(time_series, m=2, r=0.2):
    """
    Numba implementation of Sample Entropy calculation.
    """
    n = len(time_series)
    if n < m + 2:
        return 0.5  # Default for too short series
    
    # Normalize the series
    # Calculate mean
    mean_val = 0.0
    for i in range(n):
        mean_val += time_series[i]
    mean_val /= n
    
    # Calculate standard deviation
    sd = 0.0
    for i in range(n):
        sd += (time_series[i] - mean_val) ** 2
    
    sd = np.sqrt(sd / (n - 1)) if n > 1 else 0.0
    
    if sd < 1e-9:
        return 0.5  # Default for constant series
        
    # Set similarity threshold based on standard deviation
    r = r * sd
    
    # Initialize count arrays for templates of length m and m+1
    count_m = np.zeros(n - m + 1)
    count_m1 = np.zeros(n - m)
    
    # Calculate distances between vectors
    for i in range(n - m + 1):
        for j in range(i + 1, n - m + 1):
            # Calculate max distance between template_i and template_j
            max_dist = 0.0
            for k in range(m):
                dist = abs(time_series[i + k] - time_series[j + k])
                if dist > max_dist:
                    max_dist = dist
            
            # Count matches for templates of length m
            if max_dist < r:
                count_m[i] += 1
                count_m[j] += 1
            
            # Count matches for templates of length m+1
            if j < n - m and i < n - m and max_dist < r:
                # Check additional point for m+1 length template
                dist_plus = abs(time_series[i + m] - time_series[j + m])
                if dist_plus < r:
                    count_m1[i] += 1
                    count_m1[j] += 1
    
    # Calculate sample entropy
    sum_m = 0.0
    sum_m1 = 0.0
    
    for i in range(len(count_m)):
        sum_m += count_m[i]
    
    for i in range(len(count_m1)):
        sum_m1 += count_m1[i]
    
    if sum_m == 0 or sum_m1 == 0:
        return 2.0  # Default for no matches
    
    return -np.log(sum_m1 / sum_m)

# Instead of using the set() operation for unique values, we'll use a different approach
# that avoids the dynamic attribute access issues
@njit(cache=True)
def _entropy_rate_impl(sequence, lag=1):
    """
    Numba implementation of entropy rate calculation.
    """
    if len(sequence) <= lag:
        return 0.5  # Default for too short sequence
    
    # Get current and previous states
    current = sequence[lag:]
    previous = sequence[:-lag]
    n_pairs = len(current)
    
    # Count joint and previous occurrences using arrays instead of dictionaries
    max_state = int(np.max(sequence)) + 1
    
    # Initialize count arrays
    joint_counts = np.zeros((max_state, max_state), dtype=np.int32)
    prev_counts = np.zeros(max_state, dtype=np.int32)
    
    # Count occurrences
    for i in range(n_pairs):
        prev_key = int(previous[i])
        curr_key = int(current[i])
        
        if prev_key < max_state and curr_key < max_state:
            joint_counts[prev_key, curr_key] += 1
            prev_counts[prev_key] += 1
    
    # Calculate entropy rate
    entropy_rate = 0.0
    
    for prev in range(max_state):
        for curr in range(max_state):
            joint_count = joint_counts[prev, curr]
            if joint_count > 0:
                p_joint = joint_count / n_pairs
                p_prev = prev_counts[prev] / n_pairs
                
                if p_prev > 0:
                    entropy_rate -= p_joint * np.log2(p_joint / p_prev)
    
    # Normalize by maximum entropy - count unique values in current
    # Instead of using a set, count unique values using an array
    unique_count = 0
    unique_flags = np.zeros(max_state, dtype=np.int32)
    
    for i in range(len(current)):
        val = int(current[i])
        if val < max_state and unique_flags[val] == 0:
            unique_flags[val] = 1
            unique_count += 1
    
    max_entropy = np.log2(unique_count) if unique_count > 1 else 1.0
    
    if max_entropy > 0:
        return min(max(entropy_rate / max_entropy, 0), 1)
    
    return 0.5  # Default if max_entropy is 0

@njit(cache=True)
def _calc_soc_index_impl(returns, period, long_period):
    """
    Numba implementation of SOC index calculation.
    """
    n = len(returns)
    volatility = np.zeros(n)
    vol_mean_long = np.zeros(n)
    soc = np.zeros(n)
    
    # Calculate volatility (standard deviation of returns)
    for i in range(period, n):
        # Calculate mean for this window
        window_mean = 0.0
        for j in range(i-period, i):
            window_mean += returns[j]
        window_mean /= period
        
        # Calculate standard deviation
        sum_squared_diff = 0.0
        for j in range(i-period, i):
            sum_squared_diff += (returns[j] - window_mean) ** 2
        
        volatility[i] = np.sqrt(sum_squared_diff / period)
    
    # Calculate long-term average volatility
    for i in range(long_period, n):
        vol_sum = 0.0
        count = 0
        for j in range(i-long_period, i):
            if volatility[j] > 0:
                vol_sum += volatility[j]
                count += 1
        
        if count > 0:
            vol_mean_long[i] = vol_sum / count
    
    # Calculate SOC index
    for i in range(long_period, n):
        if vol_mean_long[i] > 1e-8:
            soc[i] = min(max(volatility[i] / vol_mean_long[i], 0), 1)
        else:
            soc[i] = 0.5
    
    return soc

@njit(cache=True)
def _calc_soc_momentum_impl(soc_values, period=10):
    """
    Numba implementation for calculating SOC momentum (rate of change of SOC metrics).
    
    Args:
        soc_values: Array of SOC index values
        period: Lookback period for momentum calculation
        
    Returns:
        Array of SOC momentum values normalized to [0, 1]
    """
    n = len(soc_values)
    momentum = np.zeros(n, dtype=np.float64)
    
    # Need at least 'period' elements to calculate momentum
    if n <= period:
        return momentum
    
    # Calculate momentum as rate of change over period
    for i in range(period, n):
        # Calculate momentum as normalized rate of change
        current = soc_values[i]
        previous = soc_values[i - period]
        
        # Protect against division by zero
        if abs(previous) > 1e-8:
            # Calculate raw momentum (percent change)
            raw_momentum = (current - previous) / previous
            
            # Store raw momentum value
            momentum[i] = raw_momentum
    
    # Normalize momentum to [0, 1] range using min-max scaling
    # First find valid range (non-zero values)
    valid_indices = np.where(momentum != 0)[0]
    if len(valid_indices) > 0:
        valid_momentum = momentum[valid_indices]
        min_mom = np.min(valid_momentum)
        max_mom = np.max(valid_momentum)
        
        # Check if we have a meaningful range
        mom_range = max_mom - min_mom
        if mom_range > 1e-8:
            # Apply normalization to values != 0
            for i in valid_indices:
                # Normalize to [0, 1] range
                momentum[i] = (momentum[i] - min_mom) / mom_range
        else:
            # If no range, set all non-zero values to 0.5 (neutral)
            for i in valid_indices:
                momentum[i] = 0.5
    
    return momentum

@njit(cache=True)
def _calc_soc_divergence_impl(equilibrium, fragility, period=5):
    """
    Numba implementation for calculating SOC divergence (difference between equilibrium and fragility metrics).
    
    Args:
        equilibrium: Array of SOC equilibrium values
        fragility: Array of SOC fragility values
        period: Smoothing period
        
    Returns:
        Array of SOC divergence values normalized to [0, 1]
    """
    n = len(equilibrium)
    if n != len(fragility):
        # Arrays must be the same length
        return np.zeros(n, dtype=np.float64)
    
    divergence = np.zeros(n, dtype=np.float64)
    
    # Calculate raw divergence as absolute difference between metrics
    for i in range(n):
        # Calculate raw difference
        raw_diff = abs(equilibrium[i] - fragility[i])
        
        # Store raw divergence
        divergence[i] = raw_diff
    
    # Apply smoothing - simple moving average
    if period > 1:
        smoothed = np.zeros(n, dtype=np.float64)
        for i in range(period - 1, n):
            sum_val = 0.0
            for j in range(i - (period - 1), i + 1):
                sum_val += divergence[j]
            smoothed[i] = sum_val / period
        divergence = smoothed
    
    # Scale to enhance visibility - divergence is typically small
    # Scale so that values around 0.2-0.3 become more visible
    for i in range(n):
        if divergence[i] > 0:
            # Apply non-linear scaling to enhance smaller differences
            # Square root scaling makes small values more visible
            divergence[i] = np.sqrt(divergence[i])
    
    # Normalize to ensure max value is 1.0
    max_div = np.max(divergence)
    if max_div > 1e-8:
        for i in range(n):
            divergence[i] = divergence[i] / max_div
    
    return divergence

class SOCAnalyzer:
    """
    Analyzer for Self-Organized Criticality patterns in market data.
    
    This class implements methods to detect critical states, phase transitions,
    and other SOC-related metrics in financial time series.
    """
    
    def __init__(self, cache_size: int = 100, use_jit: bool = True,
                 params: Optional[SOCParameters] = None,
                 log_level: str = "INFO"):
        """
        Initialize the SOC analyzer.
        
        Args:
            cache_size: Size of the LRU cache for expensive calculations
            use_jit: Whether to use JIT compilation
            params: Optional custom parameters for SOC calculations
            log_level: Logging level (default: INFO)
        """
        # Setup logging
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
            
        # Configuration
        self.cache_size = cache_size
        self.use_jit = use_jit and CATALYST_AVAILABLE
        self.params = params or SOCParameters()
        
        # Initialize computation cache
        self._calculation_cache = {}
        
        # Apply caching
        self._setup_cached_methods()
        
        self.logger.info(f"Initialized SOCAnalyzer (JIT: {self.use_jit}, Cache: {self.cache_size})")
    
    def _setup_cached_methods(self):
        """Setup method caching using LRU cache decorator"""
        # Apply caching to expensive calculations
        self._cached_calc_soc_index = lru_cache(maxsize=self.cache_size)(self._calc_soc_index)
        
        # Keep backward compatibility while using Numba internally
        self._jit_sample_entropy = self._sample_entropy
        self._jit_entropy_rate = self._entropy_rate
    
    def _sample_entropy(self, time_series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Sample Entropy (SampEn) of a time series.
        
        SampEn measures complexity/predictability and is more robust for short time series.
        Lower values indicate more regularity, higher values indicate more complexity.
        
        Args:
            time_series: Input time series data
            m: Template length (embedding dimension)
            r: Similarity threshold (typically 0.1-0.2 times standard deviation)
            
        Returns:
            Sample entropy value
        """
        n = len(time_series)
        if n < m + 2:
            return 0.5  # Default for too short series
        
        try:
            # Use Numba implementation
            return _sample_entropy_impl(time_series, m, r)
        except Exception as e:
            self.logger.warning(f"Numba sample entropy calculation failed: {e}, falling back to Python")
            
            # Fallback to Python implementation
            # Normalize the series
            sd = np.std(time_series)
            if sd < 1e-9:
                return 0.5  # Default for constant series
                
            # Set similarity threshold based on standard deviation
            r = r * sd
            
            # Initialize count arrays for templates of length m and m+1
            count_m = np.zeros(n - m + 1)
            count_m1 = np.zeros(n - m)
            
            # Calculate distances between vectors
            for i in range(n - m + 1):
                template_i = time_series[i:i+m]
                
                # Calculate distances to other templates
                for j in range(i+1, n - m + 1):
                    template_j = time_series[j:j+m]
                    dist = np.max(np.abs(template_i - template_j))
                    
                    # Count matches for templates of length m
                    if dist < r:
                        count_m[i] += 1
                        count_m[j] += 1
                    
                    # Count matches for templates of length m+1
                    if j < n - m and i < n - m and dist < r:
                        dist_plus = np.abs(time_series[i+m] - time_series[j+m])
                        if dist_plus < r:
                            count_m1[i] += 1
                            count_m1[j] += 1
            
            # Calculate sample entropy
            sum_m = np.sum(count_m)
            sum_m1 = np.sum(count_m1)
            
            if sum_m == 0 or sum_m1 == 0:
                return 2.0  # Default for no matches
            
            return -np.log(sum_m1 / sum_m)
    
    def _entropy_rate(self, sequence: np.ndarray, lag: int = 1) -> float:
        """
        Calculate entropy rate of a sequence based on conditional entropy.
        
        The entropy rate measures the predictability of the next value given previous values.
        Higher values indicate more randomness/unpredictability.
        
        Args:
            sequence: Input sequence (ideally categorical/discretized)
            lag: Time lag for conditional probabilities
            
        Returns:
            Entropy rate
        """
        if len(sequence) <= lag:
            return 0.5  # Default for too short sequence
        
        try:
            # Use Numba implementation
            return _entropy_rate_impl(sequence, lag)
        except Exception as e:
            self.logger.warning(f"Numba entropy rate calculation failed: {e}, falling back to Python")
            
            # Fallback to Python implementation
            # Get current and previous states
            current = sequence[lag:]
            previous = sequence[:-lag]
            
            # Count joint and previous occurrences
            joint_counts = {}
            prev_counts = {}
            
            for prev, curr in zip(previous, current):
                # Convert to hashable types for dictionary keys
                prev_key = int(prev) if isinstance(prev, (np.integer, np.floating)) else prev
                curr_key = int(curr) if isinstance(curr, (np.integer, np.floating)) else curr
                key = (prev_key, curr_key)
                
                if key not in joint_counts:
                    joint_counts[key] = 0
                joint_counts[key] += 1
                
                if prev_key not in prev_counts:
                    prev_counts[prev_key] = 0
                prev_counts[prev_key] += 1
            
            # Calculate entropy rate
            entropy_rate = 0.0
            total = len(current)
            
            for (prev, curr), joint_count in joint_counts.items():
                p_joint = joint_count / total
                p_prev = prev_counts[prev] / total
                
                if p_prev > 0 and p_joint > 0:
                    entropy_rate -= p_joint * np.log2(p_joint / p_prev)
            
            # Normalize by maximum entropy
            num_unique = len(set(current))
            max_entropy = np.log2(num_unique) if num_unique > 1 else 1.0
            
            if max_entropy > 0:
                return np.clip(entropy_rate / max_entropy, 0, 1)
            
            return 0.5  # Default if max_entropy is 0
    
    def _calc_soc_index(self, data_key: str, period: int) -> pd.Series:
        """Cached SOC calculation for LRU cache."""
        # This is a placeholder implementation that returns a dummy series
        # The actual implementation would need the real dataframe
        return pd.Series(0.5)
    
    def calculate_soc_index(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate base SOC index based on volatility characteristics.
        
        Args:
            dataframe: DataFrame with price data
            period: Integer rolling window period
            
        Returns:
            Pandas Series with the base SOC index
        """
        if len(dataframe) < period * 2:
            self.logger.debug(f"Not enough data for SOC Index ({len(dataframe)} < {period*2}). Using default.")
            return pd.Series(0.5, index=dataframe.index)
        
        try:
            # Generate cache key
            cache_key = f"soc_index_{period}_{len(dataframe)}"
            
            # Try to get from cache
            if cache_key in self._calculation_cache:
                self.logger.debug(f"Using cached SOC index calculation for period {period}")
                return self._calculation_cache[cache_key]
            
            # Ensure 'close' is numeric and positive
            safe_close = dataframe['close'].replace(0, np.nan).astype(float)
            
            # Calculate returns
            returns = safe_close.pct_change().fillna(0).values
            
            # Use Numba implementation for core calculation
            try:
                long_period = period * 2
                soc_values = _calc_soc_index_impl(returns, period, long_period)
                soc = pd.Series(soc_values, index=dataframe.index)
            except Exception as e:
                self.logger.warning(f"Numba SOC index calculation failed: {e}, falling back to Python")
                
                # Calculate volatility
                volatility = pd.Series(returns).rolling(window=period).std().fillna(0)
                
                # Calculate long-term average volatility
                vol_mean_long = volatility.rolling(window=period * 2).mean()
                
                # Avoid division by zero
                vol_mean_long_safe = vol_mean_long.replace(0, np.nan).ffill().bfill().fillna(1e-8)
                
                # Calculate SOC index
                soc = (volatility / vol_mean_long_safe).fillna(0.5).clip(0, 1)
            
            # Apply smoothing
            if len(soc) > 10:
                smooth1 = soc.ewm(span=5, min_periods=3).mean()
                smooth2 = smooth1.rolling(window=3, min_periods=1).mean()
                soc = smooth2
            
            # Cache the result
            self._calculation_cache[cache_key] = soc
            
            return soc
            
        except Exception as e:
            self.logger.error(f"Error in calculate_soc_index: {e}", exc_info=True)
            return pd.Series(0.5, index=dataframe.index)
    
    def add_soc_metrics_to_dataframe(self, dataframe: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Calculate SOC metrics and add them to dataframe.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculations
            
        Returns:
            pd.DataFrame: Market data with SOC metrics added
        """
        try:
            # Create a copy to avoid modifying the original
            df = dataframe.copy()
            
            # Calculate SOC metrics
            soc_metrics = self.calculate_soc_metrics(df, period)
            
            # Add metrics to dataframe
            df['soc_index'] = soc_metrics['soc_index']
            df['soc_equilibrium'] = soc_metrics['equilibrium']
            df['soc_fragility'] = soc_metrics['fragility']
            df['soc_complexity'] = soc_metrics['complexity']
            df['soc_regime'] = soc_metrics['regime']
            df['soc_entropy'] = soc_metrics.get('entropy', pd.Series(0.5, index=df.index))
            
            return df
            
        except Exception as e:
            self.logger.error(f"SOC metrics calculation error: {str(e)}", exc_info=True)
            # Return original dataframe with default values
            df = dataframe.copy()
            df['soc_index'] = 0.5
            df['soc_equilibrium'] = 0.5
            df['soc_fragility'] = 0.5
            df['soc_complexity'] = 0.5
            df['soc_entropy'] = 0.5
            df['soc_regime'] = 'normal'
            df['soc_momentum'] = 0.0
            df['soc_divergence'] = 0.0
            return df
        
    # Alias for backward compatibility
    add_soc_metrics = add_soc_metrics_to_dataframe
    
    def calculate_soc_metrics(self, dataframe: pd.DataFrame, period: int = 50) -> Dict[str, pd.Series]:
        """
        Calculate Self-Organized Criticality metrics.
        
        Args:
            dataframe (pd.DataFrame): Market data
            period (int): Period for calculations
            
        Returns:
            Dict[str, pd.Series]: SOC metrics
        """
        try:
            # Calculate SOC index using optimized method
            soc_index = self.calculate_soc_index(dataframe, period)
            
            # Calculate returns
            returns = dataframe['close'].pct_change(1).fillna(0)
            
            # Calculate avalanche size distribution
            avalanche_sizes = self._calculate_avalanche_sizes(returns, period)
            
            # Calculate power law fit
            power_law_metrics = self._calculate_power_law_fit(avalanche_sizes, period)
            
            # Use a mix of power_law_metrics and direct calculations
            complexity = power_law_metrics['distribution_entropy']
            equilibrium = power_law_metrics['criticality_distance']
            fragility = power_law_metrics['tail_weight']
            
            # Calculate sample entropy for complexity
            window_size = min(period, 100)
            if len(dataframe) > window_size:
                try:
                    # Extract close prices for sample entropy calculation
                    close_prices = dataframe['close'].values
                    window = close_prices[-window_size:]
                    window_std = np.std(window)
                    
                    if window_std > 1e-8:
                        # Normalize the window
                        window_norm = (window - np.mean(window)) / window_std
                        
                        # Calculate sample entropy with Numba
                        sampen = self._jit_sample_entropy(
                            window_norm, 
                            self.params.sample_entropy_m,
                            self.params.sample_entropy_r
                        )
                        
                        # Normalize to 0-1 range
                        sampen_complexity = min(1.0, sampen / 3.0) if not np.isnan(sampen) else 0.5
                        
                        # Blend the complexity measures
                        complexity = 0.7 * sampen_complexity + 0.3 * complexity
                except Exception as e:
                    self.logger.warning(f"Error calculating sample entropy: {e}")
            
            # Entropy rate calculation
            entropy_series = pd.Series(0.5, index=dataframe.index)
            try:
                # Bin the returns for entropy calculation
                n_bins = self.params.n_bins
                
                if returns.nunique() > n_bins:
                    binned_returns = pd.qcut(
                        returns, n_bins, labels=False, duplicates='drop'
                    ).fillna(0)
                else:
                    # Use fewer bins if there's not enough variation
                    effective_bins = min(returns.nunique(), 3)
                    binned_returns = pd.qcut(
                        returns, effective_bins, labels=False, duplicates='drop'
                    ).fillna(0)
                
                # Calculate entropy rate
                window_size = min(period, 50)
                if len(binned_returns) > window_size:
                    binned_values = binned_returns.values[-window_size:]
                    
                    if len(set(binned_values)) > 1:
                        # Calculate entropy rate with Numba
                        ent_rate = self._jit_entropy_rate(
                            binned_values, 
                            self.params.entropy_rate_lag
                        )
                        entropy_series.iloc[-1] = ent_rate
                        
                        # Forward fill for visualization
                        entropy_series = entropy_series.replace(0, np.nan).ffill().fillna(0.5)
            except Exception as e:
                self.logger.warning(f"Error calculating entropy rate: {e}")
            
            # Determine SOC regime
            regime = self._determine_soc_regime(equilibrium, fragility)
            
            # Convert regime to series
            regime_series = pd.Series(regime, index=dataframe.index)
            
            # Make sure all series have the same length
            soc_index = soc_index.reindex(dataframe.index).fillna(0.5)
            equilibrium = pd.Series(equilibrium, index=dataframe.index).fillna(0.5)
            fragility = pd.Series(fragility, index=dataframe.index).fillna(0.5)
            complexity = pd.Series(complexity, index=dataframe.index).fillna(0.5)
            
            return {
                'soc_index': soc_index,
                'equilibrium': equilibrium,
                'fragility': fragility,
                'complexity': complexity,
                'entropy': entropy_series,
                'regime': regime_series
            }
            
        except Exception as e:
            self.logger.error(f"SOC metrics calculation error: {str(e)}", exc_info=True)
            # Return default values
            index = dataframe.index
            return {
                'soc_index': pd.Series(0.5, index=index),
                'equilibrium': pd.Series(0.5, index=index),
                'fragility': pd.Series(0.5, index=index),
                'complexity': pd.Series(0.5, index=index),
                'entropy': pd.Series(0.5, index=index),
                'regime': pd.Series('normal', index=index)
            }

    def calculate_soc_momentum(self, dataframe: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate momentum of SOC index (rate of change).
        
        Args:
            dataframe: DataFrame with SOC metrics
            period: Period for momentum calculation
            
        Returns:
            Pandas Series with SOC momentum values
        """
        try:
            # Ensure we have the SOC index
            if 'soc_index' not in dataframe.columns:
                # Try to calculate SOC index if missing
                soc_index = self.calculate_soc_index(dataframe, period)
            else:
                soc_index = dataframe['soc_index']
            
            # Use Numba implementation for calculation
            momentum_values = _calc_soc_momentum_impl(soc_index.values, period)
            
            # Create Series with dataframe index
            momentum = pd.Series(momentum_values, index=dataframe.index)
            
            # Apply light smoothing for better visualization
            momentum = momentum.rolling(window=3, min_periods=1).mean().fillna(0)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating SOC momentum: {e}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def calculate_soc_divergence(self, dataframe: pd.DataFrame, period: int = 5) -> pd.Series:
        """
        Calculate divergence between SOC equilibrium and fragility metrics.
        
        Args:
            dataframe: DataFrame with SOC metrics
            period: Smoothing period for divergence calculation
            
        Returns:
            Pandas Series with SOC divergence values
        """
        try:
            # Ensure we have the required SOC metrics
            if 'soc_equilibrium' not in dataframe.columns or 'soc_fragility' not in dataframe.columns:
                # Try to calculate SOC metrics if missing
                soc_metrics = self.calculate_soc_metrics(dataframe, period)
                equilibrium = soc_metrics.get('equilibrium', pd.Series(0.5, index=dataframe.index))
                fragility = soc_metrics.get('fragility', pd.Series(0.5, index=dataframe.index))
            else:
                equilibrium = dataframe['soc_equilibrium']
                fragility = dataframe['soc_fragility']
            
            # Use Numba implementation for calculation
            divergence_values = _calc_soc_divergence_impl(
                equilibrium.values, 
                fragility.values, 
                period
            )
            
            # Create Series with dataframe index
            divergence = pd.Series(divergence_values, index=dataframe.index)
            
            return divergence
            
        except Exception as e:
            self.logger.error(f"Error calculating SOC divergence: {e}", exc_info=True)
            return pd.Series(0.0, index=dataframe.index)
    
    def _calculate_avalanche_sizes(self, returns: pd.Series, period: int) -> pd.Series:
        """
        Calculate avalanche sizes from returns.
        
        Args:
            returns (pd.Series): Price returns
            period (int): Period for calculation
            
        Returns:
            pd.Series: Avalanche sizes
        """
        # Initialize avalanche sizes
        avalanche_sizes = pd.Series(index=returns.index, dtype=float)
        
        # Calculate avalanches
        for i in range(period, len(returns)):
            # Extract window
            window = returns.iloc[i-period:i]
            
            # Count consecutive returns with same sign
            signs = np.sign(window)
            
            # Initialize counters
            current_size = 1
            sizes = []
            
            # Calculate avalanche sizes
            for j in range(1, len(signs)):
                if signs.iloc[j] == signs.iloc[j-1]:
                    current_size += 1
                else:
                    sizes.append(current_size)
                    current_size = 1
            
            # Add last avalanche
            sizes.append(current_size)
            
            # Calculate average avalanche size
            if sizes:
                avalanche_sizes.iloc[i] = np.mean(sizes)
            else:
                avalanche_sizes.iloc[i] = 1.0
        
        # Fill NaN values
        avalanche_sizes = avalanche_sizes.fillna(1.0)
        
        return avalanche_sizes
    
    def _calculate_power_law_fit(self, avalanche_sizes: pd.Series, period: int) -> Dict[str, pd.Series]:
        """
        Calculate power law fit metrics.
        
        Args:
            avalanche_sizes (pd.Series): Avalanche sizes
            period (int): Period for calculation
            
        Returns:
            Dict[str, pd.Series]: Power law fit metrics
        """
        # Initialize metrics
        power_law_fit = pd.Series(index=avalanche_sizes.index, dtype=float)
        criticality_distance = pd.Series(index=avalanche_sizes.index, dtype=float)
        tail_weight = pd.Series(index=avalanche_sizes.index, dtype=float)
        distribution_entropy = pd.Series(index=avalanche_sizes.index, dtype=float)
        
        # Calculate metrics for each window
        for i in range(period, len(avalanche_sizes)):
            # Extract window
            window = avalanche_sizes.iloc[i-period:i].dropna()
            
            if len(window) < 10:  # Need enough data
                continue
                
            # Calculate histogram
            hist, bin_edges = np.histogram(window, bins=10, density=True)
            
            # Calculate power law fit (simplified)
            # In a true power law, log(frequency) vs log(size) is linear
            x = np.log(bin_edges[1:])
            y = np.log(hist + 1e-10)  # Add small value to avoid log(0)
            
            try:
                # Linear regression
                slope, intercept = np.polyfit(x, y, 1)
                
                # Power law exponent
                power_law_exponent = -slope
                
                # Critical state has power law exponent around 1.5
                # Distance from critical state
                criticality_dist = np.abs(power_law_exponent - 1.5) / 1.5
                
                # Tail weight (probability of large avalanches)
                tail_weight_value = hist[-1] / np.sum(hist) if np.sum(hist) > 0 else 0
                
                # Distribution entropy
                entropy = -np.sum(hist * np.log(hist + 1e-10)) / np.log(len(hist)) if np.sum(hist) > 0 else 0
                
                # Scale metrics to [0, 1]
                # Goodness of power law fit
                r_squared = 1 - np.sum((y - (intercept + slope * x)) ** 2) / np.sum((y - np.mean(y)) ** 2)
                power_law_fit.iloc[i] = max(0, min(1, r_squared))
                
                # Criticality distance (inverted and scaled)
                criticality_distance.iloc[i] = max(0, min(1, 1 - criticality_dist))
                
                # Tail weight (scaled)
                tail_weight.iloc[i] = max(0, min(1, tail_weight_value * 10))
                
                # Distribution entropy (already in [0, 1])
                distribution_entropy.iloc[i] = max(0, min(1, entropy))
                
            except Exception as e:
                self.logger.debug(f"Power law fit calculation error: {str(e)}")
                # Use default values on error
                power_law_fit.iloc[i] = 0.5
                criticality_distance.iloc[i] = 0.5
                tail_weight.iloc[i] = 0.5
                distribution_entropy.iloc[i] = 0.5
        
        # Fill NaN values
        power_law_fit = power_law_fit.fillna(0.5)
        criticality_distance = criticality_distance.fillna(0.5)
        tail_weight = tail_weight.fillna(0.5)
        distribution_entropy = distribution_entropy.fillna(0.5)
        
        return {
            'power_law_fit': power_law_fit,
            'criticality_distance': criticality_distance,
            'tail_weight': tail_weight,
            'distribution_entropy': distribution_entropy
        }
    
    def _determine_soc_regime(self, equilibrium: pd.Series, fragility: pd.Series) -> List[str]:
        """
        Determine SOC regime from equilibrium and fragility.
        
        Args:
            equilibrium (pd.Series): Equilibrium metric
            fragility (pd.Series): Fragility metric
            
        Returns:
            List[str]: SOC regime for each point
        """
        # Initialize regime list
        regime = []
        
        # Determine regime for each point
        for i in range(len(equilibrium)):
            eq = equilibrium.iloc[i]
            frag = fragility.iloc[i]
            
            if eq > 0.8 and frag > 0.8:
                # High equilibrium, high fragility = critical state
                regime.append('critical')
            elif eq > 0.7 and frag > 0.6:
                # High equilibrium, moderate fragility = near-critical
                regime.append('near_critical')
            elif eq < 0.3 and frag > 0.7:
                # Low equilibrium, high fragility = unstable
                regime.append('unstable')
            elif eq < 0.3 and frag < 0.3:
                # Low equilibrium, low fragility = stable
                regime.append('stable')
            else:
                # Default = normal
                regime.append('normal')
        
        return regime
    
    def detect_critical_transitions(self, dataframe: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Detect critical transitions (tipping points).
        
        Args:
            dataframe (pd.DataFrame): Market data
            window (int): Window for calculation
            
        Returns:
            pd.Series: Transition probability
        """
        try:
            # Calculate returns
            returns = dataframe['close'].pct_change(1).fillna(0)
            
            # Calculate variance
            rolling_var = returns.rolling(window=window).var()
            
            # Calculate autocorrelation (lag 1)
            rolling_autocorr = returns.rolling(window=window).apply(
                lambda x: pd.Series(x).autocorr(lag=1))
            
            # Early warning signals of critical transitions:
            # 1. Increasing variance
            # 2. Increasing autocorrelation
            
            # Calculate variance trend
            var_trend = rolling_var.diff(window // 2)
            
            # Calculate autocorrelation trend
            autocorr_trend = rolling_autocorr.diff(window // 2)
            
            # Combine signals (scaled to [0, 1])
            var_signal = np.clip(var_trend / rolling_var.rolling(window=window).mean(), 0, 1)
            autocorr_signal = np.clip((autocorr_trend + 1) / 2, 0, 1)
            
            # Final transition probability
            transition_prob = 0.6 * var_signal + 0.4 * autocorr_signal
            
            # Fill NaN values
            transition_prob = transition_prob.fillna(0.5)
            
            return transition_prob
            
        except Exception as e:
            self.logger.error(f"Critical transition detection error: {str(e)}", exc_info=True)
            # Return default values
            return pd.Series(0.5, index=dataframe.index)
    
    def clear_cache(self):
        """Clears internal calculation caches."""
        # Clear the main cache dict
        self._calculation_cache.clear()
        # Reset LRU caches on methods if they were applied
        if hasattr(self, '_cached_calc_soc_index') and hasattr(self._cached_calc_soc_index, 'cache_clear'):
             try: self._cached_calc_soc_index.cache_clear()
             except Exception: pass # Ignore errors if cache doesn't exist
        # Clear other caches if added
        self.logger.debug("SOCAnalyzer cache cleared.")

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Analyze price and volume data to generate SOC-based signal.
        Expected interface for CDFA server integration.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            
        Returns:
            Dict containing signal, confidence, and analysis details
        """
        try:
            # Convert arrays to DataFrame for internal processing
            df = pd.DataFrame({
                'close': prices,
                'open': prices,  # Approximation
                'high': prices,  # Approximation
                'low': prices,   # Approximation
                'volume': volumes
            })
            
            # Calculate SOC index with default period
            period = min(20, len(df) // 2) if len(df) > 10 else 10
            soc_df = self.calculate_soc_index(df, period)
            
            # Calculate critical transition probability
            transition_prob = self.detect_critical_transitions(soc_df)
            
            # Get latest values
            latest_soc = soc_df['soc_fragility'].iloc[-1] if 'soc_fragility' in soc_df.columns else 0.5
            latest_transition = transition_prob.iloc[-1] if len(transition_prob) > 0 else 0.5
            
            # Combine SOC fragility and transition probability for signal
            # High fragility + high transition probability = higher signal
            signal = (latest_soc * 0.6 + latest_transition * 0.4)
            
            # Calculate confidence based on regime clarity and data stability
            if len(soc_df) >= 10 and 'soc_regime' in soc_df.columns:
                recent_regimes = soc_df['soc_regime'].tail(10)
                # Count how many recent values are the same regime (stability)
                regime_stability = (recent_regimes == recent_regimes.iloc[-1]).mean()
                
                # Adjust confidence based on regime type
                latest_regime = recent_regimes.iloc[-1]
                regime_confidence_map = {
                    'critical': 0.9,  # High confidence in critical regime
                    'unstable': 0.8,
                    'release': 0.7,
                    'stable': 0.6,
                    'normal': 0.5
                }
                regime_confidence = regime_confidence_map.get(latest_regime, 0.5)
                
                # Combine stability and regime confidence
                confidence = (regime_stability * 0.5 + regime_confidence * 0.5)
                confidence = max(0.1, min(1.0, confidence))
            else:
                confidence = 0.5  # Default confidence for insufficient data
            
            return {
                "signal": float(signal),
                "confidence": float(confidence),
                "soc_fragility": float(latest_soc),
                "transition_probability": float(latest_transition),
                "soc_regime": str(soc_df['soc_regime'].iloc[-1]) if 'soc_regime' in soc_df.columns else 'unknown',
                "analysis_type": "soc",
                "data_points": len(prices)
            }
            
        except Exception as e:
            self.logger.error(f"Error in SOC analyze method: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "analysis_type": "soc"
            }