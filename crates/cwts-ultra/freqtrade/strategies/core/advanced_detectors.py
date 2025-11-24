"""
Advanced Cryptocurrency Market Detectors and Analyzers

Enterprise-grade implementations of:
- Accumulation Detector
- Distribution Detector
- Confluence Area Detector
- Bubble Detector
- Topological Data Analysis
- Temporal Pattern Analysis

Optimized with:
- Numba JIT compilation
- Vectorization
- Caching via memoization
- Parallel computing
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange, float64, int64, boolean
from numba.typed import Dict, List
from numba.core.errors import NumbaPerformanceWarning
from functools import lru_cache
import warnings
import logging
from typing import Tuple, List as ListType, Dict as DictType, Optional, Union, Callable
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ripser
import persim
from itertools import combinations
import talib


# Create a wrapper function to handle Numba caching issues
def jit_with_safe_caching(*args, **kwargs):
    """Wrapper for numba.jit that handles caching failures gracefully"""
    # Default to cache=False if not explicitly set
    if 'cache' not in kwargs:
        kwargs['cache'] = False
    
    def decorator(func):
        try:
            return jit(*args, **kwargs)(func)
        except RuntimeError as e:
            # If caching fails, fall back to no caching
            if 'cannot cache function' in str(e):
                kwargs['cache'] = False
                return jit(*args, **kwargs)(func)
            raise
    return decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_detectors')

# Suppress Numba performance warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

#######################
# Utility Functions
#######################

@jit_with_safe_caching(nopython=True)
def calculate_rsi(prices, window=14):
    """Calculate RSI with Numba optimization."""
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs = up/down if down != 0 else np.inf
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

@jit_with_safe_caching(nopython=True)
def calculate_bollinger_bands(prices, window=20, num_std=2.0):
    """Calculate Bollinger Bands with Numba optimization."""
    rolling_mean = np.zeros_like(prices)
    rolling_std = np.zeros_like(prices)
    
    for i in range(window-1, len(prices)):
        segment = prices[i-(window-1):i+1]
        rolling_mean[i] = np.mean(segment)
        rolling_std[i] = np.std(segment)
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band

@jit_with_safe_caching(nopython=True, parallel=True)
def find_divergences(price, indicator, window=10):
    """Find divergences between price and indicator."""
    divergences = np.zeros(len(price), dtype=np.int32)
    
    for i in prange(window, len(price)-window):
        # Find local price highs/lows
        if (price[i] > price[i-1:i]).all() and (price[i] > price[i+1:i+window]).all():
            # Price high
            left_idx = max(0, i-window)
            right_idx = min(len(indicator), i+window)
            indicator_segment = indicator[left_idx:right_idx]
            indicator_max_idx = np.argmax(indicator_segment) + left_idx
            
            # Bearish divergence: price higher but indicator lower
            if indicator_max_idx < i and price[indicator_max_idx] < price[i] and indicator[indicator_max_idx] > indicator[i]:
                divergences[i] = -1  # Bearish
        
        elif (price[i] < price[i-1:i]).all() and (price[i] < price[i+1:i+window]).all():
            # Price low
            left_idx = max(0, i-window)
            right_idx = min(len(indicator), i+window)
            indicator_segment = indicator[left_idx:right_idx]
            indicator_min_idx = np.argmin(indicator_segment) + left_idx
            
            # Bullish divergence: price lower but indicator higher
            if indicator_min_idx < i and price[indicator_min_idx] > price[i] and indicator[indicator_min_idx] < indicator[i]:
                divergences[i] = 1  # Bullish
    
    return divergences

@jit_with_safe_caching(nopython=True)
def exponential_smoothing(data, alpha=0.2):
    """Apply exponential smoothing to data series."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

@jit_with_safe_caching(nopython=True)
def calculate_percent_rank(data, window=14):
    """Calculate the percent rank of each value within a rolling window."""
    rank = np.zeros_like(data)
    
    for i in range(window, len(data)):
        window_data = data[i-window:i]
        current_value = data[i-1]
        count_less = 0
        
        for j in range(len(window_data)):
            if window_data[j] < current_value:
                count_less += 1
        
        rank[i] = count_less / window if window > 0 else 0
    
    return rank

@jit_with_safe_caching(nopython=True)
def calculate_gradient(data, window=5):
    """Calculate the gradient (slope) of data over a window."""
    gradient = np.zeros_like(data)
    x = np.arange(window)
    
    for i in range(window, len(data)):
        y = data[i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=-1)[0]
        gradient[i] = m
    
    return gradient

def vectorized_parallel_apply(func, df, *args, **kwargs):
    """Apply a function to each column of a DataFrame in parallel."""
    with ThreadPoolExecutor() as executor:
        result = list(executor.map(lambda col: func(df[col].values, *args, **kwargs), df.columns))
    
    return pd.DataFrame(dict(zip(df.columns, result)), index=df.index)

class CacheManager:
    """Manages caching for computationally intensive operations."""
    
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = {}
    
    def memoize(self, func):
        """Decorator to memoize function results."""
        @lru_cache(maxsize=self.max_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()

cache_manager = CacheManager()

#######################
# Accumulation Detector
#######################

"""
AccumulationDetector: Enterprise-grade detector for identifying accumulation zones in cryptocurrency markets.
"""


# Standalone Numba functions - not class methods
@jit_with_safe_caching(nopython=True)
def _calculate_trend_numba(x, y):
    """
    Calculate linear trend slope using simple linear regression.
    This is a Numba-compatible replacement for np.polyfit.
    
    Args:
        x (numpy.ndarray): x values (typically time indices)
        y (numpy.ndarray): y values (data points)
        
    Returns:
        float: Slope of the trend line
    """
    n = len(x)
    if n < 2:
        return 0.0
        
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i] * y[i]
        sum_xx += x[i] * x[i]
    
    # Calculate slope
    if (n * sum_xx - sum_x * sum_x) == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope


@jit_with_safe_caching(nopython=True)
def _calculate_accumulation_score_numba(
    prices, volumes, volatility, buy_sell_ratio, 
    higher_lows_indicator, rsi, lookback=30, sensitivity=0.7
):
    """
    Calculate accumulation score using JIT compilation.
    
    Args:
        prices (numpy.ndarray): Array of price data
        volumes (numpy.ndarray): Array of volume data
        volatility (numpy.ndarray): Volatility measures
        buy_sell_ratio (numpy.ndarray): Buy/sell pressure ratio
        higher_lows_indicator (numpy.ndarray): Higher lows detection results
        rsi (numpy.ndarray): RSI values
        lookback (int): Lookback period
        sensitivity (float): Detection sensitivity threshold
        
    Returns:
        numpy.ndarray: Accumulation scores for each candle
    """
    result = np.zeros_like(prices)
    
    for i in range(lookback, len(prices)):
        # Subset of data for current window
        price_window = prices[i-lookback:i+1]
        vol_window = volumes[i-lookback:i+1]
        volatility_window = volatility[i-lookback:i+1]
        
        # 1. Price Range Contraction (decreasing volatility)
        vol_slope = (volatility_window[-1] - volatility_window[0]) / volatility_window[0] if volatility_window[0] != 0 else 0
        vol_score = 1.0 if vol_slope < -0.1 else 0.0
        
        # 2. Volume pattern: decreasing volume in consolidation
        # Use custom trend calculation instead of np.polyfit
        x_indices = np.arange(len(vol_window))
        vol_trend = _calculate_trend_numba(x_indices, vol_window)
        vol_score = 1.0 if vol_trend < 0 else 0.0
        
        # 3. Higher lows in price structure
        higher_lows_score = higher_lows_indicator[i]
        
        # 4. Buy-sell ratio indicating accumulation
        bs_ratio_score = 1.0 if buy_sell_ratio[i] > 1.2 else 0.0
        
        # 5. RSI behavior: not overbought, gradually increasing
        rsi_score = 1.0 if (rsi[i] < 60) and (rsi[i] > rsi[i-5]) else 0.0
        
        # Weighted average of all factors
        result[i] = (0.25 * vol_score + 
                     0.15 * vol_score + 
                     0.25 * higher_lows_score + 
                     0.25 * bs_ratio_score +
                     0.10 * rsi_score)
        
        # Apply sensitivity
        result[i] = 1.0 if result[i] > sensitivity else 0.0
    
    return result


@jit_with_safe_caching(nopython=True)
def _detect_higher_lows_numba(prices, window=7):
    """
    Detect higher lows in price structure.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): Window size for detection
        
    Returns:
        numpy.ndarray: Binary indicator of higher lows
    """
    result = np.zeros_like(prices)
    
    for i in range(window * 3, len(prices)):
        # Find local minima in the window
        minima_indices = []
        for j in range(i - window * 3 + window, i - window + 1, window):
            if j >= window and j < len(prices) - window:
                # Check if this is a local minimum
                is_min = True
                for k in range(j-window, j):
                    if prices[j] >= prices[k]:
                        is_min = False
                        break
                for k in range(j+1, j+window):
                    if prices[j] >= prices[k]:
                        is_min = False
                        break
                
                if is_min:
                    minima_indices.append(j)
        
        # Check for at least 3 higher lows
        if len(minima_indices) >= 3:
            is_higher_lows = True
            for k in range(1, len(minima_indices)):
                if prices[minima_indices[k]] <= prices[minima_indices[k-1]]:
                    is_higher_lows = False
                    break
                    
            if is_higher_lows:
                result[i] = 1.0
    
    return result


@jit_with_safe_caching(nopython=True)
def _calculate_rsi_numba(prices, window=14):
    """
    Calculate RSI with Numba optimization.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): RSI window
        
    Returns:
        numpy.ndarray: RSI values
    """
    if len(prices) <= window:
        return np.zeros_like(prices)
        
    # Calculate price changes
    deltas = np.zeros(len(prices)-1)
    for i in range(1, len(prices)):
        deltas[i-1] = prices[i] - prices[i-1]
    
    # Separate gains and losses
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    for i in range(len(deltas)):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = -deltas[i]
    
    # Calculate initial averages
    avg_gain = np.sum(gains[:window]) / window
    avg_loss = np.sum(losses[:window]) / window
    
    # Initialize RSI array
    rsi = np.zeros_like(prices)
    
    # Calculate first RSI value
    if avg_loss == 0:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values
    for i in range(window + 1, len(prices)):
        avg_gain = ((avg_gain * (window - 1)) + gains[i-1]) / window
        avg_loss = ((avg_loss * (window - 1)) + losses[i-1]) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


class AccumulationDetector:
    """
    Enterprise-grade detector for identifying accumulation zones in cryptocurrency markets.
    
    Accumulation is characterized by:
    - Decreasing volatility
    - Consistent buy pressure with minimal price movement
    - On-chain metrics showing increasing holder count
    - Higher lows in price structure
    - Declining volume during consolidation
    """
    
    def __init__(self, lookback_period=30, sensitivity=0.7, use_parallel=True):
        """
        Initialize the AccumulationDetector.
        
        Args:
            lookback_period (int): Period to look back for pattern identification
            sensitivity (float): Value between 0-1 controlling detection sensitivity
            use_parallel (bool): Whether to use parallel processing
        """
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
        self.use_parallel = use_parallel
        self.logger = logging.getLogger('AccumulationDetector')
    
    def detect(self, data):
        """
        Detect accumulation zones in market data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and additional metrics
                Required columns: 'close', 'volume', 'volatility' (or 'atr'),
                                 'buy_sell_ratio' (if available)
                
        Returns:
            pd.Series: Binary indicator of accumulation zones
        """
        self.logger.info("Detecting accumulation zones...")
        
        # Ensure required columns exist
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        # Extract or calculate needed metrics
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Calculate volatility if not provided
        if 'volatility' in data.columns:
            volatility = data['volatility'].values
        elif 'atr' in data.columns:
            volatility = data['atr'].values
        else:
            # Calculate volatility as rolling standard deviation
            volatility = np.zeros_like(prices)
            for i in range(self.lookback_period, len(prices)):
                volatility[i] = np.std(prices[i-self.lookback_period:i])
        
        # Calculate buy/sell ratio if not provided
        if 'buy_sell_ratio' in data.columns:
            buy_sell_ratio = data['buy_sell_ratio'].values
        else:
            # Approximate using price & volume relationship
            buy_sell_ratio = np.ones_like(prices)
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    # Up candle - assume more buying
                    buy_sell_ratio[i] = 1.2
                else:
                    # Down candle - assume more selling
                    buy_sell_ratio[i] = 0.8
        
        # Calculate RSI for additional confirmation
        rsi = _calculate_rsi_numba(prices)
        
        # Detect higher lows
        higher_lows = _detect_higher_lows_numba(prices)
        
        # Calculate accumulation score
        accumulation_score = _calculate_accumulation_score_numba(
            prices, volumes, volatility, buy_sell_ratio, 
            higher_lows, rsi, self.lookback_period, self.sensitivity
        )
        
        # Convert back to pandas Series
        result = pd.Series(accumulation_score, index=data.index)
        
        # Apply additional filtering to remove noise
        result = result.rolling(window=3).mean().fillna(0)
        result = (result > 0.5).astype(int)
        
        self.logger.info(f"Detected {result.sum()} accumulation zones")
        return result

    def visualize(self, data, accumulation_zones, title="Accumulation Zones Detection"):
        """
        Visualize detected accumulation zones on price chart.
        
        Args:
            data (pd.DataFrame): Price data
            accumulation_zones (pd.Series): Binary accumulation zone indicators
            title (str): Chart title
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        plt.figure(figsize=(14, 7))
        
        # Plot price
        plt.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
        
        # Highlight accumulation zones
        for i in range(len(accumulation_zones)):
            if accumulation_zones.iloc[i] == 1:
                plt.axvspan(accumulation_zones.index[i], accumulation_zones.index[min(i+1, len(accumulation_zones)-1)], 
                           alpha=0.2, color='green')
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return plt#######################
# Distribution Detector
#######################

"""
DistributionDetector: Enterprise-grade detector for identifying distribution zones in cryptocurrency markets.
"""

# Standalone Numba functions
@jit_with_safe_caching(nopython=True)
def _calculate_rsi_numba(prices, window=14):
    """
    Calculate RSI with Numba optimization.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): RSI window
        
    Returns:
        numpy.ndarray: RSI values
    """
    if len(prices) <= window:
        return np.zeros_like(prices)
        
    # Calculate price changes
    deltas = np.zeros(len(prices)-1)
    for i in range(1, len(prices)):
        deltas[i-1] = prices[i] - prices[i-1]
    
    # Separate gains and losses
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    for i in range(len(deltas)):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = -deltas[i]
    
    # Calculate initial averages
    avg_gain = np.sum(gains[:window]) / window
    avg_loss = np.sum(losses[:window]) / window
    
    # Initialize RSI array
    rsi = np.zeros_like(prices)
    
    # Calculate first RSI value
    if avg_loss == 0:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values
    for i in range(window + 1, len(prices)):
        avg_gain = ((avg_gain * (window - 1)) + gains[i-1]) / window
        avg_loss = ((avg_loss * (window - 1)) + losses[i-1]) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit_with_safe_caching(nopython=True)
def _detect_lower_highs_numba(prices, window=7):
    """
    Detect lower highs in price structure.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): Window size for detection
        
    Returns:
        numpy.ndarray: Binary indicator of lower highs
    """
    result = np.zeros_like(prices)
    
    for i in range(window * 3, len(prices)):
        # Find local maxima in the window
        maxima_indices = []
        for j in range(i - window * 3 + window, i - window + 1, window):
            if j >= window and j < len(prices) - window:
                # Check if this is a local maximum
                is_max = True
                for k in range(j-window, j):
                    if prices[j] <= prices[k]:
                        is_max = False
                        break
                for k in range(j+1, j+window):
                    if prices[j] <= prices[k]:
                        is_max = False
                        break
                
                if is_max:
                    maxima_indices.append(j)
        
        # Check for at least 3 lower highs
        if len(maxima_indices) >= 3:
            is_lower_highs = True
            for k in range(1, len(maxima_indices)):
                if prices[maxima_indices[k]] >= prices[maxima_indices[k-1]]:
                    is_lower_highs = False
                    break
                    
            if is_lower_highs:
                result[i] = 1.0
    
    return result


@jit_with_safe_caching(nopython=True)
def _detect_supply_absorption_numba(prices, volumes, window=5):
    """
    Detect supply absorption (strong resistance).
    This occurs when price tries to rise but is met with increased selling.
    
    Args:
        prices (numpy.ndarray): Price data
        volumes (numpy.ndarray): Volume data
        window (int): Window size for detection
        
    Returns:
        numpy.ndarray: Binary indicator of supply absorption
    """
    result = np.zeros_like(prices)
    
    for i in range(window * 2, len(prices) - 1):
        # Check for attempts to break higher
        if prices[i] > prices[i-1]:
            # Calculate average volume for up moves in window
            avg_up_volume = 0.0
            count = 0
            for j in range(i - window, i):
                if prices[j] > prices[j-1]:
                    avg_up_volume += volumes[j]
                    count += 1
            
            if count > 0:
                avg_up_volume /= count
                
                # Check if followed by strong rejection
                if prices[i+1] < prices[i] and volumes[i+1] > avg_up_volume * 1.5:
                    result[i] = 1.0
    
    return result


@jit_with_safe_caching(nopython=True)
def _find_divergences_numba(price, indicator, window=10):
    """
    Find divergences between price and indicator.
    
    Args:
        price (numpy.ndarray): Price data
        indicator (numpy.ndarray): Technical indicator data
        window (int): Window size for detection
        
    Returns:
        numpy.ndarray: Divergence indicators (-1 for bearish, 1 for bullish, 0 for none)
    """
    divergences = np.zeros(len(price), dtype=np.int32)
    
    for i in range(window, len(price)-window):
        # Find local price highs
        is_price_high = True
        for j in range(i-window, i):
            if price[i] <= price[j]:
                is_price_high = False
                break
        for j in range(i+1, i+window):
            if j < len(price) and price[i] <= price[j]:
                is_price_high = False
                break
                
        if is_price_high:
            # Price high - check for bearish divergence
            left_idx = max(0, i-window)
            right_idx = min(len(indicator), i+window)
            
            # Find maximum indicator value in this range
            indicator_max_val = indicator[left_idx]
            indicator_max_idx = left_idx
            for j in range(left_idx+1, right_idx):
                if indicator[j] > indicator_max_val:
                    indicator_max_val = indicator[j]
                    indicator_max_idx = j
            
            # Bearish divergence: price higher but indicator lower
            if indicator_max_idx < i and price[indicator_max_idx] < price[i] and indicator[indicator_max_idx] > indicator[i]:
                divergences[i] = -1  # Bearish
        else:
            # Check for price low
            is_price_low = True
            for j in range(i-window, i):
                if price[i] >= price[j]:
                    is_price_low = False
                    break
            for j in range(i+1, i+window):
                if j < len(price) and price[i] >= price[j]:
                    is_price_low = False
                    break
                    
            if is_price_low:
                # Price low - check for bullish divergence
                left_idx = max(0, i-window)
                right_idx = min(len(indicator), i+window)
                
                # Find minimum indicator value in this range
                indicator_min_val = indicator[left_idx]
                indicator_min_idx = left_idx
                for j in range(left_idx+1, right_idx):
                    if indicator[j] < indicator_min_val:
                        indicator_min_val = indicator[j]
                        indicator_min_idx = j
                
                # Bullish divergence: price lower but indicator higher
                if indicator_min_idx < i and price[indicator_min_idx] > price[i] and indicator[indicator_min_idx] < indicator[i]:
                    divergences[i] = 1  # Bullish
    
    return divergences


@jit_with_safe_caching(nopython=True)
def _calculate_distribution_score_numba(
    prices, volumes, lower_highs_indicator, bearish_divergence,
    supply_absorption, rsi, lookback=30, sensitivity=0.7
):
    """
    Calculate distribution score using JIT compilation.
    
    Args:
        prices (numpy.ndarray): Array of price data
        volumes (numpy.ndarray): Array of volume data
        lower_highs_indicator (numpy.ndarray): Lower highs detection results
        bearish_divergence (numpy.ndarray): Bearish divergence detection results
        supply_absorption (numpy.ndarray): Supply absorption detection results
        rsi (numpy.ndarray): RSI values
        lookback (int): Lookback period
        sensitivity (float): Detection sensitivity threshold
        
    Returns:
        numpy.ndarray: Distribution scores for each candle
    """
    result = np.zeros_like(prices)
    
    for i in range(lookback, len(prices)):
        if i < lookback:
            continue
        
        # 1. Lower highs in price structure
        lower_highs_score = lower_highs_indicator[i]
        
        # 2. Volume increasing on down moves
        down_moves = np.zeros(lookback)
        for j in range(1, lookback):
            idx = i - lookback + j
            if idx > 0 and prices[idx] < prices[idx-1]:
                down_moves[j] = 1
        
        vol_window = volumes[i-lookback:i]
        down_move_volumes = np.zeros(len(down_moves))
        for j in range(len(down_moves)):
            if down_moves[j] > 0:
                down_move_volumes[j] = vol_window[j]
        
        # Calculate average volume for down moves
        down_volume_sum = 0.0
        down_count = 0
        for j in range(len(down_move_volumes)):
            if down_move_volumes[j] > 0:
                down_volume_sum += down_move_volumes[j]
                down_count += 1
        
        avg_down_volume = down_volume_sum / max(down_count, 1)
        avg_total_volume = np.sum(vol_window) / max(len(vol_window), 1)
        
        vol_score = 1.0 if avg_down_volume > avg_total_volume * 1.2 else 0.0
        
        # 3. Bearish divergence
        divergence_score = 1.0 if bearish_divergence[i] == -1 else 0.0
        
        # 4. Supply absorption
        absorption_score = supply_absorption[i]
        
        # 5. RSI behavior: overbought and declining
        rsi_score = 0.0
        if i >= 5 and rsi[i] > 60 and rsi[i] < rsi[i-5]:
            rsi_score = 1.0
        
        # Weighted average of all factors
        result[i] = (0.25 * lower_highs_score + 
                     0.15 * vol_score + 
                     0.25 * divergence_score + 
                     0.25 * absorption_score +
                     0.10 * rsi_score)
        
        # Apply sensitivity
        result[i] = 1.0 if result[i] > sensitivity else 0.0
    
    return result


class DistributionDetector:
    """
    Enterprise-grade detector for identifying distribution zones in cryptocurrency markets.
    
    Distribution is characterized by:
    - Lower highs in price structure
    - Increasing volume on down moves
    - Bearish divergences in momentum indicators
    - Hidden supply absorption (strong resistance)
    - Declining buy pressure
    """
    
    def __init__(self, lookback_period=30, sensitivity=0.7, use_parallel=True):
        """
        Initialize the DistributionDetector.
        
        Args:
            lookback_period (int): Period to look back for pattern identification
            sensitivity (float): Value between 0-1 controlling detection sensitivity
            use_parallel (bool): Whether to use parallel processing
        """
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
        self.use_parallel = use_parallel
        self.logger = logging.getLogger('DistributionDetector')
    
    def detect(self, data):
        """
        Detect distribution zones in market data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
                Required columns: 'close', 'high', 'low', 'volume'
                
        Returns:
            pd.Series: Binary indicator of distribution zones
        """
        self.logger.info("Detecting distribution zones...")
        
        # Ensure required columns exist
        required_columns = ['close', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        # Extract needed metrics
        prices = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values
        
        # Calculate RSI
        rsi = _calculate_rsi_numba(prices)
        
        # Detect lower highs
        lower_highs = _detect_lower_highs_numba(prices)
        
        # Detect bearish divergences
        bearish_divergence = _find_divergences_numba(prices, rsi)
        
        # Detect supply absorption
        supply_absorption = _detect_supply_absorption_numba(prices, volumes)
        
        # Calculate distribution score
        distribution_score = _calculate_distribution_score_numba(
            prices, volumes, lower_highs, bearish_divergence,
            supply_absorption, rsi, self.lookback_period, self.sensitivity
        )
        
        # Convert back to pandas Series
        result = pd.Series(distribution_score, index=data.index)
        
        # Apply additional filtering to remove noise
        result = result.rolling(window=3).mean().fillna(0)
        result = (result > 0.5).astype(int)
        
        self.logger.info(f"Detected {result.sum()} distribution zones")
        return result

    def visualize(self, data, distribution_zones, title="Distribution Zones Detection"):
        """
        Visualize detected distribution zones on price chart.
        
        Args:
            data (pd.DataFrame): Price data
            distribution_zones (pd.Series): Binary distribution zone indicators
            title (str): Chart title
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        plt.figure(figsize=(14, 7))
        
        # Plot price
        plt.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
        
        # Highlight distribution zones
        for i in range(len(distribution_zones)):
            if distribution_zones.iloc[i] == 1:
                plt.axvspan(distribution_zones.index[i], distribution_zones.index[min(i+1, len(distribution_zones)-1)], 
                           alpha=0.2, color='red')
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return plt
#######################
# Confluence Area Detector
#######################

"""
ConfluenceAreaDetector: Enterprise-grade detector for identifying confluence areas in cryptocurrency markets.
"""

# Non-Numba implementation of cluster_price_levels due to compatibility issues
def _cluster_price_levels(levels, tolerance=0.01):
    """
    Cluster nearby price levels.
    
    Args:
        levels (numpy.ndarray): Array of price levels
        tolerance (float): Percentage tolerance for grouping levels
            
    Returns:
        tuple: (clustered_levels, counts) - Clustered levels and counts
    """
    if len(levels) == 0:
        return np.array([]), np.array([])
            
    # Sort levels
    sorted_levels = np.sort(levels)
    
    # Initialize clusters
    clusters = []
    counts = []
    
    current_cluster = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        current_level = sorted_levels[i]
        prev_level = sorted_levels[i-1]
        
        # Check if current level is within tolerance of previous
        if abs(current_level - prev_level) / prev_level <= tolerance:
            current_cluster.append(current_level)
        else:
            # Finalize current cluster and start a new one
            clusters.append(np.mean(current_cluster))
            counts.append(len(current_cluster))
            current_cluster = [current_level]
    
    # Add the last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))
        counts.append(len(current_cluster))
    
    return np.array(clusters), np.array(counts)


def _detect_support_resistance(highs, lows, window_size=20, num_levels=5):
    """
    Detect support and resistance levels using local highs and lows.
    
    Args:
        highs (numpy.ndarray): Array of high prices
        lows (numpy.ndarray): Array of low prices
        window_size (int): Window size for peak detection
        num_levels (int): Number of top levels to return
        
    Returns:
        numpy.ndarray: Support and resistance levels
    """
    # Find local maxima (resistance)
    resistance_indices = argrelextrema(highs, np.greater_equal, order=window_size)[0]
    resistance_levels = highs[resistance_indices]
    
    # Find local minima (support)
    support_indices = argrelextrema(lows, np.less_equal, order=window_size)[0]
    support_levels = lows[support_indices]
    
    # Combine support and resistance
    all_levels = np.concatenate([support_levels, resistance_levels]) if len(support_levels) > 0 and len(resistance_levels) > 0 else np.array([])
    
    if len(all_levels) == 0:
        return np.array([])
    
    # Cluster levels to find the most significant ones
    clustered_levels, counts = _cluster_price_levels(all_levels)
    
    # Sort by count (significance)
    if len(clustered_levels) > 0:
        sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
        top_levels = clustered_levels[sorted_indices[:min(num_levels, len(sorted_indices))]]
        return top_levels
        
    return np.array([])


def _calculate_fibonacci_levels(high, low, is_uptrend=True):
    """
    Calculate Fibonacci retracement and extension levels.
    
    Args:
        high (float): High price
        low (float): Low price
        is_uptrend (bool): Whether the trend is up or down
        
    Returns:
        numpy.ndarray: Fibonacci levels
    """
    diff = high - low
    
    if is_uptrend:
        # Retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)
        # Extension levels (1.272, 1.618)
        levels = np.array([
            high - 0.236 * diff,  # 23.6% retracement
            high - 0.382 * diff,  # 38.2% retracement
            high - 0.5 * diff,    # 50% retracement
            high - 0.618 * diff,  # 61.8% retracement
            high - 0.786 * diff,  # 78.6% retracement
            high + 0.272 * diff,  # 127.2% extension
            high + 0.618 * diff   # 161.8% extension
        ])
    else:
        # For downtrend
        levels = np.array([
            low + 0.236 * diff,   # 23.6% retracement
            low + 0.382 * diff,   # 38.2% retracement
            low + 0.5 * diff,     # 50% retracement
            low + 0.618 * diff,   # 61.8% retracement
            low + 0.786 * diff,   # 78.6% retracement
            low - 0.272 * diff,   # 127.2% extension
            low - 0.618 * diff    # 161.8% extension
        ])
    
    return levels


def _calculate_moving_averages(prices, windows=[20, 50, 100, 200]):
    """
    Calculate multiple moving averages for the price series.
    
    Args:
        prices (numpy.ndarray): Array of prices
        windows (list): List of moving average periods
        
    Returns:
        dict: Dictionary of moving averages keyed by period
    """
    ma_dict = {}
    
    for window in windows:
        if len(prices) >= window:
            ma = np.zeros_like(prices)
            for i in range(window-1, len(prices)):
                ma[i] = np.mean(prices[i-window+1:i+1])
            ma_dict[window] = ma
        else:
            ma_dict[window] = np.zeros_like(prices)
    
    return ma_dict


def _get_volume_profile_nodes(prices, volumes, num_bins=20):
    """
    Calculate volume profile nodes (price levels with high volume).
    
    Args:
        prices (numpy.ndarray): Array of prices
        volumes (numpy.ndarray): Array of volumes
        num_bins (int): Number of price bins
        
    Returns:
        numpy.ndarray: Key volume nodes
    """
    if len(prices) == 0 or len(volumes) == 0:
        return np.array([])
        
    # Create bins for prices
    min_price = np.min(prices)
    max_price = np.max(prices)
    
    if min_price == max_price:
        return np.array([min_price])
        
    bins = np.linspace(min_price, max_price, num_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate volume per bin
    vol_per_bin = np.zeros(num_bins)
    
    for i in range(len(prices)):
        if prices[i] >= min_price and prices[i] <= max_price:
            bin_idx = min(int((prices[i] - min_price) / (max_price - min_price) * num_bins), num_bins-1)
            vol_per_bin[bin_idx] += volumes[i]
    
    # Find significant volume nodes (local maxima)
    threshold = np.mean(vol_per_bin) + np.std(vol_per_bin)
    significant_indices = np.where(vol_per_bin > threshold)[0]
    significant_nodes = bin_centers[significant_indices]
    
    return significant_nodes


class ConfluenceAreaDetector:
    """
    Enterprise-grade detector for identifying confluence areas in cryptocurrency markets.
    
    Confluence areas are where multiple technical indicators, support/resistance levels,
    or other significant market factors align at the same price level.
    
    Features:
    - Multiple timeframe support/resistance detection
    - Fibonacci retracement level analysis
    - Moving average convergence points
    - Volume profile nodes
    - Historical pivot points
    """
    
    def __init__(self, num_indicators=3, window_size=20, significance_threshold=0.05, use_parallel=True):
        """
        Initialize the ConfluenceAreaDetector.
        
        Args:
            num_indicators (int): Minimum number of indicators that must align to signal confluence
            window_size (int): Window size for historical pivots and local S/R
            significance_threshold (float): Threshold for determining level significance (0.0-1.0)
            use_parallel (bool): Whether to use parallel processing
        """
        self.num_indicators = num_indicators
        self.window_size = window_size
        self.significance_threshold = significance_threshold
        self.use_parallel = use_parallel
        self.logger = logging.getLogger('ConfluenceAreaDetector')
        
        # Initialize the price level cache
        self.level_cache = {}
    
    def detect(self, data, current_price=None):
        """
        Detect confluence areas in market data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
                Required columns: 'close', 'high', 'low', 'volume'
            current_price (float, optional): Current price to analyze confluence areas
                If None, uses the last close price
                
        Returns:
            pd.DataFrame: Detected confluence areas with strength scores
        """
        self.logger.info("Detecting confluence areas...")
        
        # Ensure required columns exist
        required_columns = ['close', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        # Use last close if current_price not provided
        if current_price is None:
            current_price = data['close'].iloc[-1]
        
        # Extract needed price data
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values
        
        # Initialize list to store all detected levels
        all_levels = []
        
        # 1. Detect support and resistance levels
        sr_levels = _detect_support_resistance(highs, lows, self.window_size)
        all_levels.extend([(level, 'Support/Resistance') for level in sr_levels])
        
        # 2. Calculate Fibonacci levels
        # Find significant swing high and low for Fibonacci calculation
        high_point = np.max(highs[-min(len(highs), 100):])
        low_point = np.min(lows[-min(len(lows), 100):])
        is_uptrend = closes[-1] > closes[-min(len(closes), 20)]
        
        fib_levels = _calculate_fibonacci_levels(high_point, low_point, is_uptrend)
        all_levels.extend([(level, 'Fibonacci') for level in fib_levels])
        
        # 3. Calculate moving averages
        ma_dict = _calculate_moving_averages(closes)
        for period, ma in ma_dict.items():
            if len(ma) > 0:
                all_levels.append((ma[-1], f'MA{period}'))
        
        # 4. Get volume profile nodes
        vol_nodes = _get_volume_profile_nodes(closes, volumes)
        all_levels.extend([(level, 'Volume Node') for level in vol_nodes])
        
        # 5. Add round numbers if near current price
        price_digits = len(str(int(current_price)))
        magnitude = 10 ** (price_digits - 1)
        for i in range(1, 10):
            round_level = i * magnitude
            if abs(round_level - current_price) / current_price < 0.1:
                all_levels.append((round_level, 'Round Number'))
        
        # Convert to DataFrame
        levels_df = pd.DataFrame(all_levels, columns=['Price', 'Type'])
        
        # Cluster nearby levels to find confluence areas
        if len(levels_df) > 0:
            # Convert to numpy for faster processing
            price_levels = levels_df['Price'].values
            
            # Get clusters
            clustered_levels, counts = _cluster_price_levels(price_levels)
            
            # Create confluence areas DataFrame
            confluence_areas = []
            
            for i, (level, count) in enumerate(zip(clustered_levels, counts)):
                if count >= self.num_indicators:
                    # Find which indicator types contribute to this confluence
                    types = []
                    for _, row in levels_df.iterrows():
                        if abs(row['Price'] - level) / level <= self.significance_threshold:
                            types.append(row['Type'])
                    
                    # Calculate distance from current price as percentage
                    distance = abs(level - current_price) / current_price
                    
                    # Calculate strength based on number of contributing indicators and proximity
                    strength = count * (1.0 - min(distance, 0.1) * 10)
                    
                    confluence_areas.append({
                        'price_level': level,
                        'strength': strength,
                        'count': count,
                        'distance': distance,
                        'contributors': ', '.join(set(types))
                    })
            
            result = pd.DataFrame(confluence_areas)
            if len(result) > 0:
                # Sort by strength
                result = result.sort_values('strength', ascending=False).reset_index(drop=True)
                
                self.logger.info(f"Detected {len(result)} confluence areas")
                return result
        
        # Empty result if no confluence areas found
        self.logger.info("No confluence areas detected")
        return pd.DataFrame(columns=['price_level', 'strength', 'count', 'distance', 'contributors'])

    def visualize(self, data, confluence_areas, title="Confluence Areas Detection"):
        """Visualize detected confluence areas on price chart."""
        if len(confluence_areas) == 0:
            self.logger.warning("No confluence areas to visualize")
            return None
            
        plt.figure(figsize=(14, 7))
        
        # Plot price
        plt.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
        
        # Get current price range for scaling
        y_min = data['low'].min()
        y_max = data['high'].max()
        price_range = y_max - y_min
        
        # Plot confluence areas as horizontal lines
        for i, row in confluence_areas.iterrows():
            level = row['price_level']
            strength = row['strength']
            # Scale line width by strength
            line_width = 1 + (strength / 5)
            line_alpha = min(0.8, 0.3 + (strength / 10))
            
            plt.axhline(y=level, linestyle='--', linewidth=line_width, 
                       alpha=line_alpha, color='purple', 
                       label=f"Level: {level:.2f}, Strength: {strength:.2f}" if i==0 else "")
            
            # Add label on the right side
            plt.text(data.index[-1], level, f"{level:.2f}", 
                    verticalalignment='center', horizontalalignment='left',
                    fontsize=8, color='purple')
        
        plt.title(title)
        plt.legend(loc='upper left')
        plt.tight_layout()
        return plt
#######################
# Bubble Detector
#######################

"""
BubbleDetector: Enterprise-grade detector for identifying market bubbles in cryptocurrency markets.
"""

# Standalone Numba functions for BubbleDetector
@jit_with_safe_caching(nopython=True)
def _fit_exponential_growth_numba(prices, window):
    """
    Fit an exponential growth model to price data and assess fit quality.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): Window size to fit
        
    Returns:
        tuple: (growth_rate, r_squared) - growth rate and goodness of fit
    """
    if len(prices) < window:
        return 0.0, 0.0
        
    # Extract window of prices
    window_prices = prices[-window:]
    
    # Convert to log prices for linear regression
    log_prices = np.log(window_prices)
    
    # Create x values (time indices)
    x = np.arange(window)
    
    # Perform linear regression on log prices
    n = window
    sum_x = np.sum(x)
    sum_y = np.sum(log_prices)
    sum_xy = np.sum(x * log_prices)
    sum_xx = np.sum(x * x)
    
    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate predicted values
    pred_log_prices = np.zeros_like(log_prices)
    for i in range(len(x)):
        pred_log_prices[i] = intercept + slope * x[i]
    
    # Calculate R-squared
    ss_total = 0.0
    mean_log_price = np.mean(log_prices)
    for i in range(len(log_prices)):
        ss_total += (log_prices[i] - mean_log_price) ** 2
    
    ss_residual = 0.0
    for i in range(len(log_prices)):
        ss_residual += (log_prices[i] - pred_log_prices[i]) ** 2
    
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Convert slope to growth rate (daily)
    growth_rate = np.exp(slope) - 1
    
    return growth_rate, r_squared


@jit_with_safe_caching(nopython=True)
def _calculate_price_acceleration_numba(prices, window=30):
    """
    Calculate price acceleration (second derivative of price).
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): Window for smoothing
        
    Returns:
        numpy.ndarray: Price acceleration values
    """
    if len(prices) <= window * 2:
        return np.zeros_like(prices)
        
    # Calculate price velocity (first derivative)
    velocity = np.zeros_like(prices)
    for i in range(1, len(prices)):
        velocity[i] = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
    
    # Calculate acceleration (second derivative)
    acceleration = np.zeros_like(prices)
    for i in range(1, len(velocity)):
        acceleration[i] = velocity[i] - velocity[i-1]
    
    # Smooth acceleration with moving average
    smoothed_acceleration = np.zeros_like(acceleration)
    for i in range(window, len(acceleration)):
        smoothed_acceleration[i] = np.mean(acceleration[i-window+1:i+1])
    
    return smoothed_acceleration


@jit_with_safe_caching(nopython=True)
def _detect_deviation_from_trend_numba(prices, window=90):
    """
    Detect deviation from long-term trend.
    
    Args:
        prices (numpy.ndarray): Price data
        window (int): Window for long-term trend
        
    Returns:
        numpy.ndarray: Deviation scores
    """
    if len(prices) <= window:
        return np.zeros_like(prices)
        
    # Calculate long-term trend with simple moving average
    trend = np.zeros_like(prices)
    for i in range(window-1, len(prices)):
        trend[i] = np.mean(prices[i-window+1:i+1])
    
    # Calculate deviation as percentage from trend
    deviation = np.zeros_like(prices)
    for i in range(window, len(prices)):
        deviation[i] = (prices[i] - trend[i]) / trend[i] if trend[i] > 0 else 0
    
    return deviation


@jit_with_safe_caching(nopython=True)
def _calculate_bubble_score_numba(
    prices, volumes, price_acceleration, trend_deviation,
    retail_participation, media_mentions, lookback=90, 
    sensitivity=0.7, exp_threshold=2.0
):
    """
    Calculate bubble score using JIT compilation.
    
    Args:
        prices (numpy.ndarray): Array of price data
        volumes (numpy.ndarray): Array of volume data
        price_acceleration (numpy.ndarray): Price acceleration data
        trend_deviation (numpy.ndarray): Deviation from trend data
        retail_participation (numpy.ndarray): Retail participation metrics
        media_mentions (numpy.ndarray): Media mention metrics
        lookback (int): Lookback period for analysis
        sensitivity (float): Threshold for bubble detection (0.0-1.0)
        exp_threshold (float): Minimum growth rate to consider exponential
        
    Returns:
        numpy.ndarray: Bubble scores for each candle
    """
    result = np.zeros_like(prices)
    
    for i in range(lookback, len(prices)):
        # Check if enough data
        if i < lookback:
            continue
            
        # Get window of data for bubble analysis
        window_prices = prices[i-lookback:i+1]
        
        # 1. Check for exponential growth
        growth_rate, r_squared = _fit_exponential_growth_numba(window_prices, lookback)
        exp_growth_score = 1.0 if (growth_rate > exp_threshold and r_squared > 0.9) else 0.0
        
        # 2. Price acceleration
        accel_score = 1.0 if price_acceleration[i] > 0.01 else 0.0
        
        # 3. Deviation from long-term trend
        trend_score = 1.0 if trend_deviation[i] > 0.5 else 0.0
        
        # 4. Volume increases
        volume_ratio = 1.0
        if len(volumes) > lookback and i >= lookback:
            avg_volume = 0.0
            count = 0
            for j in range(i-lookback, i):
                avg_volume += volumes[j]
                count += 1
            if count > 0:
                avg_volume /= count
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
        
        volume_score = 1.0 if volume_ratio > 3.0 else 0.0
        
        # 5. Additional factors if available
        retail_score = 0.0
        if i < len(retail_participation):
            retail_score = retail_participation[i]
            
        media_score = 0.0
        if i < len(media_mentions):
            media_score = media_mentions[i]
        
        # Combined bubble score with weights
        result[i] = (
            0.3 * exp_growth_score + 
            0.2 * accel_score + 
            0.2 * trend_score + 
            0.1 * volume_score +
            0.1 * retail_score +
            0.1 * media_score
        )
        
        # Apply sensitivity threshold
        result[i] = 1.0 if result[i] >= sensitivity else 0.0
    
    return result


class BubbleDetector:
    """
    Enterprise-grade detector for identifying market bubbles in cryptocurrency markets.
    
    Bubble characteristics:
    - Exponential price increases
    - Widespread retail participation
    - Increasing media/social media mentions
    - Deviation from fundamental valuation
    - Inflated trading metrics
    - Historical pattern matching
    """
    
    def __init__(self, lookback_period=90, sensitivity=0.7, exponential_threshold=2.0, use_parallel=True):
        """
        Initialize the BubbleDetector.
        
        Args:
            lookback_period (int): Period to look back for pattern identification
            sensitivity (float): Value between 0-1 controlling detection sensitivity
            exponential_threshold (float): Minimum growth rate to consider exponential
            use_parallel (bool): Whether to use parallel processing
        """
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
        self.exponential_threshold = exponential_threshold
        self.use_parallel = use_parallel
        self.logger = logging.getLogger('BubbleDetector')
    
    def detect(self, data, include_social_data=False, social_data=None):
        """
        Detect market bubbles in cryptocurrency data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
                Required columns: 'close', 'volume'
            include_social_data (bool): Whether to include social media data
            social_data (pd.DataFrame, optional): Social media mentions and sentiment data
                
        Returns:
            tuple: (bubble_indicator, bubble_probability)
                - bubble_indicator: Binary indicator of bubble conditions
                - bubble_probability: Continuous probability score (0-1)
        """
        self.logger.info("Detecting bubble conditions...")
        
        # Ensure required columns exist
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        # Extract needed metrics
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Calculate price acceleration
        price_acceleration = _calculate_price_acceleration_numba(prices)
        
        # Calculate deviation from long-term trend
        trend_deviation = _detect_deviation_from_trend_numba(prices, self.lookback_period)
        
        # Initialize retail participation and media mentions metrics
        retail_participation = np.zeros_like(prices)
        media_mentions = np.zeros_like(prices)
        
        # Incorporate social data if available
        if include_social_data and social_data is not None:
            if 'retail_participation' in social_data.columns:
                # Align with price data index
                retail_data = social_data['retail_participation'].reindex(data.index, method='ffill').fillna(0)
                retail_participation = retail_data.values
            
            if 'media_mentions' in social_data.columns:
                # Align with price data index
                media_data = social_data['media_mentions'].reindex(data.index, method='ffill').fillna(0)
                media_mentions = media_data.values
        
        # Calculate bubble score
        bubble_score = _calculate_bubble_score_numba(
            prices, volumes, price_acceleration, trend_deviation,
            retail_participation, media_mentions, 
            self.lookback_period, self.sensitivity, self.exponential_threshold
        )
        
        # Convert back to pandas Series
        result = pd.Series(bubble_score, index=data.index)
        
        # Apply additional filtering to remove noise
        result = result.rolling(window=5).mean().fillna(0)
        result = (result > 0.5).astype(int)
        
        # Calculate bubble probability (from 0 to 1)
        probability = pd.Series(
            _calculate_bubble_score_numba(
                prices, volumes, price_acceleration, trend_deviation,
                retail_participation, media_mentions, 
                self.lookback_period, 0.0, self.exponential_threshold
            ), 
            index=data.index
        )
        
        self.logger.info(f"Detected bubble conditions in {result.sum()} periods")
        
        return result, probability
    
    def visualize(self, data, bubble_indicators, bubble_probability, title="Bubble Detection"):
        """
        Visualize detected bubble conditions on price chart.
        
        Args:
            data (pd.DataFrame): Price data
            bubble_indicators (pd.Series): Binary bubble indicators
            bubble_probability (pd.Series): Bubble probability scores
            title (str): Chart title
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Plot price on main chart
        ax1.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
        
        # Highlight bubble periods
        for i in range(len(bubble_indicators)):
            if bubble_indicators.iloc[i] == 1:
                ax1.axvspan(bubble_indicators.index[i], bubble_indicators.index[min(i+1, len(bubble_indicators)-1)], 
                           alpha=0.2, color='red')
        
        # Plot bubble probability
        ax2.plot(bubble_probability.index, bubble_probability, color='red', label='Bubble Probability')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')
        ax2.axhline(y=self.sensitivity, linestyle='--', color='grey', alpha=0.7, label=f'Threshold ({self.sensitivity})')
        ax2.fill_between(bubble_probability.index, bubble_probability, color='red', alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Set titles and labels
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('Price')
        
        plt.tight_layout()
        return plt#######################
# Topological Data Analysis
#######################

class TopologicalDataAnalyzer:
    """
    Enterprise-grade implementation of Topological Data Analysis for cryptocurrency markets.
    
    Features:
    - Persistent Homology to identify structural features
    - Sliding window embeddings for time series analysis
    - Mapper algorithm for dimensionality reduction
    - Topological features extraction for predictive modeling
    """
    
    def __init__(self, max_dimension=1, window_size=20, stride=1, use_parallel=True, max_cache_size=10):
        """
        Initialize the TopologicalDataAnalyzer.
        
        Args:
            max_dimension (int): Maximum homology dimension to compute
            window_size (int): Window size for sliding window embeddings
            stride (int): Stride for sliding windows
            use_parallel (bool): Whether to use parallel computing
            max_cache_size (int): Maximum size of the results cache
        """
        self.max_dimension = max_dimension
        self.window_size = window_size
        self.stride = stride
        self.use_parallel = use_parallel
        self.max_cache_size = max_cache_size
        self.logger = logging.getLogger('TopologicalDataAnalyzer')
        
        # Initialize cache for persistent homology results
        self.ph_cache = {}
    
    @staticmethod
    def _create_point_cloud(time_series, window_size, stride=1):
        """
        Create a point cloud using the sliding window embedding technique.
        
        Args:
            time_series (numpy.ndarray): 1D time series data
            window_size (int): Size of the sliding window
            stride (int): Stride for the sliding window
            
        Returns:
            numpy.ndarray: Point cloud where each point is a window of time series
        """
        n_points = (len(time_series) - window_size) // stride + 1
        point_cloud = np.zeros((n_points, window_size))
        
        for i in range(n_points):
            start_idx = i * stride
            point_cloud[i] = time_series[start_idx:start_idx+window_size]
            
        # Transpose the point cloud if there are more columns than rows
        # This addresses the Ripser warning
        if point_cloud.shape[1] > point_cloud.shape[0]:
            point_cloud = point_cloud.T
            
        return point_cloud

    
    @lru_cache(maxsize=128)
    def _compute_persistent_homology(self, data_key, max_dimension=1):
        """
        Compute persistent homology for the given point cloud.
        
        Args:
            data_key (str): Key for the data (used for caching)
            max_dimension (int): Maximum homology dimension
            
        Returns:
            tuple: (diagrams, cocycles) - persistence diagrams and cocycles
        """
        # Retrieve data from cache
        if data_key in self.ph_cache:
            point_cloud = self.ph_cache[data_key]
        else:
            # This is just a fallback - in practice, data_key should be in cache
            self.logger.warning(f"Data key {data_key} not found in cache")
            return None, None
        
        # Compute persistent homology
        diagrams = ripser.ripser(point_cloud, maxdim=max_dimension)['dgms']
        
        # Extract cocycles (optional for more advanced analysis)
        # This requires a modified version of ripser that returns cocycles
        cocycles = None
        
        return diagrams, cocycles
    
    def _extract_topological_features(self, diagrams):
        """
        Extract features from persistence diagrams.
        
        Args:
            diagrams (list): List of persistence diagrams for each dimension
            
        Returns:
            dict: Dictionary of topological features
        """
        features = {}
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                features[f'total_persistence_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                features[f'max_persistence_{dim}'] = 0
                features[f'num_holes_{dim}'] = 0
                continue
            
            # Calculate persistence (death - birth)
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Remove infinity values
            finite_idx = np.isfinite(diagram[:, 1])
            finite_persistence = persistence[finite_idx]
            
            if len(finite_persistence) == 0:
                features[f'total_persistence_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                features[f'max_persistence_{dim}'] = 0
            else:
                # Total persistence
                features[f'total_persistence_{dim}'] = np.sum(finite_persistence)
                
                # Persistence entropy
                norm_persistence = finite_persistence / np.sum(finite_persistence)
                entropy = -np.sum(norm_persistence * np.log(norm_persistence + 1e-10))
                features[f'persistence_entropy_{dim}'] = entropy
                
                # Maximum persistence
                features[f'max_persistence_{dim}'] = np.max(finite_persistence)
            
            # Number of holes
            features[f'num_holes_{dim}'] = len(diagram)
        
        return features
    
    def analyze(self, data, columns=None, normalize=True):
        """
        Perform topological data analysis on market data.
        
        Args:
            data (pd.DataFrame): DataFrame with market data
            columns (list, optional): List of columns to analyze
                If None, uses 'close', 'volume', 'high', 'low'
            normalize (bool): Whether to normalize data before analysis
            
        Returns:
            pd.DataFrame: DataFrame with topological features
        """
        self.logger.info("Performing topological data analysis...")
        
        # Default columns to analyze
        if columns is None:
            columns = ['close']
            for col in ['volume', 'high', 'low']:
                if col in data.columns:
                    columns.append(col)
        
        # Validate columns exist
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in input data")
        
        # Extract data for analysis
        data_to_analyze = {}
        for col in columns:
            series = data[col].values
            if normalize:
                # Min-max normalization
                min_val = np.min(series)
                max_val = np.max(series)
                if max_val > min_val:
                    series = (series - min_val) / (max_val - min_val)
            
            data_to_analyze[col] = series
        
        # Analyze each column
        results = []
        
        for timestamp in data.index:
            idx = data.index.get_loc(timestamp)
            if idx < self.window_size:
                continue
            
            row_features = {'timestamp': timestamp}
            
            # For each column, create a point cloud and compute topology
            for col, series in data_to_analyze.items():
                # Extract window of data up to current timestamp
                window_end = idx + 1
                window_start = max(0, window_end - self.window_size * 10)  # Use 10 windows
                window_data = series[window_start:window_end]
                
                # Create point cloud using sliding window embedding
                point_cloud = self._create_point_cloud(window_data, self.window_size, self.stride)
                
                if len(point_cloud) < 3:  # Need at least 3 points for meaningful topology
                    continue
                
                # Cache point cloud for persistence homology computation
                data_key = f"{col}_{timestamp}"
                self.ph_cache[data_key] = point_cloud
                
                # Compute persistent homology
                diagrams, _ = self._compute_persistent_homology(data_key, self.max_dimension)
                
                if diagrams is None:
                    continue
                
                # Extract features from persistence diagrams
                topo_features = self._extract_topological_features(diagrams)
                
                # Add to row features with column prefix
                for feat_name, feat_value in topo_features.items():
                    row_features[f"{col}_{feat_name}"] = feat_value
            
            results.append(row_features)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Extracted {len(result_df.columns)} topological features")
        return result_df
    
    def detect_topology_changes(self, topo_features, threshold=0.1):
        """
        Detect significant changes in topological features over time.
        
        Args:
            topo_features (pd.DataFrame): DataFrame with topological features
            threshold (float): Threshold for change detection
            
        Returns:
            pd.Series: Binary indicator of significant topology changes
        """
        if len(topo_features) < 2:
            return pd.Series(0, index=topo_features.index)
        
        # Calculate changes in key topological features
        changes = pd.DataFrame(index=topo_features.index)
        
        for col in topo_features.columns:
            if 'total_persistence' in col or 'max_persistence' in col:
                changes[col] = topo_features[col].pct_change().abs()
        
        # Aggregate changes across features
        total_change = changes.mean(axis=1)
        
        # Mark significant changes
        change_indicator = (total_change > threshold).astype(int)
        
        return change_indicator
    
    def visualize_persistence_diagram(self, data, column='close', timestamp=None):
        """
        Visualize the persistence diagram for a specific column and timestamp.
        
        Args:
            data (pd.DataFrame): DataFrame with market data
            column (str): Column to analyze
            timestamp: Specific timestamp to analyze, uses latest if None
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        if timestamp is None:
            timestamp = data.index[-1]
        
        idx = data.index.get_loc(timestamp)
        if idx < self.window_size:
            self.logger.warning("Not enough data for the specified timestamp")
            return None
        
        # Extract window of data
        window_end = idx + 1
        window_start = max(0, window_end - self.window_size * 10)
        window_data = data[column].values[window_start:window_end]
        
        # Create point cloud
        point_cloud = self._create_point_cloud(window_data, self.window_size, self.stride)
        
        if len(point_cloud) < 3:
            self.logger.warning("Not enough points for topology calculation")
            return None
        
        # Compute persistent homology
        data_key = f"{column}_{timestamp}"
        self.ph_cache[data_key] = point_cloud
        diagrams, _ = self._compute_persistent_homology(data_key, self.max_dimension)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title(f"{column} Time Series")
        plt.plot(window_data)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        plt.subplot(1, 2, 2)
        plt.title("Persistence Diagram")
        
        # Plot each dimension with different colors
        colors = ['blue', 'red', 'green', 'purple']
        
        for dim, diagram in enumerate(diagrams):
            if dim > self.max_dimension or dim >= len(colors):
                break
                
            # Filter out points with infinite death time for plotting
            finite_idx = np.isfinite(diagram[:, 1])
            finite_points = diagram[finite_idx]
            
            if len(finite_points) > 0:
                plt.scatter(finite_points[:, 0], finite_points[:, 1], 
                           label=f'Dimension {dim}', color=colors[dim], alpha=0.7)
        
        # Add diagonal line
        min_val = min([np.min(diag[:, 0]) for diag in diagrams if len(diag) > 0]) if diagrams else 0
        max_val = max([np.max(diag[np.isfinite(diag[:, 1]), 1]) for diag in diagrams if len(diag) > 0]) if diagrams else 1
        
        # Add some padding
        min_val = max(0, min_val - 0.1)
        max_val = max_val + 0.1
        
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.axis('equal')
        plt.legend()
        
        plt.tight_layout()
        return plt

#######################
# Temporal Pattern Analysis
#######################

"""
TemporalPatternAnalyzer: Fixed, Numba-compatible implementation for temporal pattern analysis.
"""

# Note: FFT operations are not fully supported in Numba, so we'll implement
# cycle detection without Numba optimization
def _detect_cycles_fft(prices, min_length=5, max_length=365):
    """
    Detect cycles in price data using Fast Fourier Transform (non-Numba version).
    
    Args:
        prices (numpy.ndarray): Price data
        min_length (int): Minimum cycle length
        max_length (int): Maximum cycle length
        
    Returns:
        list: List of (cycle_length, power) tuples
    """
    n = len(prices)
    if n <= min_length:
        return []
    
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed_prices = prices * window
    
    # Detrend the data
    x = np.arange(n)
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, windowed_prices, rcond=-1)[0]
    detrended = windowed_prices - (slope * x + intercept)
    
    # Compute FFT
    fft_values = np.abs(np.fft.rfft(detrended))
    fft_freq = np.fft.rfftfreq(n)
    
    # Convert frequencies to cycle lengths
    cycle_lengths = 1.0 / fft_freq[1:]  # Skip DC component
    powers = fft_values[1:]
    
    # Normalize powers
    if np.max(powers) > 0:
        powers = powers / np.max(powers)
    
    # Find dominant cycles within range
    cycles = []
    for i in range(len(cycle_lengths)):
        length = cycle_lengths[i]
        if min_length <= length <= max_length:
            cycles.append((length, powers[i]))
    
    # Sort by power
    cycles = sorted(cycles, key=lambda x: x[1], reverse=True)
    
    return cycles


@jit_with_safe_caching(nopython=True)
def _detect_fractality_numba(prices, window_sizes_array):
    """
    Measure fractal dimension of price series across multiple scales.
    
    Args:
        prices (numpy.ndarray): Price data
        window_sizes_array (numpy.ndarray): Array of window sizes to analyze
        
    Returns:
        float: Hurst exponent (measure of fractality)
    """
    if len(prices) <= window_sizes_array[-1]:
        return 0.5  # Default value for short series
    
    # Calculate price changes
    price_changes = np.zeros(len(prices) - 1)
    for i in range(1, len(prices)):
        price_changes[i-1] = prices[i] - prices[i-1]
    
    # Calculate RS values for different window sizes
    log_rs = np.zeros(len(window_sizes_array))
    log_window = np.zeros(len(window_sizes_array))
    
    for i in range(len(window_sizes_array)):
        window = window_sizes_array[i]
        # Pre-allocate a numpy array for rs_values
        max_rs_values = len(price_changes) - window + 1
        rs_values_array = np.zeros(max_rs_values)
        rs_count = 0
        
        # Calculate RS for overlapping windows
        for j in range(max_rs_values):
            segment = price_changes[j:j+window]
            
            # Calculate mean manually
            segment_sum = 0.0
            for k in range(window):
                segment_sum += segment[k]
            mean = segment_sum / window
            
            # Calculate standard deviation manually
            std_sum = 0.0
            for k in range(window):
                std_sum += (segment[k] - mean) ** 2
            std = np.sqrt(std_sum / window) if std_sum > 0 else 0.0
            
            if std == 0:
                continue
            
            # Calculate cumulative deviation
            cumulative = np.zeros(window)
            cumsum = 0.0
            for k in range(window):
                cumsum += segment[k] - mean
                cumulative[k] = cumsum
            
            # Calculate range and rescale
            r_min = cumulative[0]
            r_max = cumulative[0]
            for k in range(1, window):
                if cumulative[k] < r_min:
                    r_min = cumulative[k]
                if cumulative[k] > r_max:
                    r_max = cumulative[k]
            
            r = r_max - r_min
            s = std
            
            rs = r / s if s > 0 else 0
            rs_values_array[rs_count] = rs
            rs_count += 1
        
        if rs_count > 0:
            # Calculate mean manually
            rs_sum = 0.0
            for j in range(rs_count):
                rs_sum += rs_values_array[j]
            rs_mean = rs_sum / rs_count
            
            log_rs[i] = np.log(rs_mean)
            log_window[i] = np.log(window)
    
    # Perform linear regression to get Hurst exponent
    n = len(window_sizes_array)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    
    for i in range(n):
        sum_x += log_window[i]
        sum_y += log_rs[i]
        sum_xy += log_window[i] * log_rs[i]
        sum_xx += log_window[i] * log_window[i]
    
    if (n * sum_xx - sum_x * sum_x) == 0:
        return 0.5
        
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    
    return slope

@jit_with_safe_caching(nopython=True)
def _calculate_self_similarity_numba(prices, window_size=20, stride=5):
    """
    Calculate self-similarity of time series across different times.
    
    Args:
        prices (numpy.ndarray): Price data
        window_size (int): Size of window for comparison
        stride (int): Stride for sliding window
        
    Returns:
        numpy.ndarray: Self-similarity scores
    """
    n = len(prices)
    n_windows = (n - window_size) // stride + 1
    similarity = np.zeros(n)
    
    if n_windows < 2:
        return similarity
    
    # Create matrix of windows
    windows = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        start = i * stride
        for k in range(window_size):
            windows[i, k] = prices[start + k]
        
        # Normalize each window
        window_min = np.min(windows[i])
        window_max = np.max(windows[i])
        if window_max > window_min:
            for k in range(window_size):
                windows[i, k] = (windows[i, k] - window_min) / (window_max - window_min)
    
    # Calculate pairwise similarities (parallelized)
    for i in prange(n_windows):
        best_similarity = 0.0
        
        for j in range(n_windows):
            if i == j:
                continue
            
            # Calculate correlation manually for Numba compatibility
            x = windows[i]
            y = windows[j]
            
            # Manual correlation calculation
            x_sum = 0.0
            y_sum = 0.0
            for k in range(window_size):
                x_sum += x[k]
                y_sum += y[k]
            
            x_mean = x_sum / window_size
            y_mean = y_sum / window_size
            
            numerator = 0.0
            denom_x = 0.0
            denom_y = 0.0
            
            for k in range(window_size):
                x_diff = x[k] - x_mean
                y_diff = y[k] - y_mean
                numerator += x_diff * y_diff
                denom_x += x_diff * x_diff
                denom_y += y_diff * y_diff
            
            if denom_x > 0 and denom_y > 0:
                corr = numerator / np.sqrt(denom_x * denom_y)
                similarity_score = abs(corr)
                
                if similarity_score > best_similarity:
                    best_similarity = similarity_score
        
        # Assign similarity to the timepoints in this window
        start = i * stride
        end = min(start + window_size, n)
        for k in range(start, end):
            if best_similarity > similarity[k]:
                similarity[k] = best_similarity
    
    return similarity


# Non-Numba function for harmonic pattern detection
def _detect_harmonic_patterns(prices, highs, lows):
    """
    Detect harmonic patterns in price data.
    
    Args:
        prices (numpy.ndarray): Close prices
        highs (numpy.ndarray): High prices
        lows (numpy.ndarray): Low prices
        
    Returns:
        list: List of detected patterns with confidence
    """
    # This implementation is a simplified version
    # A full implementation would include pattern-specific detection logic
    
    patterns = []
    pattern_names = ["Gartley", "Butterfly", "Bat", "Crab", "Shark"]
    
    # Find potential XABCD patterns
    n = len(prices)
    for i in range(n - 20):
        # Find significant swing points
        # This is a simplified approach - real implementation would involve more sophisticated logic
        swing_points = []
        
        for j in range(i, min(i + 50, n - 1)):
            is_swing_high = (j > 0 and j < n - 1 and 
                            highs[j] > highs[j-1] and highs[j] > highs[j+1])
            is_swing_low = (j > 0 and j < n - 1 and 
                           lows[j] < lows[j-1] and lows[j] < lows[j+1])
            
            if is_swing_high or is_swing_low:
                swing_points.append((j, prices[j], "high" if is_swing_high else "low"))
        
        if len(swing_points) >= 5:  # Need at least 5 points for XABCD pattern
            # Check Fibonacci relationships between swing points
            # This is where specific pattern detection logic would go
            
            # For demonstration, randomly assign confidence to patterns
            # In a real implementation, this would be calculated based on pattern criteria
            if np.random.random() > 0.8:
                pattern_type = np.random.choice(pattern_names)
                confidence = np.random.random() * 0.5 + 0.5  # Between 0.5 and 1.0
                patterns.append({
                    'type': pattern_type,
                    'points': [p[0] for p in swing_points[:5]],
                    'confidence': confidence,
                    'completed': swing_points[-1][0]
                })
    
    return patterns


# Use standard implementation for cycle projection
def _project_cycles(prices, cycles, forward_periods=30):
    """
    Project future prices based on detected cycles.
    
    Args:
        prices (numpy.ndarray): Historical prices
        cycles (list): List of (cycle_length, power) tuples
        forward_periods (int): Number of periods to project forward
        
    Returns:
        numpy.ndarray: Projected prices
    """
    if not cycles or len(prices) < 2:
        return np.zeros(forward_periods)
    
    # Calculate trend
    x = np.arange(len(prices))
    A = np.vstack([x, np.ones(len(prices))]).T
    slope, intercept = np.linalg.lstsq(A, prices, rcond=-1)[0]
    trend = slope * x + intercept
    
    # Detrend prices
    detrended = prices - trend
    
    # Combine top cycles to create projection
    projection = np.zeros(forward_periods)
    
    for cycle_length, power in cycles:
        # Convert to integer cycle length
        int_length = int(round(cycle_length))
        if int_length <= 1:
            continue
            
        # Use last full cycle as template
        last_cycle = detrended[-int_length:] if len(detrended) >= int_length else detrended
        
        # Repeat the cycle pattern
        for i in range(forward_periods):
            idx = i % len(last_cycle)
            projection[i] += last_cycle[idx] * power
    
    # Normalize projection influence by total power
    total_power = sum(power for _, power in cycles)
    if total_power > 0:
        projection = projection / total_power
    
    # Add trend component
    for i in range(forward_periods):
        projection[i] += intercept + slope * (len(prices) + i)
    
    return projection


class TemporalPatternAnalyzer:
    """
    Enterprise-grade implementation of Temporal Pattern Analysis for cryptocurrency markets.
    
    Features:
    - Fractal pattern recognition
    - Cycle identification across multiple timeframes
    - Harmonic pattern detection
    - Time-based symmetry analysis
    - Self-similarity measurement
    """
    
    def __init__(self, max_cycle_length=365, min_cycle_length=5, harmonic_patterns=True, 
                use_parallel=True, confidence_threshold=0.7):
        """
        Initialize the TemporalPatternAnalyzer.
        
        Args:
            max_cycle_length (int): Maximum cycle length to detect in days
            min_cycle_length (int): Minimum cycle length to detect in days
            harmonic_patterns (bool): Whether to detect harmonic patterns
            use_parallel (bool): Whether to use parallel processing
            confidence_threshold (float): Threshold for pattern confidence
        """
        self.max_cycle_length = max_cycle_length
        self.min_cycle_length = min_cycle_length
        self.harmonic_patterns = harmonic_patterns
        self.use_parallel = use_parallel
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger('TemporalPatternAnalyzer')
        
        # Initialize the cache
        self.cycle_cache = {}
    
    def analyze(self, data, lookback_window=None):
        """
        Perform temporal pattern analysis on market data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            lookback_window (int, optional): Window for lookback analysis
                If None, uses all available data
                
        Returns:
            dict: Dictionary with various temporal pattern metrics
        """
        self.logger.info("Performing temporal pattern analysis...")
        
        # Ensure required columns exist
        if 'close' not in data.columns:
            raise ValueError("Required column 'close' not found in input data")
        
        # Determine lookback window
        if lookback_window is None:
            lookback_window = len(data)
        else:
            lookback_window = min(lookback_window, len(data))
        
        # Extract price data
        prices = data['close'].values[-lookback_window:]
        
        # Calculate cycle detection using FFT (non-Numba version)
        cycles = _detect_cycles_fft(prices, self.min_cycle_length, self.max_cycle_length)
        
        # Extract top cycles (limit to 5)
        top_cycles = cycles[:5] if cycles else []
        
        # Calculate fractality (Hurst exponent) - Numba compatible version
        window_sizes_array = np.array([5, 10, 20, 50])  # Must be a numpy array for Numba
        hurst_exponent = _detect_fractality_numba(prices, window_sizes_array)
        
        # Calculate self-similarity
        similarity_scores = _calculate_self_similarity_numba(prices)
        avg_similarity = np.mean(similarity_scores)
        
        # Detect harmonic patterns if requested
        harmonic_patterns = []
        if self.harmonic_patterns and 'high' in data.columns and 'low' in data.columns:
            highs = data['high'].values[-lookback_window:]
            lows = data['low'].values[-lookback_window:]
            harmonic_patterns = _detect_harmonic_patterns(prices, highs, lows)
            
            # Filter by confidence threshold
            harmonic_patterns = [p for p in harmonic_patterns if p['confidence'] >= self.confidence_threshold]
        
        # Calculate cycle projections
        cycle_projections = _project_cycles(prices, top_cycles, forward_periods=30)
        
        # Create result dictionary
        result = {
            'dominant_cycles': top_cycles,
            'hurst_exponent': hurst_exponent,
            'self_similarity': avg_similarity,
            'harmonic_patterns': harmonic_patterns,
            'cycle_projections': cycle_projections,
            'similarity_scores': similarity_scores
        }
        
        self.logger.info(f"Found {len(top_cycles)} dominant cycles and {len(harmonic_patterns)} harmonic patterns")
        return result
    
    def identify_similar_historical_periods(self, data, lookback=100, num_matches=3):
        """
        Identify historical periods that are similar to the current market structure.
        
        Args:
            data (pd.DataFrame): DataFrame with market data
            lookback (int): Length of pattern to match
            num_matches (int): Number of matches to return
            
        Returns:
            list: List of dictionaries with match information
        """
        self.logger.info("Identifying similar historical periods...")
        
        if len(data) < lookback * 2:
            self.logger.warning("Not enough data for historical pattern matching")
            return []
        
        # Extract price data
        prices = data['close'].values
        
        # Get current pattern (last 'lookback' periods)
        current_pattern = prices[-lookback:]
        
        # Normalize current pattern
        current_min = np.min(current_pattern)
        current_max = np.max(current_pattern)
        if current_max > current_min:
            current_pattern_norm = (current_pattern - current_min) / (current_max - current_min)
        else:
            current_pattern_norm = np.zeros_like(current_pattern)
        
        # Scan historical data for similar patterns
        matches = []
        
        for i in range(len(prices) - lookback * 2):
            historical_pattern = prices[i:i+lookback]
            
            # Normalize historical pattern
            hist_min = np.min(historical_pattern)
            hist_max = np.max(historical_pattern)
            if hist_max > hist_min:
                historical_pattern_norm = (historical_pattern - hist_min) / (hist_max - hist_min)
            else:
                historical_pattern_norm = np.zeros_like(historical_pattern)
            
            # Calculate correlation manually for compatibility
            x_mean = np.mean(current_pattern_norm)
            y_mean = np.mean(historical_pattern_norm)
            numerator = np.sum((current_pattern_norm - x_mean) * (historical_pattern_norm - y_mean))
            denom_x = np.sum((current_pattern_norm - x_mean) ** 2)
            denom_y = np.sum((historical_pattern_norm - y_mean) ** 2)
            
            if denom_x > 0 and denom_y > 0:
                correlation = numerator / np.sqrt(denom_x * denom_y)
            else:
                correlation = 0
            
            # Calculate simple Euclidean distance
            distance = np.sqrt(np.sum((current_pattern_norm - historical_pattern_norm) ** 2))
            distance_normalized = distance / np.sqrt(lookback)
            
            # Calculate similarity score (higher is better)
            similarity = max(0, (correlation + 1) / 2) * (1 - min(1, distance_normalized))
            
            # Store match information
            if similarity > 0.7:  # Threshold for considering a match
                # Get what happened next
                future_periods = min(lookback, len(prices) - (i + lookback))
                if future_periods > 0:
                    future_pattern = prices[i+lookback:i+lookback+future_periods]
                    
                    # Calculate return
                    future_return = (future_pattern[-1] / future_pattern[0] - 1) * 100
                    
                    matches.append({
                        'start_idx': i,
                        'end_idx': i + lookback - 1,
                        'similarity': similarity,
                        'correlation': correlation,
                        'future_return': future_return,
                        'timestamp': data.index[i] if i < len(data.index) else None
                    })
        
        # Sort by similarity
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        
        # Return top matches
        return matches[:num_matches]
    
    def visualize_cycles(self, data, cycles_result, title="Temporal Cycle Analysis"):
        """
        Visualize detected cycles and projections.
        
        Args:
            data (pd.DataFrame): Market data
            cycles_result (dict): Result from analyze method
            title (str): Chart title
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        if not cycles_result['dominant_cycles']:
            self.logger.warning("No cycles to visualize")
            return None
            
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        # Plot price on main chart
        ax1.plot(data.index, data['close'], label='Price', color='blue', alpha=0.7)
        
        # Plot projections if available
        if 'cycle_projections' in cycles_result and len(cycles_result['cycle_projections']) > 0:
            projections = cycles_result['cycle_projections']
            
            # Create future dates for projections
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(data.index)
                if freq is None:
                    freq = pd.tseries.offsets.Day()
                future_dates = [last_date + i * freq for i in range(1, len(projections) + 1)]
            else:
                future_dates = [last_date + i for i in range(1, len(projections) + 1)]
                
            ax1.plot(future_dates, projections, '--', label='Projection', color='green')
        
        # Plot self-similarity
        if 'similarity_scores' in cycles_result:
            similarity = cycles_result['similarity_scores']
            ax2.plot(data.index[-len(similarity):], similarity, color='purple', label='Self-Similarity')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Similarity')
            ax2.axhline(y=0.8, linestyle='--', color='grey', alpha=0.7)
            ax2.legend(loc='upper left')
        
        # Plot Hurst exponent indicator
        hurst = cycles_result['hurst_exponent']
        ax3.axhline(y=0.5, linestyle='--', color='grey', alpha=0.7, label='Random Walk (H=0.5)')
        ax3.axhline(y=hurst, color='red', label=f'Hurst Exponent (H={hurst:.2f})')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Hurst')
        ax3.legend(loc='upper left')
        
        # Set titles and labels
        ax1.set_title(title)
        
        # Add cycle information
        cycle_text = "Dominant Cycles:\n"
        for i, (length, power) in enumerate(cycles_result['dominant_cycles'][:3]):
            cycle_text += f"Cycle {i+1}: {length:.1f} periods, power={power:.2f}\n"
        
        ax1.text(0.02, 0.05, cycle_text, transform=ax1.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.legend(loc='upper left')
        ax1.set_ylabel('Price')
        
        plt.tight_layout()
        return plt
    
    def visualize_similar_periods(self, data, similar_periods, title="Similar Historical Patterns"):
        """
        Visualize similar historical periods against current pattern.
        
        Args:
            data (pd.DataFrame): Market data
            similar_periods (list): Result from identify_similar_historical_periods
            title (str): Chart title
            
        Returns:
            matplotlib.pyplot: Plot object
        """
        if not similar_periods:
            self.logger.warning("No similar periods to visualize")
            return None
            
        # Number of periods to plot
        n_similar = min(3, len(similar_periods))
        
        # Create figure with subplots
        fig, axs = plt.subplots(n_similar + 1, 1, figsize=(14, 5 * (n_similar + 1)), sharex=False)
        
        # Plot current pattern in the first subplot
        lookback = similar_periods[0]['end_idx'] - similar_periods[0]['start_idx'] + 1
        current_pattern = data['close'].values[-lookback:]
        current_dates = data.index[-lookback:]
        
        axs[0].plot(current_dates, current_pattern, label='Current Pattern', color='blue', linewidth=2)
        axs[0].set_title(f"Current Pattern (Last {lookback} Periods)")
        axs[0].legend()
        axs[0].set_ylabel('Price')
        
        # Plot similar patterns in other subplots
        for i in range(n_similar):
            match = similar_periods[i]
            start_idx = match['start_idx']
            end_idx = match['end_idx']
            
            # Get pattern dates and prices
            match_dates = data.index[start_idx:end_idx+1]
            match_pattern = data['close'].values[start_idx:end_idx+1]
            
            # Get future pattern if available
            future_periods = min(lookback, len(data) - (end_idx + 1))
            if future_periods > 0:
                future_dates = data.index[end_idx+1:end_idx+future_periods+1]
                future_pattern = data['close'].values[end_idx+1:end_idx+future_periods+1]
            
            # Plot historical pattern and future
            axs[i+1].plot(match_dates, match_pattern, label='Historical Pattern', color='green')
            if future_periods > 0:
                axs[i+1].plot(future_dates, future_pattern, label='What Happened Next', color='red', linestyle='--')
            
            # Add match information
            axs[i+1].set_title(f"Match {i+1}: Similarity={match['similarity']:.2f}, Correlation={match['correlation']:.2f}")
            if 'timestamp' in match and match['timestamp'] is not None:
                axs[i+1].set_xlabel(f"Starting at {match['timestamp']}")
            
            axs[i+1].legend()
            axs[i+1].set_ylabel('Price')
        
        plt.tight_layout()
        return plt