#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:04:48 2025

@author: ashina
"""

# fibonacci_pattern_detector.py

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import time
import threading
from functools import lru_cache
import numba as nb
from numba import njit, float64, int64, boolean, prange, types
from enum import Enum
import math
from numba.typed import List as NumbaTypedList
from numba import types as NumbaTypes


from ..analyzers.fibonacci_analyzer import FibonacciAnalyzer, FibonacciParameters


# PennyLane Catalyst JIT support - keeping for backward compatibility
try:
    import numba.roc as roc
    ROC_AVAILABLE = True
except ImportError:
    ROC_AVAILABLE = False
    # Create a stub 'roc' so decorators and calls exist but are no-ops on CPU
    class _RocStub:
        def jit(self, *args, **kwargs):
            def decorate(fn): return fn
            return decorate
        def atomic(self):
            # stub atomic namespace
            return self
        def add(self, arr, idx, val):
            # no-op stub
            return 0
        def get_global_id(self, dim):
            return 0
        def to_device(self, x):
            return x
        def device_array(self, shape, dtype=None):
            # fallback host array
            return np.zeros(shape, dtype=dtype)
        def get_num_devices(self):
            return 1
        def select_device(self, id):
            pass
    # instantiate stub so 'roc' name is always defined
    roc = _RocStub()
    def jit(self, *args, **kwargs):
            def decorate(fn): return fn
            return decorate
    def atomic(self):
            # stub atomic namespace
            return self
    def add(self, arr, idx, val):
            # no-op stub
            return 0
    def get_global_id(self, dim):
            return 0
    def to_device(self, x):
            return x
    def device_array(self, shape, dtype=None):
            # fallback host array
            return np.zeros(shape, dtype=dtype)
    def get_num_devices(self):
            return 1
    def select_device(self, id):
            pass
    roc = _RocStub()
    
from enum import Enum
    
logger = logging.getLogger(__name__)

# ========================= Data Structures =========================

@dataclass
class PatternPoint:
    """Represents a significant point in a harmonic pattern."""
    index: int  # Position in the dataframe
    price: float  # Price value
    role: str  # X, A, B, C, D, etc.
    time: Optional[pd.Timestamp] = None  # Timestamp if available
    confidence: float = 1.0  # Confidence in point identification


@dataclass
class PatternConfig:
    """Configuration for a specific harmonic pattern."""
    name: str  # Pattern name
    ratios: Dict[str, Tuple[float, float]]  # Acceptable ratio ranges (min, max)
    # Example: {'XA/AB': (0.618, 0.618), 'BC/AB': (0.382, 0.886), ...}
    trend_requirement: Optional[str] = None  # 'bullish', 'bearish', or None
    min_size: float = 0.01  # Minimum pattern size as % of price
    min_quality: float = 0.7  # Minimum quality score to be valid


@dataclass
class HarmonicPattern:
    """Detected harmonic pattern with metadata."""
    name: str  # Pattern name
    points: Dict[str, PatternPoint]  # Pattern points (X, A, B, C, D)
    quality: float  # Pattern quality score (0.0-1.0)
    completion: float  # Pattern completion percentage (0.0-1.0)
    target_price: Optional[float] = None  # Target price if pattern completes
    stop_price: Optional[float] = None  # Suggested stop loss price
    trend: str = ""  # Pattern trend ('bullish' or 'bearish')
    created_at: float = field(default_factory=time.time)  # Pattern detection time
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class PatternDetectionConfig:
    """Configuration for pattern detection."""
    # Pattern detection parameters
    min_pattern_size: float = 0.01  # Minimum size as % of price
    max_pattern_size: float = 0.5  # Maximum size as % of price
    ratio_tolerance: float = 0.05  # Tolerance for ratio matching
    min_swing_strength: float = 0.01  # Minimum swing strength
    swing_window: int = 5  # Window for swing point detection
    use_gpu: bool = True
    use_multi_gpu: bool = True
    
    # Optimization parameters
    use_numba: bool = True  # Whether to use Numba acceleration
    use_parallel: bool = True  # Whether to use parallel processing
    
    # Confidence parameters
    volume_confirmation_weight: float = 0.3  # Weight of volume confirmation
    time_persistence_weight: float = 0.2  # Weight of time persistence
    
    # Cache parameters
    cache_size: int = 100  # Cache size for expensive calculations


class PatternState(Enum):
    """Possible states of pattern development."""
    FORMING = 'forming'  # Pattern is still forming
    COMPLETED = 'completed'  # Pattern has completed
    FAILED = 'failed'  # Pattern has failed to complete
    CONFIRMED = 'confirmed'  # Pattern completed and confirmed by price action
    INVALIDATED = 'invalidated'  # Pattern was completed but invalidation criteria met

# ========================= GPU Kernel & Launcher (Single GPU) =========================
@roc.jit
def find_candidates_kernel(
    swing_high, swing_low, close,
    candidates_arr, candidate_count,
    min_range, max_range,
    n_highs, n_lows, n_close
):
    # if stub, just return immediately
    if not ROC_AVAILABLE:
        return

    i = roc.get_global_id(0)
    if i >= n_highs - 2:
        return

    x_idx = swing_high[i]
    for j in range(n_lows):
        a_idx = swing_low[j]
        if a_idx <= x_idx: continue

        # (… copy your nested loops from the CPU version …)
        # when you detect a valid pattern:
        current = roc.atomic.add(candidate_count, 0, 1)
        if current < candidates_arr.shape[0]:
            candidates_arr[current, 0] = x_idx
            # fill other columns…

def _find_pattern_candidates_gpu(
    swing_high, swing_low, close,
    min_size, max_size,
    max_candidates_est
):
    # log device availability
    logger.info(f"[GPU] ROC_AVAILABLE={ROC_AVAILABLE}, devices={roc.get_num_devices()}")
    if not ROC_AVAILABLE or roc.get_num_devices() == 0:
        raise RuntimeError("No ROCm devices detected")

    # pick device 0 (or loop/select for multi‑GPU)
    roc.select_device(0)

    # copy inputs
    d_high = roc.to_device(swing_high)
    d_low  = roc.to_device(swing_low)
    d_close= roc.to_device(close)

    # allocate outputs
    d_cands = roc.device_array((max_candidates_est, 5), dtype=np.int64)
    d_count = roc.device_array(1, dtype=np.int64)

    # compute constants
    avg_price = close.mean()
    min_range = avg_price * min_size
    max_range = avg_price * max_size
    n_highs = swing_high.shape[0]
    n_lows  = swing_low.shape[0]
    n_close = close.shape[0]

    # launch
    threads_per_block = 128
    blocks = (n_highs + threads_per_block - 1) // threads_per_block
    find_candidates_kernel[blocks, threads_per_block](
        d_high, d_low, d_close,
        d_cands, d_count,
        min_range, max_range,
        n_highs, n_lows, n_close
    )

    # retrieve
    count = int(d_count.copy_to_host()[0])
    host_cands = d_cands.copy_to_host()[:count]
    return host_cands

# ========================= Numba-optimized Core Functions =========================

@njit(cache=True)
def _normalize_prices(prices: np.ndarray) -> np.ndarray:
    """
    Normalize price series to 0-1 range for pattern comparison.
    
    Args:
        prices: Array of prices
        
    Returns:
        Normalized prices
    """
    min_price = np.min(prices)
    max_price = np.max(prices)
    
    if max_price > min_price:
        return (prices - min_price) / (max_price - min_price)
    else:
        return np.ones_like(prices) * 0.5


@njit(cache=True)
def _calculate_ratios(points: np.ndarray):
    """
    Calculate key Fibonacci ratios between pattern points.
    
    Args:
        points: Array of pattern points [X, A, B, C, D]
        
    Returns:
        2D array of (index, ratio_value) pairs
    """
    # Safety check
    if len(points) < 5:
        # Return a properly typed empty 2D array instead of empty list
        return np.zeros((0, 2), dtype=np.float64)
    
    X, A, B, C, D = points
    
    # Calculate legs
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    
    # Skip invalid patterns with zero legs
    if min(XA, AB, BC, CD) < 1e-6:
        # Return a properly typed empty 2D array
        return np.zeros((0, 2), dtype=np.float64)
    
    # Calculate standard ratios and store as a 2D array with explicit typing
    # Each row is [index, ratio_value]
    result = np.zeros((9, 2), dtype=np.float64)
    
    # Fill in ratio values
    result[0, 0] = 0
    result[0, 1] = AB / XA  # AB/XA
    
    result[1, 0] = 1
    result[1, 1] = BC / AB  # BC/AB
    
    result[2, 0] = 2
    result[2, 1] = CD / BC  # CD/BC
    
    result[3, 0] = 3
    result[3, 1] = BC / XA  # BC/XA
    
    result[4, 0] = 4
    result[4, 1] = CD / AB  # CD/AB
    
    # Advanced projections
    AD = abs(D - A)
    result[5, 0] = 5
    result[5, 1] = AD / XA  # AD/XA
    
    # Additional extensions
    XB = abs(B - X)
    XC = abs(C - X)
    XD = abs(D - X)
    
    # Only calculate these if XA > 0
    if XA > 0:
        result[6, 0] = 6
        result[6, 1] = XB / XA  # XB/XA
        
        result[7, 0] = 7
        result[7, 1] = XC / XA  # XC/XA
        
        result[8, 0] = 8
        result[8, 1] = XD / XA  # XD/XA
    
    return result




@njit(cache=True)
def _calculate_pattern_score(ratios, target_indices, target_ranges, tolerance=0.05):
    """
    Calculate pattern match score based on ratios.
    
    Args:
        ratios: 2D array of [index, value] pairs
        target_indices: Array of target ratio indices
        target_ranges: Array of (min, max) tuples for ratio ranges
        tolerance: Match tolerance
        
    Returns:
        Pattern quality score (0.0-1.0)
    """
    if len(ratios) == 0 or len(target_indices) == 0:
        return 0.0
    
    total_score = 0.0
    count = 0
    
    # For each target ratio index
    for i in range(len(target_indices)):
        target_idx = target_indices[i]
        min_ratio = target_ranges[i][0]
        max_ratio = target_ranges[i][1]
        
        # Find the ratio in our array
        found = False
        actual = 0.0
        
        for j in range(len(ratios)):
            idx = int(ratios[j, 0])
            if idx == target_idx:
                found = True
                actual = ratios[j, 1]
                break
        
        if found:
            # Calculate distance to acceptable range
            if actual < min_ratio:
                distance = (min_ratio - actual) / min_ratio
            elif actual > max_ratio:
                distance = (actual - max_ratio) / max_ratio
            else:
                distance = 0.0  # Within range
            
            # Convert distance to score (0.0-1.0)
            ratio_score = max(0.0, 1.0 - distance / tolerance)
            total_score += ratio_score
            count += 1
    
    # Return average score
    return total_score / count if count > 0 else 0.0



@njit(cache=True)
def _validate_pattern_sequence(pattern_type: str, points: np.ndarray) -> bool:
    """
    Validate if the pattern points follow the correct sequence.
    
    Args:
        pattern_type: Type of pattern
        points: Pattern points [X, A, B, C, D]
        
    Returns:
        Whether the pattern sequence is valid
    """
    if len(points) != 5:
        return False
    
    X, A, B, C, D = points
    
    # Basic validation - ensure points are in sequence
    if not (X < A < B < C < D or X > A > B > C > D):
        return False
    
    # Pattern-specific validation
    if pattern_type == 'gartley':
        # Gartley: X->A retracement, A->B retracement, B->C retracement, C->D extension
        if X < A:  # Bullish
            return (X < A and A > B and B < C and C > D and X < D)
        else:  # Bearish
            return (X > A and A < B and B > C and C < D and X > D)
            
    elif pattern_type == 'butterfly':
        # Butterfly: Similar to Gartley but with deeper retracements and extensions
        if X < A:  # Bullish
            return (X < A and A > B and B < C and C > D and D < X)
        else:  # Bearish
            return (X > A and A < B and B > C and C < D and D > X)
            
    elif pattern_type == 'bat':
        # Bat: Similar to Gartley but with different ratio requirements
        if X < A:  # Bullish
            return (X < A and A > B and B < C and C > D and D < B)
        else:  # Bearish
            return (X > A and A < B and B > C and C < D and D > B)
            
    elif pattern_type == 'crab':
        # Crab: Extreme extension pattern
        if X < A:  # Bullish
            return (X < A and A > B and B < C and C > D and D < A)
        else:  # Bearish
            return (X > A and A < B and B > C and C < D and D > A)
            
    elif pattern_type == 'shark':
        # Shark: Initial extension followed by retracements
        if X < A:  # Bullish
            return (X < A and A > B and B < C and C > D and D < B)
        else:  # Bearish
            return (X > A and A < B and B > C and C < D and D > B)
    
    # Unknown pattern type
    return True  # Default to true for custom patterns

@njit(cache=True)
def _calculate_extensions(pattern_type: str, points: np.ndarray) -> Dict[str, float]:
    """
    Calculate price extensions/targets with NaN protection
    """
    if len(points) != 5:
        return {"target_1": np.nan, "stop": np.nan}
    
    X, A, B, C, D = points
    
    # Check for valid inputs
    if not all(np.isfinite([X, A, B, C, D])):
        return {"target_1": np.nan, "stop": np.nan}
        
    # Calculate key measurements
    XA = A - X
    AB = B - A
    BC = C - B
    CD = D - C
    
    # Check for zero values that would cause divisions by zero
    if abs(XA) < 1e-9 or abs(CD) < 1e-9:
        return {"target_1": np.nan, "stop": np.nan}
    
    extensions = {}
    
    # Calculate common extension levels
    extensions['1.272'] = D + CD * 0.272
    extensions['1.414'] = D + CD * 0.414
    extensions['1.618'] = D + CD * 0.618
    extensions['2.000'] = D + CD * 1.000
    extensions['2.618'] = D + CD * 1.618
    
    # Pattern-specific extensions with safety checks
    if pattern_type == 'gartley':
        extensions['stop'] = C
        
    elif pattern_type == 'butterfly':
        extensions['target_1'] = X
        extensions['target_2'] = X + XA * 0.272 * (-1 if XA > 0 else 1)
        extensions['stop'] = D + CD * 0.2 * (-1 if CD > 0 else 1)
        
    elif pattern_type == 'bat':
        extensions['target_1'] = A + XA * 0.5 * (-1 if XA > 0 else 1)
        extensions['stop'] = C
        
    elif pattern_type == 'crab':
        extensions['target_1'] = X
        extensions['target_2'] = X + XA * 0.272 * (-1 if XA > 0 else 1)
        extensions['stop'] = D + CD * 0.1 * (-1 if CD > 0 else 1)
        
    elif pattern_type == 'shark':
        extensions['target_1'] = A + XA * 0.5 * (-1 if XA > 0 else 1)
        extensions['stop'] = D + CD * 0.2 * (-1 if CD > 0 else 1)
    
    return extensions


# ========================= GPU Kernel & Launcher (Single GPU) =========================
@roc.jit
def find_candidates_kernel(
    swing_high, swing_low, close,
    candidates_arr, candidate_count,
    min_range, max_range,
    n_highs, n_lows, n_close
):
    i = roc.get_global_id(0)
    if i >= n_highs - 2:
        return

    x_idx = swing_high[i]
    for j in range(n_lows):
        a_idx = swing_low[j]
        if a_idx <= x_idx: continue
        for k in range(i + 1, n_highs - 1):
            b_idx = swing_high[k]
            if b_idx <= a_idx: continue
            for l in range(j + 1, n_lows):
                c_idx = swing_low[l]
                if c_idx <= b_idx: continue
                for m in range(k + 1, n_highs):
                    d_idx = swing_high[m]
                    if d_idx <= c_idx: continue
                    if x_idx < n_close and a_idx < n_close and b_idx < n_close and c_idx < n_close and d_idx < n_close:
                        pr = max(close[x_idx], close[b_idx], close[d_idx]) - min(close[a_idx], close[c_idx])
                        if pr >= min_range and pr <= max_range:
                            curr = roc.atomic.add(candidate_count, 0, 1)
                            if curr < candidates_arr.shape[0]:
                                candidates_arr[curr, 0] = x_idx
                                candidates_arr[curr, 1] = a_idx
                                candidates_arr[curr, 2] = b_idx
                                candidates_arr[curr, 3] = c_idx
                                candidates_arr[curr, 4] = d_idx


def _find_pattern_candidates_gpu(high, low, close, min_size, max_size, max_candidates_est, device_id=0):
    # select device (for multi-GPU)
    roc.select_device(device_id)
    swing_high, swing_low = _detect_swing_points(close, 5)
    if len(swing_high) < 3 or len(swing_low) < 2:
        return np.empty((0,5), dtype=np.int64)

    # device arrays
    d_close = roc.to_device(close)
    d_high = roc.to_device(swing_high)
    d_low = roc.to_device(swing_low)
    d_cand = roc.device_array((max_candidates_est,5), dtype=np.int64)
    d_count = roc.device_array(1, dtype=np.int64)

    avg = close.mean()
    min_range = avg * min_size
    max_range = avg * max_size
    n_highs = swing_high.shape[0]
    n_lows = swing_low.shape[0]
    n_close = close.shape[0]

    threads_per_block = 128
    blocks = (n_highs + threads_per_block - 1) // threads_per_block
    find_candidates_kernel[blocks, threads_per_block](
        d_high, d_low, d_close,
        d_cand, d_count,
        min_range, max_range,
        n_highs, n_lows, n_close
    )
    count = int(d_count.copy_to_host()[0])
    return d_cand.copy_to_host()[:count]

# ========================= Multi-GPU Coordinator =========================
def _find_pattern_candidates_multi_gpu(high, low, close, min_size, max_size, max_candidates_est):
    # split work by index range across available GPUs
    ndev = roc.get_num_devices()
    # detect swings once on host
    swing_high, swing_low = _detect_swing_points(close, 5)
    if len(swing_high) < 3 or len(swing_low) < 2:
        return []

    # partition swing_high array into nearly equal chunks
    parts = np.array_split(swing_high, ndev)
    results = []
    lock = threading.Lock()
    
    def worker(dev_id, high_part):
        # derive corresponding low and full close for each subtask
        # use full low & close to allow patterns spanning partitions
        cands = _find_pattern_candidates_gpu(
            high, low, close,
            min_size, max_size, max_candidates_est // ndev,
            device_id=dev_id
        )
        with lock:
            results.append(cands)

    threads = []
    for dev_id, part in enumerate(parts):
        t = threading.Thread(target=worker, args=(dev_id, part))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # merge and dedupe results
    if not results:
        return []
    merged = np.vstack(results)
    # unique rows
    merged = np.unique(merged, axis=0)
    return merged


# ========================= Replace CPU version with GPU call =========================
@njit(cache=True)
def _detect_swing_points(prices: np.ndarray, window: int = 5):
    # unchanged CPU swing detection
    n = len(prices)
    if n < window*2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    highs = []
    lows = []
    for i in range(window, n-window):
        sh = True; sl = True
        for j in range(1, window+1):
            if prices[i] <= prices[i-j] or prices[i] <= prices[i+j]: sh = False
            if prices[i] >= prices[i-j] or prices[i] >= prices[i+j]: sl = False
        if sh: highs.append(i)
        if sl: lows.append(i)
    return np.array(highs, np.int64), np.array(lows, np.int64)

# Override the CPU candidate finder in module
def _find_pattern_candidates(highs, lows, close,
                             min_size=0.01, max_size=0.5,
                             max_candidates_est=10000):
    # Dispatch to GPU
    return _find_pattern_candidates_gpu(close, highs, lows,
                                        min_size, max_size,
                                        max_candidates_est)
# ========================= Pattern Detector Class =========================

class FibonacciPatternDetector:
    """
    Advanced pattern detection using Fibonacci relationships and harmonic patterns.
    
    This class implements:
    1. Detection of classic harmonic patterns (Gartley, Butterfly, Bat, Crab, Shark)
    2. Pattern quality assessment with confidence scoring
    3. Price targets and stop levels calculation
    4. Real-time pattern tracking and development
    5. Multi-timeframe confluence analysis
    """
    
    def __init__(self, config: Optional[PatternDetectionConfig] = None):
        """
        Initialize pattern detector with configuration.
        
        Args:
            config: Pattern detection configuration
        """
        self.config = config or PatternDetectionConfig()
        self.logger = logger
        
        # Initialize pattern definitions
        self.patterns = self._init_pattern_definitions()
        
        # Cache for expensive calculations
        self._swing_cache = {}
        self._pattern_cache = {}
        
        # Active patterns being tracked
        self._active_patterns = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize performance stats
        self._perf_stats = {
            'pattern_detection_time': 0.0,
            'swing_detection_time': 0.0,
            'total_patterns_detected': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.debug("FibonacciPatternDetector initialized")
    
    def _init_pattern_definitions(self) -> Dict[str, PatternConfig]:
        """
        Initialize predefined harmonic pattern configurations.
        
        Returns:
            Dictionary of pattern configurations
        """
        patterns = {}
        
        # Gartley Pattern
        patterns['gartley'] = PatternConfig(
            name='gartley',
            ratios={
                'AB/XA': (0.618, 0.618),  # AB should be 0.618 of XA
                'BC/AB': (0.382, 0.886),  # BC should be 0.382-0.886 of AB
                'CD/BC': (1.272, 1.272)   # CD should be 1.272 of BC
            },
            min_quality=0.8
        )
        
        # Butterfly Pattern
        patterns['butterfly'] = PatternConfig(
            name='butterfly',
            ratios={
                'AB/XA': (0.786, 0.786),  # AB should be 0.786 of XA
                'BC/AB': (0.382, 0.886),  # BC should be 0.382-0.886 of AB
                'CD/BC': (1.618, 2.618)   # CD should be 1.618-2.618 of BC
            },
            min_quality=0.7
        )
        
        # Bat Pattern
        patterns['bat'] = PatternConfig(
            name='bat',
            ratios={
                'AB/XA': (0.382, 0.5),    # AB should be 0.382-0.5 of XA
                'BC/AB': (0.382, 0.886),  # BC should be 0.382-0.886 of AB
                'CD/BC': (1.618, 2.618)   # CD should be 1.618-2.618 of BC
            },
            min_quality=0.75
        )
        
        # Crab Pattern
        patterns['crab'] = PatternConfig(
            name='crab',
            ratios={
                'AB/XA': (0.382, 0.618),  # AB should be 0.382-0.618 of XA
                'BC/AB': (0.382, 0.886),  # BC should be 0.382-0.886 of AB
                'CD/BC': (2.618, 3.618)   # CD should be 2.618-3.618 of BC
            },
            min_quality=0.7
        )
        
        # Shark Pattern
        patterns['shark'] = PatternConfig(
            name='shark',
            ratios={
                'AB/XA': (0.5, 0.5),      # AB should be 0.5 of XA
                'BC/AB': (1.13, 1.618),   # BC should be 1.13-1.618 of AB
                'CD/BC': (1.618, 1.618)   # CD should be 1.618 of BC
            },
            min_quality=0.75
        )
        
        # Three-Drive Pattern
        patterns['three_drive'] = PatternConfig(
            name='three_drive',
            ratios={
                'AB/XA': (0.618, 0.618),  # AB should be 0.618 of XA
                'BC/AB': (1.272, 1.272),  # BC should be 1.272 of AB
                'CD/BC': (0.786, 0.786)   # CD should be 0.786 of BC
            },
            min_quality=0.8
        )
        
        return patterns

    def detect_patterns(self, dataframe: pd.DataFrame) -> Dict[str, List[HarmonicPattern]]:
        """
        Detect all supported patterns in the dataframe.
        
        Args:
            dataframe: Input DataFrame with OHLC data
            
        Returns:
            Dictionary of pattern type to list of detected patterns
        """
        # Check required columns
        required_cols = ['high', 'low', 'close']
        if not all(col in dataframe.columns for col in required_cols):
            self.logger.warning(f"Missing required columns. Available: {dataframe.columns.tolist()}")
            return {}
        
        # Optimize for large dataframes: use only the last N candles for new pattern detection
        lookback = min(len(dataframe), 200)  # Look at most 200 candles back
        
        # Record start time
        start_time = time.time()
        
        # Generate unique key for this data
        data_key = f"{dataframe.index[-1]}-{hash(dataframe['close'].iloc[-lookback:].values.tobytes())}"
        
        # Check cache
        with self._lock:
            if data_key in self._pattern_cache:
                self._perf_stats['cache_hits'] += 1
                return self._pattern_cache[data_key]
                
            self._perf_stats['cache_misses'] += 1
        
        # Prepare results
        results = {}
        for pattern_name in self.patterns:
            results[pattern_name] = []
        
        try:
            # Extract price data
            high = dataframe['high'].values[-lookback:]
            low = dataframe['low'].values[-lookback:]
            close = dataframe['close'].values[-lookback:]
            
            # Detect swing points
            swing_start = time.time()
            swing_high_indices, swing_low_indices = self._detect_swings(high, low, close)
            
            with self._lock:
                self._perf_stats['swing_detection_time'] = time.time() - swing_start
            
            # Find pattern candidates
            candidates = self._find_candidates(high, low, close, swing_high_indices, swing_low_indices)
            
            for pattern_name, pattern_config in self.patterns.items():
                for x_idx, a_idx, b_idx, c_idx, d_idx in candidates:
                    # Adjust indices to full dataframe
                    full_x_idx = len(dataframe) - lookback + x_idx
                    full_a_idx = len(dataframe) - lookback + a_idx
                    full_b_idx = len(dataframe) - lookback + b_idx
                    full_c_idx = len(dataframe) - lookback + c_idx
                    full_d_idx = len(dataframe) - lookback + d_idx
                    
                    # Get point prices
                    points = np.array([
                        close[x_idx], close[a_idx], close[b_idx], close[c_idx], close[d_idx]
                    ])
                    
                    # Validate pattern sequence
                    if not _validate_pattern_sequence(pattern_name, points):
                        continue
                    
                    # Calculate ratios - now returns list of tuples
                    ratio_tuples = _calculate_ratios(points)
                    
                    # Skip if no valid ratios
                    if ratio_tuples.size == 0:
                        continue
                    
                    # Define target ratio indices and ranges for this pattern
                    if pattern_name == 'gartley':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.618, 0.618), (0.382, 0.886), (1.272, 1.272)]
                    elif pattern_name == 'butterfly':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.786, 0.786), (0.382, 0.886), (1.618, 2.618)]
                    elif pattern_name == 'bat':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.382, 0.5), (0.382, 0.886), (1.618, 2.618)]
                    elif pattern_name == 'crab':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.382, 0.618), (0.382, 0.886), (2.618, 3.618)]
                    elif pattern_name == 'shark':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.5, 0.5), (1.13, 1.618), (1.618, 1.618)]
                    elif pattern_name == 'three_drive':
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.618, 0.618), (1.272, 1.272), (0.786, 0.786)]
                    else:
                        # Unknown pattern, use default indices and ranges
                        target_indices = [0, 1, 2]  # AB/XA, BC/AB, CD/BC
                        target_ranges = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
                    
                    # Calculate pattern quality score
                    quality = _calculate_pattern_score(
                        ratio_tuples, target_indices, target_ranges, self.config.ratio_tolerance
                    )
                    
                    # Skip low quality patterns
                    if quality < pattern_config.min_quality:
                        continue
                        
                    # In your detect_patterns method, change:
                    ratio_tuples = _calculate_ratios(points)
                    
                    # Skip if no valid ratios
                    if len(ratio_tuples) == 0:
                        continue
                    
                    # Convert the 2D array to a dictionary
                    ratio_names = ["AB/XA", "BC/AB", "CD/BC", "BC/XA", "CD/AB", "AD/XA", "XB/XA", "XC/XA", "XD/XA"]
                    ratios = {}
                    for i in range(len(ratio_tuples)):
                        idx = int(ratio_tuples[i, 0])
                        if idx < len(ratio_names):
                            ratios[ratio_names[idx]] = ratio_tuples[i, 1]
                                        
                    # Calculate pattern direction
                    is_bullish = points[0] < points[1]  # X < A means bullish
                    trend = "bullish" if is_bullish else "bearish"
                    
                    
                    # Calculate targets and stops
                    extensions = _calculate_extensions(pattern_name, points)
                    
                    # Create PatternPoints
                    pattern_points = {
                        'X': PatternPoint(full_x_idx, points[0], 'X', dataframe.index[full_x_idx] if full_x_idx < len(dataframe) else None),
                        'A': PatternPoint(full_a_idx, points[1], 'A', dataframe.index[full_a_idx] if full_a_idx < len(dataframe) else None),
                        'B': PatternPoint(full_b_idx, points[2], 'B', dataframe.index[full_b_idx] if full_b_idx < len(dataframe) else None),
                        'C': PatternPoint(full_c_idx, points[3], 'C', dataframe.index[full_c_idx] if full_c_idx < len(dataframe) else None),
                        'D': PatternPoint(full_d_idx, points[4], 'D', dataframe.index[full_d_idx] if full_d_idx < len(dataframe) else None),
                    }
                    
                    # Create HarmonicPattern
                    pattern = HarmonicPattern(
                        name=pattern_name,
                        points=pattern_points,
                        quality=quality,
                        completion=1.0,  # Fully formed pattern
                        target_price=extensions.get('target_1', extensions.get('1.618')),
                        stop_price=extensions.get('stop', None),
                        trend=trend,
                        metadata={
                            'ratios': {k: round(v, 3) for k, v in ratios.items()},
                            'extensions': {k: round(v, 4) for k, v in extensions.items()},
                            'ideal_ratios': {k: v for k, v in pattern_config.ratios.items()}
                        }
                    )
                    
                    # Add to results
                    results[pattern_name].append(pattern)
                    
                    with self._lock:
                        self._perf_stats['total_patterns_detected'] += 1
        
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}", exc_info=True)
        
        # Update cache
        with self._lock:
            self._pattern_cache[data_key] = results
            
            # Limit cache size
            if len(self._pattern_cache) > self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._pattern_cache))
                del self._pattern_cache[oldest_key]
            
            # Update performance stats
            self._perf_stats['pattern_detection_time'] = time.time() - start_time
        
        return results

    def _detect_swings(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect swing high and low points.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        # Generate cache key
        cache_key = hash(close.tobytes())
        
        # Check cache
        with self._lock:
            if cache_key in self._swing_cache:
                return self._swing_cache[cache_key]
        
        # Detect swing points
        swing_high_indices, swing_low_indices = _detect_swing_points(close, self.config.swing_window)
        
        # Update cache
        with self._lock:
            self._swing_cache[cache_key] = (swing_high_indices, swing_low_indices)
            
            # Limit cache size
            if len(self._swing_cache) > self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._swing_cache))
                del self._swing_cache[oldest_key]
        
        return swing_high_indices, swing_low_indices

    def find_pattern_candidates_py(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                  swing_high_indices: np.ndarray, swing_low_indices: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """
        Python implementation of pattern candidate finder (for debugging).
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            swing_high_indices: Indices of swing highs
            swing_low_indices: Indices of swing lows
            
        Returns:
            List of (X, A, B, C, D) index tuples
        """
        candidates = []
        
        # Check if we have enough swing points
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 2:
            return candidates
        
        # Calculate average price for size reference
        avg_price = np.mean(close)
        min_range = avg_price * self.config.min_pattern_size
        max_range = avg_price * self.config.max_pattern_size
        
        # For each possible combination of highs and lows
        for i, x_idx in enumerate(swing_high_indices[:-2]):
            for j, a_idx in enumerate(swing_low_indices):
                # A must follow X
                if a_idx <= x_idx:
                    continue
                
                for k, b_idx in enumerate(swing_high_indices[i+1:-1]):
                    # B must follow A
                    if b_idx <= a_idx:
                        continue
                    
                    for l, c_idx in enumerate(swing_low_indices[j+1:]):
                        # C must follow B
                        if c_idx <= b_idx:
                            continue
                        
                        for m, d_idx in enumerate(swing_high_indices[i+k+2:]):
                            # D must follow C
                            if d_idx <= c_idx:
                                continue
                            
                            # Check pattern size
                            pattern_range = max(close[x_idx], close[b_idx], close[d_idx]) - min(close[a_idx], close[c_idx])
                            
                            if pattern_range >= min_range and pattern_range <= max_range:
                                candidates.append((x_idx, a_idx, b_idx, c_idx, d_idx))
        
        return candidates
    

    #def find_candidates(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, swing_high_indices: np.ndarray, swing_low_indices: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    def _find_candidates(self, high, low, close, swing_high, swing_low):
        if self.config.use_gpu and ROC_AVAILABLE:
            try:
                if self.config.use_multi_gpu:
                    arr = _find_pattern_candidates_multi_gpu(
                        high, low, close,
                        self.config.min_pattern_size,
                        self.config.max_pattern_size,
                        10000
                    )
                else:
                    arr = _find_pattern_candidates_gpu(
                        high, low, close,
                        self.config.min_pattern_size,
                        self.config.max_pattern_size,
                        10000
                    )
                return [tuple(r) for r in arr]
            except Exception as e:
                logger.error(f"GPU error, falling back: {e}")
        # fallback to CPU/Numba or Python
        return self.find_pattern_candidates_py(high, low, close, swing_high, swing_low)


    def add_pattern_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern detection columns to dataframe.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with pattern columns added
        """
        # Detect patterns
        patterns = self.detect_patterns(dataframe)
        
        # Initialize columns
        for pattern_name in self.patterns:
            dataframe[f'pattern_{pattern_name}'] = 0.0
            dataframe[f'pattern_{pattern_name}_target'] = np.nan
            dataframe[f'pattern_{pattern_name}_stop'] = np.nan
        
        dataframe['pattern_detected'] = False
        dataframe['pattern_quality'] = 0.0
        dataframe['pattern_best'] = None
        
        # Find best pattern for each row
        for pattern_name, detected in patterns.items():
            for pattern in detected:
                # Get last pattern point (D)
                d_point = pattern.points['D']
                
                # Update dataframe at D point
                if 0 <= d_point.index < len(dataframe):
                    dataframe.loc[dataframe.index[d_point.index], f'pattern_{pattern_name}'] = pattern.quality
                    dataframe.loc[dataframe.index[d_point.index], 'pattern_detected'] = True
                    dataframe.loc[dataframe.index[d_point.index], 'pattern_quality'] = pattern.quality
                    
                    # Set target and stop prices
                    if pattern.target_price is not None:
                        dataframe.loc[dataframe.index[d_point.index], f'pattern_{pattern_name}_target'] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        dataframe.loc[dataframe.index[d_point.index], f'pattern_{pattern_name}_stop'] = pattern.stop_price
                    
                    # Update best pattern if quality is higher
                    current_best = dataframe.loc[dataframe.index[d_point.index], 'pattern_best']
                    current_quality = dataframe.loc[dataframe.index[d_point.index], 'pattern_quality']
                    
                    if current_best is None or pattern.quality > current_quality:
                        dataframe.loc[dataframe.index[d_point.index], 'pattern_best'] = pattern_name
        
        return dataframe

    def detect_developing_patterns(self, dataframe: pd.DataFrame) -> Dict[str, List[HarmonicPattern]]:
        """
        Detect patterns that are still developing (not yet complete).
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Dictionary of pattern type to list of developing patterns
        """
        # Implement developing pattern detection logic
        # This would identify partial patterns (XAB, XABC) that might complete in the future
        # For now we'll return empty results
        return {pattern_name: [] for pattern_name in self.patterns}

    def pattern_to_signal(self, pattern: HarmonicPattern, current_price: float) -> Dict[str, Any]:
        """
        Convert pattern to actionable trading signal.
        
        Args:
            pattern: Detected pattern
            current_price: Current price
            
        Returns:
            Signal dictionary with entry, target, stop, etc.
        """
        signal = {
            'pattern': pattern.name,
            'quality': pattern.quality,
            'trend': pattern.trend,
            'action': 'buy' if pattern.trend == 'bullish' else 'sell',
            'entry_price': pattern.points['D'].price,
            'target_price': pattern.target_price,
            'stop_price': pattern.stop_price,
            'risk_reward': None,
            'confidence': pattern.quality
        }
        
        # Calculate risk/reward ratio if targets and stops are available
        if pattern.target_price is not None and pattern.stop_price is not None:
            # For bullish patterns
            if pattern.trend == 'bullish':
                reward = pattern.target_price - current_price
                risk = current_price - pattern.stop_price
            else:  # Bearish patterns
                reward = current_price - pattern.target_price
                risk = pattern.stop_price - current_price
            
            # Calculate ratio if risk is non-zero
            if abs(risk) > 1e-6:
                signal['risk_reward'] = abs(reward / risk)
        
        return signal

    def calculate_confluence(self, patterns: Dict[str, List[HarmonicPattern]], 
                           price: float, tolerance: float = 0.005) -> float:
        """
        Calculate confluence score based on proximity to pattern levels.
        
        Args:
            patterns: Detected patterns
            price: Current price
            tolerance: Proximity tolerance
            
        Returns:
            Confluence score (0-1)
        """
        if not patterns:
            return 0.0
        
        # Collect all relevant price levels
        price_levels = []
        
        for pattern_list in patterns.values():
            for pattern in pattern_list:
                # Add key pattern points
                for point in pattern.points.values():
                    price_levels.append(point.price)
                
                # Add target and stop levels
                if pattern.target_price is not None:
                    price_levels.append(pattern.target_price)
                
                if pattern.stop_price is not None:
                    price_levels.append(pattern.stop_price)
                
                # Add extension levels from metadata
                extensions = pattern.metadata.get('extensions', {})
                for ext_price in extensions.values():
                    if isinstance(ext_price, (int, float)):
                        price_levels.append(ext_price)
        
        # Calculate proximity to each level
        if not price_levels or price <= 0:
            return 0.0
        
        min_distance = float('inf')
        for level in price_levels:
            if level > 0:
                distance = abs(price - level) / price
                min_distance = min(min_distance, distance)
        
        # Convert to confluence score
        if min_distance == float('inf'):
            return 0.0
            
        return max(0.0, 1.0 - min_distance / tolerance)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        with self._lock:
            return dict(self._perf_stats)

    def get_active_patterns(self) -> Dict[str, List[HarmonicPattern]]:
        """
        Get currently active patterns.
        
        Returns:
            Dictionary of active patterns
        """
        with self._lock:
            return {k: list(v) for k, v in self._active_patterns.items()}

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self._perf_stats = {
                'pattern_detection_time': 0.0,
                'swing_detection_time': 0.0,
                'total_patterns_detected': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }

    def reset_cache(self) -> None:
        """Clear caches."""
        with self._lock:
            self._swing_cache.clear()
            self._pattern_cache.clear()

    def detect_signals(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect harmonic pattern signals for CDFA integration.
        Expected interface method for pattern detectors.
        
        Args:
            dataframe: Input DataFrame with OHLC data
            
        Returns:
            Dict containing pattern detection results and signals
        """
        try:
            # Detect harmonic patterns
            patterns = self.detect_patterns(dataframe)
            
            # Count total patterns detected
            total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
            
            # Calculate signal strength based on pattern presence and quality
            signal_strength = 0.5  # Default neutral
            confidence = 0.5
            
            if total_patterns > 0:
                # Calculate average pattern quality
                all_patterns = []
                for pattern_list in patterns.values():
                    all_patterns.extend(pattern_list)
                
                if all_patterns:
                    avg_quality = np.mean([p.quality for p in all_patterns])
                    recent_patterns = [p for p in all_patterns if p.completion_index >= len(dataframe) - 5]
                    
                    # Signal strength based on pattern quality and recency
                    if recent_patterns:
                        recent_avg_quality = np.mean([p.quality for p in recent_patterns])
                        signal_strength = min(0.9, 0.5 + (recent_avg_quality - 0.5))
                        confidence = min(0.9, avg_quality)
                    else:
                        signal_strength = min(0.8, 0.5 + (avg_quality - 0.5) * 0.5)
                        confidence = avg_quality * 0.8
            
            # Prepare pattern summary for output
            pattern_summary = {}
            for pattern_type, pattern_list in patterns.items():
                if pattern_list:
                    pattern_summary[pattern_type] = {
                        'count': len(pattern_list),
                        'avg_quality': np.mean([p.quality for p in pattern_list]),
                        'recent_patterns': len([p for p in pattern_list if p.completion_index >= len(dataframe) - 10])
                    }
            
            return {
                "signal": float(signal_strength),
                "confidence": float(confidence),
                "total_patterns": total_patterns,
                "pattern_types": list(patterns.keys()),
                "pattern_summary": pattern_summary,
                "detection_type": "fibonacci_harmonic",
                "data_points": len(dataframe)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci pattern signal detection: {e}")
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "detection_type": "fibonacci_harmonic"
            }


# ========================= Advanced Pattern Analysis =========================

class FibonacciPatternAnalyzer:
    """
    Advanced pattern analysis integrating with FibonacciAnalyzer.
    
    This class provides:
    1. Integration with FibonacciAnalyzer
    2. Multi-timeframe pattern confluence analysis
    3. Pattern confirmation with volume and price action
    4. Risk/reward assessment for detected patterns
    5. Pattern statistics and performance tracking
    """
    
    def __init__(self, fibonacci_analyzer=None, config: Optional[PatternDetectionConfig] = None):
        """
        Initialize pattern analyzer.
        
        Args:
            fibonacci_analyzer: FibonacciAnalyzer instance
            config: Pattern detection configuration
        """
        self.fibonacci_analyzer = fibonacci_analyzer
        self.detector = FibonacciPatternDetector(config)
        self.logger = logger
        
        # Pattern statistics
        self._pattern_stats = {
            pattern_name: {
                'total_detected': 0,
                'successful': 0,
                'failed': 0,
                'avg_quality': 0.0,
                'avg_profit_factor': 0.0
            }
            for pattern_name in self.detector.patterns
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.debug("FibonacciPatternAnalyzer initialized")
    
    def analyze_dataframe(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Analyze dataframe for patterns and add analysis columns.
        
        Args:
            dataframe: Input DataFrame
            metadata: Additional metadata
            
        Returns:
            DataFrame with pattern analysis columns
        """
        # Add pattern columns from detector
        dataframe = self.detector.add_pattern_columns(dataframe)
        
        # If fibonacci_analyzer is available, add retracement interactions
        if self.fibonacci_analyzer is not None:
            dataframe = self._add_fib_confluence(dataframe)
        
        # Add volume confirmation
        if 'volume' in dataframe.columns:
            dataframe = self._add_volume_confirmation(dataframe)
        
        # Add performance columns
        dataframe = self._add_performance_columns(dataframe)
        
        return dataframe
    
    def _add_fib_confluence(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fibonacci confluence analysis to dataframe.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with confluence columns
        """
        # Check for retracement columns from fibonacci_analyzer
        fib_cols = [col for col in dataframe.columns if col.startswith('fib_retr_') or col.startswith('fib_ext_')]
        
        if not fib_cols:
            dataframe['fib_pattern_confluence'] = 0.0
            return dataframe
        
        # Initialize confluence column
        dataframe['fib_pattern_confluence'] = 0.0
        
        # For each pattern type
        for pattern_name in self.detector.patterns:
            pattern_col = f'pattern_{pattern_name}'
            
            if pattern_col not in dataframe.columns:
                continue
                
            # Find rows with detected patterns
            pattern_rows = dataframe[dataframe[pattern_col] > 0].index
            
            for row in pattern_rows:
                # Get pattern quality
                quality = dataframe.loc[row, pattern_col]
                
                # Get the current price
                price = dataframe.loc[row, 'close']
                
                # Get Fibonacci levels
                fib_levels = [dataframe.loc[row, col] for col in fib_cols if pd.notna(dataframe.loc[row, col])]
                
                # Calculate minimum distance to any Fibonacci level
                min_distance = float('inf')
                for level in fib_levels:
                    if level > 0:
                        distance = abs(price - level) / price
                        min_distance = min(min_distance, distance)
                
                # Convert to confluence score
                if min_distance < float('inf'):
                    tolerance = 0.005  # 0.5% tolerance
                    confluence = max(0.0, 1.0 - min_distance / tolerance)
                    
                    # Update dataframe
                    dataframe.loc[row, 'fib_pattern_confluence'] = confluence
        
        return dataframe
    
    def _add_volume_confirmation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume confirmation for patterns.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with volume confirmation columns
        """
        # Initialize volume confirmation column
        dataframe['pattern_volume_confirmation'] = 0.0
        
        # Calculate average volume
        dataframe['avg_volume'] = dataframe['volume'].rolling(window=20).mean()
        
        # For each pattern type
        for pattern_name in self.detector.patterns:
            pattern_col = f'pattern_{pattern_name}'
            
            if pattern_col not in dataframe.columns:
                continue
                
            # Find rows with detected patterns
            pattern_rows = dataframe[dataframe[pattern_col] > 0].index
            
            for row in pattern_rows:
                # Check if we have volume data
                if row < 0 or row >= len(dataframe):
                    continue
                    
                # Calculate volume ratio at pattern completion (D point)
                volume = dataframe.loc[row, 'volume']
                avg_volume = dataframe.loc[row, 'avg_volume']
                
                if avg_volume > 0:
                    volume_ratio = volume / avg_volume
                    
                    # Calculate confirmation score (0-1)
                    # Higher volume = better confirmation
                    confirmation = min(1.0, volume_ratio / 2.0)
                    
                    # Update dataframe
                    dataframe.loc[row, 'pattern_volume_confirmation'] = confirmation
        
        return dataframe
    
    def _add_performance_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern performance analysis columns.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with performance columns
        """
        # Initialize performance columns
        dataframe['pattern_signal'] = 0.0
        dataframe['pattern_risk_reward'] = np.nan
        dataframe['pattern_profit_potential'] = np.nan
        
        # For each pattern type
        for pattern_name in self.detector.patterns:
            pattern_col = f'pattern_{pattern_name}'
            target_col = f'pattern_{pattern_name}_target'
            stop_col = f'pattern_{pattern_name}_stop'
            
            if not all(col in dataframe.columns for col in [pattern_col, target_col, stop_col]):
                continue
                
            # Find rows with detected patterns
            pattern_rows = dataframe[dataframe[pattern_col] > 0].index
            
            for row in pattern_rows:
                # Skip invalid rows
                if row < 0 or row >= len(dataframe):
                    continue
                    
                # Get pattern quality
                quality = dataframe.loc[row, pattern_col]
                
                # Get current price, target and stop
                price = dataframe.loc[row, 'close']
                target = dataframe.loc[row, target_col]
                stop = dataframe.loc[row, stop_col]
                
                # Calculate risk/reward ratio
                if pd.notna(target) and pd.notna(stop) and price > 0:
                    # Determine if bullish or bearish
                    is_bullish = True  # Default
                    
                    # Use best pattern column if available
                    if 'pattern_best' in dataframe.columns and dataframe.loc[row, 'pattern_best'] is not None:
                        best_pattern = dataframe.loc[row, 'pattern_best']
                        
                        # Check pattern trend from detected patterns
                        patterns = self.detector.detect_patterns(dataframe)
                        for pattern in patterns.get(best_pattern, []):
                            if pattern.points['D'].index == row:
                                is_bullish = pattern.trend == 'bullish'
                                break
                    
                    # Calculate risk and reward
                    if is_bullish:
                        reward = target - price
                        risk = price - stop
                    else:
                        reward = price - target
                        risk = stop - price
                    
                    # Calculate ratio if risk is non-zero
                    if abs(risk) > 1e-6:
                        risk_reward = abs(reward / risk)
                        dataframe.loc[row, 'pattern_risk_reward'] = risk_reward
                        
                        # Calculate profit potential
                        profit_potential = reward / price
                        dataframe.loc[row, 'pattern_profit_potential'] = profit_potential
                        
                        # Calculate overall signal strength
                        # Combine quality, confluence, volume confirmation and risk/reward
                        signal_strength = 0.0
                        component_count = 0
                        
                        # Pattern quality
                        signal_strength += quality
                        component_count += 1
                        
                        # Fibonacci confluence if available
                        if 'fib_pattern_confluence' in dataframe.columns:
                            confluence = dataframe.loc[row, 'fib_pattern_confluence']
                            signal_strength += confluence
                            component_count += 1
                        
                        # Volume confirmation if available
                        if 'pattern_volume_confirmation' in dataframe.columns:
                            volume_conf = dataframe.loc[row, 'pattern_volume_confirmation']
                            signal_strength += volume_conf
                            component_count += 1
                        
                        # Risk/reward factor (capped at 1.0)
                        rr_factor = min(1.0, risk_reward / 3.0)
                        signal_strength += rr_factor
                        component_count += 1
                        
                        # Calculate average
                        if component_count > 0:
                            signal_strength /= component_count
                            
                            # Set signal direction (-1 to 1)
                            signal = signal_strength * (1.0 if is_bullish else -1.0)
                            dataframe.loc[row, 'pattern_signal'] = signal
        
        return dataframe
    
    def multi_timeframe_analysis(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform multi-timeframe pattern analysis.
        
        Args:
            dataframes: Dictionary of timeframe to dataframe
            
        Returns:
            Multi-timeframe analysis results
        """
        results = {
            'confluent_patterns': [],
            'strongest_signal': None,
            'mtf_strength': 0.0,
            'timeframe_weights': {},
            'confluence_levels': []
        }
        
        # Analyze each timeframe
        timeframe_patterns = {}
        timeframe_signals = {}
        
        for timeframe, dataframe in dataframes.items():
            # Detect patterns
            patterns = self.detector.detect_patterns(dataframe)
            timeframe_patterns[timeframe] = patterns
            
            # Get last row signal
            if not dataframe.empty and 'pattern_signal' in dataframe.columns:
                signal = dataframe['pattern_signal'].iloc[-1]
                timeframe_signals[timeframe] = signal
        
        # Calculate timeframe weights
        total_weight = 0.0
        
        # Assign weights based on timeframe (higher timeframes get higher weights)
        for timeframe in dataframes:
            # Parse timeframe into minutes
            minutes = self._timeframe_to_minutes(timeframe)
            # Weight is logarithmic with timeframe minutes
            weight = math.log(minutes + 1) / math.log(1440 + 1)  # Normalize by log(1440) (daily)
            results['timeframe_weights'][timeframe] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for timeframe in results['timeframe_weights']:
                results['timeframe_weights'][timeframe] /= total_weight
        
        # Find confluent patterns (same pattern across multiple timeframes)
        pattern_timecounts = {}
        
        for timeframe, patterns in timeframe_patterns.items():
            for pattern_name, pattern_list in patterns.items():
                if pattern_list:  # If we have detected patterns
                    if pattern_name not in pattern_timecounts:
                        pattern_timecounts[pattern_name] = 0
                    pattern_timecounts[pattern_name] += 1
        
        # Add patterns found in multiple timeframes
        for pattern_name, count in pattern_timecounts.items():
            if count > 1:
                pattern_instances = []
                for timeframe, patterns in timeframe_patterns.items():
                    pattern_instances.extend(patterns.get(pattern_name, []))
                
                results['confluent_patterns'].append({
                    'pattern': pattern_name,
                    'timeframes': count,
                    'instances': len(pattern_instances),
                    'avg_quality': sum(p.quality for p in pattern_instances) / len(pattern_instances) if pattern_instances else 0.0
                })
        
        # Calculate weighted signal strength across timeframes
        weighted_signal = 0.0
        total_signal_weight = 0.0
        
        for timeframe, signal in timeframe_signals.items():
            if pd.notna(signal):
                weight = results['timeframe_weights'].get(timeframe, 0.0)
                weighted_signal += signal * weight
                total_signal_weight += weight
        
        if total_signal_weight > 0:
            mtf_strength = weighted_signal / total_signal_weight
            results['mtf_strength'] = mtf_strength
            
            # Determine strongest signal
            if abs(mtf_strength) > 0.3:  # Minimum threshold
                direction = 'bullish' if mtf_strength > 0 else 'bearish'
                results['strongest_signal'] = {
                    'direction': direction,
                    'strength': abs(mtf_strength),
                    'timeframes': len(timeframe_signals)
                }
        
        # Find confluence price levels
        all_levels = []
        
        for timeframe, dataframe in dataframes.items():
            # Get Fibonacci levels
            fib_cols = [col for col in dataframe.columns if col.startswith('fib_retr_') or col.startswith('fib_ext_')]
            
            if dataframe.empty or not fib_cols:
                continue
                
            # Get last row levels
            row = dataframe.iloc[-1]
            for col in fib_cols:
                if pd.notna(row[col]):
                    level = row[col]
                    all_levels.append((level, timeframe, col))
        
        # Find clusters of levels (levels within 0.5% of each other)
        clusters = []
        tolerance = 0.005  # 0.5%
        
        for level, timeframe, col_name in all_levels:
            # Find or create cluster
            found_cluster = False
            
            for cluster in clusters:
                cluster_avg = sum(l[0] for l in cluster) / len(cluster)
                
                if abs(level - cluster_avg) / cluster_avg < tolerance:
                    cluster.append((level, timeframe, col_name))
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append([(level, timeframe, col_name)])
        
        # Add confluence levels (clusters with more than one level)
        for cluster in clusters:
            if len(cluster) > 1:
                # Calculate average level
                avg_level = sum(l[0] for l in cluster) / len(cluster)
                
                # Count unique timeframes
                timeframes = set(l[1] for l in cluster)
                
                results['confluence_levels'].append({
                    'price': avg_level,
                    'count': len(cluster),
                    'timeframes': len(timeframes),
                    'timeframe_list': list(timeframes),
                    'deviation': max(abs(l[0] - avg_level) / avg_level for l in cluster)
                })
        
        # Sort confluence levels by count (descending)
        results['confluence_levels'].sort(key=lambda x: x['count'], reverse=True)
        
        return results
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '15m', '1d')
            
        Returns:
            Minutes represented by timeframe
        """
        # Default to 60 minutes (1h) if parsing fails
        minutes = 60
        
        try:
            # Parse timeframe
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                minutes = int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                minutes = int(timeframe[:-1]) * 1440  # 24 * 60
            elif timeframe.endswith('w'):
                minutes = int(timeframe[:-1]) * 10080  # 7 * 24 * 60
        except ValueError:
            pass
            
        return minutes
    
    def track_pattern_performance(self, dataframe: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Track and analyze the performance of detected patterns.
        
        Args:
            dataframe: Input DataFrame
            lookback: Number of candles to look back for pattern outcomes
            
        Returns:
            Pattern performance statistics
        """
        results = {
            'pattern_stats': {},
            'total_patterns': 0,
            'successful_rate': 0.0,
            'avg_risk_reward': 0.0,
            'avg_profit_factor': 0.0
        }
        
        # For each pattern type
        for pattern_name in self.detector.patterns:
            pattern_col = f'pattern_{pattern_name}'
            target_col = f'pattern_{pattern_name}_target'
            stop_col = f'pattern_{pattern_name}_stop'
            
            if not all(col in dataframe.columns for col in [pattern_col, target_col, stop_col]):
                continue
                
            # Find rows with detected patterns
            pattern_rows = dataframe[dataframe[pattern_col] > 0].index
            
            # Initialize stats
            pattern_stats = {
                'total': len(pattern_rows),
                'target_reached': 0,
                'stop_reached': 0,
                'inconclusive': 0,
                'avg_risk_reward': 0.0,
                'success_rate': 0.0,
                'avg_bars_to_target': 0.0,
                'avg_bars_to_stop': 0.0
            }
            
            # Analyze each pattern
            for i, row in enumerate(pattern_rows):
                # Skip patterns in the last lookback candles
                row_idx = dataframe.index.get_loc(row)
                if row_idx >= len(dataframe) - lookback:
                    continue
                
                # Get pattern data
                price = dataframe.loc[row, 'close']
                target = dataframe.loc[row, target_col]
                stop = dataframe.loc[row, stop_col]
                quality = dataframe.loc[row, pattern_col]
                
                # Skip if target or stop is missing
                if pd.isna(target) or pd.isna(stop):
                    continue
                
                # Determine if bullish or bearish
                is_bullish = True  # Default
                
                # Use best pattern column if available
                if 'pattern_best' in dataframe.columns and dataframe.loc[row, 'pattern_best'] is not None:
                    best_pattern = dataframe.loc[row, 'pattern_best']
                    
                    # Check pattern trend from detected patterns
                    patterns = self.detector.detect_patterns(dataframe.iloc[:row_idx+1])
                    for pattern in patterns.get(best_pattern, []):
                        if pattern.points['D'].index == row_idx:
                            is_bullish = pattern.trend == 'bullish'
                            break
                
                # Check future price action
                bars_to_target = None
                bars_to_stop = None
                
                for future_idx in range(row_idx + 1, min(row_idx + lookback + 1, len(dataframe))):
                    future_price = dataframe['close'].iloc[future_idx]
                    future_high = dataframe['high'].iloc[future_idx]
                    future_low = dataframe['low'].iloc[future_idx]
                    
                    # Check if target reached
                    if is_bullish and future_high >= target and bars_to_target is None:
                        bars_to_target = future_idx - row_idx
                    elif not is_bullish and future_low <= target and bars_to_target is None:
                        bars_to_target = future_idx - row_idx
                    
                    # Check if stop reached
                    if is_bullish and future_low <= stop and bars_to_stop is None:
                        bars_to_stop = future_idx - row_idx
                    elif not is_bullish and future_high >= stop and bars_to_stop is None:
                        bars_to_stop = future_idx - row_idx
                
                # Determine outcome
                if bars_to_target is not None and (bars_to_stop is None or bars_to_target <= bars_to_stop):
                    # Target reached first
                    pattern_stats['target_reached'] += 1
                    pattern_stats['avg_bars_to_target'] += bars_to_target
                elif bars_to_stop is not None:
                    # Stop reached first
                    pattern_stats['stop_reached'] += 1
                    pattern_stats['avg_bars_to_stop'] += bars_to_stop
                else:
                    # Neither reached within lookback
                    pattern_stats['inconclusive'] += 1
            
            # Calculate averages
            if pattern_stats['target_reached'] > 0:
                pattern_stats['avg_bars_to_target'] /= pattern_stats['target_reached']
            
            if pattern_stats['stop_reached'] > 0:
                pattern_stats['avg_bars_to_stop'] /= pattern_stats['stop_reached']
            
            # Calculate success rate
            conclusive = pattern_stats['target_reached'] + pattern_stats['stop_reached']
            if conclusive > 0:
                pattern_stats['success_rate'] = pattern_stats['target_reached'] / conclusive
            
            # Calculate average risk/reward
            risk_rewards = []
            for row in pattern_rows:
                price = dataframe.loc[row, 'close']
                target = dataframe.loc[row, target_col]
                stop = dataframe.loc[row, stop_col]
                
                if pd.notna(target) and pd.notna(stop) and price > 0:
                    is_bullish = True  # Default assumption
                    
                    # Calculate risk and reward
                    if is_bullish:
                        reward = target - price
                        risk = price - stop
                    else:
                        reward = price - target
                        risk = stop - price
                    
                    # Calculate ratio if risk is non-zero
                    if abs(risk) > 1e-6:
                        risk_rewards.append(abs(reward / risk))
            
            if risk_rewards:
                pattern_stats['avg_risk_reward'] = sum(risk_rewards) / len(risk_rewards)
            
            # Add to results
            results['pattern_stats'][pattern_name] = pattern_stats
            results['total_patterns'] += pattern_stats['total']
        
        # Calculate overall stats
        target_reached = sum(stats['target_reached'] for stats in results['pattern_stats'].values())
        stop_reached = sum(stats['stop_reached'] for stats in results['pattern_stats'].values())
        
        # Calculate overall success rate
        conclusive = target_reached + stop_reached
        if conclusive > 0:
            results['successful_rate'] = target_reached / conclusive
        
        # Calculate average risk/reward
        avg_rr = [stats['avg_risk_reward'] for stats in results['pattern_stats'].values() if stats['avg_risk_reward'] > 0]
        if avg_rr:
            results['avg_risk_reward'] = sum(avg_rr) / len(avg_rr)
        
        # Calculate profit factor
        if stop_reached > 0:
            results['avg_profit_factor'] = (target_reached * results['avg_risk_reward']) / stop_reached
        
        return results


# ========================= Fibonacci Analyzer Integration =========================

def enhance_fibonacci_analyzer(analyzer):
    """
    Enhance FibonacciAnalyzer with pattern detection capabilities.
    
    Args:
        analyzer: FibonacciAnalyzer instance
    """
    # Create pattern analyzer
    pattern_analyzer = FibonacciPatternAnalyzer(analyzer)
    
    # Add analyzer as attribute
    analyzer.pattern_analyzer = pattern_analyzer
    
    # Add pattern detection method
    def detect_patterns(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Detect harmonic patterns in dataframe.
        
        Args:
            dataframe: Input DataFrame
            metadata: Additional metadata
            
        Returns:
            DataFrame with pattern detection columns
        """
        return self.pattern_analyzer.analyze_dataframe(dataframe, metadata)
    
    # Add multi-timeframe pattern analysis method
    def analyze_pattern_confluence(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze pattern confluence across multiple timeframes.
        
        Args:
            dataframes: Dictionary of timeframe to dataframe
            
        Returns:
            Pattern confluence analysis results
        """
        return self.pattern_analyzer.multi_timeframe_analysis(dataframes)
    
    # Add pattern performance tracking method
    def track_pattern_performance(self, dataframe: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Track and analyze pattern performance.
        
        Args:
            dataframe: Input DataFrame
            lookback: Number of candles to look back
            
        Returns:
            Pattern performance statistics
        """
        return self.pattern_analyzer.track_pattern_performance(dataframe, lookback)
    
    # Add pattern to original Fibonacci analyzer
    import types
    analyzer.detect_patterns = types.MethodType(detect_patterns, analyzer)
    analyzer.analyze_pattern_confluence = types.MethodType(analyze_pattern_confluence, analyzer)
    analyzer.track_pattern_performance = types.MethodType(track_pattern_performance, analyzer)
    
    return analyzer