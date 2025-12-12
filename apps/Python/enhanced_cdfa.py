#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Cognitive Diversity Fusion Analysis (CDFA) System

This module implements a comprehensive system for signal fusion and analysis
using the principles of cognitive diversity, self-organized criticality,
and complex adaptive systems. It provides integration with multiple analyzers,
machine learning capabilities, and visualization tools.

Key features:
- Numba JIT acceleration for performance-critical functions
- Vectorized operations using NumPy
- Redis integration for distributed communication
- ML/RL-based signal processing and weight optimization
- Adaptive learning with feedback loops
- Visual analytics for performance monitoring

Author: ashina (original), Enhanced version created on May 4, 2025
"""

import numpy as np
import warnings
import logging
import time
import threading
import math
import hashlib
import weakref
import queue
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TypeVar, Set, NamedTuple, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, partial, wraps
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# ---- Optional dependencies with graceful fallbacks ----

# SciPy for statistical computations
try:
    from scipy.stats import spearmanr, kendalltau, pearsonr, entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some diversity methods (Spearman, Pearson, KL) might be approximate or unavailable.")
    def spearmanr(*args, **kwargs): return (np.nan, np.nan)
    def kendalltau(*args, **kwargs): return (np.nan, np.nan)
    def pearsonr(*args, **kwargs): return (np.nan, np.nan)
    def scipy_entropy(*args, **kwargs): return np.nan

# Numba for JIT compilation
try:
    import numba as nb
    from numba import njit, float64, int64, boolean, prange
    NUMBA_AVAILABLE = True
    
    # Define Numba accelerated functions
    @njit(float64[:](float64[:]), cache=True, fastmath=True)
    def _normalize_scores_numba(scores):
        """Accelerated score normalization"""
        result = np.zeros_like(scores)
        # Find min/max of finite values
        min_val = np.inf
        max_val = -np.inf
        has_finite = False
        
        for i in range(len(scores)):
            if np.isfinite(scores[i]):
                has_finite = True
                if scores[i] < min_val:
                    min_val = scores[i]
                if scores[i] > max_val:
                    max_val = scores[i]
        
        # If no finite values or range is zero, return neutral values
        if not has_finite or max_val - min_val < 1e-9:
            for i in range(len(scores)):
                result[i] = 0.5
            return result
            
        # Normalize finite values
        val_range = max_val - min_val
        for i in range(len(scores)):
            if np.isfinite(scores[i]):
                result[i] = max(0.0, min(1.0, (scores[i] - min_val) / val_range))
            else:
                result[i] = 0.5
                
        return result
        
    @njit(float64(float64[:], float64[:]), cache=True, fastmath=True)
    def _kendall_distance_numba(a, b):
        """Numba-accelerated Kendall tau distance calculation"""
        n = len(a)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i+1, n):
                a_diff = a[i] - a[j]
                b_diff = b[i] - b[j]
                
                if a_diff * b_diff > 0:
                    concordant += 1
                elif a_diff * b_diff < 0:
                    discordant += 1
                    
        # Return 0.5 if no (dis)concordant pairs found
        if concordant == 0 and discordant == 0:
            return 0.5
            
        # Calculate tau and normalize to [0,1]
        tau = (concordant - discordant) / (concordant + discordant)
        return (1.0 - tau) / 2.0
        
    @njit(cache=True)
    def _calculate_volatility_clustering(squared_returns):
        """JIT-optimized volatility clustering calculation"""
        n = len(squared_returns)
        if n <= 1: 
            return 0.1
            
        finite_sq_returns = squared_returns[np.isfinite(squared_returns)]
        n_finite = len(finite_sq_returns)
        
        if n_finite <= 1: 
            return 0.1
            
        mean_sq = np.mean(finite_sq_returns)
        numerator = 0.0
        variance_sum_sq = np.sum((finite_sq_returns - mean_sq)**2)
        
        for i in range(1, n_finite):
            diff_i = finite_sq_returns[i] - mean_sq
            diff_i_1 = finite_sq_returns[i-1] - mean_sq
            numerator += diff_i * diff_i_1
            
        if variance_sum_sq <= 1e-9: 
            return 0.1
            
        autocorr = numerator / variance_sum_sq
        return min(0.9, abs(autocorr))
    
    @njit(cache=True)
    def _calculate_hill_estimator(sorted_data):
        """JIT-optimized Hill estimator calculation"""
        n = len(sorted_data)
        if n <= 1: 
            return 3.0  # Default tail index
            
        positive_data = sorted_data[sorted_data > 1e-12]
        n_pos = len(positive_data)
        
        if n_pos <= 1: 
            return 3.0
            
        log_data = np.log(positive_data)
        log_threshold = log_data[-1]
        mean_excess = 0.0
        
        for i in range(n_pos): 
            mean_excess += log_data[i] - log_threshold
            
        mean_excess /= n_pos
        
        if mean_excess <= 1e-9: 
            return 3.0
            
        return 1.0 / mean_excess
        
    @njit(cache=True)
    def _dtw_distance_numba(s1, s2, window):
        """Numba-accelerated DTW distance calculation"""
        n, m = len(s1), len(s2)
        if n == 0 or m == 0:
            return np.inf
            
        # Create DTW matrix (use float32 to reduce memory usage)
        dtw_matrix = np.full((n+1, m+1), np.inf, dtype=np.float32)
        dtw_matrix[0, 0] = 0.0
        
        # Adjust window size (Sakoe-Chiba band)
        w = max(window, abs(n - m))
        
        # Fill the matrix
        for i in range(1, n+1):
            # Apply window constraint
            start_j = max(1, i-w)
            end_j = min(m+1, i+w+1)
            
            for j in range(start_j, end_j):
                cost = abs(s1[i-1] - s2[j-1])
                last_min = min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
                dtw_matrix[i, j] = cost + last_min
                
        # Return normalized distance
        return dtw_matrix[n, m] / (n + m)
    
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Performance optimizations will be disabled.")
    
    # Define dummy functions for graceful fallback
    def _normalize_scores_numba(scores):
        """Fallback normalization function"""
        return scores
        
    def _kendall_distance_numba(a, b):
        """Fallback Kendall distance function"""
        return 0.0
        
    def _calculate_volatility_clustering(squared_returns):
        """Fallback volatility clustering function"""
        return 0.1
        
    def _calculate_hill_estimator(sorted_data):
        """Fallback Hill estimator function"""
        return 3.0
        
    def _dtw_distance_numba(s1, s2, window):
        """Fallback DTW distance function"""
        return np.inf
        
    # Create dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    # Define dummy types and functions
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
            
    nb = DummyModule()
    prange = range
    float64 = int64 = boolean = lambda x: x

# Redis for distributed communication
try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Integration features will be disabled.")

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")
    
    # Create dummy figure class for typing
    class Figure:
        pass

# ML Libraries
try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ML features will be limited.")
    
    # Create dummy class for typing
    class BaseEstimator:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural network features will be limited.")

# ---- Core Enumerations and Configuration ----
class DiversityMethod(Enum):
    """Methods to calculate cognitive diversity between systems."""
    KENDALL = auto()
    SPEARMAN = auto()
    PEARSON = auto()
    RSC = auto()  # Rank Score Characteristic
    KL_DIVERGENCE = auto()
    JSD = auto()  # Jensen-Shannon Divergence
    
    def __str__(self) -> str: 
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'DiversityMethod':
        """Create enum from string representation."""
        s_upper = s.upper().replace("-", "_")
        for item in cls:
            if item.name == s_upper: 
                return item
        for item in cls:
            if item.name.startswith(s_upper): 
                return item
        warnings.warn(f"Unknown DiversityMethod '{s}', defaulting to KENDALL.")
        return cls.KENDALL

class FusionType(Enum):
    """Types of fusion methods available."""
    SCORE = auto()  # Score-based fusion
    RANK = auto()   # Rank-based fusion
    HYBRID = auto() # Hybrid score/rank fusion
    LAYERED = auto() # Multi-layer fusion
    ML = auto()     # Machine learning based fusion
    RL = auto()     # Reinforcement learning based fusion
    
    def __str__(self) -> str: 
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'FusionType':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper: 
                return item
        for item in cls:
            if item.name.startswith(s_upper): 
                return item
        raise ValueError(f"Unknown FusionType: {s}")

from enum import Enum, auto

class SignalType(Enum):
    """
    Enum defining the types of signals that can be generated by the analysis system.
    Used to categorize signals by their output format and interpretation.
    """
    # Basic signal types
    BINARY = auto()         # Simple buy/sell signals (-1, 0, 1)
    CONTINUOUS = auto()     # Continuous values typically in range [-1, 1] or [0, 1]
    CATEGORICAL = auto()    # Multiple discrete categories (e.g. regimes)
    
    # Advanced signal types
    PROBABILITY = auto()    # Probability scores [0, 1]
    MOMENTUM = auto()       # Directional strength indicators
    TREND = auto()          # Trend identification signals
    OSCILLATOR = auto()     # Oscillator signals (typically mean-reverting)
    VOLATILITY = auto()     # Volatility-based signals
    REGIME = auto()         # Market regime indicators
    CYCLE = auto()          # Cycle-based timing signals
    DIVERGENCE = auto()     # Divergence signals between components
    CORRELATION = auto()    # Correlation-based signals
    ANOMALY = auto()        # Anomaly/outlier detection signals
    PATTERN = auto()        # Pattern recognition signals
    COMPOSITE = auto()      # Combined signals from multiple sources
    
    @classmethod
    def get_normalization(cls, signal_type):
        """
        Returns the appropriate normalization range for a given signal type.
        
        Returns:
            tuple: (min_value, max_value) representing the normalization range
        """
        normalization_ranges = {
            cls.BINARY: (-1, 1),
            cls.CONTINUOUS: (-1, 1),
            cls.CATEGORICAL: (0, None),  # Upper bound depends on categories
            cls.PROBABILITY: (0, 1),
            cls.MOMENTUM: (-1, 1),
            cls.TREND: (-1, 1),
            cls.OSCILLATOR: (-1, 1),
            cls.VOLATILITY: (0, None),  # Unbounded upper limit
            cls.REGIME: (0, None),      # Depends on regime encoding
            cls.CYCLE: (0, 1),         # Normalized cycle position
            cls.DIVERGENCE: (-1, 1),
            cls.CORRELATION: (-1, 1),
            cls.ANOMALY: (0, 1),
            cls.PATTERN: (0, 1),
            cls.COMPOSITE: (-1, 1)
        }
        return normalization_ranges.get(signal_type, (-1, 1))
    
    @classmethod
    def requires_thresholds(cls, signal_type):
        """
        Determines if a signal type typically requires thresholds for interpretation.
        
        Returns:
            bool: True if thresholds are typically used
        """
        threshold_types = {
            cls.CONTINUOUS, cls.PROBABILITY, cls.MOMENTUM, 
            cls.OSCILLATOR, cls.VOLATILITY, cls.DIVERGENCE,
            cls.CORRELATION, cls.ANOMALY, cls.PATTERN
        }
        return signal_type in threshold_types
    
    @classmethod
    def get_default_thresholds(cls, signal_type):
        """
        Returns default threshold values for signal interpretation.
        
        Returns:
            dict: Dictionary of threshold levels
        """
        if signal_type == cls.BINARY:
            return {}  # Binary signals don't need thresholds
            
        elif signal_type in (cls.CONTINUOUS, cls.MOMENTUM, cls.TREND):
            return {
                "strong_buy": 0.7,
                "buy": 0.3,
                "neutral": 0.0,
                "sell": -0.3,
                "strong_sell": -0.7
            }
            
        elif signal_type in (cls.PROBABILITY, cls.OSCILLATOR, cls.ANOMALY, cls.PATTERN):
            return {
                "overbought": 0.8,
                "high": 0.7,
                "neutral_high": 0.6,
                "neutral": 0.5,
                "neutral_low": 0.4,
                "low": 0.3,
                "oversold": 0.2
            }
            
        elif signal_type == cls.VOLATILITY:
            return {
                "very_high": 2.0,
                "high": 1.5,
                "elevated": 1.2,
                "normal": 1.0,
                "low": 0.8,
                "very_low": 0.5
            }
            
        elif signal_type == cls.CORRELATION:
            return {
                "strong_positive": 0.7,
                "moderate_positive": 0.3,
                "uncorrelated": 0.0,
                "moderate_negative": -0.3,
                "strong_negative": -0.7
            }
            
        # Default thresholds
        return {
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3
        }
    
class MarketRegime(Enum):
    """Adaptive cycle phases from Panarchy theory."""
    GROWTH = auto()       # r-phase: Exploitation/Growth
    CONSERVATION = auto() # K-phase: Conservation
    RELEASE = auto()      # Ω-phase: Release/Creative Destruction
    REORGANIZATION = auto() # α-phase: Reorganization
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, s: str) -> 'MarketRegime':
        """Create enum from string representation."""
        s_lower = s.lower()
        
        # Direct mapping
        if s_lower in ("growth", "r-phase", "exploitation", "r"):
            return cls.GROWTH
        if s_lower in ("conservation", "k-phase", "k"):
            return cls.CONSERVATION
        if s_lower in ("release", "omega", "omega-phase", "ω", "ω-phase", "creative-destruction"):
            return cls.RELEASE
        if s_lower in ("reorganization", "alpha", "alpha-phase", "α", "α-phase", "renewal"):
            return cls.REORGANIZATION
            
        # Fallback to unknown
        return cls.UNKNOWN

class PatternRecWindow(Enum):
    """Window sizes for pattern recognition."""
    SMALL = 10
    MEDIUM = 20
    LARGE = 40
    XLARGE = 80

@dataclass
class ScoreData:
    """
    Enhanced ScoreData class with vectorized operations and Numba acceleration.
    """
    raw_scores: List[float]
    normalized_scores: Optional[List[float]] = None
    ranks: Optional[List[Union[int, float]]] = None  # Allow float for avg rank

    def __post_init__(self):
        if self.normalized_scores is None: 
            self._normalize_scores()
        if self.ranks is None: 
            self._calculate_ranks()

    def _normalize_scores(self):
        """Normalize scores with vectorized operations and optional Numba acceleration"""
        if not self.raw_scores:
            self.normalized_scores = []
            return
            
        # Convert to numpy array for vectorized operations
        scores_array = np.array(self.raw_scores, dtype=float)
        
        # Use Numba if available and enough data to be worthwhile
        if NUMBA_AVAILABLE and len(scores_array) > 10:
            # Use Numba-accelerated function
            normalized = _normalize_scores_numba(scores_array)
            self.normalized_scores = normalized.tolist()
        else:
            # Original implementation for small arrays or when Numba unavailable
            finite_mask = np.isfinite(scores_array)
            if not np.any(finite_mask):
                self.normalized_scores = [0.5] * len(self.raw_scores)
                return
                
            # Get min/max of finite values only
            finite_values = scores_array[finite_mask]
            min_s = np.min(finite_values)
            max_s = np.max(finite_values)
            s_range = max_s - min_s
            
            if s_range < 1e-9:
                self.normalized_scores = [0.5] * len(self.raw_scores)
            else:
                # Create normalized array with vectorized operations
                normalized = np.full_like(scores_array, 0.5)
                normalized[finite_mask] = np.clip((scores_array[finite_mask] - min_s) / s_range, 0.0, 1.0)
                self.normalized_scores = normalized.tolist()

    def _calculate_ranks(self):
        """Calculate ranks with vectorized operations"""
        if not self.raw_scores:
            self.ranks = []
            return
            
        # Convert to numpy array with indices for stable sorting
        n = len(self.raw_scores)
        scores_array = np.array(self.raw_scores, dtype=float)
        indices = np.arange(n)
        
        # Replace non-finite values with -inf for sorting
        sorted_pairs = np.column_stack((
            np.where(np.isfinite(scores_array), scores_array, -np.inf),
            indices
        ))
        
        # Sort descending by score, ascending by index for stable ranking
        sorted_pairs = sorted_pairs[np.lexsort((-sorted_pairs[:, 0], sorted_pairs[:, 1]))]
        
        # Calculate ranks (with handling for ties)
        ranks = np.zeros(n, dtype=float)
        i = 0
        
        while i < n:
            j = i
            current_score = sorted_pairs[i, 0]
            
            # Find tied values
            while j < n-1 and sorted_pairs[j+1, 0] == current_score:
                j += 1
                
            # Calculate average rank for tied group
            avg_rank = (i + j + 2) / 2  # +2 because rank is 1-based
            
            # Assign to all tied elements
            for k in range(i, j+1):
                orig_idx = int(sorted_pairs[k, 1])
                ranks[orig_idx] = avg_rank
                
            # Move to next group
            i = j + 1
            
        # Convert to appropriate type (int if whole number, float otherwise)
        self.ranks = [(int(r) if r == int(r) else r) for r in ranks]

class DiversityResult(NamedTuple):
    """Result from diversity calculation."""
    value: float
    method: DiversityMethod
    confidence: float

@dataclass
class CDFAConfig:
    """Enhanced Configuration for Cognitive Diversity & Fusion Analysis."""
    # Core parameters
    diversity_threshold: float = 0.3
    performance_threshold: float = 0.6
    default_diversity_method: DiversityMethod = DiversityMethod.KENDALL
    default_fusion_type: FusionType = FusionType.HYBRID
    
    # System behavior
    enable_caching: bool = True
    cache_size: int = 256
    parallelization_threshold: int = 5
    max_workers: int = 4
    min_signals_required: int = 2
    
    # Algorithm parameters
    rsc_scale_factor: float = 4.0
    expansion_factor: int = 2
    reduction_ratio: float = 0.5
    kl_epsilon: float = 1e-9
    kl_num_bins: int = 10
    adaptive_alpha_vol_sensitivity: float = 0.4
    
    # Weighting configuration
    diversity_weighting_scheme: str = "multiplicative"
    additive_weighting_perf_bias: float = 0.6
    
    # Optimization flags
    use_numba: bool = True
    use_vectorization: bool = True
    
    # Logging
    enable_logging: bool = True
    log_level: int = logging.INFO
    
    # Redis integration
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_channel_prefix: str = "cdfa:"
    signal_ttl: int = 3600  # Time-to-live for cached signals (seconds)
    update_queue_size: int = 100
    
    # ML/RL configuration
    enable_ml: bool = False
    ml_model_type: str = "rf"  # 'rf', 'sgd', 'nn', 'qstar'
    ml_update_interval: int = 300  # Seconds
    ml_batch_size: int = 64
    ml_learning_rate: float = 0.01
    ml_update_strategy: str = "sample"  # 'sample', 'window', 'importance'
    
    # Adaptive learning
    enable_adaptive_learning: bool = False
    feedback_window: int = 100
    learning_rate: float = 0.05
    performance_decay: float = 0.95
    
    # Visualization
    enable_visualization: bool = False
    plot_style: str = "darkgrid"
    max_plots_history: int = 50

# --- Redis Client for Communication ---
if REDIS_AVAILABLE:
    class CDFARedisClient:
        """Redis client for communication with other system components"""
        
        def __init__(self, host="localhost", port=6379, db=0, password=None, channel_prefix="cdfa:"):
            """Initialize Redis client for CDFA"""
            self.logger = logging.getLogger(f"{__name__}.CDFARedisClient")
            
            # Connection parameters
            self.host = host
            self.port = port
            self.db = db
            self.password = password
            self.channel_prefix = channel_prefix
            
            # Runtime state
            self._redis = None
            self._pubsub = None
            self._connected = False
            self._running = True
            self._lock = threading.RLock()
            
            # Message handling
            self._subscribers = {}
            self._listener_thread = None
            self._message_queue = queue.Queue(maxsize=500)
            self._processor_thread = None
            
            # Try to connect
            self.connect()
            
        def connect(self):
            """Establish connection to Redis"""
            try:
                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=3.0,
                    decode_responses=False
                )
                
                # Test connection
                self._redis.ping()
                self._connected = True
                
                # Setup pub/sub
                self._setup_pubsub()
                
                self.logger.info(f"Connected to Redis at {self.host}:{self.port}")
                return True
                
            except redis.RedisError as e:
                self._connected = False
                self.logger.error(f"Failed to connect to Redis: {e}")
                return False
                
        def _setup_pubsub(self):
            """Setup PubSub and subscription handling"""
            if not self._connected:
                return False
                
            # Create PubSub object
            self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
            
            # Start listener thread if not running
            if not self._listener_thread or not self._listener_thread.is_alive():
                self._listener_thread = threading.Thread(
                    target=self._listen_for_messages,
                    daemon=True
                )
                self._listener_thread.start()
                
            # Start processor thread if not running
            if not self._processor_thread or not self._processor_thread.is_alive():
                self._processor_thread = threading.Thread(
                    target=self._process_messages,
                    daemon=True
                )
                self._processor_thread.start()
                
            # Re-subscribe to channels
            with self._lock:
                for channel in self._subscribers.keys():
                    full_channel = f"{self.channel_prefix}{channel}"
                    self._pubsub.subscribe(full_channel)
                    
            return True
            
        def _listen_for_messages(self):
            """Background thread to listen for Redis messages"""
            self.logger.debug("Redis listener thread started")
            
            while self._running:
                try:
                    if not self._connected:
                        time.sleep(1.0)
                        continue
                        
                    # Get message with timeout
                    message = self._pubsub.get_message(timeout=0.5)
                    if message:
                        try:
                            # Queue message for processing
                            self._message_queue.put(message, block=False)
                        except queue.Full:
                            self.logger.warning("Message queue full, dropping message")
                            
                except redis.RedisError as e:
                    self.logger.error(f"Redis error in listener: {e}")
                    self._connected = False
                    time.sleep(1.0)
                    self.connect()
                    
                except Exception as e:
                    self.logger.error(f"Error in Redis listener: {e}")
                    time.sleep(0.1)
                    
            self.logger.debug("Redis listener thread stopped")
            
        def _process_messages(self):
            """Background thread to process queued messages"""
            self.logger.debug("Redis message processor thread started")
            
            while self._running:
                try:
                    # Get message from queue with timeout
                    message = self._message_queue.get(timeout=0.5)
                    
                    # Process message
                    self._handle_message(message)
                    
                    # Mark task as done
                    self._message_queue.task_done()
                    
                except queue.Empty:
                    pass
                    
                except Exception as e:
                    self.logger.error(f"Error processing Redis message: {e}")
                    time.sleep(0.1)
                    
            self.logger.debug("Redis message processor thread stopped")
            
        def _handle_message(self, message):
            """Process a single Redis message"""
            try:
                # Extract channel from message
                channel = message.get('channel', b'').decode('utf-8')
                
                # Remove prefix from channel
                if channel.startswith(self.channel_prefix):
                    channel = channel[len(self.channel_prefix):]
                else:
                    return
                    
                # Extract data
                data = message.get('data')
                if not data:
                    return
                    
                # Decode data with MessagePack
                try:
                    data = msgpack.unpackb(data, raw=False)
                except (TypeError, ValueError):
                    # Try to decode as string
                    try:
                        data = data.decode('utf-8')
                    except (AttributeError, UnicodeDecodeError):
                        self.logger.warning(f"Failed to decode message on channel {channel}")
                        return
                
                # Invoke subscribers
                with self._lock:
                    subscribers = self._subscribers.get(channel, [])
                    
                for callback in subscribers:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in Redis callback: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error handling Redis message: {e}")
                
        def subscribe(self, channel, callback):
            """Subscribe to a Redis channel"""
            if not callable(callback):
                raise ValueError("Callback must be callable")
                
            with self._lock:
                # Add callback to subscribers
                if channel not in self._subscribers:
                    self._subscribers[channel] = []
                    
                    # Subscribe to channel
                    if self._connected:
                        full_channel = f"{self.channel_prefix}{channel}"
                        self._pubsub.subscribe(full_channel)
                        
                self._subscribers[channel].append(callback)
                
            return True
            
        def unsubscribe(self, channel, callback=None):
            """Unsubscribe from a Redis channel"""
            with self._lock:
                # Check if channel exists
                if channel not in self._subscribers:
                    return False
                    
                if callback is None:
                    # Remove all subscribers
                    self._subscribers.pop(channel)
                    
                    # Unsubscribe from channel
                    if self._connected:
                        full_channel = f"{self.channel_prefix}{channel}"
                        self._pubsub.unsubscribe(full_channel)
                else:
                    # Remove specific callback
                    if callback in self._subscribers[channel]:
                        self._subscribers[channel].remove(callback)
                        
                    # If no subscribers left, unsubscribe
                    if not self._subscribers[channel]:
                        self._subscribers.pop(channel)
                        
                        # Unsubscribe from channel
                        if self._connected:
                            full_channel = f"{self.channel_prefix}{channel}"
                            self._pubsub.unsubscribe(full_channel)
                            
            return True
            
        def publish(self, channel, data):
            """Publish data to a Redis channel"""
            if not self._connected and not self.connect():
                return False
                
            try:
                # Encode data with MessagePack
                packed = msgpack.packb(data, use_bin_type=True)
                
                # Publish message
                full_channel = f"{self.channel_prefix}{channel}"
                self._redis.publish(full_channel, packed)
                
                return True
                
            except redis.RedisError as e:
                self.logger.error(f"Redis error in publish: {e}")
                self._connected = False
                return False
                
            except Exception as e:
                self.logger.error(f"Error publishing to Redis: {e}")
                return False
                
        def store_fusion_result(self, key, data, ttl=3600):
            """Store fusion result in Redis with expiration"""
            if not self._connected and not self.connect():
                return False
                
            try:
                # Encode data with MessagePack
                packed = msgpack.packb(data, use_bin_type=True)
                
                # Store data with expiration
                full_key = f"{self.channel_prefix}fusion:{key}"
                self._redis.set(full_key, packed, ex=ttl)
                
                return True
                
            except redis.RedisError as e:
                self.logger.error(f"Redis error in store_fusion_result: {e}")
                self._connected = False
                return False
                
            except Exception as e:
                self.logger.error(f"Error storing fusion result: {e}")
                return False
                
        def get_fusion_result(self, key):
            """Retrieve fusion result from Redis"""
            if not self._connected and not self.connect():
                return None
                
            try:
                # Get data
                full_key = f"{self.channel_prefix}fusion:{key}"
                data = self._redis.get(full_key)
                
                if not data:
                    return None
                    
                # Decode data with MessagePack
                return msgpack.unpackb(data, raw=False)
                
            except redis.RedisError as e:
                self.logger.error(f"Redis error in get_fusion_result: {e}")
                self._connected = False
                return None
                
            except Exception as e:
                self.logger.error(f"Error retrieving fusion result: {e}")
                return None
                
        def close(self):
            """Close Redis connections and stop threads"""
            self.logger.debug("Closing Redis client")
            
            # Stop threads
            self._running = False
            
            # Close PubSub
            if self._pubsub:
                try:
                    self._pubsub.close()
                except:
                    pass
                    
            # Close Redis connection
            if self._redis:
                try:
                    self._redis.close()
                except:
                    pass
                    
            # Wait for threads to stop
            if self._listener_thread and self._listener_thread.is_alive():
                self._listener_thread.join(timeout=0.5)
                
            if self._processor_thread and self._processor_thread.is_alive():
                self._processor_thread.join(timeout=0.5)
                
            self._connected = False
            self.logger.info("Redis client closed")

# --- ML/RL Integration Classes ---

class MLSignalProcessor:
    """
    Machine learning based signal processor that can enhance and optimize fusion weights.
    """
    def __init__(self, config: CDFAConfig = None):
        """Initialize ML signal processor with configuration."""
        self.config = config or CDFAConfig()
        self.logger = logging.getLogger(f"{__name__}.MLSignalProcessor")
        self._models = {}  # symbol -> model
        self._history = {}  # symbol -> historical data
        self._lock = threading.RLock()
        self._last_update = {}  # symbol -> timestamp
        
        # Check for ML libraries
        self._has_sklearn = SKLEARN_AVAILABLE
        self._has_torch = TORCH_AVAILABLE
        
        if not (self._has_sklearn or self._has_torch):
            self.logger.warning("ML libraries not available. ML functionality disabled.")
            
        # Initialize models dictionary
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models based on config."""
        if not (self._has_sklearn or self._has_torch):
            return
            
        # Clear models
        with self._lock:
            self._models = {}
        
    def _create_model(self, model_type: str, input_dim: int):
        """Create a new model of specified type."""
        if model_type == "rf" and self._has_sklearn:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "sgd" and self._has_sklearn:
            return SGDRegressor(
                loss="huber",
                learning_rate="adaptive",
                eta0=self.config.ml_learning_rate,
                max_iter=1000,
                random_state=42
            )
        elif model_type == "nn" and self._has_torch:
            return MLNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                output_dim=1,
                learning_rate=self.config.ml_learning_rate
            )
        else:
            self.logger.warning(f"Unknown model type '{model_type}' or required library unavailable.")
            # Default to RF if sklearn available
            if self._has_sklearn:
                return RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            return None
    
    def can_process(self) -> bool:
        """Check if ML processing is available."""
        return (self._has_sklearn or self._has_torch) and self.config.enable_ml
    
    def add_training_sample(self, symbol: str, features: Dict[str, float], target: float, timestamp: float = None):
        """Add a training sample to the history."""
        if not self.can_process():
            return False
        
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            if symbol not in self._history:
                self._history[symbol] = {
                    'features': [],
                    'targets': [],
                    'timestamps': []
                }
                
            self._history[symbol]['features'].append(features)
            self._history[symbol]['targets'].append(target)
            self._history[symbol]['timestamps'].append(timestamp)
            
        return True
        
    def train_model(self, symbol: str, force: bool = False) -> bool:
        """Train or update model with collected data."""
        if not self.can_process():
            return False
            
        # Check if we have enough data
        with self._lock:
            if symbol not in self._history:
                return False
                
            history = self._history[symbol]
            if len(history['features']) < 10:  # Need at least 10 samples
                return False
                
            # Check if update is needed
            current_time = time.time()
            last_update = self._last_update.get(symbol, 0)
            
            if not force and current_time - last_update < self.config.ml_update_interval:
                return False  # Skip update if recent
                
            # Prepare training data
            features_list = history['features']
            targets = history['targets']
            timestamps = history['timestamps']
            
        # Convert features to uniform format
        feature_names = set()
        for feature_dict in features_list:
            feature_names.update(feature_dict.keys())
            
        # Create feature matrix
        feature_names = sorted(list(feature_names))
        X = np.zeros((len(features_list), len(feature_names)))
        y = np.array(targets)
        
        for i, feature_dict in enumerate(features_list):
            for j, name in enumerate(feature_names):
                X[i, j] = feature_dict.get(name, 0.0)
                
        # Train or update model
        try:
            with self._lock:
                if symbol not in self._models or self._models[symbol] is None:
                    # Create new model
                    model = self._create_model(self.config.ml_model_type, len(feature_names))
                    if model is None:
                        return False
                        
                    # Train model
                    if isinstance(model, nn.Module):
                        model.fit(X, y, epochs=100)
                    else:
                        model.fit(X, y)
                        
                    # Store model and metadata
                    self._models[symbol] = {
                        'model': model,
                        'feature_names': feature_names
                    }
                else:
                    # Update existing model
                    model_data = self._models[symbol]
                    model = model_data['model']
                    old_features = model_data['feature_names']
                    
                    # Handle feature name changes
                    if old_features != feature_names:
                        # Need to recreate model with new features
                        model = self._create_model(self.config.ml_model_type, len(feature_names))
                        if model is None:
                            return False
                            
                        # Train from scratch
                        if isinstance(model, nn.Module):
                            model.fit(X, y, epochs=100)
                        else:
                            model.fit(X, y)
                    else:
                        # Update existing model (incremental learning)
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X, y)
                        elif isinstance(model, nn.Module):
                            model.update(X, y)
                        else:
                            # Refit for models without incremental learning
                            model.fit(X, y)
                            
                    # Update model data
                    self._models[symbol] = {
                        'model': model,
                        'feature_names': feature_names
                    }
                    
                # Update timestamp
                self._last_update[symbol] = current_time
                
            self.logger.info(f"Model for {symbol} trained/updated with {len(features_list)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            return False
            
    def predict_weights(self, symbol: str, features: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Predict optimal weights for fusion using the trained model.
        
        Args:
            symbol: Symbol identifier
            features: Dictionary of features with values
            
        Returns:
            Dictionary of predicted weights for each feature/signal
        """
        if not self.can_process():
            return {}
            
        with self._lock:
            if symbol not in self._models or self._models[symbol] is None:
                return {}
                
            model_data = self._models[symbol]
            model = model_data['model']
            feature_names = model_data['feature_names']
            
        # Convert features to matrix
        feature_values = []
        for name in feature_names:
            if name in features:
                values = features[name]
                if values:
                    feature_values.append(values[-1])  # Use last value
                else:
                    feature_values.append(0.0)
            else:
                feature_values.append(0.0)
                
        # Reshape for single prediction
        X = np.array(feature_values).reshape(1, -1)
        
        try:
            # Make prediction
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    prediction = model(X_tensor).item()
            else:
                prediction = model.predict(X)[0]
                
            # Convert to weights - extract feature importance if available
            weights = {}
            
            if hasattr(model, 'feature_importances_'):
                # For random forest and similar models
                importances = model.feature_importances_
                total_importance = importances.sum()
                
                if total_importance > 0:
                    for i, name in enumerate(feature_names):
                        weights[name] = importances[i] / total_importance
            else:
                # Fallback to equal weights for features with non-zero values
                non_zero_features = [name for i, name in enumerate(feature_names) if X[0, i] != 0]
                if non_zero_features:
                    weight = 1.0 / len(non_zero_features)
                    weights = {name: weight for name in non_zero_features}
                    
            return weights
                
        except Exception as e:
            self.logger.error(f"Error predicting weights for {symbol}: {e}")
            return {}
            
    def add_feedback(self, symbol: str, prediction: float, actual: float, features: Dict[str, float]):
        """
        Add feedback for reinforcement learning.
        
        Args:
            symbol: Symbol identifier
            prediction: Predicted value
            actual: Actual value
            features: Features used for prediction
        """
        if not self.can_process():
            return
            
        # Calculate reward/error
        error = abs(prediction - actual)
        
        # Add as training sample
        self.add_training_sample(symbol, features, actual)
        
        # Update model if enough feedback received
        with self._lock:
            if symbol in self._history and len(self._history[symbol]['targets']) >= self.config.ml_batch_size:
                self.train_model(symbol, force=True)
                
    def clean_history(self, max_age: float = None):
        """
        Clean old history entries to prevent memory buildup.
        
        Args:
            max_age: Maximum age in seconds (default: 7 days)
        """
        if max_age is None:
            max_age = 7 * 24 * 3600  # 7 days
            
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        with self._lock:
            for symbol in list(self._history.keys()):
                history = self._history[symbol]
                timestamps = history['timestamps']
                
                if not timestamps:
                    continue
                    
                # Find indices to keep
                keep_indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]
                
                if len(keep_indices) == len(timestamps):
                    continue  # No cleanup needed
                    
                # Create new history with only recent entries
                new_history = {
                    'features': [history['features'][i] for i in keep_indices],
                    'targets': [history['targets'][i] for i in keep_indices],
                    'timestamps': [timestamps[i] for i in keep_indices]
                }
                
                self._history[symbol] = new_history
                
        self.logger.debug(f"Cleaned history older than {max_age} seconds")

class MLNeuralNetwork(nn.Module):
    """Neural network for ML-based signal processing."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, learning_rate: float = 0.01):
        """
        Initialize neural network for ML-based signal processing.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
        
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Train the network with data.
        
        Args:
            X: Input features (numpy array)
            y: Target values (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Convert numpy arrays to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.train()  # Set to training mode
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_X)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
    def update(self, X, y, epochs=10, batch_size=32):
        """
        Update the network with new data (incremental learning).
        
        Args:
            X: Input features (numpy array)
            y: Target values (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Use smaller number of epochs for updates
        self.fit(X, y, epochs=epochs, batch_size=batch_size)
        
    def predict(self, X):
        """
        Make predictions with the network.
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Predictions as numpy array
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self(X_tensor)
            return predictions.numpy()

class AdaptiveFusionLearner:
    """
    Implements reinforcement learning for fusion weight optimization.
    """
    def __init__(self, config: CDFAConfig = None):
        """Initialize the adaptive fusion learner."""
        self.config = config or CDFAConfig()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveFusionLearner")
        
        # State tracking
        self._history = {}  # symbol -> experience history
        self._performance_metrics = {}  # symbol -> signal -> metrics
        self._last_weights = {}  # symbol -> weights
        self._lock = threading.RLock()
        
    def record_experience(self, symbol: str, weights: Dict[str, float], signals: Dict[str, List[float]], 
                         result: List[float], reward: float, timestamp: float = None):
        """
        Record experience for learning.
        
        Args:
            symbol: Symbol identifier
            weights: Weights used for fusion
            signals: Signal values
            result: Fusion result
            reward: Reward/performance measure
            timestamp: Optional timestamp (defaults to current time)
        """
        if not self.config.enable_adaptive_learning:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        # Extract last values from signals
        signal_values = {}
        for name, values in signals.items():
            if values:
                signal_values[name] = values[-1]
                
        # Store experience
        experience = {
            'weights': weights.copy(),
            'signals': signal_values,
            'result': result[-1] if result else 0.5,
            'reward': reward,
            'timestamp': timestamp
        }
        
        with self._lock:
            if symbol not in self._history:
                self._history[symbol] = []
                
            # Add to history
            self._history[symbol].append(experience)
            
            # Limit history size
            if len(self._history[symbol]) > self.config.feedback_window:
                self._history[symbol] = self._history[symbol][-self.config.feedback_window:]
                
            # Store last weights
            self._last_weights[symbol] = weights.copy()
                
    def update_performance_metrics(self, symbol: str):
        """
        Update performance metrics based on recorded experience.
        
        Args:
            symbol: Symbol identifier
        """
        if not self.config.enable_adaptive_learning:
            return {}
            
        with self._lock:
            if symbol not in self._history or len(self._history[symbol]) < 5:
                return {}
                
            history = self._history[symbol]
            
            # Extract signal names
            signal_names = set()
            for experience in history:
                signal_names.update(experience['signals'].keys())
                
            # Calculate correlation-based performance for each signal
            performance = {}
            
            for name in signal_names:
                # Extract signal values and rewards
                signal_values = []
                rewards = []
                
                for experience in history:
                    if name in experience['signals']:
                        signal_values.append(experience['signals'][name])
                        rewards.append(experience['reward'])
                        
                if len(signal_values) < 3:
                    continue
                    
                # Calculate correlation
                try:
                    corr, _ = pearsonr(signal_values, rewards)
                    if np.isnan(corr):
                        corr = 0.0
                        
                    # Transform to [0,1] range and adjust for direction
                    perf = abs(corr)
                    
                    # Store in performance metrics
                    if symbol not in self._performance_metrics:
                        self._performance_metrics[symbol] = {}
                        
                    # Update with exponential smoothing
                    old_perf = self._performance_metrics[symbol].get(name, perf)
                    learning_rate = self.config.learning_rate
                    new_perf = old_perf * (1 - learning_rate) + perf * learning_rate
                    
                    self._performance_metrics[symbol][name] = new_perf
                    performance[name] = new_perf
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating performance for {name}: {e}")
                    
            return performance
                
    def optimize_weights(self, symbol: str, signals: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Optimize weights based on learned performance.
        
        Args:
            symbol: Symbol identifier
            signals: Current signal values
            
        Returns:
            Optimized weights for fusion
        """
        if not self.config.enable_adaptive_learning:
            return {}
            
        # Update performance metrics
        self.update_performance_metrics(symbol)
        
        with self._lock:
            # Get performance metrics
            if symbol not in self._performance_metrics:
                return {}
                
            performance = self._performance_metrics[symbol]
            
            # Calculate weights based on performance
            weights = {}
            total_perf = sum(performance.values())
            
            if total_perf > 0:
                for name, perf in performance.items():
                    if name in signals and signals[name]:
                        weights[name] = perf / total_perf
                        
            # Apply smoothing with last weights if available
            if symbol in self._last_weights:
                last_weights = self._last_weights[symbol]
                decay = self.config.performance_decay
                
                for name in set(weights.keys()) | set(last_weights.keys()):
                    if name in weights and name in last_weights:
                        weights[name] = weights[name] * (1 - decay) + last_weights[name] * decay
                    elif name in last_weights:
                        weights[name] = last_weights[name] * decay
                        
            return weights
            
    def get_performance_metrics(self, symbol: str) -> Dict[str, float]:
        """Get current performance metrics for a symbol."""
        with self._lock:
            if symbol not in self._performance_metrics:
                return {}
                
            return self._performance_metrics[symbol].copy()

class FusionVisualizer:
    """
    Provides visualization tools for fusion metrics and performance.
    """
    def __init__(self, config: CDFAConfig = None):
        """Initialize the fusion visualizer."""
        self.config = config or CDFAConfig()
        self.logger = logging.getLogger(f"{__name__}.FusionVisualizer")
        
        self._can_visualize = MATPLOTLIB_AVAILABLE
        if not self._can_visualize:
            self.logger.warning("Matplotlib not available. Visualization disabled.")
            return
            
        # Set style
        if hasattr(sns, 'set_style'):
            sns.set_style(self.config.plot_style)
            
        # Plot history
        self._plot_history = {}  # symbol -> list of figures
        self._lock = threading.RLock()
        
    def can_visualize(self) -> bool:
        """Check if visualization is available."""
        return self._can_visualize and self.config.enable_visualization
        
    def create_diversity_matrix_plot(self, diversity_matrix: pd.DataFrame) -> Optional[Figure]:
        """
        Create a heatmap visualization of the diversity matrix.
        
        Args:
            diversity_matrix: Pairwise diversity matrix
            
        Returns:
            Matplotlib figure or None if visualization unavailable
        """
        if not self.can_visualize():
            return None
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap with seaborn
            mask = np.zeros_like(diversity_matrix, dtype=bool)
            np.fill_diagonal(mask, True)  # Mask diagonal values
            
            sns.heatmap(
                diversity_matrix,
                annot=True,
                cmap="viridis",
                mask=mask,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "Diversity Score"}
            )
            
            # Set title and labels
            ax.set_title("Cognitive Diversity Matrix")
            
            # Rotate y-axis labels for better visibility
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating diversity matrix plot: {e}")
            return None
            
    def create_fusion_performance_plot(self, symbol: str, fusion_history: List[Dict[str, Any]]) -> Optional[Figure]:
        """
        Create a performance visualization of fusion results.
        
        Args:
            symbol: Symbol identifier
            fusion_history: List of fusion results
            
        Returns:
            Matplotlib figure or None if visualization unavailable
        """
        if not self.can_visualize() or not fusion_history:
            return None
            
        try:
            # Filter history for symbol
            symbol_history = [entry for entry in fusion_history if entry.get('symbol') == symbol]
            
            if not symbol_history:
                return None
                
            # Extract data
            timestamps = [entry.get('timestamp', 0) for entry in symbol_history]
            fused_signals = [entry.get('fused_signal', [0.5])[-1] for entry in symbol_history]
            confidences = [entry.get('confidence', 0.5) for entry in symbol_history]
            
            # Convert timestamps to datetime
            dates = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            
            # Create figure with dual y-axis
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot fused signal
            ax1.plot(dates, fused_signals, 'b-', label='Fused Signal')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Fused Signal', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim(0, 1)
            
            # Create second y-axis for confidence
            ax2 = ax1.twinx()
            ax2.plot(dates, confidences, 'r--', label='Confidence')
            ax2.set_ylabel('Confidence', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 1)
            
            # Add title and grid
            plt.title(f'Fusion Performance for {symbol}')
            ax1.grid(True, alpha=0.3)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            # Store in history
            with self._lock:
                if symbol not in self._plot_history:
                    self._plot_history[symbol] = []
                    
                self._plot_history[symbol].append(fig)
                
                # Limit history size
                if len(self._plot_history[symbol]) > self.config.max_plots_history:
                    # Close old figure to free memory
                    old_fig = self._plot_history[symbol].pop(0)
                    plt.close(old_fig)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating fusion performance plot: {e}")
            return None
            
    def create_weight_distribution_plot(self, symbol: str, weights: Dict[str, float]) -> Optional[Figure]:
        """
        Create a visualization of fusion weight distribution.
        
        Args:
            symbol: Symbol identifier
            weights: Dictionary of signal weights
            
        Returns:
            Matplotlib figure or None if visualization unavailable
        """
        if not self.can_visualize() or not weights:
            return None
            
        try:
            # Sort weights for better visualization
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            names = [item[0] for item in sorted_weights]
            values = [item[1] for item in sorted_weights]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar chart
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, align='center', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            
            # Add labels and title
            ax.set_xlabel('Weight')
            ax.set_title(f'Fusion Weight Distribution for {symbol}')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                
            # Set x-axis limit
            ax.set_xlim(0, max(values) * 1.1)
            
            # Add grid
            ax.grid(True, axis='x', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating weight distribution plot: {e}")
            return None
            
    def create_signal_comparison_plot(self, symbol: str, signals: Dict[str, List[float]], 
                                    fusion_result: List[float], timestamps: List[float] = None) -> Optional[Figure]:
        """
        Create a comparison visualization of signals and fusion result.
        
        Args:
            symbol: Symbol identifier
            signals: Dictionary of signals
            fusion_result: Fusion result
            timestamps: Optional list of timestamps
            
        Returns:
            Matplotlib figure or None if visualization unavailable
        """
        if not self.can_visualize() or not signals or not fusion_result:
            return None
            
        try:
            # Determine common length
            min_length = min([len(values) for values in signals.values()] + [len(fusion_result)])
            
            if min_length < 2:
                return None
                
            # Create x-axis - timestamps or indices
            if timestamps and len(timestamps) >= min_length:
                x = [pd.to_datetime(ts, unit='s') for ts in timestamps[-min_length:]]
            else:
                x = np.arange(min_length)
                
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot each signal
            for name, values in signals.items():
                values_subset = values[-min_length:]
                ax.plot(x, values_subset, alpha=0.5, label=name)
                
            # Plot fusion result with thicker line
            fusion_subset = fusion_result[-min_length:]
            ax.plot(x, fusion_subset, 'k-', linewidth=2, label='Fusion')
            
            # Add labels and title
            ax.set_xlabel('Time' if timestamps else 'Index')
            ax.set_ylabel('Signal Value')
            ax.set_title(f'Signal Comparison for {symbol}')
            
            # Add legend with smaller font
            ax.legend(loc='upper left', fontsize='small')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating signal comparison plot: {e}")
            return None
            
    def save_plot(self, fig: Figure, filename: str) -> bool:
        """
        Save a plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.can_visualize() or fig is None:
            return False
            
        try:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")
            return False
            
    def close_all_plots(self):
        """Close all plots to free memory."""
        if not self.can_visualize():
            return
            
        with self._lock:
            for symbol, plots in self._plot_history.items():
                for fig in plots:
                    plt.close(fig)
                    
            self._plot_history = {}

# --- Main CDFA Class ---
class CognitiveDiversityFusionAnalysis:
    """
    Enhanced implementation of Cognitive Diversity Fusion Analysis (CDFA)
    providing robust, memory-efficient, and production-ready analysis and fusion methods.
    Includes KL/JSD diversity, adaptive fusion, enhanced weighting, and DTW.
    """
    def __init__(self, config: Optional[CDFAConfig] = None):
        self.config = config if config is not None else CDFAConfig()
        self._initialize_logging()
        self._execution_times: Dict[str, List[float]] = {}
        self._call_counts: Dict[str, int] = {}
        self._diversity_cache: Dict[Tuple, float] = {}
        self._rsc_cache: Dict[Tuple, Dict[int, float]] = {}
        self._dtw_cache: Dict[Tuple, Dict[str, float]] = {}
        self._fusion_history: List[Dict[str, Any]] = []
        self._diversity_matrices: Dict[str, weakref.ReferenceType[pd.DataFrame]] = {}
        self._lock = threading.RLock()  # Thread safety lock
        
        # Redis integration (optional)
        self.redis_client = None
        self._signal_cache = {}  # symbol -> source -> SignalData
        self._market_info = {}   # symbol -> market info
        
        # Connector instances for external components
        self._soc_analyzer = None
        self._panarchy_analyzer = None
        self._fibonacci_analyzer = None
        self._antifragility_analyzer = None
        self._pattern_recognizer = None
        self._whale_detector = None
        self._black_swan_detector = None
        self._fibonacci_detector = None
        
        # ML/RL integration
        self._ml_processor = MLSignalProcessor(self.config)
        self._adaptive_learner = AdaptiveFusionLearner(self.config)
        self._visualizer = FusionVisualizer(self.config)
        
        # Initialize Redis if enabled
        if self.config.enable_redis and REDIS_AVAILABLE:
            self._initialize_redis()
            
        self.logger.info(f"CognitiveDiversityFusionAnalysis (Enhanced) initialized with Numba: {NUMBA_AVAILABLE}")

    # --- Helpers (Logging, Timing, Caching) ---
    def _initialize_logging(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        try:
            # Access attributes using dot notation
            if getattr(self.config, 'enable_logging', True) and not self.logger.handlers:
                log_level = getattr(self.config, 'log_level', logging.INFO) # Use logging.INFO for default
                log_format = getattr(
                    self.config,
                    'log_format',
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                
                # Set up file handler if log file is specified
                if hasattr(self.config, 'log_file') and self.config.log_file:
                    file_handler = logging.FileHandler(self.config['log_file'])
                    file_handler.setLevel(log_level)
                    file_formatter = logging.Formatter(log_format)
                    file_handler.setFormatter(file_formatter)
                    self.logger.addHandler(file_handler)
                
                # Set up console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_formatter = logging.Formatter(log_format)
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
                
                self.logger.setLevel(log_level)
                self.logger.propagate = False
                
        except Exception as e:
            print(f"Error initializing logging: {e}")
            # Fall back to basic logging if there's an error
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
        self.logger.setLevel(self.config.log_level)

    def _time_execution(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            func_name = func.__name__
            with self._lock:
                if func_name not in self._execution_times:
                    self._execution_times[func_name] = []
                if func_name not in self._call_counts:
                    self._call_counts[func_name] = 0
                self._execution_times[func_name].append(execution_time)
                self._call_counts[func_name] += 1
                if len(self._execution_times[func_name]) > 100:
                    self._execution_times[func_name].pop(0)
            return result
        return wrapper

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for function execution times"""
        with self._lock:
            metrics = {}
            for func_name, times in self._execution_times.items():
                if times: 
                    metrics[func_name] = {
                        'avg_time': sum(times)/len(times), 
                        'max_time': max(times), 
                        'min_time': min(times), 
                        'call_count': self._call_counts.get(func_name,0)
                    }
            return metrics

    def _get_internal_cache(self, cache_dict: dict, key: tuple) -> Optional[Any]:
        """Thread-safe cache retrieval"""
        if self.config.enable_caching:
            with self._lock:
                return cache_dict.get(key)
        return None

    def _put_internal_cache(self, cache_dict: dict, key: tuple, value: Any):
        """Thread-safe cache storage with size management"""
        if self.config.enable_caching:
            with self._lock:
                cache_dict[key] = value
                # Evict oldest items if cache exceeds size limit
                if len(cache_dict) > self.config.cache_size:
                     try:
                         cache_dict.pop(next(iter(cache_dict)))
                     except StopIteration:
                         pass
                     
    def _initialize_redis(self):
        """Initialize Redis client for integration"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis library not available. Integration disabled.")
            return False
            
        try:
            # Create Redis client
            self.redis_client = CDFARedisClient(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                channel_prefix=self.config.redis_channel_prefix
            )
            
            # Subscribe to signal channels
            signals = [
                "signals:antifragility", 
                "signals:soc", 
                "signals:panarchy",
                "signals:whale", 
                "signals:blackswan", 
                "signals:fibonacci"
            ]
            
            for channel in signals:
                self.redis_client.subscribe(channel, self._handle_signal)
                
            # Subscribe to control channel
            self.redis_client.subscribe("control", self._handle_control)
            
            self.logger.info("Redis integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis integration: {e}")
            self.redis_client = None
            return False

    # --- Diversity Methods ---
    @_time_execution
    def calculate_rank_score_characteristic(self, scores: List[float]) -> Dict[int, float]:
        """
        Calculate rank score characteristic with Numba acceleration.
        """
        # Create cache key for consistent caching
        try:
            cache_key = hash(tuple(scores))
            cached = self._get_internal_cache(self._rsc_cache, cache_key)
            if cached is not None: 
                return cached
        except TypeError:
            # Unhashable type - bypass cache
            cache_key = None
        
        if not scores: 
            return {}
        
        # Convert to numpy array for faster processing
        scores_array = np.array([float(s) for s in scores], dtype=np.float64)
        
        # Filter finite values
        finite_mask = np.isfinite(scores_array)
        if not np.any(finite_mask):
            result = {i+1: 0.5 for i in range(len(scores))}
            if cache_key:
                self._put_internal_cache(self._rsc_cache, cache_key, result)
            return result
            
        # Get finite scores with indices
        finite_scores = scores_array[finite_mask]
        finite_indices = np.arange(len(scores_array))[finite_mask]
        
        # Sort in descending order
        sorted_indices = np.argsort(-finite_scores)
        
        # Calculate range for normalization
        min_s, max_s = np.min(finite_scores), np.max(finite_scores)
        s_range = max_s - min_s
        
        # Create result dictionary
        rsc = {}
        
        for rank, idx in enumerate(sorted_indices, 1):
            orig_idx = finite_indices[idx]
            score = finite_scores[idx]
            
            # Normalize score
            if s_range < 1e-9:
                norm_score = 0.5
            else:
                norm_score = np.clip((score - min_s) / s_range, 0.0, 1.0)
                
            rsc[rank] = float(norm_score)
            
        # Cache result if caching enabled
        if cache_key:
            self._put_internal_cache(self._rsc_cache, cache_key, rsc)
            
        return rsc

    def _calculate_kl_jsd(self, p: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """
        Vectorized implementation of KL divergence and JSD calculation.
        """
        # Apply small epsilon to avoid log(0)
        eps = self.config.kl_epsilon
        
        # Ensure valid probability distributions
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        kl_pq = np.sum(p * np.log2(p / q))
        
        # Calculate Jensen-Shannon Divergence
        m = 0.5 * (p + q)
        m = np.maximum(m, eps)
        m = m / np.sum(m)
        
        jsd = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
        
        return float(kl_pq), float(jsd)

    def _scores_to_prob_dist(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert scores to probability distribution using vectorized histogram.
        """
        if len(scores) == 0:
            return np.array([1.0])
            
        # Calculate histogram with numpy
        hist, _ = np.histogram(scores, bins=self.config.kl_num_bins, range=(0, 1), density=True)
        
        # Ensure valid distribution
        hist_sum = np.sum(hist)
        if hist_sum > 1e-9:
            prob_dist = hist / hist_sum
        else:
            prob_dist = np.ones(self.config.kl_num_bins) / self.config.kl_num_bins
            
        return prob_dist

    @_time_execution
    def calculate_cognitive_diversity(self, system_a_scores: List[float], system_b_scores: List[float], method: Union[str, DiversityMethod] = None) -> float:
        """
        Calculate diversity metrics with Numba acceleration.
        """
        if method is None: 
            method = self.config.default_diversity_method
        elif isinstance(method, str): 
            method = DiversityMethod.from_string(method)

        # Use internal cache
        try: 
            cache_key = (tuple(system_a_scores), tuple(system_b_scores), method)
            cached = self._get_internal_cache(self._diversity_cache, cache_key)
            if cached is not None: 
                return cached
        except TypeError: 
            self.logger.warning("Unhashable scores for diversity cache.")
            cache_key = None

        # Basic validation
        if len(system_a_scores) != len(system_b_scores) or len(system_a_scores) < 2: 
            return 0.0
            
        # Convert to numpy arrays
        a_array = np.array([float(s) for s in system_a_scores], dtype=np.float64)
        b_array = np.array([float(s) for s in system_b_scores], dtype=np.float64)
        
        # Filter finite values
        finite_mask = np.isfinite(a_array) & np.isfinite(b_array)
        a_clean = a_array[finite_mask]
        b_clean = b_array[finite_mask]
        
        if len(a_clean) < 2: 
            return 0.0
            
        distance = 0.0
        try:
            if method == DiversityMethod.KENDALL:
                # Use Numba if available and appropriate
                if NUMBA_AVAILABLE and self.config.use_numba and len(a_clean) > 10:
                    distance = _kendall_distance_numba(a_clean, b_clean)
                else:
                    tau, _ = kendalltau(a_clean, b_clean)
                    distance = 0.0 if np.isnan(tau) else max(0.0, (1 - tau) / 2)
                    
            elif method == DiversityMethod.SPEARMAN:
                rho, _ = spearmanr(a_clean, b_clean)
                distance = 0.0 if np.isnan(rho) else max(0.0, (1 - rho) / 2)
                
            elif method == DiversityMethod.PEARSON:
                if np.std(a_clean) < 1e-9 or np.std(b_clean) < 1e-9:
                    distance = 0.0
                else:
                    r, _ = pearsonr(a_clean, b_clean)
                    distance = 0.0 if np.isnan(r) else max(0.0, (1 - r) / 2)
                    
            elif method == DiversityMethod.RSC:
                rsc_a = self.calculate_rank_score_characteristic(a_clean.tolist())
                rsc_b = self.calculate_rank_score_characteristic(b_clean.tolist())
                common_ranks = set(rsc_a.keys()) & set(rsc_b.keys())
                
                if not common_ranks: 
                    return 0.0
                    
                sq_diff = sum((rsc_a[rank] - rsc_b[rank])**2 for rank in common_ranks)
                mse = sq_diff / len(common_ranks)
                distance = np.clip(mse * self.config.rsc_scale_factor, 0.0, 1.0)
                
            elif method in [DiversityMethod.KL_DIVERGENCE, DiversityMethod.JSD]:
                # Prepare data
                norm_a = np.clip(a_clean, 0, 1)
                norm_b = np.clip(b_clean, 0, 1)
                
                # Calculate probability distributions
                dist_p = self._scores_to_prob_dist(norm_a)
                dist_q = self._scores_to_prob_dist(norm_b)
                
                # Calculate divergence
                kl_pq, jsd = self._calculate_kl_jsd(dist_p, dist_q)
                
                if method == DiversityMethod.KL_DIVERGENCE:
                    distance = np.clip(1.0 - math.exp(-kl_pq), 0.0, 1.0)
                else:
                    distance = np.clip(jsd, 0.0, 1.0)
                    
            else:
                self.logger.error(f"Unknown diversity method: {method}")
                distance = 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating diversity ({method}): {e}", exc_info=False)
            distance = 0.0

        # Cache result
        if cache_key:
            self._put_internal_cache(self._diversity_cache, cache_key, distance)
            
        return distance

    @_time_execution
    def calculate_multiple_diversity_metrics(self, system_a_scores: List[float], system_b_scores: List[float]) -> List[DiversityResult]:
        """Calculate multiple diversity metrics with confidence values"""
        results = []
        for method in DiversityMethod:
             try:
                 diversity = self.calculate_cognitive_diversity(system_a_scores, system_b_scores, method)
                 n_samples = len(system_a_scores) # Use original length for confidence base
                 sample_conf = min(1.0, n_samples / 20)
                 value_conf = 1.0 - abs(diversity - 0.5) * 0.5
                 confidence = sample_conf * value_conf
                 results.append(DiversityResult(diversity, method, confidence))
             except Exception as e:
                 self.logger.error(f"Error calculating {method} diversity: {e}")
                 
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    @_time_execution
    def calculate_pairwise_diversity_matrix(self, system_scores: Dict[str, List[float]], method: Union[str, DiversityMethod] = None) -> pd.DataFrame:
        """
        Calculate pairwise diversity matrix with parallel processing optimization.
        """
        if method is None:
            method = self.config.default_diversity_method
        elif isinstance(method, str):
            method = DiversityMethod.from_string(method)
            
        systems = list(system_scores.keys())
        n_systems = len(systems)
        
        if n_systems == 0:
            return pd.DataFrame()
            
        # Create empty matrix
        div_matrix = np.zeros((n_systems, n_systems))
        
        # Determine if parallel execution should be used
        use_parallel = (n_systems >= self.config.parallelization_threshold and 
                       self.config.max_workers > 1)
                       
        # Define computation function
        def compute_pair(pair_indices):
            """Compute diversity for a pair of systems"""
            i, j = pair_indices
            # Use deep copy to avoid race conditions
            scores_i = system_scores.get(systems[i], []).copy()
            scores_j = system_scores.get(systems[j], []).copy()
            diversity = self.calculate_cognitive_diversity(scores_i, scores_j, method)
            return (i, j, diversity)
            
        # Generate upper triangle indices
        pairs = [(i, j) for i in range(n_systems) for j in range(i + 1, n_systems)]
        
        if use_parallel:
            # Use ThreadPoolExecutor for parallel computation
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Process all pairs in parallel
                results = list(executor.map(compute_pair, pairs))
                
                # Fill matrix with results
                for i, j, diversity in results:
                    div_matrix[i, j] = div_matrix[j, i] = diversity
        else:
            # Sequential computation
            for i, j in pairs:
                diversity = self.calculate_cognitive_diversity(
                    system_scores[systems[i]], 
                    system_scores[systems[j]],
                    method
                )
                div_matrix[i, j] = div_matrix[j, i] = diversity
                
        # Convert to DataFrame
        return pd.DataFrame(div_matrix, index=systems, columns=systems)

    @_time_execution
    def average_diversity(self, system_name: str, diversity_matrix: pd.DataFrame) -> float:
        """Calculate average diversity for a system"""
        if system_name not in diversity_matrix.index:
            self.logger.warning(f"System {system_name} not found in matrix")
            return 0.0
            
        # Get all diversity values except self-comparison
        diversities = diversity_matrix.loc[system_name].drop(system_name) if system_name in diversity_matrix.columns else diversity_matrix.loc[system_name]
        
        return float(np.mean(diversities)) if len(diversities) > 0 else 0.0

    @_time_execution
    def calculate_system_dispersion(self, system_scores: Dict[str, List[float]]) -> float:
        """Calculate system dispersion using vectorized operations"""
        if len(system_scores) <= 1:
            return 0.0
            
        # Create ScoreData objects
        score_data = {name: ScoreData(scores) for name, scores in system_scores.items() if scores}
        
        if len(score_data) <= 1:
            return 0.0
            
        # Determine common length
        try:
            n_items = len(next(iter(score_data.values())).normalized_scores)
        except StopIteration:
            return 0.0
            
        # Calculate variance for each item efficiently
        item_variances = []
        
        for i in range(n_items):
            # Gather scores for this item from all systems
            item_scores = []
            
            for data in score_data.values():
                if i < len(data.normalized_scores):
                    item_scores.append(data.normalized_scores[i])
                    
            # Calculate variance if we have enough scores
            if len(item_scores) > 1:
                # Convert to numpy array for vectorized calculation
                scores_array = np.array(item_scores, dtype=float)
                item_variances.append(np.var(scores_array))
                
        # Calculate mean variance
        if not item_variances:
            return 0.0
            
        mean_variance = float(np.mean(item_variances))
        
        # Normalize to [0,1]
        return min(1.0, mean_variance * 4)

    # --- Fusion Methods ---
    @_time_execution
    def fusion_score_combination(self, system_scores: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Optimized score-based fusion with vectorized operations.
        """
        if not system_scores:
            return []
            
        # Create ScoreData objects
        score_data = {name: ScoreData(scores) for name, scores in system_scores.items() if scores}
        
        if not score_data:
            return []
            
        # Get common length
        try:
            n_items = len(next(iter(score_data.values())).normalized_scores)
        except StopIteration:
            return []
            
        # Default equal weights if none provided
        if weights is None:
            weights = {system: 1.0 for system in score_data}
            
        # Apply weights to each system
        active_weights = {system: weights.get(system, 0.0) for system in score_data}
        total_weight = sum(active_weights.values())
        
        if total_weight < 1e-9:
            return [0.5] * n_items
            
        # Normalize weights
        norm_weights = {sys: w / total_weight for sys, w in active_weights.items()}
        
        # Use vectorized operations for efficiency
        combined_scores = np.zeros(n_items)
        
        # For each system, add weighted contribution
        for system, weight in norm_weights.items():
            data = score_data[system]
            
            # Get normalized scores (pad with 0.5 if shorter)
            if len(data.normalized_scores) < n_items:
                scores = data.normalized_scores + [0.5] * (n_items - len(data.normalized_scores))
            else:
                scores = data.normalized_scores[:n_items]
                
            # Add weighted contribution
            combined_scores += np.array(scores) * weight
            
        return combined_scores.tolist()

    @_time_execution
    def fusion_rank_combination(self, system_scores: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Optimized rank-based fusion with vectorized operations.
        """
        if not system_scores:
            return []
            
        # Create ScoreData objects
        score_data = {name: ScoreData(scores) for name, scores in system_scores.items() if scores}
        
        if not score_data:
            return []
            
        # Get common length
        try:
            n_items = len(next(iter(score_data.values())).ranks)
        except StopIteration:
            return []
            
        # Default equal weights if none provided
        if weights is None:
            weights = {system: 1.0 for system in score_data}
            
        # Apply weights to each system
        active_weights = {system: weights.get(system, 0.0) for system in score_data}
        total_weight = sum(active_weights.values())
        
        if total_weight < 1e-9:
            return [0.5] * n_items
            
        # Normalize weights
        norm_weights = {sys: w / total_weight for sys, w in active_weights.items()}
        
        # Use max rank for Borda count
        max_rank = n_items
        
        # Initialize Borda scores
        borda_scores = np.zeros(n_items)
        
        # For each system, add weighted Borda scores
        for system, weight in norm_weights.items():
            data = score_data[system]
            
            # Get ranks (pad if shorter)
            if len(data.ranks) < n_items:
                ranks = data.ranks + [max_rank] * (n_items - len(data.ranks))
            else:
                ranks = data.ranks[:n_items]
                
            # Convert to Borda scores (max_rank - rank + 1)
            borda = np.array([max_rank - r + 1 if r is not None else 0 for r in ranks])
            
            # Add weighted contribution
            borda_scores += borda * weight
            
        # Normalize to [0,1]
        min_b, max_b = np.min(borda_scores), np.max(borda_scores)
        range_b = max_b - min_b
        
        if range_b < 1e-9:
            return [0.5] * n_items
            
        normalized_scores = (borda_scores - min_b) / range_b
        
        return normalized_scores.tolist()

    @_time_execution
    def fusion_hybrid_combination(self, system_scores: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None, alpha: float = 0.5) -> List[float]:
        """
        Optimized hybrid fusion combining score and rank methods.
        """
        # Ensure alpha is in valid range
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # Calculate score and rank fusion
        score_f = self.fusion_score_combination(system_scores, weights)
        rank_f = self.fusion_rank_combination(system_scores, weights)
        
        # Handle edge cases
        if not score_f and not rank_f:
            return []
            
        if not score_f:
            return rank_f
            
        if not rank_f:
            return score_f
            
        if len(score_f) != len(rank_f):
            self.logger.warning(f"Hybrid fusion length mismatch: {len(score_f)} vs {len(rank_f)}")
            return score_f
            
        # Use vectorized operations for efficiency
        score_array = np.array(score_f)
        rank_array = np.array(rank_f)
        
        hybrid = alpha * score_array + (1.0 - alpha) * rank_array
        
        return hybrid.tolist()

    @_time_execution
    def multi_layer_fusion(self, system_scores: Dict[str, List[float]], performance_metrics: Dict[str, float], method: Union[str, DiversityMethod] = None, expansion_factor: Optional[int] = None) -> List[float]:
        """
        Enhanced multi-layer fusion with optimization.
        """
        # --- Validate inputs ---
        if not system_scores or len(system_scores) < 2:
            return list(system_scores.values())[0] if system_scores else []
            
        if method is None:
            method = self.config.default_diversity_method
        elif isinstance(method, str):
            method = DiversityMethod.from_string(method)
            
        if expansion_factor is None:
            expansion_factor = self.config.expansion_factor
            
        # --- Filter systems by performance threshold ---
        filtered_systems = {
            name: scores 
            for name, scores in system_scores.items() 
            if performance_metrics.get(name, 0) >= self.config.performance_threshold
        }
        
        if not filtered_systems:
            filtered_systems = system_scores
            self.logger.warning("No systems meet perf threshold, using all.")
            
        if len(filtered_systems) < 2:
            return list(filtered_systems.values())[0] if filtered_systems else []
            
        # --- Calculate diversity matrix ---
        diversity_matrix = self.calculate_pairwise_diversity_matrix(filtered_systems, method)
        
        # --- Create virtual systems ---
        virtual_systems = {}
        original_count = len(filtered_systems)
        target_virtual_count = original_count * expansion_factor
        
        # Find diverse pairs
        pairs = []
        f_systems_list = list(filtered_systems.keys())
        
        for i in range(original_count):
            for j in range(i + 1, original_count):
                sys1, sys2 = f_systems_list[i], f_systems_list[j]
                div = diversity_matrix.loc[sys1, sys2]
                
                if div >= self.config.diversity_threshold:
                    perf1 = performance_metrics.get(sys1, 0)
                    perf2 = performance_metrics.get(sys2, 0)
                    pairs.append((sys1, sys2, div * (perf1 + perf2) / 2))
                    
        # Sort pairs by combined score
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Create virtual systems from diverse pairs
        for idx, (sys1, sys2, _) in enumerate(pairs):
            if len(virtual_systems) >= target_virtual_count:
                break
                
            subset = {sys1: filtered_systems[sys1], sys2: filtered_systems[sys2]}
            
            # Skip empty signals
            if not subset[sys1] or not subset[sys2]:
                continue
                
            # Create virtual systems with different fusion types
            virtual_systems[f"VS_{idx}_score"] = self.fusion_score_combination(
                subset, {sys1: 0.6, sys2: 0.4}
            )
            
            virtual_systems[f"VS_{idx}_rank"] = self.fusion_rank_combination(
                subset, {sys1: 0.4, sys2: 0.6}
            )
            
            virtual_systems[f"VS_{idx}_hybrid"] = self.fusion_hybrid_combination(
                subset, {sys1: 0.5, sys2: 0.5}
            )
            
        # Combine original and virtual systems
        all_systems = {**filtered_systems, **virtual_systems}
        
        if not all_systems:
            self.logger.warning("MCF: No systems available after expansion.")
            return []
            
        # Estimate performance metrics for virtual systems
        all_performance = performance_metrics.copy()
        
        for vs_name in virtual_systems:
            try:
                # Parse component names from virtual system name
                parts = vs_name.split('_')
                if len(parts) >= 3:
                    component_idx = parts[1]  # Index part
                    if component_idx.isdigit():
                        idx = int(component_idx)
                        if idx < len(pairs):
                            sys1, sys2, _ = pairs[idx]
                            
                            # Average performance of component systems
                            perf1 = performance_metrics.get(sys1, 0)
                            perf2 = performance_metrics.get(sys2, 0)
                            avg_perf = (perf1 + perf2) / 2
                            
                            # Slightly reduce for virtual systems
                            all_performance[vs_name] = avg_perf * 0.9
            except Exception as e:
                self.logger.debug(f"Error estimating performance for {vs_name}: {e}")
                all_performance[vs_name] = 0.5  # Default
            
        # Calculate diversity for all systems
        all_diversity = self.calculate_pairwise_diversity_matrix(all_systems, method)
        avg_diversity = {name: self.average_diversity(name, all_diversity) for name in all_systems}
        
        # Combine metrics
        combined_metrics = {
            name: all_performance.get(name, 0) * avg_diversity.get(name, 0) 
            for name in all_systems
        }
        
        # Rank systems by combined metrics
        ranked_systems = sorted(combined_metrics.items(), key=lambda x: x[1], reverse=True)
        
        # Select top systems
        n_final = max(2, min(int(np.sqrt(len(all_systems))), original_count * 2))
        final_systems = {
            name: all_systems[name] 
            for name, _ in ranked_systems[:n_final] 
            if name in all_systems
        }
        
        # Fallback if reduction yields nothing
        if not final_systems:
            self.logger.warning("MCF reduction resulted in empty set, using top 2 ranked.")
            final_systems = {
                name: all_systems[name] 
                for name, _ in ranked_systems[:2] 
                if name in all_systems
            }
            
        # Ultimate fallback
        if not final_systems:
            self.logger.error("MCF could not select any final systems.")
            return list(filtered_systems.values())[0] if filtered_systems else []
            
        # Calculate final weights
        final_weights = {name: combined_metrics[name] for name in final_systems}
        
        # Normalize weights
        total_weight = sum(final_weights.values())
        
        if total_weight > 1e-9:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            final_weights = {k: 1.0 / len(final_weights) for k in final_weights}
            
        self.logger.info(f"Multi-layer fusion using {len(final_systems)} systems from total {len(all_systems)}")
        
        # Final fusion
        return self.fusion_hybrid_combination(final_systems, final_weights)
        
    def visualize_fusion_results(self, symbol: str) -> Optional[Figure]:
        """
        Create visualization of fusion results for a symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Matplotlib figure if visualization is available, None otherwise
        """
        if not hasattr(self, '_visualizer') or not self._visualizer.can_visualize():
            self.logger.warning("Visualization not available")
            return None
            
        # Filter fusion history for symbol
        fusion_history = [entry for entry in self._fusion_history if entry.get('symbol') == symbol]
        
        if not fusion_history:
            self.logger.warning(f"No fusion history for symbol: {symbol}")
            return None
            
        # Create performance visualization
        return self._visualizer.create_fusion_performance_plot(symbol, fusion_history)
        
    def visualize_diversity_matrix(self, symbol: str) -> Optional[Figure]:
        """
        Create visualization of diversity matrix for a symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Matplotlib figure if visualization is available, None otherwise
        """
        if not hasattr(self, '_visualizer') or not self._visualizer.can_visualize():
            self.logger.warning("Visualization not available")
            return None
            
        with self._lock:
            if symbol not in self._signal_cache:
                self.logger.warning(f"No signals for symbol: {symbol}")
                return None
                
            # Extract signals
            signals = {
                name: signal.get('values', [])
                for name, signal in self._signal_cache[symbol].items()
            }
            
        # Calculate diversity matrix
        diversity_matrix = self.calculate_pairwise_diversity_matrix(signals)
        
        # Create diversity matrix visualization
        return self._visualizer.create_diversity_matrix_plot(diversity_matrix)
        
    def visualize_weight_distribution(self, symbol: str) -> Optional[Figure]:
        """
        Create visualization of weight distribution for a symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Matplotlib figure if visualization is available, None otherwise
        """
        if not hasattr(self, '_visualizer') or not self._visualizer.can_visualize():
            self.logger.warning("Visualization not available")
            return None
            
        # Find weights used for symbol
        weights = {}
        with self._lock:
            for entry in reversed(self._fusion_history):
                if entry.get('symbol') == symbol:
                    # Attempt to extract weights from entry
                    if 'weights' in entry:
                        weights = entry['weights']
                    else:
                        # If weights not stored, try to extract performance metrics
                        weights = {
                            name: perf
                            for name, perf in entry.get('performance_metrics', {}).items()
                        }
                    break
                    
        if not weights:
            self.logger.warning(f"No weights found for symbol: {symbol}")
            return None
            
        # Create weight distribution visualization
        return self._visualizer.create_weight_distribution_plot(symbol, weights)
        
    def visualize_signal_comparison(self, symbol: str) -> Optional[Figure]:
        """
        Create visualization comparing signals and fusion result for a symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Matplotlib figure if visualization is available, None otherwise
        """
        if not hasattr(self, '_visualizer') or not self._visualizer.can_visualize():
            self.logger.warning("Visualization not available")
            return None
            
        with self._lock:
            if symbol not in self._signal_cache:
                self.logger.warning(f"No signals for symbol: {symbol}")
                return None
                
            # Extract signals
            signals = {
                name: signal.get('values', [])
                for name, signal in self._signal_cache[symbol].items()
            }
            
            # Get fusion result
            fusion_result = []
            for entry in reversed(self._fusion_history):
                if entry.get('symbol') == symbol:
                    fusion_result = entry.get('fused_signal', [])
                    break
                    
        if not fusion_result:
            self.logger.warning(f"No fusion result for symbol: {symbol}")
            return None
            
        # Create signal comparison visualization
        return self._visualizer.create_signal_comparison_plot(symbol, signals, fusion_result)
    


# ----- CORE PATTERN DETECTION METHODS -----

    @_time_execution
    def detect_dtw_patterns(self, series: np.ndarray, templates: Dict[str, np.ndarray], window_size: int = 20) -> Dict[str, float]:
        """
        Enhanced DTW pattern detection with Numba acceleration.
        
        Args:
            series: Input time series
            templates: Dictionary of pattern templates
            window_size: Window size for DTW algorithm
            
        Returns:
            Dictionary of pattern name to similarity score
        """
        # Convert input to numpy array
        series = np.asarray(series, dtype=np.float64)
        
        if len(series) == 0:
            return {}
            
        # Create cache key
        try:
            cache_key = hash(series.tobytes())
            cached = self._get_internal_cache(self._dtw_cache, cache_key)
            if cached is not None:
                return cached
        except Exception:
            cache_key = None
            
        # Normalize series
        norm_series = self._normalize_series(series)
        results = {}
        
        for name, template in templates.items():
            # Skip invalid templates
            if len(template) == 0 or len(template) > len(series):
                results[name] = 0.0
                continue
                
            # Normalize template
            template = np.asarray(template, dtype=np.float64)
            norm_template = self._normalize_series(template)
            
            # Compare template against the end of the series
            series_segment = norm_series[-len(norm_template):]
            
            # Check for non-finite values
            if not np.all(np.isfinite(series_segment)) or not np.all(np.isfinite(norm_template)):
                self.logger.warning(f"DTW: Non-finite values in series or template for '{name}'")
                results[name] = 0.0
                continue
                
            # Calculate DTW distance (with Numba acceleration if available)
            if NUMBA_AVAILABLE and self.config.use_numba:
                distance = _dtw_distance_numba(series_segment, norm_template, window_size)
            else:
                # Classic DTW implementation
                n, m = len(series_segment), len(norm_template)
                dtw_matrix = np.full((n+1, m+1), np.inf)
                dtw_matrix[0, 0] = 0
                
                # Use Sakoe-Chiba band
                w = max(window_size, abs(n - m))
                
                for i in range(1, n+1):
                    start_j = max(1, i-w)
                    end_j = min(m+1, i+w+1)
                    
                    for j in range(start_j, end_j):
                        cost = abs(series_segment[i-1] - norm_template[j-1])
                        dtw_matrix[i, j] = cost + min(
                            dtw_matrix[i-1, j],
                            dtw_matrix[i, j-1],
                            dtw_matrix[i-1, j-1]
                        )
                        
                distance = dtw_matrix[n, m] / (n + m)
                
            # Convert distance to similarity score
            similarity = max(0.0, 1.0 - distance) if np.isfinite(distance) else 0.0
            results[name] = similarity
            
        # Cache results
        if cache_key:
            self._put_internal_cache(self._dtw_cache, cache_key, results)
            
        return results
    
    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """
        Normalize series to [0,1] range (vectorized).
        
        Args:
            series: Input time series
            
        Returns:
            Normalized series
        """
        if len(series) == 0:
            return np.array([])
            
        # Create mask for finite values
        finite_mask = np.isfinite(series)
        
        if not np.any(finite_mask):
            return np.full_like(series, 0.5)
            
        # Get min/max of finite values
        finite_values = series[finite_mask]
        min_s = np.min(finite_values)
        max_s = np.max(finite_values)
        s_range = max_s - min_s
        
        if s_range < 1e-9:
            return np.full_like(series, 0.5)
            
        # Normalize using vectorized operations
        normalized = np.full_like(series, 0.5)
        normalized[finite_mask] = np.clip((series[finite_mask] - min_s) / s_range, 0.0, 1.0)
        
        return normalized
    
    # ----- EXTERNAL ANALYZER INTEGRATION -----
    
    def integrate_external_analyzers(self):
        """Setup connections to external analyzers and detectors"""
        # Initialize connector attributes if not already done
        self._soc_analyzer = None
        self._panarchy_analyzer = None
        self._fibonacci_analyzer = None
        self._antifragility_analyzer = None
        self._pattern_recognizer = None
        
        self.logger.info("CDFA external analyzer integration initialized")
        return True
    
    def connect_soc_analyzer(self, analyzer):
        """Connect to SOCAnalyzer instance for direct integration"""
        self._soc_analyzer = analyzer
        self.logger.info("SOCAnalyzer connected to CDFA")
        
    def connect_panarchy_analyzer(self, analyzer):
        """Connect to PanarchyAnalyzer instance for direct integration"""
        self._panarchy_analyzer = analyzer
        self.logger.info("PanarchyAnalyzer connected to CDFA")
        
    def connect_fibonacci_analyzer(self, analyzer):
        """Connect to FibonacciAnalyzer instance for direct integration"""
        self._fibonacci_analyzer = analyzer
        self.logger.info("FibonacciAnalyzer connected to CDFA")
        
    def connect_antifragility_analyzer(self, analyzer):
        """Connect to AntifragilityAnalyzer instance for direct integration"""
        self._antifragility_analyzer = analyzer
        self.logger.info("AntifragilityAnalyzer connected to CDFA")
        
    def connect_pattern_recognizer(self, recognizer):
        """Connect to PatternRecognizer instance for direct integration"""
        self._pattern_recognizer = recognizer
        self.logger.info("PatternRecognizer connected to CDFA")
    
    # ----- EXTERNAL DETECTOR INTEGRATION -----
    
    def analyze(self, data: Union[np.ndarray, pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        General analysis method for compatibility with CDFA server interface
        
        Args:
            data: Input data (DataFrame, numpy array, or signal dictionary)
            
        Returns:
            Dictionary with analysis results including final_score and confidence
        """
        try:
            start_time = time.time()
            
            # Handle different input types
            if isinstance(data, dict):
                # Signal dictionary format (signal_1, signal_2, etc.)
                signals = {}
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signals[key] = np.array(value)
                    else:
                        # Convert single value to array
                        signals[key] = np.array([value])
                
                # Simple fusion of signals
                if signals:
                    signal_values = []
                    for signal_data in signals.values():
                        if len(signal_data) > 0:
                            signal_values.append(np.mean(signal_data))
                    
                    if signal_values:
                        final_score = np.mean(signal_values)
                        confidence = 1.0 - (np.std(signal_values) if len(signal_values) > 1 else 0.0)
                    else:
                        final_score = 0.5
                        confidence = 0.0
                else:
                    final_score = 0.5
                    confidence = 0.0
                
                processing_time = time.time() - start_time
                
                return {
                    "final_score": float(np.clip(final_score, 0.0, 1.0)),
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "components": signals,
                    "processing_time": processing_time
                }
                
            elif isinstance(data, pd.DataFrame):
                # DataFrame format - use existing processing method
                try:
                    result = self.process_signals_from_dataframe(data, "unknown", calculate_fusion=True)
                    
                    # Extract relevant data
                    fusion_result = result.get("fusion_result", {})
                    if isinstance(fusion_result, (list, np.ndarray)) and len(fusion_result) > 0:
                        final_score = float(fusion_result[-1])
                    else:
                        final_score = 0.5
                    
                    # Calculate confidence based on signal diversity
                    signals = result.get("signals", {})
                    if signals:
                        signal_values = []
                        for signal_data in signals.values():
                            if isinstance(signal_data, (list, np.ndarray)) and len(signal_data) > 0:
                                signal_values.append(signal_data[-1] if hasattr(signal_data, '__getitem__') else signal_data)
                        
                        if len(signal_values) > 1:
                            confidence = 1.0 - (np.std(signal_values) / np.mean(signal_values) if np.mean(signal_values) != 0 else 1.0)
                        else:
                            confidence = 0.5
                    else:
                        confidence = 0.5
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "final_score": float(np.clip(final_score, 0.0, 1.0)),
                        "confidence": float(np.clip(confidence, 0.0, 1.0)),
                        "components": signals,
                        "processing_time": processing_time
                    }
                    
                except Exception as e:
                    self.logger.debug(f"DataFrame processing failed, using fallback: {e}")
                    # Fallback for DataFrame
                    if 'close' in data.columns and len(data) > 0:
                        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] if data['close'].iloc[0] != 0 else 0.0
                        final_score = 0.5 + np.tanh(price_change) * 0.3
                        confidence = min(1.0, abs(price_change) * 10)
                    else:
                        final_score = 0.5
                        confidence = 0.0
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "final_score": float(np.clip(final_score, 0.0, 1.0)),
                        "confidence": float(np.clip(confidence, 0.0, 1.0)),
                        "components": {},
                        "processing_time": processing_time
                    }
                    
            elif isinstance(data, np.ndarray):
                # Numpy array format
                if len(data.shape) == 1:
                    # 1D array - treat as price series
                    if len(data) > 1:
                        price_change = (data[-1] - data[0]) / data[0] if data[0] != 0 else 0.0
                        volatility = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0.0
                        
                        final_score = 0.5 + np.tanh(price_change) * 0.3
                        confidence = min(1.0, volatility * 2.0)
                    else:
                        final_score = 0.5
                        confidence = 0.0
                else:
                    # Multi-dimensional array - use mean across dimensions
                    final_score = np.mean(data)
                    confidence = 1.0 - np.std(data) if np.std(data) < 1.0 else 0.0
                
                processing_time = time.time() - start_time
                
                return {
                    "final_score": float(np.clip(final_score, 0.0, 1.0)),
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "components": {"raw_data": data.tolist()},
                    "processing_time": processing_time
                }
                
            else:
                # Unknown data type
                self.logger.warning(f"Unknown data type for analysis: {type(data)}")
                return {
                    "final_score": 0.5,
                    "confidence": 0.0,
                    "components": {},
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            self.logger.error(f"Error in analyze method: {e}")
            return {
                "final_score": 0.5,
                "confidence": 0.0,
                "components": {},
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def integrate_external_detectors(self):
        """Setup connections to external detector components"""
        # Initialize detector attributes
        self._whale_detector = None
        self._black_swan_detector = None
        self._fibonacci_detector = None
        
        self.logger.info("CDFA external detector integration initialized")
        return True
        
    def integrate_whale_detector(self, detector):
        """
        Integrate WhaleDetector for direct signal processing.
        
        Args:
            detector: WhaleDetector instance
        """
        self._whale_detector = detector
        self.logger.info("WhaleDetector integrated with CDFA")
        
    def integrate_black_swan_detector(self, detector):
        """
        Integrate BlackSwanDetector for direct signal processing.
        
        Args:
            detector: BlackSwanDetector instance
        """
        self._black_swan_detector = detector
        self.logger.info("BlackSwanDetector integrated with CDFA")
        
    def integrate_fibonacci_detector(self, detector):
        """
        Integrate FibonacciPatternDetector for direct signal processing.
        
        Args:
            detector: FibonacciPatternDetector instance
        """
        self._fibonacci_detector = detector
        self.logger.info("FibonacciPatternDetector integrated with CDFA")
    
    # ----- DATA PROCESSING METHODS -----
    
    def process_signals_from_dataframe(self, dataframe: pd.DataFrame, symbol: str, calculate_fusion: bool = True) -> Dict[str, Any]:
        """
        Extract signals from dataframe using connected analyzers and optionally calculate fusion.
        """
        signals = {}
        performance_metrics = {}
        
        # Add debug logging
        self.logger.info(f"Processing signals for {symbol} with {len(dataframe)} candles")
        
        # Log dataframe structure
        self.logger.debug(f"DataFrame columns: {dataframe.columns.tolist()}")
        self.logger.debug(f"DataFrame has NaN values: {dataframe.isna().sum().sum() > 0}")
        
        
        # Extract market regime and volatility information
        market_regime = "unknown"
        volatility = 0.5
        
        # --- CRITICAL FIX: Add basic signal generation ---
        # Generate basic price-based signals even if analyzers fail
        if 'close' in dataframe.columns and len(dataframe) > 20:
            try:
                # Simple moving averages
                dataframe['sma_20'] = dataframe['close'].rolling(window=20).mean()
                dataframe['sma_50'] = dataframe['close'].rolling(window=50).mean()
                
                # Basic trend signal (1 = uptrend, 0 = downtrend)
                trend = (dataframe['sma_20'] > dataframe['sma_50']).astype(float)
                signals["basic_trend"] = trend.fillna(0.5).tolist()
                performance_metrics["basic_trend"] = 0.7
                
                # Momentum signal
                momentum = dataframe['close'].pct_change(14)
                norm_momentum = (momentum - momentum.rolling(30).min()) / (momentum.rolling(30).max() - momentum.rolling(30).min())
                signals["basic_momentum"] = norm_momentum.fillna(0.5).tolist()
                performance_metrics["basic_momentum"] = 0.65
                
                # Volatility estimation
                returns = dataframe['close'].pct_change().dropna()
                rolling_std = returns.rolling(window=20).std()
                norm_vol = (rolling_std - rolling_std.rolling(50).min()) / (rolling_std.rolling(50).max() - rolling_std.rolling(50).min())
                signals["basic_volatility"] = norm_vol.fillna(0.5).tolist()
                performance_metrics["basic_volatility"] = 0.6
                
                # Use last value for volatility estimate
                if not norm_vol.empty:
                    volatility = float(norm_vol.iloc[-1])
                    
                self.logger.info(f"Generated basic signals for {symbol}")
            except Exception as e:
                self.logger.error(f"Error generating basic signals: {e}")
            
        # ---- SOC Analyzer Integration ----
        if self._soc_analyzer is not None:
            try:
                # Calculate SOC metrics if not already in dataframe
                if 'soc_index' not in dataframe.columns:
                    # Period for SOC calculations (30 or 50 are typical values)
                    period = 30
                    # Get SOC metrics
                    soc_metrics = self._soc_analyzer.calculate_soc_metrics(dataframe, period)
                    
                    # Extract signals
                    signals["soc_index"] = soc_metrics['soc_index'].tolist()
                    signals["soc_complexity"] = soc_metrics['complexity'].tolist()
                    signals["soc_equilibrium"] = soc_metrics['equilibrium'].tolist()
                    signals["soc_fragility"] = soc_metrics['fragility'].tolist()
                    
                    # Get SOC momentum
                    soc_momentum = self._soc_analyzer.calculate_soc_momentum(dataframe, period)
                    signals["soc_momentum"] = soc_momentum.tolist()
                    
                    # Get SOC divergence
                    soc_divergence = self._soc_analyzer.calculate_soc_divergence(dataframe, period)
                    signals["soc_divergence"] = soc_divergence.tolist()
                    
                    # Get SOC regime and use it for market regime if available
                    if 'regime' in soc_metrics and len(soc_metrics['regime']) > 0:
                        last_regime = soc_metrics['regime'].iloc[-1]
                        if isinstance(last_regime, str):
                            market_regime = last_regime
                    
                    # Set performance metrics
                    performance_metrics["soc_index"] = 0.75
                    performance_metrics["soc_complexity"] = 0.70
                    performance_metrics["soc_equilibrium"] = 0.70
                    performance_metrics["soc_fragility"] = 0.70
                else:
                    # Extract from existing dataframe
                    signals["soc_index"] = dataframe['soc_index'].tolist()
                    
                    # Extract additional metrics if available
                    for column in ['soc_complexity', 'soc_equilibrium', 'soc_fragility', 
                                   'soc_momentum', 'soc_divergence']:
                        if column in dataframe.columns:
                            signals[column] = dataframe[column].tolist()
                    
                    # Get SOC regime for market regime
                    if 'soc_regime' in dataframe.columns:
                        last_regime = dataframe['soc_regime'].iloc[-1]
                        if isinstance(last_regime, str):
                            market_regime = last_regime
                            
                    # Set performance metrics
                    performance_metrics["soc_index"] = 0.75
            except Exception as e:
                self.logger.error(f"Error getting SOC signals: {e}")
        
        # ---- Panarchy Analyzer Integration ----
        if self._panarchy_analyzer is not None:
            try:
                # Calculate Panarchy components if not already in dataframe
                if 'panarchy_P' not in dataframe.columns:
                    # Period for Panarchy calculations
                    period = 50
                    # Calculate PCR components
                    panarchy_df = self._panarchy_analyzer.calculate_pcr_components(dataframe, period)
                    
                    # Extract signals
                    signals["panarchy_potential"] = panarchy_df['panarchy_P'].tolist()
                    signals["panarchy_connectedness"] = panarchy_df['panarchy_C'].tolist()
                    signals["panarchy_resilience"] = panarchy_df['panarchy_R'].tolist()
                    
                    # Identify regime
                    panarchy_df = self._panarchy_analyzer.identify_regime(panarchy_df, period)
                    
                    # Extract regime data
                    signals["panarchy_regime_score"] = panarchy_df['panarchy_regime_score'].tolist()
                    
                    # Use panarchy phase for market regime if available
                    if 'panarchy_phase' in panarchy_df.columns:
                        last_phase = panarchy_df['panarchy_phase'].iloc[-1]
                        if isinstance(last_phase, str):
                            market_regime = last_phase
                    
                    # Set performance metrics
                    performance_metrics["panarchy_potential"] = 0.70
                    performance_metrics["panarchy_connectedness"] = 0.70
                    performance_metrics["panarchy_resilience"] = 0.70
                    performance_metrics["panarchy_regime_score"] = 0.75
                else:
                    # Extract from existing dataframe
                    for column in ['panarchy_P', 'panarchy_C', 'panarchy_R', 'panarchy_regime_score']:
                        if column in dataframe.columns:
                            col_name = column.lower().replace('panarchy_', 'panarchy_')
                            if column == 'panarchy_P':
                                col_name = 'panarchy_potential'
                            elif column == 'panarchy_C':
                                col_name = 'panarchy_connectedness'
                            elif column == 'panarchy_R':
                                col_name = 'panarchy_resilience'
                                
                            signals[col_name] = dataframe[column].tolist()
                    
                    # Use panarchy phase for market regime
                    if 'panarchy_phase' in dataframe.columns:
                        last_phase = dataframe['panarchy_phase'].iloc[-1]
                        if isinstance(last_phase, str):
                            market_regime = last_phase
                    
                    # Set performance metrics
                    performance_metrics["panarchy_regime_score"] = 0.75
            except Exception as e:
                self.logger.error(f"Error getting Panarchy signals: {e}")
        
        # ---- Fibonacci Analyzer Integration ----
        if self._fibonacci_analyzer is not None:
            try:
                # Calculate Fibonacci metrics if not already in dataframe
                if 'fib_alignment_score' not in dataframe.columns:
                    # Set periods for swing point detection and alignment score
                    period = 20
                    
                    # Identify swing points
                    fib_df = self._fibonacci_analyzer.identify_swing_points(dataframe.copy(), period)
                    
                    # Calculate retracements
                    fib_df = self._fibonacci_analyzer.calculate_retracements(fib_df, period)
                    
                    # Calculate extensions
                    fib_df = self._fibonacci_analyzer.calculate_extensions(fib_df)
                    
                    # Calculate alignment score
                    fib_df = self._fibonacci_analyzer.calculate_alignment_score(fib_df, period)
                    
                    # Extract signals
                    signals["fibonacci_alignment"] = fib_df['fib_alignment_score'].tolist()
                    
                    # Set performance metrics
                    performance_metrics["fibonacci_alignment"] = 0.65
                else:
                    # Extract from existing dataframe
                    signals["fibonacci_alignment"] = dataframe['fib_alignment_score'].tolist()
                    
                    # Set performance metrics
                    performance_metrics["fibonacci_alignment"] = 0.65
                    
                # Calculate MTF confluence if metadata available
                if hasattr(dataframe, 'metadata') and isinstance(dataframe.metadata, dict):
                    metadata = dataframe.metadata
                    timeframes = metadata.get('timeframes', ['1h'])
                    
                    # Calculate confluence
                    fib_confluence = self._fibonacci_analyzer.calculate_mtf_confluence(
                        dataframe, timeframes, metadata)
                    
                    signals["fibonacci_confluence"] = fib_confluence.get('fib_mtf_confluence', [0.0] * len(dataframe))
                    performance_metrics["fibonacci_confluence"] = 0.70
            except Exception as e:
                self.logger.error(f"Error getting Fibonacci signals: {e}")
        
        # ---- Antifragility Analyzer Integration ----
        if self._antifragility_analyzer is not None:
            try:
                # Calculate Antifragility metrics if not already in dataframe
                if 'antifragility' not in dataframe.columns and 'convexity' not in dataframe.columns:
                    # Calculate convexity first (this is the public method)
                    convexity = self._antifragility_analyzer.calculate_convexity(dataframe)
                    signals["convexity"] = convexity.tolist()
                    
                    # If dataframe doesn't have convexity yet, add it for antifragility calculation
                    df_copy = dataframe.copy()
                    df_copy['convexity'] = convexity
                    
                    # Calculate volatility metrics
                    vol_metrics = self._antifragility_analyzer.calculate_robust_volatility(df_copy)
                    vol_regime = vol_metrics['vol_regime']
                    signals["volatility_regime"] = vol_regime.tolist()
                    
                    # If volatility is not set yet, use this value
                    if isinstance(vol_regime, pd.Series) and len(vol_regime) > 0:
                        volatility = float(vol_regime.iloc[-1])
                    
                    # Calculate antifragility index
                    antifragility = self._antifragility_analyzer.calculate_antifragility_index(df_copy)
                    signals["antifragility"] = antifragility.tolist()
                    
                    # Calculate fragility score
                    fragility = self._antifragility_analyzer.calculate_fragility_score(df_copy)
                    signals["antifragility_fragility"] = fragility.tolist()
                    
                    # Set performance metrics
                    performance_metrics["convexity"] = 0.80
                    performance_metrics["antifragility"] = 0.85
                    performance_metrics["antifragility_fragility"] = 0.75
                else:
                    # Extract from existing dataframe
                    if 'convexity' in dataframe.columns:
                        signals["convexity"] = dataframe['convexity'].tolist()
                        performance_metrics["convexity"] = 0.80
                    
                    if 'antifragility' in dataframe.columns:
                        signals["antifragility"] = dataframe['antifragility'].tolist()
                        performance_metrics["antifragility"] = 0.85
                    
                    # Get volatility regime if available
                    if 'volatility_regime' in dataframe.columns:
                        signals["volatility_regime"] = dataframe['volatility_regime'].tolist()
                        # Update volatility value
                        vol_series = dataframe['volatility_regime']
                        if len(vol_series) > 0:
                            volatility = float(vol_series.iloc[-1])
                            
                    # Get fragility score if available
                    if 'fragility_score' in dataframe.columns:
                        signals["antifragility_fragility"] = dataframe['fragility_score'].tolist()
                        performance_metrics["antifragility_fragility"] = 0.75
            except Exception as e:
                self.logger.error(f"Error getting Antifragility signals: {e}")
        
        # ---- Pattern Recognizer Integration ----
        if self._pattern_recognizer is not None and 'close' in dataframe.columns:
            try:
                # Extract price data
                high = dataframe['high'].values if 'high' in dataframe.columns else None
                low = dataframe['low'].values if 'low' in dataframe.columns else None
                close = dataframe['close'].values
                
                # Define templates for pattern detection
                templates = {
                    "head_shoulders": np.array([0.3, 0.6, 0.4, 0.8, 0.4, 0.6, 0.3]),
                    "double_top": np.array([0.3, 0.8, 0.5, 0.8, 0.3]),
                    "double_bottom": np.array([0.8, 0.3, 0.5, 0.3, 0.8]),
                    "triangle": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]),
                    "flag": np.array([0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7])
                }
                
                # Use the PatternRecognizer to detect patterns
                # Use its window sizes
                window = PatternRecWindow.MEDIUM.value
                pattern_results = self._pattern_recognizer.detect_dtw_patterns(
                    close, templates, window_size=window)
                
                # Add pattern signals
                for pattern_name, similarity in pattern_results.items():
                    signal_name = f"pattern_{pattern_name}"
                    # Repeat the similarity value for all candles
                    signals[signal_name] = [similarity] * len(dataframe)
                    # Set performance metrics based on pattern type
                    performance_metrics[signal_name] = 0.70
                    
                # If available, detect more sophisticated patterns using 
                # class-specific methods from PatternRecognizer
                if hasattr(self._pattern_recognizer, 'detect_shapelet_patterns'):
                    shapelets = self._pattern_recognizer.detect_shapelet_patterns(close, templates)
                    for pattern_name, similarity in shapelets.items():
                        signal_name = f"shapelet_{pattern_name}"
                        signals[signal_name] = [similarity] * len(dataframe)
                        performance_metrics[signal_name] = 0.75
            except Exception as e:
                self.logger.error(f"Error getting pattern signals: {e}")
        
        # Process with detector integration components if available
        detector_signals = {}
        
        # WhaleDetector
        if hasattr(self, '_whale_detector') and self._whale_detector is not None:
            whale_signals = self.get_whale_signals(dataframe)
            detector_signals.update(whale_signals)
            
        # BlackSwanDetector
        if hasattr(self, '_black_swan_detector') and self._black_swan_detector is not None:
            bs_signals = self.get_black_swan_signals(dataframe)
            detector_signals.update(bs_signals)
            
        # FibonacciDetector
        if hasattr(self, '_fibonacci_detector') and self._fibonacci_detector is not None:
            fib_signals = self.get_fibonacci_signals(dataframe)
            detector_signals.update(fib_signals)
            
        # Add detector signals to main signals dict
        signals.update(detector_signals)
            

        # Apply ML enhancements if available
        if self.config.enable_ml and hasattr(self, '_ml_processor') and self._ml_processor.can_process():
            try:
                # Extract features for ML (time-based signals)
                feature_dict = {}
                for name, values in signals.items():
                    # --- Original code checks if values exist, which can cause the error if 'values' is a Series ---
                    # --- This check should be safe IF the previous steps guarantee 'values' are lists ---
                    # --- The fix below ensures that ML-added signals are lists ---
                    if values: # Keep this original check, but ensure values are list-like upstream
                        feature_dict[name] = values # Assuming 'values' is a list here

                # Periodically train model with accumulated data
                self._ml_processor.train_model(symbol)

                # Try to predict additional signals using ML
                ml_signals = {} # Initialize the dict for ML signals for this iteration
                ml_predictor = None

                # Check for Pulsar integration
                if hasattr(self._ml_processor, '_pulsar_predictor'):
                    ml_predictor = self._ml_processor._pulsar_predictor

                # Check if ml_predictor was successfully assigned AND has the required method
                if ml_predictor and hasattr(ml_predictor, 'predict_signals'):
                    # Use Pulsar predictor for signal generation
                    prediction = ml_predictor.predict_signals(feature_dict) # feature_dict should contain lists

                    if isinstance(prediction, dict) and 'signals' in prediction:
                        for name, values in prediction['signals'].items():
                            # --- FIX: Ensure values are lists before adding ---
                            if isinstance(values, pd.Series):
                                # Convert Series to list
                                ml_signals[f"ml_{name}"] = values.tolist()
                            elif isinstance(values, np.ndarray):
                                # Convert numpy array to list
                                ml_signals[f"ml_{name}"] = values.tolist()
                            else:
                                # Assume it's already list-like or compatible
                                ml_signals[f"ml_{name}"] = values
                            # --- End Fix ---

                            # Assign performance metrics for the ML signal
                            performance_metrics[f"ml_{name}"] = prediction.get('confidences', {}).get(name, 0.6)
                # --- End Check for ml_predictor and predict_signals ---

                # Add ML signals to main signals dict (only signals generated in this pass)
                if ml_signals: # Check if any ML signals were actually generated
                     signals.update(ml_signals) # Update the main signals dictionary

            except Exception as e:
                # Log the specific error encountered during ML processing
                self.logger.error(f"Error applying ML enhancements: {e}", exc_info=True) # Added exc_info=True
        
        # Store signals and market info in cache
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            for name, values in signals.items():
                perf = performance_metrics.get(name, 0.7)
                self._signal_cache[symbol][name] = {
                    "timestamp": time.time(),
                    "values": values,
                    "performance": perf,
                    "metadata": {}
                }
                
            # Update market info
            self._market_info[symbol] = {
                "market_regime": market_regime,
                "volatility": volatility
            }
            
        # Calculate fusion if requested
        result = {
            "signals": signals,
            "performance_metrics": performance_metrics,
            "market_regime": market_regime,
            "volatility": volatility,
            "signals_count": len(signals)  # Add this for debugging
        }
        
        if calculate_fusion and signals:
            fusion_result = self._process_fusion(symbol)
            if fusion_result:
                result["fusion_result"] = fusion_result
        
        return result
    
    def process_with_detectors(self, dataframe: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Process dataframe with all integrated detectors and return signals.
        
        Args:
            dataframe: Market data dataframe
            symbol: Symbol identifier
            
        Returns:
            Dictionary with signals and metadata
        """
        # Extract signals from all connected detectors
        signals = {}
        
        # WhaleDetector
        whale_signals = self.get_whale_signals(dataframe)
        signals.update(whale_signals)
        
        # BlackSwanDetector
        bs_signals = self.get_black_swan_signals(dataframe)
        signals.update(bs_signals)
        
        # FibonacciDetector
        fib_signals = self.get_fibonacci_signals(dataframe)
        signals.update(fib_signals)
        
        # Get market regime and volatility
        market_regime = "unknown"
        volatility = 0.5
        
        if "panarchy_phase" in dataframe.columns:
            market_regime = dataframe["panarchy_phase"].iloc[-1]
            
        if "volatility_regime" in dataframe.columns:
            volatility = float(dataframe["volatility_regime"].iloc[-1])
            
        # Store signals in cache
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            for name, values in signals.items():
                self._signal_cache[symbol][name] = {
                    "timestamp": time.time(),
                    "values": values,
                    "performance": 0.7,  # Default performance (can be refined)
                    "metadata": {}
                }
                
            # Update market info
            self._market_info[symbol] = {
                "market_regime": market_regime,
                "volatility": volatility
            }
        
        # Process fusion using stored signals
        fusion_result = self._process_fusion(symbol)
        
        return {
            "signals": signals,
            "fusion_result": fusion_result,
            "market_regime": market_regime,
            "volatility": volatility
        }
    
    def process_signals_for_trading(self, dataframe: pd.DataFrame, 
                                 symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Process signals from detectors into a comprehensive trading signal.
        
        This method provides a full integration between CDFA and the detector
        components, producing a unified signal for trading systems.
        
        Args:
            dataframe: Market data dataframe
            symbol: Symbol identifier
            timeframe: Timeframe string (e.g., '1h', '1d')
            
        Returns:
            Dictionary with processed signals and trading recommendations
        """
        # Process with all detectors
        detector_results = self.process_with_detectors(dataframe, symbol)
        signals = detector_results.get('signals', {})
        fusion = detector_results.get('fusion_result', {}).get('fused_signal', [0.5])
        
        # Get market regime and volatility
        market_regime = detector_results.get('market_regime', 'unknown')
        volatility = detector_results.get('volatility', 0.5)
        
        # Extract the most recent signal values
        latest = {}
        for name, values in signals.items():
            if values:
                latest[name] = values[-1]
        
        # Calculate trading signal based on detector and fusion results
        trading_signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': time.time(),
            'market_regime': market_regime,
            'volatility': volatility,
            'fusion_value': fusion[-1] if fusion else 0.5,
            'whale_activity': latest.get('whale_activity', 0.0),
            'black_swan_probability': latest.get('black_swan_probability', 0.0),
            'fibonacci_pattern': max([v for k, v in latest.items() if k.startswith('fibonacci_')], default=0.0),
            'recommendation': self._generate_action_recommendation(signals, fusion, market_regime)
        }
        
        # Calculate risk factors
        risk_factors = {
            'volatility_risk': volatility,
            'black_swan_risk': latest.get('black_swan_probability', 0.0),
            'whale_risk': 1.0 - latest.get('whale_confidence', 0.5),
            'pattern_risk': 1.0 - trading_signal['fibonacci_pattern']
        }
        
        # Calculate overall risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        trading_signal['risk_score'] = risk_score
        
        # Use ML/RL for final recommendation if available
        if self.config.enable_ml and hasattr(self, '_ml_processor') and self._ml_processor.can_process():
            try:
                # Create feature dict for ML prediction
                features = {name: value for name, value in latest.items() if np.isfinite(value)}
                
                # Check for Pulsar integration
                if hasattr(self._ml_processor, '_pulsar_predictor'):
                    ml_predictor = self._ml_processor._pulsar_predictor
                    
                    if hasattr(ml_predictor, 'predict_trading_action'):
                        # Use Pulsar predictor for trading action
                        ml_action = ml_predictor.predict_trading_action(features)
                        
                        # Override recommendation if confidence is high enough
                        if ml_action.get('confidence', 0) > 0.7:
                            trading_signal['recommendation'] = ml_action.get('action', trading_signal['recommendation'])
                            trading_signal['ml_confidence'] = ml_action.get('confidence')
                            trading_signal['ml_enhanced'] = True
            except Exception as e:
                self.logger.error(f"Error applying ML to trading recommendation: {e}")
        
        # Publish to Redis if available
        if hasattr(self, 'redis_client') and self.redis_client:
            self.redis_client.publish('trading:signals', trading_signal)
            self.redis_client.store_fusion_result(
                f"{symbol}_{timeframe}", 
                trading_signal, 
                ttl=self.config.signal_ttl
            )
        
        return trading_signal
    
    # ----- DETECTOR SIGNAL EXTRACTION -----
    
    def get_whale_signals(self, dataframe: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Extract whale activity signals from dataframe using integrated detector.
        
        Args:
            dataframe: Market data dataframe
            
        Returns:
            Dictionary of signal name to values
        """
        signals = {}
        
        if not hasattr(self, '_whale_detector') or self._whale_detector is None:
            self.logger.warning("WhaleDetector not integrated")
            return signals
            
        try:
            # Detect whale activity
            activity = self._whale_detector.detect_whale_activity(dataframe)
            if isinstance(activity, pd.Series):
                signals['whale_activity'] = activity.values.tolist()
                
            # Detect whale direction
            direction = self._whale_detector.detect_whale_direction(dataframe)
            if isinstance(direction, pd.Series):
                # Normalize direction to 0-1 range
                norm_direction = ((direction + 1) / 2).values.tolist()
                signals['whale_direction'] = norm_direction
                
            # Get confidence if available
            if len(activity) > 0:
                last_activity = activity.iloc[-1]
                confidence = self._whale_detector.get_whale_confidence(last_activity)
                signals['whale_confidence'] = [confidence] * len(dataframe)
                
        except Exception as e:
            self.logger.error(f"Error getting whale signals: {e}")
            
        return signals
        
    def get_black_swan_signals(self, dataframe: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Extract black swan signals from dataframe using integrated detector.
        
        Args:
            dataframe: Market data dataframe
            
        Returns:
            Dictionary of signal name to values
        """
        signals = {}
        
        if not hasattr(self, '_black_swan_detector') or self._black_swan_detector is None:
            self.logger.warning("BlackSwanDetector not integrated")
            return signals
            
        try:
            # Calculate black swan probability
            probability = self._black_swan_detector.calculate_black_swan_probability(dataframe)
            if isinstance(probability, pd.Series):
                signals['black_swan_probability'] = probability.values.tolist()
                
            # Get direction if available
            if hasattr(self._black_swan_detector, 'get_black_swan_direction'):
                direction = self._black_swan_detector.get_black_swan_direction(dataframe)
                if isinstance(direction, pd.Series):
                    # Normalize direction to 0-1 range
                    norm_direction = ((direction + 1) / 2).values.tolist()
                    signals['black_swan_direction'] = norm_direction
                    
            # Estimate severity if available and probability is significant
            if len(probability) > 0 and probability.iloc[-1] > 0.3:
                last_prob = probability.iloc[-1]
                last_dir = direction.iloc[-1] if 'direction' in locals() else 0
                
                severity = self._black_swan_detector.estimate_black_swan_severity(
                    dataframe, last_prob, last_dir
                )
                
                # Extract impact as signal
                if isinstance(severity, dict) and 'estimated_impact' in severity:
                    impact = abs(severity['estimated_impact'])
                    signals['black_swan_impact'] = [impact] * len(dataframe)
                    
        except Exception as e:
            self.logger.error(f"Error getting black swan signals: {e}")
            
        return signals
        
    def get_fibonacci_signals(self, dataframe: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Extract Fibonacci pattern signals from dataframe using integrated detector.
        
        Args:
            dataframe: Market data dataframe
            
        Returns:
            Dictionary of signal name to values
        """
        signals = {}
        
        if not hasattr(self, '_fibonacci_detector') or self._fibonacci_detector is None:
            self.logger.warning("FibonacciPatternDetector not integrated")
            return signals
            
        try:
            # Detect patterns
            patterns = self._fibonacci_detector.detect_patterns(dataframe)
            
            if isinstance(patterns, dict):
                # Process each pattern type
                for pattern_name, pattern_list in patterns.items():
                    if pattern_list:
                        # Find highest quality pattern for each type
                        best_pattern = max(pattern_list, key=lambda p: p.quality)
                        signals[f'fibonacci_{pattern_name}'] = [best_pattern.quality] * len(dataframe)
                        
                        # Add pattern target if available
                        if best_pattern.target_price is not None:
                            signals[f'fibonacci_{pattern_name}_target'] = [
                                1.0 if dataframe['close'].iloc[i] > best_pattern.target_price else 0.0
                                for i in range(len(dataframe))
                            ]
                            
            # If no patterns detected but detector has add_pattern_columns method
            if not signals and hasattr(self._fibonacci_detector, 'add_pattern_columns'):
                # Use add_pattern_columns to get pattern data
                pattern_df = self._fibonacci_detector.add_pattern_columns(dataframe.copy())
                
                # Extract relevant columns
                for col in pattern_df.columns:
                    if col.startswith('pattern_') and col not in dataframe.columns:
                        # Convert to list and ensure all values are numeric
                        col_values = pattern_df[col].values
                        if isinstance(col_values[0], (int, float)) or np.issubdtype(col_values.dtype, np.number):
                            signals[col] = col_values.tolist()
                            
        except Exception as e:
            self.logger.error(f"Error getting Fibonacci signals: {e}")
            
        return signals
    
    # ----- SIGNAL HANDLING METHODS -----
    
    def _handle_signal(self, data):
        """Handle incoming signal from Redis with enhanced analyzer support"""
        if not isinstance(data, dict):
            self.logger.warning(f"Received non-dict signal: {type(data)}")
            return
            
        try:
            # Extract signal metadata
            source = data.get("source", "unknown")
            symbol = data.get("symbol", "unknown")
            timestamp = data.get("timestamp", time.time())
            
            # Handle different signal types by source
            if source in ["soc", "soc_analyzer"]:
                # SOC Analyzer signals
                self._handle_soc_signal(data, symbol, timestamp)
            elif source in ["panarchy", "panarchy_analyzer"]:
                # Panarchy Analyzer signals
                self._handle_panarchy_signal(data, symbol, timestamp)
            elif source in ["fibonacci", "fibonacci_analyzer"]:
                # Fibonacci Analyzer signals
                self._handle_fibonacci_signal(data, symbol, timestamp)
            elif source in ["antifragility", "antifragility_analyzer"]:
                # Antifragility Analyzer signals
                self._handle_antifragility_signal(data, symbol, timestamp)
            elif source in ["pattern", "pattern_recognizer"]:
                # Pattern Recognizer signals
                self._handle_pattern_signal(data, symbol, timestamp)
            else:
                # Generic signal handling
                values = data.get("values", [])
                
                # Store in signal cache
                with self._lock:
                    if symbol not in self._signal_cache:
                        self._signal_cache[symbol] = {}
                        
                    self._signal_cache[symbol][source] = {
                        "timestamp": timestamp,
                        "values": values,
                        "performance": data.get("performance", 0.5),
                        "metadata": data.get("metadata", {})
                    }
                    
                    # Update market info if available
                    market_regime = data.get("market_regime")
                    volatility = data.get("volatility")
                    
                    if market_regime is not None or volatility is not None:
                        if symbol not in self._market_info:
                            self._market_info[symbol] = {}
                            
                        if market_regime is not None:
                            self._market_info[symbol]["market_regime"] = market_regime
                            
                        if volatility is not None:
                            self._market_info[symbol]["volatility"] = volatility
                            
            # Process fusion if requested
            if data.get("process_fusion", False):
                self._process_fusion(symbol)
                
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
    
    def _handle_control(self, data):
        """Handle control messages from Redis"""
        if not isinstance(data, dict):
            self.logger.warning(f"Received non-dict control message: {type(data)}")
            return
            
        command = data.get("command", "").lower()
        
        if command == "fusion":
            # Process fusion for specified symbol
            symbol = data.get("symbol")
            if symbol:
                self._process_fusion(symbol)
                
        elif command == "clear_cache":
            # Clear internal caches
            self.clear_cache()
            
        elif command == "recover":
            # Trigger recovery
            self.recover()
            
        elif command == "config":
            # Update configuration
            config_updates = data.get("config", {})
            if config_updates:
                self._update_config(config_updates)
                
        elif command == "visualize":
            # Generate visualization if requested
            if hasattr(self, '_visualizer') and self._visualizer.can_visualize():
                symbol = data.get("symbol")
                viz_type = data.get("type", "diversity")
                output_file = data.get("output", None)
                
                if symbol and symbol in self._signal_cache:
                    signals = {
                        name: signal.get("values", [])
                        for name, signal in self._signal_cache[symbol].items()
                    }
                    
                    if viz_type == "diversity":
                        # Create diversity matrix visualization
                        viz = self._visualizer.create_diversity_matrix_plot(
                            self.calculate_pairwise_diversity_matrix(signals)
                        )
                    elif viz_type == "weights":
                        # Get weights
                        weights = {}
                        for entry in reversed(self._fusion_history):
                            if entry.get("symbol") == symbol:
                                weights = entry.get("weights", {})
                                break
                                
                        # Create weight distribution visualization
                        viz = self._visualizer.create_weight_distribution_plot(
                            symbol, weights
                        )
                    elif viz_type == "performance":
                        # Create performance visualization
                        viz = self._visualizer.create_fusion_performance_plot(
                            symbol, self._fusion_history
                        )
                    elif viz_type == "comparison":
                        # Get fusion result
                        fusion_result = []
                        for entry in reversed(self._fusion_history):
                            if entry.get("symbol") == symbol:
                                fusion_result = entry.get("fused_signal", [])
                                break
                                
                        # Create signal comparison visualization
                        viz = self._visualizer.create_signal_comparison_plot(
                            symbol, signals, fusion_result
                        )
                    else:
                        viz = None
                        
                    # Save visualization if requested
                    if viz and output_file:
                        self._visualizer.save_plot(viz, output_file)
                        
                    # Publish success message
                    if self.redis_client:
                        self.redis_client.publish("visualization:result", {
                            "symbol": symbol,
                            "type": viz_type,
                            "success": viz is not None,
                            "output": output_file
                        })
    
    def _handle_soc_signal(self, data, symbol, timestamp):
        """Handle SOC analyzer signal"""
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            # Store SOC index
            if "soc_index" in data:
                self._signal_cache[symbol]["soc_index"] = {
                    "timestamp": timestamp,
                    "values": data.get("soc_index", []),
                    "performance": data.get("soc_index_performance", 0.75),
                    "metadata": {}
                }
                
            # Store SOC equilibrium
            if "equilibrium" in data:
                self._signal_cache[symbol]["soc_equilibrium"] = {
                    "timestamp": timestamp,
                    "values": data.get("equilibrium", []),
                    "performance": data.get("equilibrium_performance", 0.70),
                    "metadata": {}
                }
                
            # Store SOC fragility
            if "fragility" in data:
                self._signal_cache[symbol]["soc_fragility"] = {
                    "timestamp": timestamp,
                    "values": data.get("fragility", []),
                    "performance": data.get("fragility_performance", 0.70),
                    "metadata": {}
                }
                
            # Store SOC complexity
            if "complexity" in data:
                self._signal_cache[symbol]["soc_complexity"] = {
                    "timestamp": timestamp,
                    "values": data.get("complexity", []),
                    "performance": data.get("complexity_performance", 0.70),
                    "metadata": {}
                }
                
            # Update market info
            if "regime" in data:
                if symbol not in self._market_info:
                    self._market_info[symbol] = {}
                    
                self._market_info[symbol]["market_regime"] = data.get("regime")
    
    def _handle_panarchy_signal(self, data, symbol, timestamp):
        """Handle Panarchy analyzer signal"""
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            # Store Panarchy potential
            if "potential" in data:
                self._signal_cache[symbol]["panarchy_potential"] = {
                    "timestamp": timestamp,
                    "values": data.get("potential", []),
                    "performance": data.get("potential_performance", 0.70),
                    "metadata": {}
                }
                
            # Store Panarchy connectedness
            if "connectedness" in data:
                self._signal_cache[symbol]["panarchy_connectedness"] = {
                    "timestamp": timestamp,
                    "values": data.get("connectedness", []),
                    "performance": data.get("connectedness_performance", 0.70),
                    "metadata": {}
                }
                
            # Store Panarchy resilience
            if "resilience" in data:
                self._signal_cache[symbol]["panarchy_resilience"] = {
                    "timestamp": timestamp,
                    "values": data.get("resilience", []),
                    "performance": data.get("resilience_performance", 0.70),
                    "metadata": {}
                }
                
            # Store Panarchy regime score
            if "regime_score" in data:
                self._signal_cache[symbol]["panarchy_regime_score"] = {
                    "timestamp": timestamp,
                    "values": data.get("regime_score", []),
                    "performance": data.get("regime_score_performance", 0.75),
                    "metadata": {}
                }
                
            # Update market info
            if "phase" in data:
                if symbol not in self._market_info:
                    self._market_info[symbol] = {}
                    
                self._market_info[symbol]["market_regime"] = data.get("phase")
    
    def _handle_fibonacci_signal(self, data, symbol, timestamp):
        """Handle Fibonacci analyzer signal"""
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            # Store Fibonacci alignment score
            if "alignment_score" in data:
                self._signal_cache[symbol]["fibonacci_alignment"] = {
                    "timestamp": timestamp,
                    "values": data.get("alignment_score", []),
                    "performance": data.get("alignment_performance", 0.65),
                    "metadata": {}
                }
                
            # Store Fibonacci confluence
            if "confluence" in data:
                self._signal_cache[symbol]["fibonacci_confluence"] = {
                    "timestamp": timestamp,
                    "values": data.get("confluence", []),
                    "performance": data.get("confluence_performance", 0.70),
                    "metadata": {}
                }
                
            # Store any pattern detections
            for key in data:
                if key.startswith("pattern_"):
                    pattern_name = key[8:]  # Remove "pattern_" prefix
                    self._signal_cache[symbol][f"fibonacci_{pattern_name}"] = {
                        "timestamp": timestamp,
                        "values": data.get(key, []),
                        "performance": data.get(f"{key}_performance", 0.70),
                        "metadata": {}
                    }
    
    def _handle_antifragility_signal(self, data, symbol, timestamp):
        """Handle Antifragility analyzer signal"""
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            # Store convexity
            if "convexity" in data:
                self._signal_cache[symbol]["convexity"] = {
                    "timestamp": timestamp,
                    "values": data.get("convexity", []),
                    "performance": data.get("convexity_performance", 0.80),
                    "metadata": {}
                }
                
            # Store antifragility index
            if "antifragility" in data:
                self._signal_cache[symbol]["antifragility"] = {
                    "timestamp": timestamp,
                    "values": data.get("antifragility", []),
                    "performance": data.get("antifragility_performance", 0.85),
                    "metadata": {}
                }
                
            # Store fragility score
            if "fragility" in data:
                self._signal_cache[symbol]["antifragility_fragility"] = {
                    "timestamp": timestamp,
                    "values": data.get("fragility", []),
                    "performance": data.get("fragility_performance", 0.75),
                    "metadata": {}
                }
                
            # Update volatility info
            if "volatility_regime" in data:
                if symbol not in self._market_info:
                    self._market_info[symbol] = {}
                    
                volatility = data.get("volatility_regime")
                if isinstance(volatility, list) and volatility:
                    volatility = volatility[-1]
                    
                self._market_info[symbol]["volatility"] = float(volatility)
    
    def _handle_pattern_signal(self, data, symbol, timestamp):
        """Handle Pattern recognizer signal"""
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            # Store pattern detections
            patterns = data.get("patterns", {})
            if isinstance(patterns, dict):
                for pattern_name, similarity in patterns.items():
                    # Convert similarity to list if it's a single value
                    if not isinstance(similarity, list):
                        similarity = [similarity] * data.get("length", 1)
                        
                    self._signal_cache[symbol][f"pattern_{pattern_name}"] = {
                        "timestamp": timestamp,
                        "values": similarity,
                        "performance": data.get(f"{pattern_name}_performance", 0.70),
                        "metadata": {}
                    }
    
    def _handle_panarchy_feedback(self, feedback):
        """Process feedback from Panarchy system for adaptive learning"""
        if not isinstance(feedback, dict):
            self.logger.warning(f"Invalid Panarchy feedback format: {type(feedback)}")
            return
            
        try:
            # Extract feedback data
            symbol = feedback.get('symbol')
            signal_quality = feedback.get('signal_quality')
            performance_metrics = feedback.get('performance_metrics', {})
            
            if not symbol or signal_quality is None:
                return
                
            # Update performance metrics for signals
            with self._lock:
                if symbol in self._signal_cache:
                    for source, signal in self._signal_cache[symbol].items():
                        # Update performance based on feedback
                        current_perf = signal.get('performance', 0.5)
                        
                        # Adaptive learning rate (0.1-0.3)
                        learning_rate = 0.1 + (signal_quality * 0.2)
                        
                        # Update with exponential moving average
                        new_perf = current_perf * (1 - learning_rate) + signal_quality * learning_rate
                        
                        # Store updated performance
                        self._signal_cache[symbol][source]['performance'] = new_perf
                        
            # Apply feedback to ML components if available
            if self.config.enable_ml and hasattr(self, '_ml_processor') and self._ml_processor.can_process():
                # Extract fusion features for ML training
                with self._lock:
                    if symbol in self._fusion_history:
                        # Find the most recent fusion result for this symbol
                        for entry in reversed(self._fusion_history):
                            if entry.get('symbol') == symbol:
                                # Create feature dict from signals
                                features = {}
                                for name, sig in self._signal_cache.get(symbol, {}).items():
                                    values = sig.get('values', [])
                                    if values:
                                        features[name] = values[-1]
                                
                                # Add feedback as training sample
                                self._ml_processor.add_training_sample(
                                    symbol=symbol,
                                    features=features,
                                    target=signal_quality
                                )
                                break
                
            # Apply feedback to adaptive learning component
            if self.config.enable_adaptive_learning and hasattr(self, '_adaptive_learner'):
                with self._lock:
                    if symbol in self._fusion_history:
                        # Find the most recent fusion result for this symbol
                        for entry in reversed(self._fusion_history):
                            if entry.get('symbol') == symbol:
                                # Extract signals, fusion result, and weights used
                                signals = {}
                                for name, sig in self._signal_cache.get(symbol, {}).items():
                                    signals[name] = sig.get('values', [])
                                
                                # Record experience with reward
                                self._adaptive_learner.record_experience(
                                    symbol=symbol,
                                    weights=entry.get('weights', {}),
                                    signals=signals,
                                    result=entry.get('fused_signal', []),
                                    reward=signal_quality
                                )
                                break
            
            # Log feedback
            self.logger.info(f"Received Panarchy feedback for {symbol}: quality={signal_quality:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing Panarchy feedback: {e}")
    
    # ----- FUSION PROCESSING -----
    
    def adaptive_fusion(self,
                      system_scores: Dict[str, List[float]],
                      performance_metrics: Dict[str, float],
                      market_regime: str, # Expects string like "trending", "crisis" etc.
                      volatility: float,  # Expects float 0-1
                      method: Union[str, DiversityMethod] = None) -> List[float]:
        """
        Enhanced adaptive fusion with optimal parameter selection.
        
        Args:
            system_scores: Dictionary of system name to signal values
            performance_metrics: Dictionary of system name to performance metric
            market_regime: Market regime indicator
            volatility: Volatility level (0-1)
            method: Diversity method to use (optional)
            
        Returns:
            List of fused signal values
        """
        if not system_scores:
            return []
            
        # Ensure volatility is valid
        volatility = np.clip(volatility, 0.0, 1.0)
        
        # Filter performance metrics to ensure only valid values
        valid_performance_metrics = {}
        for k, v in performance_metrics.items():
            try:
                # Convert to float if possible
                if isinstance(v, (str, bool, dict, list)):
                    continue  # Skip non-numeric types
                v_float = float(v)
                if np.isfinite(v_float):  # Now check if finite
                    valid_performance_metrics[k] = v_float
            except (TypeError, ValueError):
                # Skip any values that can't be converted to float
                continue
            
        # Try ML-based fusion if enabled
        if self.config.enable_ml and hasattr(self, '_ml_processor') and self._ml_processor.can_process():
            ml_weights = self._ml_processor.predict_weights(
                symbol="default",  # Use default if no specific symbol provided
                features=system_scores
            )
            
            if ml_weights:
                self.logger.debug(f"Using ML-based weights: {ml_weights}")
                return self.fusion_hybrid_combination(system_scores, ml_weights)
                
        # Try adaptive learning if enabled
        if self.config.enable_adaptive_learning and hasattr(self, '_adaptive_learner'):
            adaptive_weights = self._adaptive_learner.optimize_weights(
                symbol="default",  # Use default if no specific symbol provided
                signals=system_scores
            )
            
            if adaptive_weights:
                self.logger.debug(f"Using adaptive learning weights: {adaptive_weights}")
                return self.fusion_hybrid_combination(system_scores, adaptive_weights)
        
        # Determine fusion type based on market regime and volatility
        market_regime_lower = market_regime.lower()
        
        if market_regime_lower == "crisis":
            fusion_type = FusionType.LAYERED
        elif market_regime_lower == "growth" and volatility < 0.4:
            fusion_type = FusionType.SCORE
        elif market_regime_lower == "conservation":
            fusion_type = FusionType.HYBRID
        elif market_regime_lower == "release" and volatility > 0.6:
            fusion_type = FusionType.RANK
        else:
            # Default based on volatility
            fusion_type = FusionType.HYBRID if volatility < 0.7 else FusionType.RANK
            
            # Determine adaptive alpha for hybrid fusion
            adaptive_alpha = 0.5
            if fusion_type == FusionType.HYBRID:
                sensitivity = 1.0  # Default value
                if hasattr(self.config, 'adaptive_alpha_vol_sensitivity'):
                    # For object-style config
                    sensitivity = self.config.adaptive_alpha_vol_sensitivity
                elif isinstance(self.config, dict):
                    # For dictionary-style config
                    sensitivity = self.config.get('adaptive_alpha_vol_sensitivity', 1.0)
                elif hasattr(self.config, '__getattr__'):
                    # For custom config objects with __getattr__
                    try:
                        sensitivity = self.config.__getattr__('adaptive_alpha_vol_sensitivity')
                    except (AttributeError, KeyError):
                        pass
            adaptive_alpha = np.clip(0.5 + (0.5 - volatility) * sensitivity, 0.0, 1.0)
            
        # Adjust diversity threshold based on market regime
        original_threshold = self.config.diversity_threshold
        self.config.diversity_threshold = max(0.1, original_threshold * (
            0.7 if market_regime_lower == "crisis" else (1 - volatility * 0.5)
        ))
        
        # Calculate weights
        if method is None:
            method = self.config.default_diversity_method
        elif isinstance(method, str):
            method = DiversityMethod.from_string(method)
            
        final_weights = self._calculate_adaptive_weights(
            system_scores,
            valid_performance_metrics,
            method
        )
        
        # Perform fusion based on type
        result = []
        try:
            if fusion_type == FusionType.LAYERED:
                result = self.multi_layer_fusion(
                    system_scores,
                    valid_performance_metrics,
                    method=method,
                    expansion_factor=(3 if market_regime_lower == "crisis" else self.config.expansion_factor)
                )
            elif fusion_type == FusionType.HYBRID:
                result = self.fusion_hybrid_combination(
                    system_scores,
                    final_weights,
                    alpha=adaptive_alpha
                )
            elif fusion_type == FusionType.SCORE:
                result = self.fusion_score_combination(
                    system_scores,
                    final_weights
                )
            elif fusion_type == FusionType.RANK:
                result = self.fusion_rank_combination(
                    system_scores,
                    final_weights
                )
        finally:
            # Restore original threshold
            self.config.diversity_threshold = original_threshold
            
        self.logger.info(
            f"AdaptiveFusion: Regime='{market_regime}', Vol={volatility:.2f} -> "
            f"Type={fusion_type.name}, Alpha={adaptive_alpha:.2f}, "
            f"DivThresh={self.config.diversity_threshold:.2f}"
        )
        
        return result

    def _calculate_adaptive_weights(self,
                                  system_scores: Dict[str, List[float]],
                                  performance_metrics: Dict[str, float],
                                  method: DiversityMethod) -> Dict[str, float]:
        """
        Enhanced weight calculation with vectorized operations.
        """
        if not system_scores:
            return {}
            
        # Calculate diversity matrix
        diversity_matrix = self.calculate_pairwise_diversity_matrix(system_scores, method)
        
        # Calculate average diversity for each system (vectorized)
        avg_diversity = {}
        for name in system_scores:
            if name in diversity_matrix.index:
                # Get diversities excluding self-comparison
                diversities = diversity_matrix.loc[name].drop(name) if name in diversity_matrix.columns else diversity_matrix.loc[name]
                avg_diversity[name] = float(np.mean(diversities)) if len(diversities) > 0 else 0.0
            else:
                avg_diversity[name] = 0.0
        
        # Extract configuration parameters
        weights = {}
        perf_thresh = self.config.performance_threshold
        div_thresh = self.config.diversity_threshold
        weighting_scheme = self.config.diversity_weighting_scheme
        
        # Calculate weights based on scheme
        for system in system_scores:
            perf = performance_metrics.get(system, 0.0)
            div = avg_diversity.get(system, 0.0)
            weight = 0.0 # Default weight
            
            if weighting_scheme == "multiplicative":
                if perf >= perf_thresh and div >= div_thresh:
                    weight = perf * div
            elif weighting_scheme == "perf_thresholded":
                if perf >= perf_thresh:
                    weight = div
            elif weighting_scheme == "additive":
                perf_bias = self.config.additive_weighting_perf_bias
                div_bias = 1.0 - perf_bias
                if perf >= perf_thresh and div >= div_thresh:
                    weight = perf_bias * perf + div_bias * div
            else: # Fallback to multiplicative
                if perf >= perf_thresh and div >= div_thresh:
                    weight = perf * div
                    
            weights[system] = max(0.0, weight) # Ensure non-negative
            
        # Handle case where no systems meet thresholds
        total_weight = sum(weights.values())
        if total_weight < 1e-9:
            self.logger.warning(f"No systems meet thresholds for scheme '{weighting_scheme}'. Using equal weights.")
            num_valid_systems = sum(1 for s in system_scores if system_scores[s])
            equal_weight = 1.0 / num_valid_systems if num_valid_systems > 0 else 0.0
            return {system: equal_weight for system in system_scores if system_scores[system]}
            
        return weights
        
    def _process_fusion(self, symbol):
        """Process fusion for a specific symbol"""
        # Check if we have signals for this symbol
        with self._lock:
            if symbol not in self._signal_cache:
                self.logger.warning(f"No signals for symbol: {symbol}")
                return False
                
            # Extract system scores
            system_scores = {}
            performance_metrics = {}
            
            signals = self._signal_cache[symbol]
            for source, signal in signals.items():
                values = signal.get("values", [])
                if values:
                    system_scores[source] = values
                    performance_metrics[source] = signal.get("performance", 0.5)
                    
            # Get market info
            market_info = self._market_info.get(symbol, {})
            market_regime = market_info.get("market_regime", "unknown")
            volatility = market_info.get("volatility", 0.5)
            
        # Check if we have enough signals
        if len(system_scores) < self.config.min_signals_required:
            self.logger.warning(f"Not enough signals for fusion: {len(system_scores)}/{self.config.min_signals_required}")
            return False
            
        # Process fusion
        fused_signal = self.adaptive_fusion(
            system_scores,
            performance_metrics,
            market_regime,
            volatility
        )
        
        if not fused_signal:
            self.logger.warning(f"Fusion failed for symbol: {symbol}")
            return False
            
        # Calculate system dispersion for confidence
        dispersion = self.calculate_system_dispersion(system_scores)
        confidence = 1.0 - dispersion * 0.5
        
        # If ML is enabled, add training sample for future optimization
        if self.config.enable_ml and hasattr(self, '_ml_processor') and self._ml_processor.can_process():
            # Extract features for ML (last values of each signal)
            features = {}
            for name, values in system_scores.items():
                if values:
                    features[name] = values[-1]
                    
            # Add to training data with current fusion result as target
            self._ml_processor.add_training_sample(
                symbol=symbol,
                features=features,
                target=fused_signal[-1] if fused_signal else 0.5
            )
            
            # Periodically train model
            last_update = self._ml_processor._last_update.get(symbol, 0)
            if time.time() - last_update > self.config.ml_update_interval:
                self._ml_processor.train_model(symbol)
        
        # Create fusion result
        fusion_result = {
            "symbol": symbol,
            "timestamp": time.time(),
            "fused_signal": fused_signal,
            "confidence": confidence,
            "market_regime": market_regime,
            "volatility": volatility,
            "num_systems": len(system_scores),
            "performance_metrics": performance_metrics
        }
        
        # Store in history
        with self._lock:
            self._fusion_history.append(fusion_result)
            if len(self._fusion_history) > 100:
                self._fusion_history = self._fusion_history[-100:]
                
        # Publish to Redis if available
        if self.redis_client:
            # Publish fusion result
            self.redis_client.publish("fusion", fusion_result)
            
            # Store with TTL
            self.redis_client.store_fusion_result(symbol, fusion_result, ttl=self.config.signal_ttl)
            
        return fusion_result
    
    def _generate_action_recommendation(self, signals, fusion, market_regime) -> Dict[str, Any]:
        """
        Generate recommendation based on fusion result.
        
        Args:
            signals: Dictionary of signal values
            fusion: Fusion result signal
            market_regime: Current market regime
            
        Returns:
            Dictionary with recommendation
        """
        if not fusion:
            return {"action": "none", "strength": 0.0}
            
        # Get latest signal value
        latest_signal = fusion[-1]
        
        # Calculate confidence based on signal strength
        signal_confidence = abs(latest_signal - 0.5) * 2.0
        
        # Adjust thresholds based on market regime
        buy_threshold = 0.7
        sell_threshold = 0.3
        
        market_regime_lower = market_regime.lower()
        if market_regime_lower == "growth":
            # More sensitive to buying in growth regime
            buy_threshold = 0.65
            sell_threshold = 0.25
        elif market_regime_lower == "conservation":
            # More balanced in conservation regime
            buy_threshold = 0.7
            sell_threshold = 0.3
        elif market_regime_lower == "release":
            # More sensitive to selling in release regime
            buy_threshold = 0.75
            sell_threshold = 0.35
        elif market_regime_lower == "reorganization":
            # More cautious in reorganization regime
            buy_threshold = 0.75
            sell_threshold = 0.25
            
        # Calculate black swan impact if available
        black_swan_prob = 0.0
        for name, values in signals.items():
            if name == "black_swan_probability" and values:
                black_swan_prob = values[-1]
                break
                
        # Adjust thresholds based on black swan probability
        if black_swan_prob > 0.5:
            # More cautious when black swan probability is high
            buy_threshold += black_swan_prob * 0.1
            sell_threshold -= black_swan_prob * 0.1
        
        # Determine action based on signal strength
        if latest_signal > buy_threshold and signal_confidence > 0.6:
            action = "buy"
            strength = (latest_signal - buy_threshold) * (1.0 / (1.0 - buy_threshold)) * signal_confidence
        elif latest_signal < sell_threshold and signal_confidence > 0.6:
            action = "sell"
            strength = (sell_threshold - latest_signal) * (1.0 / sell_threshold) * signal_confidence
        else:
            action = "hold"
            strength = 0.5
            
        return {
            "action": action,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "signal_value": float(latest_signal),
            "confidence": float(signal_confidence),
            "thresholds": {
                "buy": float(buy_threshold),
                "sell": float(sell_threshold)
            }
        }
    
    # ----- PANARCHY INTEGRATION -----
    
    def publish_to_panarchy(self, symbol: str, fusion_result: Optional[Dict[str, Any]] = None):
        """
        Publish fusion results to Panarchy Adaptive Decision System.
        
        Args:
            symbol: Symbol identifier
            fusion_result: Optional fusion result (if None, will use latest)
            
        Returns:
            True if published successfully
        """
        if not self.redis_client:
            self.logger.warning("Redis client not available. Cannot publish to Panarchy.")
            return False
            
        try:
            # Get latest fusion result if not provided
            if fusion_result is None:
                # Process fusion to get latest result
                if not self._process_fusion(symbol):
                    self.logger.warning(f"No fusion result available for {symbol}")
                    return False
                    
                # Get latest fusion history entry for this symbol
                with self._lock:
                    for entry in reversed(self._fusion_history):
                        if entry.get("symbol") == symbol:
                            fusion_result = entry
                            break
                            
                if fusion_result is None:
                    self.logger.warning(f"No fusion history entry found for {symbol}")
                    return False
                    
            # Create recommendation
            recommendation = self._generate_recommendation(fusion_result)
                    
            # Publish to Panarchy channel
            return self.redis_client.publish("panarchy:decision", {
                "symbol": symbol,
                "timestamp": fusion_result.get("timestamp", time.time()),
                "fusion_signal": fusion_result.get("fused_signal", []),
                "confidence": fusion_result.get("confidence", 0.5),
                "market_regime": fusion_result.get("market_regime", "unknown"),
                "volatility": fusion_result.get("volatility", 0.5),
                "source": "cdfa",
                "recommendation": recommendation,
                "metadata": {
                    "num_systems": fusion_result.get("num_systems", 0),
                    "fusion_history_length": len(self._fusion_history)
                }
            })
        except Exception as e:
            self.logger.error(f"Error publishing to Panarchy: {e}")
            return False
    
    def _generate_recommendation(self, fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendation based on fusion result.
        
        Args:
            fusion_result: Fusion result
            
        Returns:
            Dictionary with recommendation
        """
        fused_signal = fusion_result.get("fused_signal", [])
        if not fused_signal:
            return {"action": "none", "strength": 0.0}
            
        # Get latest signal value
        latest_signal = fused_signal[-1]
        confidence = fusion_result.get("confidence", 0.5)
        
        # Determine action based on signal strength
        if latest_signal > 0.7 and confidence > 0.6:
            action = "buy"
            strength = (latest_signal - 0.7) * (1.0 / 0.3) * confidence
        elif latest_signal < 0.3 and confidence > 0.6:
            action = "sell"
            strength = (0.3 - latest_signal) * (1.0 / 0.3) * confidence
        else:
            action = "hold"
            strength = 0.5
            
        return {
            "action": action,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "signal_value": float(latest_signal),
            "confidence": float(confidence)
        }
    
    def integrate_with_panarchy(self, config: Dict[str, Any] = None):
        """
        Set up integration with Panarchy Adaptive Decision System.
        
        Args:
            config: Optional configuration parameters
        
        Returns:
            Success flag
        """
        if not hasattr(self, 'redis_client') or not self.redis_client:
            self.logger.warning("Redis client not available. Cannot integrate with Panarchy.")
            return False
            
        # Default configuration
        default_config = {
            'channel': 'panarchy:input',
            'response_channel': 'panarchy:feedback',
            'signal_format': 'detailed',  # 'detailed' or 'simple'
            'update_interval': 300,       # seconds
            'enable_feedback': True
        }
        
        # Merge with provided config
        panarchy_config = {**default_config, **(config or {})}
        
        # Store config
        self._panarchy_config = panarchy_config
        
        # Subscribe to feedback channel if enabled
        if panarchy_config['enable_feedback']:
            self.redis_client.subscribe(
                panarchy_config['response_channel'],
                self._handle_panarchy_feedback
            )
            
        self.logger.info(f"Panarchy integration configured with channel {panarchy_config['channel']}")
        
        return True
    
    # ----- ANALYZER REQUEST METHODS -----
    
    def request_soc_analysis(self, dataframe: pd.DataFrame, period: int = 30) -> Dict[str, pd.Series]:
        """
        Request SOC analysis from connected SOCAnalyzer.
        
        Args:
            dataframe: Input dataframe with price data
            period: Period for SOC calculations
            
        Returns:
            Dictionary with SOC metrics
        """
        if self._soc_analyzer is None:
            self.logger.warning("SOCAnalyzer not connected")
            return {}
            
        try:
            # Calculate SOC metrics
            soc_metrics = self._soc_analyzer.calculate_soc_metrics(dataframe, period)
            
            # Calculate additional metrics
            soc_momentum = self._soc_analyzer.calculate_soc_momentum(dataframe, period)
            soc_divergence = self._soc_analyzer.calculate_soc_divergence(dataframe, period)
            
            # Add to results
            combined_metrics = soc_metrics.copy()
            combined_metrics['momentum'] = soc_momentum
            combined_metrics['divergence'] = soc_divergence
            
            return combined_metrics
        except Exception as e:
            self.logger.error(f"Error in SOC analysis: {e}")
            return {}
    
    def request_panarchy_analysis(self, dataframe: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Request Panarchy analysis from connected PanarchyAnalyzer.
        
        Args:
            dataframe: Input dataframe with price data
            period: Period for Panarchy calculations
            
        Returns:
            DataFrame with Panarchy metrics added
        """
        if self._panarchy_analyzer is None:
            self.logger.warning("PanarchyAnalyzer not connected")
            return dataframe
            
        try:
            # Calculate PCR components
            result_df = self._panarchy_analyzer.calculate_pcr_components(dataframe.copy(), period)
            
            # Identify regime
            result_df = self._panarchy_analyzer.identify_regime(result_df, period)
            
            # Calculate regime score
            result_df = self._panarchy_analyzer.calculate_regime_score(result_df)
            
            return result_df
        except Exception as e:
            self.logger.error(f"Error in Panarchy analysis: {e}")
            return dataframe
    
    def request_fibonacci_analysis(self, dataframe: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Request Fibonacci analysis from connected FibonacciAnalyzer.
        
        Args:
            dataframe: Input dataframe with price data
            period: Period for Fibonacci calculations
            
        Returns:
            DataFrame with Fibonacci metrics added
        """
        if self._fibonacci_analyzer is None:
            self.logger.warning("FibonacciAnalyzer not connected")
            return dataframe
            
        try:
            # Identify swing points
            result_df = self._fibonacci_analyzer.identify_swing_points(dataframe.copy(), period)
            
            # Calculate retracements
            result_df = self._fibonacci_analyzer.calculate_retracements(result_df, period)
            
            # Calculate extensions
            result_df = self._fibonacci_analyzer.calculate_extensions(result_df)
            
            # Calculate alignment score
            result_df = self._fibonacci_analyzer.calculate_alignment_score(result_df, period)
            
            return result_df
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analysis: {e}")
            return dataframe
    
    def request_antifragility_analysis(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Request Antifragility analysis from connected AntifragilityAnalyzer.
        
        Args:
            dataframe: Input dataframe with price data
            
        Returns:
            DataFrame with Antifragility metrics added
        """
        if self._antifragility_analyzer is None:
            self.logger.warning("AntifragilityAnalyzer not connected")
            return dataframe
            
        try:
            # Calculate convexity
            convexity = self._antifragility_analyzer.calculate_convexity(dataframe)
            result_df = dataframe.copy()
            result_df['convexity'] = convexity
            
            # Calculate volatility metrics
            vol_metrics = self._antifragility_analyzer.calculate_robust_volatility(result_df)
            result_df['volatility_regime'] = vol_metrics['vol_regime']
            result_df['combined_volatility'] = vol_metrics['combined_vol']
            
            # Calculate antifragility index
            antifragility = self._antifragility_analyzer.calculate_antifragility_index(result_df)
            result_df['antifragility'] = antifragility
            
            # Calculate fragility score
            fragility = self._antifragility_analyzer.calculate_fragility_score(result_df)
            result_df['fragility_score'] = fragility
            
            return result_df
        except Exception as e:
            self.logger.error(f"Error in Antifragility analysis: {e}")
            return dataframe
    
    def request_pattern_recognition(self, series: Union[List[float], np.ndarray], 
                                  templates: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Request pattern recognition from connected PatternRecognizer.
        
        Args:
            series: Input time series data
            templates: Optional custom templates for pattern matching
            
        Returns:
            Dictionary of pattern name -> similarity score
        """
        if self._pattern_recognizer is None:
            # Fall back to internal DTW if PatternRecognizer not connected
            if templates:
                return self.detect_dtw_patterns(series, templates)
            else:
                self.logger.warning("PatternRecognizer not connected and no templates provided")
                return {}
                
        try:
            # Convert to numpy array if needed
            if isinstance(series, list):
                series = np.array(series)
                
            # Use default templates if none provided
            if templates is None:
                # Create default templates similar to PatternRecognizer
                templates = {
                    "head_shoulders": np.array([0.3, 0.6, 0.4, 0.8, 0.4, 0.6, 0.3]),
                    "double_top": np.array([0.3, 0.8, 0.5, 0.8, 0.3]),
                    "double_bottom": np.array([0.8, 0.3, 0.5, 0.3, 0.8]),
                    "triangle": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]),
                    "flag": np.array([0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7])
                }
                
            # Use PatternRecognizer with medium window
            window = PatternRecWindow.MEDIUM.value
            result = self._pattern_recognizer.detect_dtw_patterns(series, templates, window_size=window)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in pattern recognition: {e}")
            
            # Fall back to internal DTW if error occurs
            if templates:
                return self.detect_dtw_patterns(series, templates)
            else:
                return {}
    
    # ----- SYSTEM MANAGEMENT METHODS -----
    
    def clear_cache(self):
        """Clears all internal caches with thread safety"""
        with self._lock:
            self._diversity_cache.clear()
            self._rsc_cache.clear()
            self._dtw_cache.clear()
            self._diversity_matrices.clear()
            self.logger.info("CDFA internal caches cleared.")
            
    def recover(self):
        """Recovers the CDFA component."""
        self.logger.warning("CDFA recovery triggered!")
        try:
            # 1. Clear caches and history
            self.clear_cache()
            
            with self._lock:
                self._fusion_history.clear()
                self._signal_cache.clear()
                self._market_info.clear()
                
            # 2. Reinitialize Redis if needed
            if self.config.enable_redis and REDIS_AVAILABLE:
                if self.redis_client:
                    self.redis_client.close()
                self._initialize_redis()
                
            # 3. Reset ML components
            if self.config.enable_ml and hasattr(self, '_ml_processor'):
                self._ml_processor = MLSignalProcessor(self.config)
                
            # 4. Reset adaptive learning components
            if self.config.enable_adaptive_learning and hasattr(self, '_adaptive_learner'):
                self._adaptive_learner = AdaptiveFusionLearner(self.config)
                
            # 5. Close and recreate visualizer
            if hasattr(self, '_visualizer'):
                self._visualizer.close_all_plots()
                self._visualizer = FusionVisualizer(self.config)
                
            self.logger.info("CDFA recovery attempt finished successfully.")
        except Exception as e_cdfa_rec:
             self.logger.error(f"Error during CDFA recovery: {e_cdfa_rec}", exc_info=True)
    
    # ----- ML/RL INTEGRATION -----
    
    def integrate_with_pulsar(self, pulsar_module):
        """
        Integrate with Pulsar ML/RL system for advanced signal processing.
        
        Args:
            pulsar_module: Pulsar module instance
            
        Returns:
            Success flag
        """
        try:
            self.logger.info("Integrating with Pulsar ML/RL system")
            
            # Check for required attributes
            required_components = [
                "QStarLearningAgent",
                "RiverOnlineML",
                "CerebellumSNN",
                "QuantumOptimizer",
                "QStarRiverPredictor"
            ]
            
            for component in required_components:
                if not hasattr(pulsar_module, component):
                    self.logger.warning(f"Pulsar module missing required component: {component}")
                    
            # Create and initialize ML processor with Pulsar components
            if hasattr(pulsar_module, "QStarRiverPredictor"):
                # Use Pulsar's QStarRiverPredictor for ML-based signal processing
                qstar_predictor = pulsar_module.QStarRiverPredictor()
                
                # Enhance ML processor with Pulsar capabilities
                self._ml_processor._pulsar_predictor = qstar_predictor
                self._ml_processor._has_pulsar = True
                
                # Create prediction method wrapper
                def predict_with_pulsar(symbol, features):
                    # Convert features to format expected by QStarRiverPredictor
                    feature_dict = {}
                    for name, values in features.items():
                        if values:
                            feature_dict[name] = values[-1]
                            
                    # Make prediction using Pulsar
                    result = qstar_predictor.predict(feature_dict)
                    
                    # Return weights
                    return result.get("weights", {})
                    
                # Replace default prediction method
                self._ml_processor._original_predict = self._ml_processor.predict_weights
                self._ml_processor.predict_weights = predict_with_pulsar
                
            # Set up adaptive learning with Pulsar RL components
            if hasattr(pulsar_module, "QStarLearningAgent"):
                # Use Pulsar's QStarLearningAgent for reinforcement learning
                qstar_agent = pulsar_module.QStarLearningAgent()
                
                # Enhance adaptive learner with Pulsar capabilities
                self._adaptive_learner._rl_agent = qstar_agent
                self._adaptive_learner._has_pulsar = True
                
                # Create optimization method wrapper
                def optimize_with_pulsar(symbol, signals):
                    # Convert signals to state representation for QStarLearningAgent
                    state = {}
                    for name, values in signals.items():
                        if values:
                            state[name] = values[-1]
                            
                    # Get action (weights) from agent
                    action = qstar_agent.get_action(state)
                    
                    # Return weights
                    return action.get("weights", {})
                    
                # Replace default optimization method
                self._adaptive_learner._original_optimize = self._adaptive_learner.optimize_weights
                self._adaptive_learner.optimize_weights = optimize_with_pulsar
                
            # Store reference to Pulsar module
            self._pulsar_module = pulsar_module
            
            self.logger.info("Pulsar ML/RL system integration complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error integrating with Pulsar: {e}")
            return False
    
    def register_analyzer_signal(self, symbol: str, source: str, values: List[float], 
                               performance: float = 0.7, market_regime: Optional[str] = None,
                               volatility: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Register a signal from an external analyzer component.
        
        Args:
            symbol: Symbol identifier
            source: Signal source identifier (e.g., 'soc', 'panarchy')
            values: Signal values
            performance: Performance metric for this signal (0-1)
            market_regime: Optional market regime information
            volatility: Optional volatility value (0-1)
            metadata: Optional additional metadata
            
        Returns:
            True if signal was registered successfully
        """
        if not values:
            return False
            
        # Store in signal cache
        with self._lock:
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                
            self._signal_cache[symbol][source] = {
                "timestamp": time.time(),
                "values": values,
                "performance": performance,
                "metadata": metadata or {}
            }
            
            # Update market info if provided
            if market_regime is not None or volatility is not None:
                if symbol not in self._market_info:
                    self._market_info[symbol] = {}
                    
                if market_regime is not None:
                    self._market_info[symbol]["market_regime"] = market_regime
                    
                if volatility is not None:
                    self._market_info[symbol]["volatility"] = volatility
                    
        return True

    def debug_dataframe_status(self, dataframe: pd.DataFrame, stage: str = "") -> Dict[str, Any]:
        """Debug helper to analyze dataframe status"""
        if dataframe is None:
            return {"status": "error", "message": "DataFrame is None", "stage": stage}
            
        if dataframe.empty:
            return {"status": "error", "message": "DataFrame is empty", "stage": stage}
            
        result = {
            "status": "info",
            "stage": stage,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_list": dataframe.columns.tolist(),
            "nan_columns": {}
        }
        
        # Check for NaN values
        for col in dataframe.columns:
            nan_count = dataframe[col].isna().sum()
            if nan_count > 0:
                nan_percent = nan_count / len(dataframe) * 100
                result["nan_columns"][col] = {
                    "count": int(nan_count),
                    "percent": float(nan_percent)
                }
                
        # If there are NaNs, set warning status
        if result["nan_columns"]:
            result["status"] = "warning"
            
        # Check for key columns
        key_column_groups = {
            "price": ["open", "high", "low", "close"],
            "soc": ["soc_index", "soc_equilibrium", "soc_fragility", "soc_momentum"],
            "panarchy": ["panarchy_potential", "panarchy_connectedness", "panarchy_resilience"],
            "antifragility": ["convexity", "antifragility"],
            "pattern": ["pattern_quality"]
        }
        
        result["missing_column_groups"] = {}
        
        for group_name, columns in key_column_groups.items():
            missing = [col for col in columns if col not in dataframe.columns]
            if missing:
                result["missing_column_groups"][group_name] = missing
                result["status"] = "warning"
                
        return result

    def fuse_signals(self, signals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse signals from multiple sources using adaptive fusion.
        API compatibility method for pipeline integration.
        
        Args:
            signals_dict: Dictionary of signals from various sources
            
        Returns:
            Dict containing fused signal, confidence, and metadata
        """
        try:
            # Convert signals_dict to the format expected by adaptive_fusion
            signals = {}
            context = {}
            
            for key, value in signals_dict.items():
                if isinstance(value, (int, float)):
                    signals[key] = float(value)
                elif isinstance(value, list) and value:
                    signals[key] = float(value[-1])  # Use last value from list
                elif isinstance(value, dict) and 'signal' in value:
                    signals[key] = float(value['signal'])
                else:
                    signals[key] = 0.5  # Default neutral signal
            
            # Use adaptive fusion with current context
            if signals:
                result = self.adaptive_fusion(signals, context, "mixed", 0.5)
            else:
                result = {'fused_signal': 0.5, 'confidence': 0.5, 'diversity_score': 0.0}
            
            return {
                'signal': result.get('fused_signal', 0.5),
                'confidence': result.get('confidence', 0.5),
                'diversity': result.get('diversity_score', 0.0),
                'fusion_method': result.get('fusion_method', 'adaptive'),
                'metadata': result.get('metadata', {}),
                'processing_time': result.get('processing_time', 0.0),
                'source_count': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error in fuse_signals: {e}")
            return {
                'signal': 0.5,
                'confidence': 0.0,
                'diversity': 0.0,
                'fusion_method': 'error',
                'error': str(e),
                'source_count': 0
            }

    def register_source(self, source_name: str, source_function: callable) -> None:
        """
        Register a new signal source with the CDFA system.
        API compatibility method for dynamic source registration.
        
        Args:
            source_name: Name of the signal source
            source_function: Function that generates signals
        """
        try:
            # Add to registered signal sources
            if not hasattr(self, '_registered_sources'):
                self._registered_sources = {}
            
            self._registered_sources[source_name] = source_function
            self.logger.info(f"Registered signal source: {source_name}")
            
        except Exception as e:
            self.logger.error(f"Error registering source {source_name}: {e}")