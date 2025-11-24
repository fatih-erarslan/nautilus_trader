"""
Enhanced Logarithmic Market Scoring Rule Implementation.

This module implements Hanson's Logarithmic Market Scoring Rule (LMSR) for
probability aggregation in trading decision systems, with enterprise-grade
robustness, performance optimizations, and advanced features.

The LMSR provides a mathematically sound framework for:
1. Converting raw indicator values to probabilities
2. Aggregating multiple probability estimates coherently
3. Maintaining proper conditional independence relationships
4. Deriving market-implied probabilities from trade volumes

Features:
- Numba njit acceleration for computational hotspots
- Vectorized implementations for batch operations
- Hardware-aware optimizations
- Low-latency processing for high-frequency use cases

References:
- Hanson, R. (2007). Logarithmic market scoring rules for modular 
  combinatorial information aggregation. The Journal of Prediction Markets, 1(1), 3-15.
- Othman, A., & Sandholm, T. (2010). Automated market-making in the large: 
  The Gates Hillman prediction market. In Proceedings of the 11th ACM conference 
  on Electronic commerce (pp. 367-376).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar, Protocol, Callable
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# New imports
import numba
from numba import njit, prange, vectorize, float64, boolean

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')
Number = Union[int, float]

class ProbabilityConversionMethod(Enum):
    """Methods for converting indicator values to probabilities."""
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'
    THRESHOLD = 'threshold'
    EXPONENTIAL = 'exponential'
    
    @classmethod
    def from_string(cls, method_str: str) -> 'ProbabilityConversionMethod':
        """Convert string to method enum, with fuzzy matching."""
        method_str = method_str.upper()
        for method in cls:
            if method.name.upper() == method_str:
                return method
        
        # Fuzzy matching
        for method in cls:
            if method.name.upper().startswith(method_str):
                return method
                
        # Default to sigmoid
        warnings.warn(f"Unknown conversion method: {method_str}, using SIGMOID")
        return cls.SIGMOID

class AggregationMethod(Enum):
    """Methods for aggregating probabilities."""
    LOG_ODDS = 'log_odds'      # Standard log-odds addition
    WEIGHTED = 'weighted'      # Weighted log-odds addition
    BAYESIAN = 'bayesian'      # Sequential Bayesian updates
    GEOMETRIC = 'geometric'    # Geometric mean of probabilities
    
    @classmethod
    def from_string(cls, method_str: str) -> 'AggregationMethod':
        """Convert string to method enum, with fuzzy matching."""
        method_str = method_str.upper()
        for method in cls:
            if method.name.upper() == method_str:
                return method
        
        # Fuzzy matching
        for method in cls:
            if method.name.upper().startswith(method_str):
                return method
                
        # Default to log-odds
        warnings.warn(f"Unknown aggregation method: {method_str}, using LOG_ODDS")
        return cls.LOG_ODDS

# Numba-accelerated core functions
@njit(fastmath=True)
def _normalize_probability_njit(prob: float, min_prob: float, max_prob: float) -> float:
    """Numba-accelerated version of normalize_probability."""
    # No need for NaN/inf checking in njit - will be done at Python level
    return max(min(prob, max_prob), min_prob)

@vectorize([float64(float64, float64, float64)], fastmath=True)
def _normalize_probability_vec(prob: np.ndarray, min_prob: float, max_prob: float) -> np.ndarray:
    """Vectorized version of normalize_probability."""
    return max(min(prob, max_prob), min_prob)

@njit(fastmath=True)
def _to_log_odds_njit(prob: float) -> float:
    """Numba-accelerated version of to_log_odds."""
    return np.log(prob / (1.0 - prob))

@vectorize([float64(float64)], fastmath=True)
def _to_log_odds_vec(prob: np.ndarray) -> np.ndarray:
    """Vectorized version of to_log_odds."""
    return np.log(prob / (1.0 - prob))

@njit(fastmath=True)
def _from_log_odds_njit(log_odds: float, min_prob: float, max_prob: float) -> float:
    """Numba-accelerated version of from_log_odds."""
    # Handle extreme values
    if log_odds > 709.0:
        return max_prob
    elif log_odds < -709.0:
        return min_prob
    
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    return max(min(prob, max_prob), min_prob)

@vectorize([float64(float64, float64, float64)], fastmath=True)
def _from_log_odds_vec(log_odds: float64, min_prob: float64, max_prob: float64) -> float64:
    """Vectorized version of from_log_odds."""
    # Clip extreme values using min/max instead of np.clip for Numba compatibility
    log_odds_clipped = max(min(log_odds, 709.0), -709.0)
    prob = 1.0 / (1.0 + np.exp(-log_odds_clipped))
    return max(min(prob, max_prob), min_prob)

@njit(fastmath=True)
def _sigmoid_conversion_njit(value: float, min_val: float, max_val: float, 
                           center: float, steepness: float) -> float:
    """Numba-accelerated sigmoid conversion."""
    normalized = (value - center) / max(abs(max_val - center), abs(min_val - center))
    return 1.0 / (1.0 + np.exp(-steepness * normalized * 4.0))

@vectorize([float64(float64, float64, float64, float64, float64)], fastmath=True)
def _sigmoid_conversion_vec(value: np.ndarray, min_val: float, max_val: float,
                          center: float, steepness: float) -> np.ndarray:
    """Vectorized sigmoid conversion."""
    normalized = (value - center) / max(abs(max_val - center), abs(min_val - center))
    return 1.0 / (1.0 + np.exp(-steepness * normalized * 4.0))

@njit(fastmath=True)
def _linear_conversion_njit(value: float, min_val: float, max_val: float) -> float:
    """Numba-accelerated linear conversion."""
    if value <= min_val:
        return 0.0
    elif value >= max_val:
        return 1.0
    else:
        return (value - min_val) / (max_val - min_val)

@vectorize([float64(float64, float64, float64)], fastmath=True)
def _linear_conversion_vec(value: float64, min_val: float64, max_val: float64) -> float64:
    """Vectorized linear conversion."""
    # Using min/max instead of np.clip for Numba compatibility
    normalized = (value - min_val) / (max_val - min_val)
    return max(min(normalized, 1.0), 0.0)

@njit(fastmath=True)
def _cost_function_njit(quantities: np.ndarray, liquidity: float) -> float:
    """Numba-accelerated LMSR cost function."""
    # Find maximum quantity for numerical stability
    max_q = quantities.max()
    
    # Calculate cost using log-sum-exp trick
    exp_sum = 0.0
    for q in quantities:
        exp_sum += np.exp((q - max_q) / liquidity)
    
    return liquidity * (max_q / liquidity + np.log(exp_sum))

@njit(fastmath=True)
def _calc_market_probabilities_njit(quantities: np.ndarray, liquidity: float) -> np.ndarray:
    """Numba-accelerated calculation of all market probabilities."""
    # Calculate exponential terms
    exp_terms = np.empty_like(quantities)
    for i in range(len(quantities)):
        exp_terms[i] = np.exp(quantities[i] / liquidity)
    
    # Calculate sum of exponentials
    exp_sum = np.sum(exp_terms)
    
    # Calculate probabilities
    probabilities = np.empty_like(quantities)
    for i in range(len(quantities)):
        probabilities[i] = exp_terms[i] / exp_sum
    
    return probabilities

@njit(fastmath=True, parallel=True)
def _aggregate_log_odds_njit(probs: np.ndarray) -> float:
    """Numba-accelerated log-odds aggregation."""
    # Convert to log-odds
    log_odds_sum = 0.0
    for i in prange(len(probs)):
        log_odds_sum += np.log(probs[i] / (1.0 - probs[i]))
    
    # Return average
    avg_log_odds = log_odds_sum / len(probs)
    
    # Convert back to probability
    return 1.0 / (1.0 + np.exp(-avg_log_odds))

@dataclass
class LMSRConfig:
    """Configuration parameters for LMSR system."""
    liquidity_parameter: float = 100.0
    min_probability: float = 0.001
    max_probability: float = 0.999
    default_conversion_method: ProbabilityConversionMethod = ProbabilityConversionMethod.SIGMOID
    default_aggregation_method: AggregationMethod = AggregationMethod.LOG_ODDS
    enable_caching: bool = True
    cache_size: int = 128
    enable_parallel: bool = True
    max_workers: int = 4
    log_level: int = logging.INFO
    # New configuration parameters
    use_numba: bool = True
    use_vectorization: bool = True
    batch_size: int = 1024
    hardware_aware_parallelism: bool = True
    prefetch_cache: bool = True
    adaptive_precision: bool = True

class LogarithmicMarketScoringRule:
    """
    Enterprise-grade implementation of Hanson's Logarithmic Market Scoring Rule (LMSR) for
    probability aggregation in trading decision systems.
    
    Features:
    1. Thread-safe probability normalization and conversion
    2. Multiple indicator-to-probability conversion methods
    3. Multiple probability aggregation methods
    4. Performance optimizations with caching and parallel processing
    5. Market-based probability derivation with proper cost functions
    6. Numba acceleration for computational hotspots
    7. Vectorized implementations for batch operations
    8. Hardware-aware optimizations
    """
    
    def __init__(self, config: Optional[LMSRConfig] = None):
        """
        Initialize the LMSR system with configuration parameters.
        
        Args:
            config: Optional configuration object
        """
        self.config = config if config is not None else LMSRConfig()
        
        # Initialize logging
        self._initialize_logging()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._execution_times = {}
        self._call_counts = {}
        
        # Caching
        if self.config.enable_caching:
            self._setup_caching()
        
        # History of probability transformations
        self._history = []
        self._history_lock = threading.Lock()
        
        # Initialize numba cache if enabled
        if self.config.use_numba and self.config.prefetch_cache:
            self._warm_numba_cache()
        
        logger.info(f"LogarithmicMarketScoringRule initialized with config: {self.config}")
    
    def _initialize_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.log_level)
        
        # Configure hardware-aware settings
        if self.config.hardware_aware_parallelism:
            self._configure_hardware_awareness()
    
    def _configure_hardware_awareness(self):
        """Configure settings based on hardware capabilities."""
        try:
            import multiprocessing
            
            # Get CPU information
            cpu_count = multiprocessing.cpu_count()
            
            # Adjust worker count based on available cores
            if cpu_count > 1:
                # Use 75% of available cores, minimum 2
                recommended_workers = max(2, int(cpu_count * 0.75))
                if recommended_workers != self.config.max_workers:
                    self.logger.info(f"Adjusting worker count from {self.config.max_workers} to {recommended_workers} based on {cpu_count} available cores")
                    self.config.max_workers = recommended_workers
            
            # Try to get more detailed system info if psutil is available
            try:
                import psutil
                
                # Check available memory
                mem_info = psutil.virtual_memory()
                mem_gb = mem_info.total / (1024**3)  # Convert to GB
                
                # Adjust batch size based on available memory
                if mem_gb < 4:  # Low memory system
                    self.config.batch_size = min(self.config.batch_size, 512)
                    self.logger.info(f"Low memory system detected ({mem_gb:.1f} GB). Reducing batch size to {self.config.batch_size}")
                elif mem_gb > 16:  # High memory system
                    self.config.batch_size = max(self.config.batch_size, 4096)
                    self.logger.info(f"High memory system detected ({mem_gb:.1f} GB). Increasing batch size to {self.config.batch_size}")
            except ImportError:
                self.logger.debug("psutil module not available, skipping memory detection")
            
            # Check CPU capabilities for SIMD support
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                
                # Enable advanced vectorization if supported
                if 'avx2' in info.get('flags', []) or 'avx512f' in info.get('flags', []):
                    self.logger.info(f"Advanced vector extensions detected (AVX2/AVX512). Enabling optimized vectorization.")
                    # Numba can leverage these automatically
            except ImportError:
                self.logger.debug("cpuinfo module not available, skipping CPU feature detection")
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Error during hardware detection: {e}. Using default settings.")
    
    def _warm_numba_cache(self):
        """Pre-compile numba functions by executing them once with sample data."""
        try:
            # Warm up the most used njit functions
            test_val = 0.5
            test_arr = np.array([0.1, 0.5, 0.9])
            
            _ = _normalize_probability_njit(test_val, self.config.min_probability, self.config.max_probability)
            _ = _normalize_probability_vec(test_arr, self.config.min_probability, self.config.max_probability)
            _ = _to_log_odds_njit(test_val)
            _ = _to_log_odds_vec(test_arr)
            _ = _from_log_odds_njit(0.0, self.config.min_probability, self.config.max_probability)
            _ = _from_log_odds_vec(np.array([0.0]), self.config.min_probability, self.config.max_probability)
            _ = _sigmoid_conversion_njit(0.5, 0.0, 1.0, 0.5, 1.0)
            _ = _sigmoid_conversion_vec(test_arr, 0.0, 1.0, 0.5, 1.0)
            _ = _cost_function_njit(test_arr, self.config.liquidity_parameter)
            _ = _calc_market_probabilities_njit(test_arr, self.config.liquidity_parameter)
            
            self.logger.info("Numba cache warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Error warming numba cache: {e}")
    
    def _setup_caching(self):
        """Apply LRU caching to expensive functions."""
        # Apply caching to pure functions that benefit from it
        self.normalize_probability = lru_cache(maxsize=self.config.cache_size)(
            self.normalize_probability
        )
        self.to_log_odds = lru_cache(maxsize=self.config.cache_size)(
            self.to_log_odds
        )
        self.from_log_odds = lru_cache(maxsize=self.config.cache_size)(
            self.from_log_odds
        )
    
    def _time_execution(func):
        """Decorator to track execution time of methods."""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update metrics
            func_name = func.__name__
            if func_name not in self._execution_times:
                self._execution_times[func_name] = []
            if func_name not in self._call_counts:
                self._call_counts[func_name] = 0
                
            self._execution_times[func_name].append(execution_time)
            self._call_counts[func_name] += 1
            
            # Limit stored times to prevent memory growth
            if len(self._execution_times[func_name]) > 100:
                self._execution_times[func_name].pop(0)
                
            return result
        return wrapper
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        metrics = {}
        for func_name, times in self._execution_times.items():
            if times:
                metrics[func_name] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'call_count': self._call_counts.get(func_name, 0)
                }
        return metrics
    
    def normalize_probability(self, prob: float) -> float:
        """
        Clip probability to allowed range to prevent numerical issues.
        
        This method is thread-safe and handles edge cases appropriately.
        
        Args:
            prob: Input probability value
            
        Returns:
            Normalized probability within configured bounds
        """
        try:
            # Handle NaN and infinity
            if np.isnan(prob) or np.isinf(prob):
                self.logger.warning(f"Invalid probability value: {prob}, using default 0.5")
                return 0.5
            
            # Use njit version if enabled
            if self.config.use_numba:
                return _normalize_probability_njit(
                    float(prob), 
                    self.config.min_probability, 
                    self.config.max_probability
                )
            else:
                # Original implementation
                return max(min(float(prob), self.config.max_probability), self.config.min_probability)
        except Exception as e:
            self.logger.error(f"Error normalizing probability: {e}")
            return 0.5
    
    def normalize_probabilities(self, probs: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Vectorized version of normalize_probability for multiple probabilities.
        
        Args:
            probs: List or array of probability values
            
        Returns:
            Normalized array of probabilities
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(probs, np.ndarray):
                probs = np.array(probs, dtype=np.float64)
            
            # Handle NaN and infinity
            mask_invalid = np.isnan(probs) | np.isinf(probs)
            if np.any(mask_invalid):
                self.logger.warning(f"Invalid probability values found, replacing with 0.5")
                probs = probs.copy()  # Create a copy to avoid modifying the original
                probs[mask_invalid] = 0.5
            
            # Use vectorized version if enabled
            if self.config.use_numba and self.config.use_vectorization:
                return _normalize_probability_vec(
                    probs, 
                    self.config.min_probability, 
                    self.config.max_probability
                )
            else:
                # Non-vectorized fallback - manually create bounds arrays for element-wise min/max
                min_bounds = np.full_like(probs, self.config.min_probability)
                max_bounds = np.full_like(probs, self.config.max_probability)
                # Apply min and max element-wise
                return np.maximum(np.minimum(probs, max_bounds), min_bounds)
        except Exception as e:
            self.logger.error(f"Error normalizing probabilities: {e}")
            return np.full_like(probs, 0.5) if isinstance(probs, np.ndarray) else np.array([0.5] * len(probs))
    
    @_time_execution
    def to_log_odds(self, probability: float) -> float:
        """
        Convert probability to log-odds.
        
        Log-odds is the natural logarithm of the odds ratio: ln(p/(1-p))
        
        Args:
            probability: Input probability
            
        Returns:
            Log-odds value
        """
        with self._lock:
            probability = self.normalize_probability(probability)
            if self.config.use_numba:
                return _to_log_odds_njit(probability)
            else:
                return np.log(probability / (1 - probability))
    
    def to_log_odds_vectorized(self, probabilities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Vectorized version of to_log_odds for multiple probabilities.
        
        Args:
            probabilities: List or array of probability values
            
        Returns:
            Array of log-odds values
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(probabilities, np.ndarray):
                probabilities = np.array(probabilities, dtype=np.float64)
            
            # Normalize probabilities
            normalized = self.normalize_probabilities(probabilities)
            
            # Use vectorized version if enabled
            if self.config.use_numba and self.config.use_vectorization:
                return _to_log_odds_vec(normalized)
            else:
                # Fallback to numpy vectorized operations
                return np.log(normalized / (1.0 - normalized))
        except Exception as e:
            self.logger.error(f"Error converting probabilities to log-odds: {e}")
            return np.zeros_like(probabilities) if isinstance(probabilities, np.ndarray) else np.zeros(len(probabilities))
    
    @_time_execution
    def from_log_odds(self, log_odds: float) -> float:
        """
        Convert log-odds back to probability.
        
        Inverse of to_log_odds: p = 1/(1+exp(-log_odds))
        
        Args:
            log_odds: Input log-odds value
            
        Returns:
            Probability value
        """
        with self._lock:
            try:
                if self.config.use_numba:
                    return _from_log_odds_njit(
                        log_odds,
                        self.config.min_probability,
                        self.config.max_probability
                    )
                else:
                    # Original implementation
                    if log_odds > 709:  # Near upper limit for np.exp
                        return self.config.max_probability
                    elif log_odds < -709:  # Near lower limit
                        return self.config.min_probability
                    
                    return self.normalize_probability(1 / (1 + np.exp(-log_odds)))
            except Exception as e:
                self.logger.error(f"Error in from_log_odds: {e}")
                return 0.5
    
    def from_log_odds_vectorized(self, log_odds: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Vectorized version of from_log_odds for multiple log-odds values.
        
        Args:
            log_odds: List or array of log-odds values
            
        Returns:
            Array of probability values
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(log_odds, np.ndarray):
                log_odds = np.array(log_odds, dtype=np.float64)
            
            # Use vectorized version if enabled
            if self.config.use_numba and self.config.use_vectorization:
                return _from_log_odds_vec(
                    log_odds,
                    self.config.min_probability,
                    self.config.max_probability
                )
            else:
                # Create bounds arrays for element-wise operations
                min_log_odds = np.full_like(log_odds, -709.0)
                max_log_odds = np.full_like(log_odds, 709.0)
                min_prob = np.full_like(log_odds, self.config.min_probability)
                max_prob = np.full_like(log_odds, self.config.max_probability)
                
                # Clip extreme values using element-wise min/max
                log_odds_clipped = np.maximum(np.minimum(log_odds, max_log_odds), min_log_odds)
                prob = 1.0 / (1.0 + np.exp(-log_odds_clipped))
                return np.maximum(np.minimum(prob, max_prob), min_prob)
        except Exception as e:
            self.logger.error(f"Error converting log-odds to probabilities: {e}")
            return np.full_like(log_odds, 0.5) if isinstance(log_odds, np.ndarray) else np.array([0.5] * len(log_odds))
    
    @_time_execution
    def indicator_to_probability(self, 
                               value: float, 
                               min_val: float, 
                               max_val: float, 
                               center: Optional[float] = None,
                               steepness: float = 1.0,
                               method: Optional[Union[str, ProbabilityConversionMethod]] = None) -> float:
        """
        Convert a raw indicator value to a probability estimate.
        
        This flexible method supports multiple conversion approaches:
        1. Sigmoid: Smooth S-curve with adjustable steepness
        2. Linear: Simple linear mapping from value range to [0,1]
        3. Threshold: Step function with binary output
        4. Exponential: Exponential mapping for emphasis on extreme values
        
        Args:
            value: Raw indicator value
            min_val: Minimum expected value of the indicator
            max_val: Maximum expected value of the indicator
            center: Value at which probability should be 0.5 (default: midpoint)
            steepness: Controls how quickly probability changes around center
            method: Conversion method to use
            
        Returns:
            Probability estimate between min_prob and max_prob
        """
        # Validate inputs
        try:
            value = float(value)
            min_val = float(min_val)
            max_val = float(max_val)
            steepness = float(steepness)
        except (ValueError, TypeError):
            self.logger.error(f"Invalid numeric inputs: value={value}, min={min_val}, max={max_val}")
            return 0.5
        
        # Handle out-of-range or invalid inputs
        if np.isnan(value) or np.isinf(value):
            self.logger.warning(f"Invalid indicator value: {value}, using default 0.5")
            return 0.5
            
        if min_val >= max_val:
            self.logger.warning(f"Invalid range: min={min_val}, max={max_val}, using default 0.5")
            return 0.5
        
        # Use default center if not provided
        if center is None:
            center = (min_val + max_val) / 2
        
        # Use default method if not specified
        if method is None:
            method = self.config.default_conversion_method
        elif isinstance(method, str):
            method = ProbabilityConversionMethod.from_string(method)
        
        # Calculate probability based on specified method
        try:
            if method == ProbabilityConversionMethod.SIGMOID:
                if self.config.use_numba:
                    probability = _sigmoid_conversion_njit(value, min_val, max_val, center, steepness)
                else:
                    # Normalize to [-1, 1] range relative to center
                    normalized = (value - center) / max(abs(max_val - center), abs(min_val - center))
                    
                    # Apply sigmoid transformation
                    probability = 1 / (1 + np.exp(-steepness * normalized * 4))
                
            elif method == ProbabilityConversionMethod.LINEAR:
                if self.config.use_numba:
                    probability = _linear_conversion_njit(value, min_val, max_val)
                else:
                    # Simple linear mapping
                    if value <= min_val:
                        probability = 0.0
                    elif value >= max_val:
                        probability = 1.0
                    else:
                        probability = (value - min_val) / (max_val - min_val)
                    
            elif method == ProbabilityConversionMethod.THRESHOLD:
                # Step function
                probability = 1.0 if value >= center else 0.0
                
            elif method == ProbabilityConversionMethod.EXPONENTIAL:
                # Exponential mapping
                if value <= min_val:
                    probability = 0.0
                elif value >= max_val:
                    probability = 1.0
                else:
                    # Normalize to [0, 1]
                    norm_val = (value - min_val) / (max_val - min_val)
                    # Apply exponential transformation
                    # steepness > 1: emphasize high values
                    # steepness < 1: emphasize low values
                    probability = norm_val ** steepness
            else:
                # Fallback to linear
                self.logger.warning(f"Unknown method: {method}, falling back to linear")
                probability = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                
            # Record transformation
            with self._history_lock:
                if len(self._history) < 1000:  # Limit history size
                    self._history.append({
                        'type': 'indicator_to_probability',
                        'method': str(method),
                        'input': value,
                        'output': probability,
                        'timestamp': time.time()
                    })
            
            return self.normalize_probability(probability)
            
        except Exception as e:
            self.logger.error(f"Error in indicator_to_probability: {e}")
            return 0.5
    
    def indicator_to_probabilities_vectorized(self,
                                           values: Union[List[float], np.ndarray],
                                           min_val: float,
                                           max_val: float,
                                           center: Optional[float] = None,
                                           steepness: float = 1.0,
                                           method: Optional[Union[str, ProbabilityConversionMethod]] = None) -> np.ndarray:
        """
        Vectorized version of indicator_to_probability for multiple values.
        
        Args:
            values: List or array of indicator values
            min_val: Minimum expected value of the indicator
            max_val: Maximum expected value of the indicator
            center: Value at which probability should be 0.5 (default: midpoint)
            steepness: Controls how quickly probability changes around center
            method: Conversion method to use
            
        Returns:
            Array of probability estimates
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=np.float64)
            
            # Validate inputs
            if np.isnan(min_val) or np.isinf(min_val) or np.isnan(max_val) or np.isinf(max_val):
                self.logger.error(f"Invalid range: min={min_val}, max={max_val}")
                return np.full_like(values, 0.5)
                
            if min_val >= max_val:
                self.logger.warning(f"Invalid range: min={min_val}, max={max_val}")
                return np.full_like(values, 0.5)
            
            # Handle NaN and infinity in values
            mask_invalid = np.isnan(values) | np.isinf(values)
            if np.any(mask_invalid):
                self.logger.warning(f"Invalid indicator values found, replacing with default")
                values = values.copy()  # Create a copy to avoid modifying the original
                values[mask_invalid] = (min_val + max_val) / 2
            
            # Use default center if not provided
            if center is None:
                center = (min_val + max_val) / 2
            
            # Use default method if not specified
            if method is None:
                method = self.config.default_conversion_method
            elif isinstance(method, str):
                method = ProbabilityConversionMethod.from_string(method)
            
            # Calculate probabilities based on specified method
            if method == ProbabilityConversionMethod.SIGMOID:
                if self.config.use_numba and self.config.use_vectorization:
                    probabilities = _sigmoid_conversion_vec(values, min_val, max_val, center, steepness)
                else:
                    # Normalize to [-1, 1] range relative to center
                    normalized = (values - center) / max(abs(max_val - center), abs(min_val - center))
                    
                    # Apply sigmoid transformation
                    probabilities = 1.0 / (1.0 + np.exp(-steepness * normalized * 4.0))
                
            elif method == ProbabilityConversionMethod.LINEAR:
                if self.config.use_numba and self.config.use_vectorization:
                    probabilities = _linear_conversion_vec(values, min_val, max_val)
                else:
                    # Simple linear mapping with element-wise min/max instead of clip
                    normalized = (values - min_val) / (max_val - min_val)
                    zeros = np.zeros_like(normalized)
                    ones = np.ones_like(normalized)
                    probabilities = np.maximum(np.minimum(normalized, ones), zeros)
                    
            elif method == ProbabilityConversionMethod.THRESHOLD:
                # Step function
                probabilities = np.where(values >= center, 1.0, 0.0)
                
            elif method == ProbabilityConversionMethod.EXPONENTIAL:
                # Exponential mapping
                # Normalize to [0, 1] using element-wise min/max
                normalized = (values - min_val) / (max_val - min_val)
                zeros = np.zeros_like(normalized)
                ones = np.ones_like(normalized)
                norm_vals = np.maximum(np.minimum(normalized, ones), zeros)
                # Apply exponential transformation
                probabilities = norm_vals ** steepness
                
            else:
                # Fallback to linear
                self.logger.warning(f"Unknown method: {method}, falling back to linear")
                normalized = (values - min_val) / (max_val - min_val)
                zeros = np.zeros_like(normalized)
                ones = np.ones_like(normalized)
                probabilities = np.maximum(np.minimum(normalized, ones), zeros)
            
            # Normalize probabilities
            return self.normalize_probabilities(probabilities)
            
        except Exception as e:
            self.logger.error(f"Error in indicator_to_probabilities_vectorized: {e}")
            return np.full_like(values, 0.5) if isinstance(values, np.ndarray) else np.array([0.5] * len(values))
    
    @_time_execution
    def aggregate_probabilities(self, 
                              probabilities: List[float],
                              method: Optional[Union[str, AggregationMethod]] = None) -> float:
        """
        Aggregate multiple probability estimates using selected method.
        
        Args:
            probabilities: List of probability estimates
            method: Aggregation method to use
            
        Returns:
            Aggregated probability
        """
        if not probabilities:
            return 0.5
        
        # Filter out invalid values
        valid_probs = [p for p in probabilities if not (np.isnan(p) or np.isinf(p))]
        if not valid_probs:
            return 0.5
        
        # Convert to numpy array for performance
        if self.config.use_numba or self.config.use_vectorization:
            valid_probs = np.array(valid_probs, dtype=np.float64)
        
        # Normalize all probabilities
        if isinstance(valid_probs, np.ndarray):
            normalized_probs = self.normalize_probabilities(valid_probs)
        else:
            normalized_probs = [self.normalize_probability(p) for p in valid_probs]
        
        # Use default method if not specified
        if method is None:
            method = self.config.default_aggregation_method
        elif isinstance(method, str):
            method = AggregationMethod.from_string(method)
        
        try:
            if method == AggregationMethod.LOG_ODDS:
                if self.config.use_numba and isinstance(normalized_probs, np.ndarray):
                    return _aggregate_log_odds_njit(normalized_probs)
                else:
                    # Convert to log-odds, sum, then convert back
                    if isinstance(normalized_probs, np.ndarray):
                        log_odds = self.to_log_odds_vectorized(normalized_probs)
                        avg_log_odds = np.mean(log_odds)
                    else:
                        log_odds_sum = sum(self.to_log_odds(p) for p in normalized_probs)
                        avg_log_odds = log_odds_sum / len(normalized_probs)
                    
                    # Return to probability space
                    return self.from_log_odds(avg_log_odds)
                    
            elif method == AggregationMethod.WEIGHTED:
                # Default to equal weights
                weights = np.ones_like(normalized_probs) if isinstance(normalized_probs, np.ndarray) else [1.0] * len(normalized_probs)
                return self.weighted_aggregate(normalized_probs, weights)
                    
            elif method == AggregationMethod.BAYESIAN:
                # Sequential Bayesian updates
                posterior = 0.5  # Start with uniform prior
                for p in normalized_probs:
                    posterior = self.update_with_evidence(posterior, p)
                return posterior
                    
            elif method == AggregationMethod.GEOMETRIC:
                # Geometric mean preserves certain probabilistic properties
                if isinstance(normalized_probs, np.ndarray):
                    log_product = np.sum(np.log(normalized_probs))
                else:
                    log_product = sum(np.log(p) for p in normalized_probs)
                    
                geom_mean = np.exp(log_product / len(normalized_probs))
                return self.normalize_probability(geom_mean)
                    
            else:
                self.logger.warning(f"Unknown aggregation method: {method}, using LOG_ODDS")
                # Fall back to log-odds method
                if isinstance(normalized_probs, np.ndarray):
                    log_odds = self.to_log_odds_vectorized(normalized_probs)
                    avg_log_odds = np.mean(log_odds)
                else:
                    log_odds_sum = sum(self.to_log_odds(p) for p in normalized_probs)
                    avg_log_odds = log_odds_sum / len(normalized_probs)
                    
                return self.from_log_odds(avg_log_odds)
                    
        except Exception as e:
            self.logger.error(f"Error in aggregate_probabilities: {e}")
            # Fall back to simple average in case of error
            if isinstance(normalized_probs, np.ndarray):
                return float(np.mean(normalized_probs))
            else:
                return sum(normalized_probs) / len(normalized_probs)
    
    @_time_execution
    def weighted_aggregate(self, 
                          probabilities: List[float], 
                          weights: List[float]) -> float:
        """
        Perform weighted aggregation of probabilities.
        
        Weighted aggregation gives more influence to probabilities with higher weights,
        allowing for emphasis on more reliable or important probability estimates.
        
        Args:
            probabilities: List of probability estimates
            weights: List of weights for each probability
            
        Returns:
            Weighted aggregated probability
        """
        if not probabilities or not weights or len(probabilities) != len(weights):
            self.logger.warning("Invalid inputs for weighted aggregation")
            return 0.5
            
        # Filter out invalid values
        valid_pairs = [(p, w) for p, w in zip(probabilities, weights) 
                      if not (np.isnan(p) or np.isinf(p) or np.isnan(w) or np.isinf(w))]
        
        if not valid_pairs:
            return 0.5
        
        # Convert to numpy for vectorized operations if enabled
        if self.config.use_vectorization and isinstance(probabilities, np.ndarray) and isinstance(weights, np.ndarray):
            # Create masks for valid values
            valid_mask = ~(np.isnan(probabilities) | np.isinf(probabilities) | 
                          np.isnan(weights) | np.isinf(weights))
            
            if not np.any(valid_mask):
                return 0.5
                
            valid_probs = probabilities[valid_mask]
            valid_weights = weights[valid_mask]
            
            # Normalize probabilities
            normalized_probs = self.normalize_probabilities(valid_probs)
            
            # Normalize weights
            total_weight = np.sum(valid_weights)
            if total_weight <= 0:
                normalized_weights = np.ones_like(valid_weights) / len(valid_weights)
            else:
                normalized_weights = valid_weights / total_weight
            
            try:
                # Convert to log-odds, apply weights, then convert back
                log_odds = self.to_log_odds_vectorized(normalized_probs)
                weighted_log_odds = np.sum(normalized_weights * log_odds)
                
                return self.from_log_odds(float(weighted_log_odds))  # Cast to float for safety
            except Exception as e:
                self.logger.error(f"Error in vectorized weighted_aggregate: {e}")
                # Fall back to weighted average
                return float(np.sum(normalized_weights * normalized_probs))
        else:
            # Unzip the valid pairs
            valid_probs, valid_weights = zip(*valid_pairs) if valid_pairs else ([], [])
            
            # Normalize all probabilities
            normalized_probs = [self.normalize_probability(p) for p in valid_probs]
            
            # Normalize weights
            total_weight = sum(valid_weights)
            if total_weight <= 0:
                # If all weights are zero or negative, use equal weights
                normalized_weights = [1.0 / len(valid_weights)] * len(valid_weights)
            else:
                normalized_weights = [w / total_weight for w in valid_weights]
            
            try:
                # Convert to log-odds, apply weights, then convert back
                weighted_log_odds = sum(w * self.to_log_odds(p) 
                                      for w, p in zip(normalized_weights, normalized_probs))
                
                return self.from_log_odds(weighted_log_odds)
            except Exception as e:
                self.logger.error(f"Error in weighted_aggregate: {e}")
                # Fall back to weighted average in case of error
                return sum(w * p for w, p in zip(normalized_weights, normalized_probs))
    
    @_time_execution
    def update_with_evidence(self, 
                           prior_probability: float, 
                           evidence_probability: float) -> float:
        """
        Update a prior probability with new evidence using Bayes' rule.
        
        This implements P(H|E) = P(H) * P(E|H) / P(E), where:
        - P(H) is the prior probability of hypothesis H
        - P(E|H) is the probability of evidence E given H is true
        - P(H|E) is the posterior probability of H given evidence E
        
        Args:
            prior_probability: Initial probability estimate (P(H))
            evidence_probability: New evidence as a probability estimate (P(E|H))
            
        Returns:
            Updated probability
        """
        with self._lock:
            try:
                # Normalize inputs
                prior = self.normalize_probability(prior_probability)
                evidence = self.normalize_probability(evidence_probability)
                
                # Calculate posterior using Bayes' rule (odds form)
                prior_odds = prior / (1 - prior)
                evidence_odds = evidence / (1 - evidence)
                
                posterior_odds = prior_odds * evidence_odds
                
                # Convert back to probability
                posterior = posterior_odds / (1 + posterior_odds)
                
                # Record transformation
                with self._history_lock:
                    if len(self._history) < 1000:  # Limit history size
                        self._history.append({
                            'type': 'bayesian_update',
                            'prior': prior,
                            'evidence': evidence,
                            'posterior': posterior,
                            'timestamp': time.time()
                        })
                
                return self.normalize_probability(posterior)
                
            except Exception as e:
                self.logger.error(f"Error in update_with_evidence: {e}")
                # In case of error, slightly bias toward evidence
                return 0.7 * prior_probability + 0.3 * evidence_probability
    
    @_time_execution
    def cost_function(self, quantities: List[float]) -> float:
        """
        LMSR cost function C(q) = b * log(sum(exp(q_i/b)))
        
        The cost function determines how much it costs to move the market
        from state q to q'. The liquidity parameter b controls how quickly
        prices change with trading volume.
        
        Args:
            quantities: Vector of quantities for each outcome
            
        Returns:
            Cost according to LMSR
        """
        if not quantities:
            return 0.0
            
        try:
            # Convert to numpy array for numba
            if self.config.use_numba:
                quantities_arr = np.array(quantities, dtype=np.float64)
                return _cost_function_njit(quantities_arr, self.config.liquidity_parameter)
            else:
                # The original LMSR cost function
                b = self.config.liquidity_parameter
                
                # Use a numerically stable implementation
                max_q = max(quantities)
                
                # For numerical stability:
                # log(sum(exp(q_i))) = max_q + log(sum(exp(q_i - max_q)))
                exp_sum = sum(np.exp((q - max_q) / b) for q in quantities)
                cost = b * (max_q / b + np.log(exp_sum))
                
                return cost
                
        except Exception as e:
            self.logger.error(f"Error calculating cost function: {e}")
            return 0.0
    
    @_time_execution
    def get_market_probability(self, quantities: List[float], index: int) -> float:
        """
        Derive implicit probability from current market state.
        
        The LMSR market maker's current prices represent the market's
        expectation of outcome probabilities.
        
        Args:
            quantities: Vector of quantities for each outcome
            index: Index of the outcome to get probability for
            
        Returns:
            Market probability for the specified outcome
        """
        if not quantities or index < 0 or index >= len(quantities):
            self.logger.error(f"Invalid inputs: quantities={quantities}, index={index}")
            return 0.5
            
        try:
            if self.config.use_numba:
                # Use the numba accelerated function for all probabilities
                quantities_arr = np.array(quantities, dtype=np.float64)
                all_probs = _calc_market_probabilities_njit(
                    quantities_arr, 
                    self.config.liquidity_parameter
                )
                return all_probs[index]
            else:
                b = self.config.liquidity_parameter
                
                # Calculate numerator
                exp_term = np.exp(quantities[index] / b)
                
                # Calculate denominator (sum of exponentials)
                exp_sum = sum(np.exp(q / b) for q in quantities)
                
                # Calculate probability
                probability = exp_term / exp_sum
                
                return self.normalize_probability(probability)
                
        except Exception as e:
            self.logger.error(f"Error calculating market probability: {e}")
            return 0.5
    
    @_time_execution
    def get_all_market_probabilities(self, quantities: List[float]) -> List[float]:
        """
        Calculate probabilities for all outcomes based on current market state.
        
        Args:
            quantities: Vector of quantities for each outcome
            
        Returns:
            List of probabilities for all outcomes
        """
        if not quantities:
            return []
            
        try:
            # Use numba for large markets
            if self.config.use_numba:
                quantities_arr = np.array(quantities, dtype=np.float64)
                probabilities = _calc_market_probabilities_njit(
                    quantities_arr, 
                    self.config.liquidity_parameter
                )
                return probabilities.tolist()
            else:
                # Use parallel processing for large markets
                if len(quantities) > 10 and self.config.enable_parallel:
                    with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                        probabilities = list(executor.map(
                            lambda i: self.get_market_probability(quantities, i),
                            range(len(quantities))
                        ))
                else:
                    probabilities = [
                        self.get_market_probability(quantities, i)
                        for i in range(len(quantities))
                    ]
                    
                # Ensure probabilities sum to 1.0
                total = sum(probabilities)
                if total > 0:
                    return [p / total for p in probabilities]
                else:
                    # Uniform distribution if all zero
                    return [1.0 / len(quantities)] * len(quantities)
                    
        except Exception as e:
            self.logger.error(f"Error calculating all market probabilities: {e}")
            return [1.0 / len(quantities)] * len(quantities)
    
    @_time_execution
    def calculate_cost_to_move(self, 
                             current_quantities: List[float],
                             target_probability: float,
                             outcome_index: int) -> float:
        """
        Calculate the cost to move the market probability to a target.
        
        Args:
            current_quantities: Current market state
            target_probability: Desired probability for the specified outcome
            outcome_index: Index of the outcome to move
            
        Returns:
            Cost to move the market to the target probability
        """
        if not current_quantities or outcome_index < 0 or outcome_index >= len(current_quantities):
            self.logger.error("Invalid inputs")
            return float('inf')
            
        try:
            # Current market probability
            current_prob = self.get_market_probability(current_quantities, outcome_index)
            
            if abs(current_prob - target_probability) < 0.001:
                return 0.0  # Already at target
                
            # Binary search to find the quantity that gives the target probability
            b = self.config.liquidity_parameter
            
            # Calculate required price movement
            log_odds_diff = self.to_log_odds(target_probability) - self.to_log_odds(current_prob)
            
            # Approximate the quantity change needed
            quantity_change = b * log_odds_diff
            
            # Create new quantities array
            new_quantities = current_quantities.copy()
            new_quantities[outcome_index] += quantity_change
            
            # Calculate cost difference
            old_cost = self.cost_function(current_quantities)
            new_cost = self.cost_function(new_quantities)
            
            return new_cost - old_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating cost to move: {e}")
            return float('inf')
    
    @_time_execution
    def combine_conditional_probabilities(self,
                                        joint_probs: Dict[Tuple[Any, Any], float]) -> Dict[Any, float]:
        """
        Combine conditional probabilities from a joint probability table.
        
        This method handles the combination of probabilities that have 
        conditional dependencies, maintaining proper probabilistic semantics.
        
        Args:
            joint_probs: Dictionary mapping (condition, outcome) tuples to probabilities
            
        Returns:
            Dictionary of marginal probabilities for each outcome
        """
        if not joint_probs:
            return {}
            
        try:
            # Extract all unique conditions and outcomes
            conditions = set()
            outcomes = set()
            
            for (condition, outcome) in joint_probs:
                conditions.add(condition)
                outcomes.add(outcome)
                
            # Calculate marginal probabilities for conditions
            condition_probs = {}
            for condition in conditions:
                # Sum probabilities for this condition across all outcomes
                total_prob = sum(
                    joint_probs.get((condition, outcome), 0)
                    for outcome in outcomes
                )
                condition_probs[condition] = total_prob
                
            # Calculate marginal probabilities for outcomes
            outcome_probs = {}
            for outcome in outcomes:
                # Sum probabilities for this outcome across all conditions
                total_prob = sum(
                    joint_probs.get((condition, outcome), 0)
                    for condition in conditions
                )
                outcome_probs[outcome] = total_prob
                
            # Normalize to ensure sum to 1.0
            total = sum(outcome_probs.values())
            if total > 0:
                outcome_probs = {k: v / total for k, v in outcome_probs.items()}
                
            return outcome_probs
            
        except Exception as e:
            self.logger.error(f"Error combining conditional probabilities: {e}")
            return {}
    
    @_time_execution
    def calculate_information_gain(self, 
                                 prior_probabilities: List[float],
                                 posterior_probabilities: List[float]) -> float:
        """
        Calculate Kullback-Leibler divergence (information gain) between distributions.
        
        KL divergence measures how much information is gained by updating from the
        prior to the posterior distribution.
        
        Args:
            prior_probabilities: Initial probability distribution
            posterior_probabilities: Updated probability distribution
            
        Returns:
            KL divergence (information gain) in bits
        """
        if not prior_probabilities or not posterior_probabilities or len(prior_probabilities) != len(posterior_probabilities):
            self.logger.error("Invalid inputs")
            return 0.0
            
        try:
            # Convert to numpy for vectorized operations
            if self.config.use_vectorization:
                priors = np.array(prior_probabilities, dtype=np.float64)
                posteriors = np.array(posterior_probabilities, dtype=np.float64)
                
                # Ensure probabilities are normalized
                prior_sum = np.sum(priors)
                posterior_sum = np.sum(posteriors)
                
                if prior_sum <= 0 or posterior_sum <= 0:
                    return 0.0
                    
                normalized_prior = priors / prior_sum
                normalized_posterior = posteriors / posterior_sum
                
                # Create mask for valid values (avoid log(0)) - use small epsilon
                epsilon = 1e-10
                valid_mask = (normalized_posterior > epsilon) & (normalized_prior > epsilon)
                
                if not np.any(valid_mask):
                    return 0.0
                
                # Extract valid values to avoid in-place operations on masked arrays
                valid_post = normalized_posterior[valid_mask]
                valid_prior = normalized_prior[valid_mask]
                    
                # Calculate KL divergence only for valid values
                ratio = valid_post / valid_prior
                log_ratio = np.log2(ratio)
                kl_terms = valid_post * log_ratio
                
                return float(np.sum(kl_terms))
            else:
                # Ensure probabilities are normalized
                prior_sum = sum(prior_probabilities)
                posterior_sum = sum(posterior_probabilities)
                
                if prior_sum <= 0 or posterior_sum <= 0:
                    return 0.0
                    
                normalized_prior = [p / prior_sum for p in prior_probabilities]
                normalized_posterior = [p / posterior_sum for p in posterior_probabilities]
                
                # Calculate KL divergence with small epsilon to avoid log(0)
                epsilon = 1e-10
                kl_divergence = sum(
                    p * np.log2(p / max(q, epsilon)) if p > epsilon else 0
                    for p, q in zip(normalized_posterior, normalized_prior)
                )
                
                return kl_divergence
                
        except Exception as e:
            self.logger.error(f"Error calculating information gain: {e}")
            return 0.0
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of probability transformations for analysis.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of history entries
        """
        with self._history_lock:
            return self._history[-limit:] if self._history else []
    
    @_time_execution
    def batch_process_indicators(self,
                               indicators: Dict[str, List[float]],
                               indicator_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Batch process multiple indicators to probabilities.
        
        Args:
            indicators: Dictionary mapping indicator names to value lists
            indicator_configs: Dictionary mapping indicator names to conversion configs
            
        Returns:
            Dictionary mapping indicator names to probability lists
        """
        if not indicators or not indicator_configs:
            return {}
            
        result = {}
        
        # Process in parallel with efficient chunking
        if self.config.enable_parallel and len(indicators) > 1:
            # Determine if we should use ThreadPoolExecutor or process indicators in chunks
            use_threading = len(indicators) > 5
            
            if use_threading:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {}
                    
                    for name, values in indicators.items():
                        if name in indicator_configs:
                            config = indicator_configs[name]
                            futures[executor.submit(self._process_indicator_optimized, values, config)] = name
                    
                    for future in futures:
                        name = futures[future]
                        try:
                            result[name] = future.result()
                        except Exception as e:
                            self.logger.error(f"Error processing indicator {name}: {e}")
                            result[name] = [0.5] * len(indicators[name])
            else:
                # Process indicators sequentially but vectorize the processing
                for name, values in indicators.items():
                    if name in indicator_configs:
                        config = indicator_configs[name]
                        try:
                            result[name] = self._process_indicator_optimized(values, config)
                        except Exception as e:
                            self.logger.error(f"Error processing indicator {name}: {e}")
                            result[name] = [0.5] * len(values)
        else:
            # Sequential processing with vectorization
            for name, values in indicators.items():
                if name in indicator_configs:
                    config = indicator_configs[name]
                    try:
                        result[name] = self._process_indicator_optimized(values, config)
                    except Exception as e:
                        self.logger.error(f"Error processing indicator {name}: {e}")
                        result[name] = [0.5] * len(values)
        
        return result
    
    def _process_indicator_optimized(self, values: List[float], config: Dict[str, Any]) -> List[float]:
        """Optimized helper method to process a single indicator's values to probabilities."""
        try:
            # Convert to numpy array for vectorization
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=np.float64)
                
            min_val = config.get('min_val', float(np.min(values)) if len(values) > 0 else 0)
            max_val = config.get('max_val', float(np.max(values)) if len(values) > 0 else 1)
            center = config.get('center', (min_val + max_val) / 2)
            steepness = config.get('steepness', 1.0)
            method = config.get('method', self.config.default_conversion_method)
            
            # Process in chunks for very large arrays to avoid memory issues
            if len(values) > self.config.batch_size:
                result = np.empty_like(values)
                for i in range(0, len(values), self.config.batch_size):
                    chunk = values[i:i + self.config.batch_size]
                    result[i:i + self.config.batch_size] = self.indicator_to_probabilities_vectorized(
                        chunk, min_val, max_val, center, steepness, method
                    )
                return result.tolist()
            else:
                # Process the entire array at once
                return self.indicator_to_probabilities_vectorized(
                    values, min_val, max_val, center, steepness, method
                ).tolist()
        except Exception as e:
            self.logger.error(f"Error in _process_indicator_optimized: {e}")
            return [0.5] * (len(values) if hasattr(values, '__len__') else 1)
