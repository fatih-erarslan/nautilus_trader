#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Resolution Analysis (MRA) for CDFA Extensions

Provides advanced multi-resolution analysis capabilities for financial data:
- Decomposition of time series into multiple resolution levels
- Trend-cycle-noise separation using wavelet-based methods
- Regime detection based on energy distribution across scales
- Cross-scale interactions analysis
- Structural change point detection
- Non-stationary correlation analysis

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import os
from datetime import datetime, timedelta
import uuid
import math
from collections import deque

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator
from .wavelet_processor import WaveletProcessor, WaveletDecompResult, WaveletFamily

# ---- Optional dependencies with graceful fallbacks ----

# PyWavelets for wavelet transform
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. MRA capabilities will be limited.")

# PyTorch for accelerated computation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Accelerated MRA will be limited.")

# SciPy for signal processing and statistics
try:
    from scipy import signal
    from scipy.stats import kurtosis, skew
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced MRA capabilities will be limited.")

# Numba for JIT acceleration
try:
    import numba as nb
    from numba import njit, prange, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. JIT acceleration will be limited.")
    
    # Define dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range
    float64 = int64 = lambda x: x

# Statsmodels for advanced statistical tests
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Some statistical tests will be limited.")

class MRAMode(Enum):
    """Multi-resolution analysis modes."""
    ADDITIVE = auto()      # Additive decomposition (y = trend + cycle + noise)
    MULTIPLICATIVE = auto() # Multiplicative decomposition (y = trend * cycle * noise)
    LOG_ADDITIVE = auto()  # Log-additive (log(y) = log(trend) + log(cycle) + log(noise))
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'MRAMode':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown MRAMode: {s}")

class ChangePointMethod(Enum):
    """Methods for change point detection."""
    ENERGY = auto()       # Energy distribution change
    WAVELET_VARIANCE = auto() # Wavelet variance change
    DETAIL_KURTOSIS = auto() # Kurtosis of detail coefficients
    MULTISCALE = auto()    # Combined multiscale approach
    BAYESIAN = auto()     # Bayesian change point detection
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'ChangePointMethod':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown ChangePointMethod: {s}")

@dataclass
class MRADecomposition:
    """Result of MRA decomposition."""
    original: np.ndarray        # Original signal
    trend: np.ndarray           # Trend component
    cycle: np.ndarray           # Cyclical component
    noise: np.ndarray           # Noise component
    details: Dict[int, np.ndarray] # Detail components by level
    approximation: np.ndarray   # Approximation component
    trend_level: int            # Level used for trend
    cycle_levels: List[int]     # Levels used for cycle
    noise_levels: List[int]     # Levels used for noise
    mode: MRAMode              # Decomposition mode
    wavelet: str               # Wavelet used
    energy_distribution: Dict[str, float] = field(default_factory=dict) # Energy distribution
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeAnalysisResult:
    """Result of regime analysis."""
    regimes: List[str]           # Identified regimes
    change_points: List[int]     # Change point indices
    probabilities: Dict[str, List[float]]  # Regime probabilities
    confidence: List[float]      # Confidence in regime assignments
    measures: Dict[str, List[float]] = field(default_factory=dict) # Regime measures
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossScaleResult:
    """Result of cross-scale interaction analysis."""
    interactions: Dict[Tuple[int, int], float]  # Interactions between scales
    information_flow: Dict[Tuple[int, int], float]  # Directed information flow
    causality: Dict[Tuple[int, int], float]     # Granger causality p-values
    synchronization: Dict[int, float]           # Synchronization by scale
    phase_locking: Dict[Tuple[int, int], float] # Phase locking between scales
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiResolutionAnalyzer:
    """
    Advanced Multi-Resolution Analysis (MRA) for financial time series.
    
    Provides tools for decomposing time series into components at different 
    resolution levels, analyzing cross-scale interactions, detecting regime
    changes, and separating trend, cycle, and noise components.
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                wavelet_processor: Optional[WaveletProcessor] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the MRA analyzer.
        
        Args:
            hw_accelerator: Optional hardware accelerator
            wavelet_processor: Optional wavelet processor (creates one if None)
            config: Configuration parameters
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Initialize hardware accelerator
        self.hw_accelerator = hw_accelerator if hw_accelerator is not None else HardwareAccelerator()
        
        # Initialize wavelet processor
        if wavelet_processor is None:
            self.logger.info("Creating internal WaveletProcessor")
            self.wavelet_processor = WaveletProcessor(
                hw_accelerator=self.hw_accelerator,
                log_level=log_level
            )
        else:
            self.wavelet_processor = wavelet_processor
            
        # Default configuration
        self.default_config = {
            # Decomposition parameters
            "default_wavelet": "sym5",
            "default_mode": MRAMode.ADDITIVE,
            "trend_level": 5,
            "cycle_levels": [2, 3, 4],
            "noise_levels": [1],
            "boundary_treatment": "reflection",
            "adaptive_level_selection": True,
            
            # Analysis parameters
            "energy_threshold": 0.95,  # Cumulative energy threshold
            "smoothness_measure": "kurtosis",  # kurtosis or variance
            "synchronization_method": "phase",  # phase or amplitude
            "causality_lag": 5,  # Maximum lag for Granger causality
            "significance_level": 0.05,  # Statistical significance level
            
            # Regime detection
            "regime_change_threshold": 0.3,  # Threshold for regime change detection
            "min_regime_duration": 10,  # Minimum number of points in a regime
            "regime_types": ["trend", "cycle", "noise", "mixed"],
            "change_point_method": ChangePointMethod.MULTISCALE,
            "window_size": 50,  # Window size for rolling analysis
            "step_size": 10,    # Step size for rolling analysis
            
            # Cross-scale analysis
            "cross_scale_measure": "correlation",  # correlation, mutual_info, or transfer_entropy
            "phase_extraction": "hilbert",  # hilbert or wavelet
            "direction_measure": "transfer_entropy",  # granger or transfer_entropy
            
            # Performance parameters
            "parallel_computation": True,
            "cache_results": True,
            "cache_ttl": 3600,  # 1 hour
            "adaptive_computation": True,  # Adjust computation based on data size
            "use_numba": True,
            "use_torch": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}

        # Logger setup
        self.logger = logging.getLogger("AdvancedMRA")
        self.logger.setLevel(getattr(logging, getattr(self.config, 'log_level', 'INFO'), logging.INFO))
        if getattr(self.config, 'verbose', False):
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                force=True
            )        
        
        # Validate wavelet family
        wavelet_family = getattr(self.config, 'wavelet_family', 'db4')
        if wavelet_family not in pywt.wavelist():
            self.logger.warning(f"Wavelet family {wavelet_family} not found. Falling back to db4.")
            setattr(self.config, 'wavelet_family', 'db4')
            
            
        # Initialize state
        self._lock = threading.RLock()
        self._decomp_cache = {}  # key -> (decomp, timestamp)
        self._regime_cache = {}  # key -> (result, timestamp)
        self._cross_scale_cache = {}  # key -> (result, timestamp)
        
        # Check available backends
        self.has_pywavelets = PYWAVELETS_AVAILABLE
        self.has_torch = TORCH_AVAILABLE
        self.has_scipy = SCIPY_AVAILABLE
        self.has_numba = NUMBA_AVAILABLE
        self.has_statsmodels = STATSMODELS_AVAILABLE
        

        
        # Initialize regime classification thresholds
        self._init_regime_thresholds()
        
        # Initialize state storage
        self._last_decomposition = None
        self._regime_history = []
        self._cycle_periods = []
        # Initialize GPU support if available
        self._init_gpu_support()
        
        # Initialize multiprocessing pool if needed
        # Set up multiprocessing if enabled
        if getattr(self.config, 'use_multiprocessing', False):
            self._init_multiprocessing()
            
        # Initialize caches
        self._decomposition_cache = {}
        self._coefficient_cache = {}
        
        # Initialize thresholding parameters
        self.thresholds = {}
        
        self.logger.info(f"AdvancedMRA initialized with wavelet: {getattr(self.config, 'wavelet_type', 'db4')}")

        if not self.has_pywavelets:
            self.logger.warning("PyWavelets not available. Most features will be disabled.")
        else:
            self.logger.info("MRA analyzer initialized")
            
            if self.has_torch and self.config["use_torch"]:
                self.logger.info("Using PyTorch acceleration for MRA")
                
            if self.has_numba and self.config["use_numba"]:
                self.logger.info("Using Numba JIT acceleration for MRA")

    def _init_gpu_support(self):
        """Initialize GPU support if available."""
        self.gpu_available = False
        if getattr(self.config, 'use_gpu', False):
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_available = True
                self.logger.info("GPU acceleration enabled")
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU")
    
    def _init_multiprocessing(self):
        """Initialize multiprocessing pool."""
        try:
            from multiprocessing import Pool, cpu_count
            num_workers = min(getattr(self.config, 'num_workers', 4), cpu_count())
            self._pool = Pool(processes=num_workers)
            self.logger.info(f"Initialized multiprocessing pool with {num_workers} workers")
        except Exception as e:
            self.logger.warning(f"Failed to initialize multiprocessing: {e}")
            self._pool = None
    
    def _init_regime_thresholds(self):
        """Initialize regime classification thresholds."""
        self.regime_thresholds = {
            # Thresholds for Growth regime
            "growth": {
                "trend_strength": 0.6,      # High trend strength
                "volatility_ratio": 1.5,    # Higher volatility in approximation than details
                "energy_ratio": 0.65        # Higher energy in approximation
            },
            # Thresholds for Conservation regime
            "conservation": {
                "oscillation_strength": 0.6, # Strong oscillatory behavior
                "energy_upper_limit": 0.4,   # Lower energy in approximation
                "volatility_deviation": 0.7  # More volatility in details
            },
            # Thresholds for Transition regime
            "transition": {
                "stability_score": 0.4,     # Lower stability
                "entropy_threshold": 0.7,   # Higher entropy
                "regime_change_lag": 5      # Number of periods for regime change momentum
            }
        }
                
    def _get_cached_result(self, cache_dict: Dict[Any, Tuple[Any, float]], key: Any) -> Optional[Any]:
        """
        Get cached result if valid.
        
        Args:
            cache_dict: Cache dictionary
            key: Cache key
            
        Returns:
            Cached result or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        with self._lock:
            # Check if result is in cache
            cache_entry = cache_dict.get(key)
            
            if cache_entry is None:
                return None
                
            result, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                cache_dict.pop(key, None)
                return None
                
            return result
            
    def _cache_result(self, cache_dict: Dict[Any, Tuple[Any, float]], key: Any, result: Any):
        """
        Cache result for future use.
        
        Args:
            cache_dict: Cache dictionary
            key: Cache key
            result: Result to cache
        """
        if not self.config["cache_results"]:
            return
            
        with self._lock:
            cache_dict[key] = (result, time.time())
    
    # ----- MRA Decomposition Methods -----
    
    def decompose(self, data: np.ndarray, wavelet: Optional[str] = None,
                mode: Optional[Union[str, MRAMode]] = None,
                trend_level: Optional[int] = None,
                cycle_levels: Optional[List[int]] = None,
                noise_levels: Optional[List[int]] = None) -> MRADecomposition:
        """
        Decompose time series using multi-resolution analysis.
        
        Args:
            data: Input time series
            wavelet: Wavelet to use (default from config)
            mode: Decomposition mode (default from config)
            trend_level: Level for trend component (default from config)
            cycle_levels: Levels for cycle component (default from config)
            noise_levels: Levels for noise component (default from config)
            
        Returns:
            MRA decomposition result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot perform MRA decomposition")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        if mode is None:
            mode = self.config["default_mode"]
        elif isinstance(mode, str):
            mode = MRAMode.from_string(mode)
            
        trend_level = trend_level or self.config["trend_level"]
        cycle_levels = cycle_levels or self.config["cycle_levels"]
        noise_levels = noise_levels or self.config["noise_levels"]
        
        # Handle data preprocessing based on mode
        if mode == MRAMode.MULTIPLICATIVE:
            # Take log for multiplicative decomposition
            processed_data = np.log(data)
            mode_for_decomp = MRAMode.ADDITIVE
        elif mode == MRAMode.LOG_ADDITIVE:
            # Take log directly
            processed_data = np.log(data)
            mode_for_decomp = MRAMode.ADDITIVE
        else:
            # Additive decomposition (no preprocessing)
            processed_data = data.copy()
            mode_for_decomp = mode
            
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, str(mode), 
                   trend_level, tuple(cycle_levels), tuple(noise_levels))
        
        # Check cache
        cached_result = self._get_cached_result(self._decomp_cache, cache_key)
        if cached_result is not None:
            return cached_result
            
        # Adaptive level selection if enabled
        if self.config["adaptive_level_selection"]:
            max_level = pywt.dwt_max_level(len(data), wavelet)
            
            if trend_level > max_level:
                self.logger.warning(f"Trend level {trend_level} exceeds maximum {max_level}, adjusting")
                trend_level = max_level
                
            # Ensure cycle and noise levels are valid
            cycle_levels = [level for level in cycle_levels if level <= max_level and level > 0]
            noise_levels = [level for level in noise_levels if level <= max_level and level > 0]
            
            if not cycle_levels:
                self.logger.warning("No valid cycle levels, using defaults")
                cycle_levels = [level for level in self.config["cycle_levels"] 
                             if level <= max_level and level > 0]
                
            if not noise_levels:
                self.logger.warning("No valid noise levels, using defaults")
                noise_levels = [level for level in self.config["noise_levels"] 
                             if level <= max_level and level > 0]
                
        # Perform wavelet decomposition
        decomp = self.wavelet_processor.decompose_signal(
            processed_data, wavelet=wavelet, level=max(trend_level, max(cycle_levels + noise_levels))
        )
        
        if decomp is None:
            self.logger.error("Failed to decompose signal")
            return None
            
        # Get all detail levels and approximation
        details = {}
        for i, detail in enumerate(decomp.details):
            level = i + 1  # Detail levels start at 1
            details[level] = detail
            
        approximation = decomp.approximation
        
        # Create trend component (approximation at trend_level)
        trend_coeffs = [approximation if level == 0 else np.zeros_like(details[level]) 
                      for level in range(trend_level + 1)]
        trend = pywt.waverec(trend_coeffs, wavelet)[:len(data)]
        
        # Create cycle component (selected detail levels)
        cycle_coeffs = []
        for level in range(trend_level + 1):
            if level == 0:
                # Approximation coefficient
                cycle_coeffs.append(np.zeros_like(approximation))
            elif level in cycle_levels:
                # Include this level in cycle
                cycle_coeffs.append(details[level])
            else:
                # Zero out other levels
                cycle_coeffs.append(np.zeros_like(details[level]))
                
        cycle = pywt.waverec(cycle_coeffs, wavelet)[:len(data)]
        
        # Create noise component (selected detail levels)
        noise_coeffs = []
        for level in range(trend_level + 1):
            if level == 0:
                # Approximation coefficient
                noise_coeffs.append(np.zeros_like(approximation))
            elif level in noise_levels:
                # Include this level in noise
                noise_coeffs.append(details[level])
            else:
                # Zero out other levels
                noise_coeffs.append(np.zeros_like(details[level]))
                
        noise = pywt.waverec(noise_coeffs, wavelet)[:len(data)]
        
        # Reconstruct all detail levels individually
        reconstructed_details = {}
        for level in range(1, trend_level + 1):
            level_coeffs = []
            for i in range(trend_level + 1):
                if i == 0:
                    # Approximation coefficient
                    level_coeffs.append(np.zeros_like(approximation))
                elif i == level:
                    # Include this level
                    level_coeffs.append(details[i])
                else:
                    # Zero out other levels
                    level_coeffs.append(np.zeros_like(details[i]))
                    
            reconstructed_details[level] = pywt.waverec(level_coeffs, wavelet)[:len(data)]
            
        # Reconstruct approximation
        approx_coeffs = []
        for level in range(trend_level + 1):
            if level == 0:
                # Approximation coefficient
                approx_coeffs.append(approximation)
            else:
                # Zero out details
                approx_coeffs.append(np.zeros_like(details[level]))
                
        reconstructed_approx = pywt.waverec(approx_coeffs, wavelet)[:len(data)]
        
        # Handle inverse transformation for different modes
        if mode == MRAMode.MULTIPLICATIVE:
            # Exponentiate components
            trend = np.exp(trend)
            cycle = np.exp(cycle)
            noise = np.exp(noise)
            reconstructed_approx = np.exp(reconstructed_approx)
            
            for level in reconstructed_details:
                reconstructed_details[level] = np.exp(reconstructed_details[level])
                
        elif mode == MRAMode.LOG_ADDITIVE:
            # No need to transform back for log-additive
            pass
            
        # Calculate energy distribution
        energy_distribution = {}
        
        # Energy of approximation
        energy_approx = np.sum(approximation**2)
        energy_distribution["approximation"] = energy_approx
        
        # Energy of details by level
        for level, detail in details.items():
            energy_distribution[f"detail_{level}"] = np.sum(detail**2)
            
        # Total energy
        total_energy = energy_approx + sum(energy_distribution[f"detail_{level}"] for level in details.keys())
        
        # Convert to relative energy
        if total_energy > 0:
            relative_energy = {
                k: v / total_energy for k, v in energy_distribution.items()
            }
        else:
            relative_energy = {
                k: 0.0 for k in energy_distribution
            }
            
        # Create result
        result = MRADecomposition(
            original=data,
            trend=trend,
            cycle=cycle,
            noise=noise,
            details=reconstructed_details,
            approximation=reconstructed_approx,
            trend_level=trend_level,
            cycle_levels=cycle_levels,
            noise_levels=noise_levels,
            mode=mode,
            wavelet=wavelet,
            energy_distribution=relative_energy,
            metadata={
                "data_length": len(data),
                "timestamp": time.time(),
                "max_level": max(trend_level, max(cycle_levels + noise_levels))
            }
        )
        
        # Cache result
        self._cache_result(self._decomp_cache, cache_key, result)
        
        return result
    
    def adaptive_decompose(self, data: np.ndarray, wavelet: Optional[str] = None,
                         mode: Optional[Union[str, MRAMode]] = None,
                         num_levels: Optional[int] = None) -> MRADecomposition:
        """
        Perform adaptive MRA decomposition with automatic level selection.
        
        Args:
            data: Input time series
            wavelet: Wavelet to use (default from config)
            mode: Decomposition mode (default from config)
            num_levels: Number of levels to use (default: auto)
            
        Returns:
            MRA decomposition result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot perform adaptive decomposition")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        if mode is None:
            mode = self.config["default_mode"]
        elif isinstance(mode, str):
            mode = MRAMode.from_string(mode)
            
        # Determine maximum possible level
        max_possible = pywt.dwt_max_level(len(data), wavelet)
        
        if num_levels is None:
            # Auto determine number of levels based on data length
            # Use log2(N) as a heuristic (maximum sensible number of levels)
            log2_n = int(np.log2(len(data)))
            num_levels = min(max_possible, log2_n - 2)  # Leave some margin
            
        # Ensure we don't exceed maximum
        num_levels = min(num_levels, max_possible)
        
        # Perform full decomposition to analyze energy distribution
        full_decomp = self.wavelet_processor.decompose_signal(
            data, wavelet=wavelet, level=num_levels
        )
        
        if full_decomp is None:
            self.logger.error("Failed to decompose signal for adaptive analysis")
            return None
            
        # Calculate energy of each level
        energy_approx = np.sum(full_decomp.approximation**2)
        energy_details = {
            i+1: np.sum(detail**2) for i, detail in enumerate(full_decomp.details)
        }
        
        # Total energy
        total_energy = energy_approx + sum(energy_details.values())
        
        # Convert to relative energy
        if total_energy > 0:
            rel_energy_approx = energy_approx / total_energy
            rel_energy_details = {
                level: energy / total_energy for level, energy in energy_details.items()
            }
        else:
            rel_energy_approx = 0.0
            rel_energy_details = {
                level: 0.0 for level in energy_details.keys()
            }
            
        # Determine trend level based on approximation energy
        # If approximation has significant energy, it captures the trend
        if rel_energy_approx > 0.1:  # Arbitrary threshold
            trend_level = num_levels
        else:
            # Find the highest level that has significant energy
            for level in range(num_levels, 0, -1):
                if level in rel_energy_details and rel_energy_details[level] > 0.05:
                    trend_level = level
                    break
            else:
                # Default if no significant level found
                trend_level = num_levels // 2
                
        # Determine cycle levels based on spectral characteristics
        # Look for peaks in energy distribution
        energy_values = [(level, energy) for level, energy in rel_energy_details.items()]
        energy_values.sort(key=lambda x: x[1], reverse=True)
        
        # Take top energy levels as cycle (excluding highest which may be trend)
        cycle_levels = [level for level, energy in energy_values 
                     if level < trend_level and energy > 0.05][:3]
        
        # Make sure we have at least one cycle level
        if not cycle_levels and num_levels > 1:
            # Default to middle levels
            mid_level = num_levels // 2
            cycle_levels = [max(1, mid_level - 1), mid_level]
            
        # Noise levels are typically the highest frequency (lowest scale) components
        noise_levels = [level for level in range(1, min(3, num_levels) + 1) 
                     if level not in cycle_levels and level < trend_level]
        
        # Ensure we have at least one noise level
        if not noise_levels and num_levels > 0:
            noise_levels = [1]  # Highest frequency detail
            
        # Now perform the actual decomposition with the determined levels
        return self.decompose(
            data, 
            wavelet=wavelet,
            mode=mode,
            trend_level=trend_level,
            cycle_levels=cycle_levels,
            noise_levels=noise_levels
        )
    
    def rolling_decompose(self, data: np.ndarray, window_size: Optional[int] = None,
                       step_size: Optional[int] = None, wavelet: Optional[str] = None,
                       mode: Optional[Union[str, MRAMode]] = None,
                       adaptive: bool = True) -> Dict[int, MRADecomposition]:
        """
        Perform rolling MRA decomposition over a time series.
        
        Args:
            data: Input time series
            window_size: Window size for rolling analysis (default from config)
            step_size: Step size for rolling analysis (default from config)
            wavelet: Wavelet to use (default from config)
            mode: Decomposition mode (default from config)
            adaptive: Whether to use adaptive decomposition
            
        Returns:
            Dictionary of position to MRA decomposition
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot perform rolling decomposition")
            return {}
            
        # Get defaults from config if not provided
        window_size = window_size or self.config["window_size"]
        step_size = step_size or self.config["step_size"]
        wavelet = wavelet or self.config["default_wavelet"]
        
        if mode is None:
            mode = self.config["default_mode"]
        elif isinstance(mode, str):
            mode = MRAMode.from_string(mode)
            
        # Ensure window size is valid
        if window_size > len(data):
            self.logger.warning(f"Window size {window_size} exceeds data length {len(data)}, reducing")
            window_size = len(data)
            
        # Calculate number of windows
        n_windows = 1 + (len(data) - window_size) // step_size
        
        # Initialize result dictionary
        results = {}
        
        # Process each window
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # Extract window data
            window_data = data[start_idx:end_idx]
            
            # Perform decomposition
            try:
                if adaptive:
                    decomp = self.adaptive_decompose(window_data, wavelet=wavelet, mode=mode)
                else:
                    decomp = self.decompose(window_data, wavelet=wavelet, mode=mode)
                    
                if decomp is not None:
                    # Store with position as key
                    results[start_idx] = decomp
                    
            except Exception as e:
                self.logger.error(f"Error decomposing window at position {start_idx}: {e}")
                
        return results
    
    # ----- Regime Analysis Methods -----
    
    def analyze_regimes(self, data: np.ndarray, wavelet: Optional[str] = None,
                     method: Optional[Union[str, ChangePointMethod]] = None,
                     window_size: Optional[int] = None) -> RegimeAnalysisResult:
        """
        Analyze regimes in time series using MRA.
        
        Args:
            data: Input time series
            wavelet: Wavelet to use (default from config)
            method: Change point detection method (default from config)
            window_size: Window size for analysis (default from config)
            
        Returns:
            Regime analysis result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze regimes")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        window_size = window_size or self.config["window_size"]
        
        if method is None:
            method = self.config["change_point_method"]
        elif isinstance(method, str):
            method = ChangePointMethod.from_string(method)
            
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, str(method), window_size)
        
        # Check cache
        cached_result = self._get_cached_result(self._regime_cache, cache_key)
        if cached_result is not None:
            return cached_result
            
        # Perform MRA decomposition
        decomp = self.adaptive_decompose(data, wavelet=wavelet)
        
        if decomp is None:
            self.logger.error("Failed to decompose signal for regime analysis")
            return None
            
        # Detect change points based on method
        change_points = []
        
        if method == ChangePointMethod.ENERGY:
            # Energy distribution change
            change_points = self._detect_energy_changes(decomp, window_size)
        elif method == ChangePointMethod.WAVELET_VARIANCE:
            # Wavelet variance change
            change_points = self._detect_variance_changes(decomp, window_size)
        elif method == ChangePointMethod.DETAIL_KURTOSIS:
            # Kurtosis of detail coefficients
            change_points = self._detect_kurtosis_changes(decomp, window_size)
        elif method == ChangePointMethod.BAYESIAN:
            # Bayesian change point detection
            if self.has_statsmodels:
                change_points = self._detect_bayesian_changes(data, window_size)
            else:
                self.logger.warning("Statsmodels not available, falling back to multiscale method")
                change_points = self._detect_multiscale_changes(decomp, window_size)
        else:
            # Default: Multiscale approach
            change_points = self._detect_multiscale_changes(decomp, window_size)
            
        # Filter change points based on minimum regime duration
        if len(change_points) > 1:
            filtered_change_points = [change_points[0]]
            
            for cp in change_points[1:]:
                if cp - filtered_change_points[-1] >= self.config["min_regime_duration"]:
                    filtered_change_points.append(cp)
                    
            change_points = filtered_change_points
            
        # Determine regimes between change points
        regimes = []
        confidence = []
        regime_probabilities = {regime: [] for regime in self.config["regime_types"]}
        
        # Add start point if not included
        if not change_points or change_points[0] > 0:
            change_points = [0] + change_points
            
        # Add end point if not included
        if not change_points or change_points[-1] < len(data) - 1:
            change_points = change_points + [len(data) - 1]
            
        # Create measures dictionary
        measures = {
            "trend_strength": [],
            "cycle_strength": [],
            "noise_strength": [],
            "regularity": []
        }
        
        # Analyze each regime
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i+1]
            
            # Extract regime data
            regime_data = data[start_idx:end_idx]
            
            # Analyze characteristics for regime classification
            decomp_regime = self.adaptive_decompose(regime_data, wavelet=wavelet)
            
            if decomp_regime is None:
                # Skip if decomposition fails
                continue
                
            # Calculate energy in trend, cycle, and noise components
            trend_energy = np.sum(decomp_regime.trend**2)
            cycle_energy = np.sum(decomp_regime.cycle**2)
            noise_energy = np.sum(decomp_regime.noise**2)
            total_energy = trend_energy + cycle_energy + noise_energy
            
            if total_energy > 0:
                trend_strength = trend_energy / total_energy
                cycle_strength = cycle_energy / total_energy
                noise_strength = noise_energy / total_energy
            else:
                trend_strength = cycle_strength = noise_strength = 0.0
                
            # Add to measures
            measures["trend_strength"].extend([trend_strength] * (end_idx - start_idx))
            measures["cycle_strength"].extend([cycle_strength] * (end_idx - start_idx))
            measures["noise_strength"].extend([noise_strength] * (end_idx - start_idx))
            
            # Calculate regularity (inverse of entropy)
            if self.has_scipy:
                from scipy.stats import entropy
                
                # Use detail coefficients energy distribution
                energy_values = [decomp_regime.energy_distribution.get(f"detail_{level}", 0.0) 
                              for level in range(1, decomp_regime.trend_level + 1)]
                
                if sum(energy_values) > 0:
                    normalized_energy = [e / sum(energy_values) for e in energy_values]
                    regularity = 1.0 - entropy(normalized_energy) / np.log(len(energy_values))
                else:
                    regularity = 0.5
            else:
                regularity = 0.5
                
            measures["regularity"].extend([regularity] * (end_idx - start_idx))
            
            # Classify regime based on energy distribution
            if trend_strength > 0.6:
                regime_type = "trend"
                confidence_value = trend_strength
            elif cycle_strength > 0.4:
                regime_type = "cycle"
                confidence_value = cycle_strength
            elif noise_strength > 0.4:
                regime_type = "noise"
                confidence_value = noise_strength
            else:
                regime_type = "mixed"
                confidence_value = 0.5
                
            # Add regime classification
            regimes.extend([regime_type] * (end_idx - start_idx))
            confidence.extend([confidence_value] * (end_idx - start_idx))
            
            # Calculate regime probabilities
            for regime_type in self.config["regime_types"]:
                if regime_type == "trend":
                    probability = trend_strength
                elif regime_type == "cycle":
                    probability = cycle_strength
                elif regime_type == "noise":
                    probability = noise_strength
                elif regime_type == "mixed":
                    # Mixed is high when others are balanced
                    max_component = max(trend_strength, cycle_strength, noise_strength)
                    probability = 1.0 - max_component
                else:
                    probability = 0.0
                    
                regime_probabilities[regime_type].extend([probability] * (end_idx - start_idx))
                
        # Create result
        result = RegimeAnalysisResult(
            regimes=regimes,
            change_points=change_points,
            probabilities=regime_probabilities,
            confidence=confidence,
            measures=measures,
            metadata={
                "data_length": len(data),
                "method": str(method),
                "wavelet": wavelet,
                "timestamp": time.time()
            }
        )
        
        # Cache result
        self._cache_result(self._regime_cache, cache_key, result)
        
        return result

    
    def _detect_energy_changes(self, decomp: MRADecomposition, window_size: int) -> List[int]:
        """
        Detect change points based on energy distribution changes.
        
        Args:
            decomp: MRA decomposition
            window_size: Window size for analysis
            
        Returns:
            List of change point indices
        """
        # Get number of points
        n = len(decomp.original)
        
        # Initialize result
        change_points = []
        
        # Minimum window size
        min_window = min(window_size, n // 10)
        
        if min_window < 5:
            # Too few points for analysis
            return change_points
            
        # Compute rolling energy distribution
        energies = {}
        
        # For each detail level
        for level in range(1, decomp.trend_level + 1):
            detail = decomp.details.get(level)
            
            if detail is not None:
                # Calculate rolling energy
                rolling_energy = np.zeros(n)
                
                for i in range(n):
                    start_idx = max(0, i - min_window // 2)
                    end_idx = min(n, i + min_window // 2)
                    
                    # Calculate energy in window
                    rolling_energy[i] = np.sum(detail[start_idx:end_idx]**2)
                    
                energies[f"detail_{level}"] = rolling_energy
                
        # Calculate rolling approximation energy
        rolling_approx_energy = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - min_window // 2)
            end_idx = min(n, i + min_window // 2)
            
            # Calculate energy in window
            rolling_approx_energy[i] = np.sum(decomp.approximation[start_idx:end_idx]**2)
            
        energies["approximation"] = rolling_approx_energy
        
        # Calculate total energy
        total_energy = sum(energies.values())
        
        # Normalize energy
        for key in energies:
            if np.sum(total_energy) > 0:
                energies[key] = energies[key] / total_energy
                
        # Calculate energy distribution distance between windows
        distance = np.zeros(n)
        
        for i in range(min_window, n - min_window):
            # Calculate energy distribution in left and right windows
            left_dist = {}
            right_dist = {}
            
            for key, values in energies.items():
                left_dist[key] = np.mean(values[i-min_window:i])
                right_dist[key] = np.mean(values[i:i+min_window])
                
            # Calculate distance between distributions
            dist_sum = 0.0
            for key in left_dist:
                dist_sum += (left_dist[key] - right_dist[key])**2
                
            distance[i] = np.sqrt(dist_sum)
            
        # Find peaks in distance
        if self.has_scipy:
            from scipy.signal import find_peaks
            
            # Find peaks with minimum distance
            peaks, _ = find_peaks(distance, distance=min_window, height=self.config["regime_change_threshold"])
            change_points = peaks.tolist()
        else:
            # Simple peak detection
            change_points = []
            for i in range(min_window, n - min_window):
                if distance[i] > self.config["regime_change_threshold"]:
                    if not change_points or i - change_points[-1] >= min_window:
                        change_points.append(i)
                        
        return change_points
    
    def _detect_variance_changes(self, decomp: MRADecomposition, window_size: int) -> List[int]:
        """
        Detect change points based on wavelet variance changes.
        
        Args:
            decomp: MRA decomposition
            window_size: Window size for analysis
            
        Returns:
            List of change point indices
        """
        # Get number of points
        n = len(decomp.original)
        
        # Initialize result
        change_points = []
        
        # Minimum window size
        min_window = min(window_size, n // 10)
        
        if min_window < 5:
            # Too few points for analysis
            return change_points
            
        # Compute rolling variance at each scale
        variances = {}
        
        # For each detail level
        for level in range(1, decomp.trend_level + 1):
            detail = decomp.details.get(level)
            
            if detail is not None:
                # Calculate rolling variance
                rolling_var = np.zeros(n)
                
                for i in range(n):
                    start_idx = max(0, i - min_window // 2)
                    end_idx = min(n, i + min_window // 2)
                    
                    # Calculate variance in window
                    rolling_var[i] = np.var(detail[start_idx:end_idx])
                    
                variances[f"detail_{level}"] = rolling_var
                
        # Calculate change in variance
        variance_change = np.zeros(n)
        
        for i in range(min_window, n - min_window):
            # Calculate variance ratio between windows
            ratio_sum = 0.0
            count = 0
            
            for key, values in variances.items():
                left_var = np.mean(values[i-min_window:i])
                right_var = np.mean(values[i:i+min_window])
                
                if min(left_var, right_var) > 1e-10:
                    # Log variance ratio
                    ratio = max(left_var, right_var) / min(left_var, right_var)
                    ratio_sum += np.log(ratio)
                    count += 1
                    
            if count > 0:
                variance_change[i] = ratio_sum / count
                
        # Find peaks in variance change
        if self.has_scipy:
            from scipy.signal import find_peaks
            
            # Find peaks with minimum distance
            peaks, _ = find_peaks(variance_change, distance=min_window, 
                               height=self.config["regime_change_threshold"])
            change_points = peaks.tolist()
        else:
            # Simple peak detection
            change_points = []
            for i in range(min_window, n - min_window):
                if variance_change[i] > self.config["regime_change_threshold"]:
                    if not change_points or i - change_points[-1] >= min_window:
                        change_points.append(i)
                        
        return change_points
    
    def _detect_kurtosis_changes(self, decomp: MRADecomposition, window_size: int) -> List[int]:
        """
        Detect change points based on kurtosis of detail coefficients.
        
        Args:
            decomp: MRA decomposition
            window_size: Window size for analysis
            
        Returns:
            List of change point indices
        """
        if not self.has_scipy:
            self.logger.warning("SciPy not available for kurtosis calculation, falling back to variance")
            return self._detect_variance_changes(decomp, window_size)
            
        from scipy.stats import kurtosis
        
        # Get number of points
        n = len(decomp.original)
        
        # Initialize result
        change_points = []
        
        # Minimum window size
        min_window = min(window_size, n // 10)
        
        if min_window < 5:
            # Too few points for analysis
            return change_points
            
        # Compute rolling kurtosis at each scale
        kurtosis_values = {}
        
        # For each detail level
        for level in range(1, decomp.trend_level + 1):
            detail = decomp.details.get(level)
            
            if detail is not None:
                # Calculate rolling kurtosis
                rolling_kurt = np.zeros(n)
                
                for i in range(n):
                    start_idx = max(0, i - min_window // 2)
                    end_idx = min(n, i + min_window // 2)
                    
                    # Calculate kurtosis in window
                    if end_idx - start_idx >= 4:  # Need at least 4 points for kurtosis
                        rolling_kurt[i] = kurtosis(detail[start_idx:end_idx])
                    else:
                        rolling_kurt[i] = 0.0
                        
                kurtosis_values[f"detail_{level}"] = rolling_kurt
                
        # Calculate change in kurtosis
        kurtosis_change = np.zeros(n)
        
        for i in range(min_window, n - min_window):
            # Calculate absolute change between windows
            change_sum = 0.0
            count = 0
            
            for key, values in kurtosis_values.items():
                left_kurt = np.mean(values[i-min_window:i])
                right_kurt = np.mean(values[i:i+min_window])
                
                # Absolute change
                change_sum += abs(left_kurt - right_kurt)
                count += 1
                
            if count > 0:
                kurtosis_change[i] = change_sum / count
                
        # Find peaks in kurtosis change
        from scipy.signal import find_peaks
        
        # Find peaks with minimum distance
        peaks, _ = find_peaks(kurtosis_change, distance=min_window, 
                           height=self.config["regime_change_threshold"])
        change_points = peaks.tolist()
                        
        return change_points
    
    def _detect_bayesian_changes(self, data: np.ndarray, window_size: int) -> List[int]:
        """
        Detect change points using Bayesian methods.
        
        Args:
            data: Input time series
            window_size: Window size for analysis
            
        Returns:
            List of change point indices
        """
        if not self.has_statsmodels:
            self.logger.warning("Statsmodels not available for Bayesian change point detection")
            return []
            
        try:
            # Use basic Bayesian tests on data segments
            n = len(data)
            
            # Initialize change points
            change_points = []
            
            # Minimum window size
            min_window = min(window_size, n // 10)
            
            if min_window < 10:
                # Too few points for analysis
                return change_points
                
            # Calculate test statistics in rolling windows
            test_stats = np.zeros(n)
            
            # Use CUSUM test for change detection
            for i in range(min_window, n - min_window):
                left_window = data[i-min_window:i]
                right_window = data[i:i+min_window]
                
                # Calculate means
                left_mean = np.mean(left_window)
                right_mean = np.mean(right_window)
                
                # Calculate variances
                left_var = np.var(left_window)
                right_var = np.var(right_window)
                
                # Calculate pooled variance
                pooled_var = (left_var * min_window + right_var * min_window) / (2 * min_window)
                
                if pooled_var > 0:
                    # Calculate test statistic
                    test_stats[i] = abs(left_mean - right_mean) / np.sqrt(pooled_var / min_window)
                    
            # Find peaks in test statistic
            if self.has_scipy:
                from scipy.signal import find_peaks
                
                # Find peaks with minimum distance
                peaks, _ = find_peaks(test_stats, distance=min_window, 
                                   height=3.0)  # Critical value for significance
                change_points = peaks.tolist()
            else:
                # Simple peak detection
                change_points = []
                for i in range(min_window, n - min_window):
                    if test_stats[i] > 3.0:  # Critical value for significance
                        if not change_points or i - change_points[-1] >= min_window:
                            change_points.append(i)
                            
            return change_points
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian change point detection: {e}")
            return []
    
    def _detect_multiscale_changes(self, decomp: MRADecomposition, window_size: int) -> List[int]:
        """
        Detect change points using a multiscale approach.
        
        Args:
            decomp: MRA decomposition
            window_size: Window size for analysis
            
        Returns:
            List of change point indices
        """
        # Combine change points from multiple methods
        methods = [
            self._detect_energy_changes, 
            self._detect_variance_changes
        ]
        
        if self.has_scipy:
            methods.append(self._detect_kurtosis_changes)
            
        # Collect change points from all methods
        all_change_points = []
        
        for method in methods:
            try:
                change_points = method(decomp, window_size)
                all_change_points.extend(change_points)
            except Exception as e:
                self.logger.error(f"Error in change point detection method {method.__name__}: {e}")
                
        # Sort change points
        all_change_points.sort()
        
        # Filter nearby change points (cluster them)
        if not all_change_points:
            return []
            
        # Minimum distance between change points
        min_distance = min(window_size // 2, len(decomp.original) // 20)
        
        # Cluster change points
        clusters = []
        current_cluster = [all_change_points[0]]
        
        for cp in all_change_points[1:]:
            if cp - current_cluster[-1] < min_distance:
                # Add to current cluster
                current_cluster.append(cp)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [cp]
                
        # Add last cluster
        clusters.append(current_cluster)
        
        # Take median of each cluster
        change_points = [int(np.median(cluster)) for cluster in clusters]
        
        return change_points
    
    def _calculate_trend_strength(self, approximation: np.ndarray) -> float:
        """Calculate trend strength from approximation component."""
        if len(approximation) < 2:
            return 0.0
            
        # Calculate directional changes
        diff = np.diff(approximation)
        
        # Trend strength as ratio of absolute sum to sum of absolutes
        abs_sum = np.abs(np.sum(diff))
        sum_abs = np.sum(np.abs(diff))
        
        if sum_abs > 0:
            return float(abs_sum / sum_abs)
        return 0.0
    
    def _calculate_oscillator_strength(self, details: Dict[str, np.ndarray]) -> float:
        """Calculate oscillator strength from detail components."""
        if not details:
            return 0.0
            
        # Sum energies of all detail levels
        detail_energy = 0
        for detail in details.values():
            detail_energy += np.sum(detail**2)
            
        return float(min(1.0, detail_energy / (1e5 + detail_energy)))
    
    def _calculate_energy_ratio(self, approximation: np.ndarray, details: Dict[str, np.ndarray]) -> float:
        """Calculate ratio of approximation energy to total energy."""
        approx_energy = np.sum(approximation**2)
        
        total_energy = approx_energy
        for detail in details.values():
            if len(detail) == len(approximation):
                total_energy += np.sum(detail**2)
        
        if total_energy > 0:
            return float(approx_energy / total_energy)
        return 0.5
    
    def _calculate_momentum(self, approximation: np.ndarray) -> float:
        """Calculate momentum indicator from approximation trend."""
        if len(approximation) < 10:
            return 0.0
            
        # Simple momentum: ratio of recent trend to overall trend
        recent = approximation[-5:]
        overall = approximation[-20:] if len(approximation) >= 20 else approximation
        
        recent_trend = np.mean(np.diff(recent)) if len(recent) > 1 else 0
        overall_trend = np.mean(np.diff(overall)) if len(overall) > 1 else 0
        
        # Normalize to [-1, 1] range
        if abs(overall_trend) > 1e-10:
            momentum = recent_trend / (abs(overall_trend) + 1e-10)
            return float(max(-1.0, min(1.0, momentum)))
        elif recent_trend > 0:
            return 1.0
        elif recent_trend < 0:
            return -1.0
        return 0.0
    
    def _calculate_volatility_ratio(self, approximation: np.ndarray, details: Dict[str, np.ndarray]) -> float:
        """Calculate ratio of approximation volatility to detail volatility."""
        approx_vol = np.std(approximation)
        
        # Calculate average volatility of detail components
        detail_vols = []
        for detail in details.values():
            if len(detail) == len(approximation):
                detail_vols.append(np.std(detail))
        
        avg_detail_vol = np.mean(detail_vols) if detail_vols else 1.0
        
        if avg_detail_vol > 0:
            return float(approx_vol / avg_detail_vol)
        return 1.0
    
    def _calculate_regime_score(self) -> float:
        """Calculate a numerical score for the current regime."""
        if not self._regime_history:
            return 0.5
            
        current_regime = self._regime_history[-1]
        
        # Map regimes to numerical scores
        regime_scores = {
            "growth": 1.0,
            "conservation": 0.5,
            "release": 0.0,
            "reorganization": 0.25
        }
        
        return float(regime_scores.get(current_regime, 0.5))
    
    def _calculate_cycle_ratio(self) -> float:
        """Calculate ratio between dominant and secondary cycles."""
        if len(self._cycle_periods) < 2:
            return 1.0
            
        # Sort periods in descending order of strength
        if hasattr(self, "_cycle_strengths") and self._cycle_strengths:
            # Use cycle strengths if available
            strengths = self._cycle_strengths
            sorted_periods = [p for _, p in sorted(zip(strengths, self._cycle_periods), reverse=True)]
        else:
            # Otherwise, use periods as is
            sorted_periods = sorted(self._cycle_periods)
            
        # Calculate ratio between top two cycles
        if len(sorted_periods) >= 2 and sorted_periods[1] > 0:
            return float(sorted_periods[0] / sorted_periods[1])
        return 1.0
    
    def _calculate_wavelet_entropy(self) -> float:
        """Calculate wavelet entropy as a measure of signal complexity."""
        if not self._last_decomposition or "coefficients" not in self._last_decomposition:
            return 0.5
            
        coeffs = self._last_decomposition["coefficients"]
        
        # Calculate energy at each decomposition level
        energies = [np.sum(coef**2) for coef in coeffs]
        total_energy = sum(energies)
        
        if total_energy > 0:
            # Calculate probability distribution
            probs = [energy / total_energy for energy in energies]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
            
            # Normalize by maximum possible entropy (log2(levels))
            max_entropy = np.log2(len(coeffs))
            if max_entropy > 0:
                return float(entropy / max_entropy)
        
        return 0.5
    
    def _calculate_singularity_measure(self, signal: np.ndarray) -> float:
        """Calculate a measure of signal singularities/discontinuities."""
        if len(signal) < 4:
            return 0.0
            
        # Use the maximum local variation as singularity measure
        diffs = np.abs(np.diff(signal))
        local_maxima = np.maximum(0, diffs[1:] - diffs[:-1])
        local_var = np.max(local_maxima) / (np.mean(diffs) + 1e-10)
        
        # Normalize to [0, 1] range
        return float(min(1.0, local_var / 10.0))
    
    def _calculate_self_similarity(self, signal: np.ndarray) -> float:
        """Calculate self-similarity measure (simplified Hurst exponent estimation)."""
        if len(signal) < 10:
            return 0.5
            
        # Simple R/S analysis (simplified)
        window_sizes = [4, 8, 16, 32]
        window_sizes = [w for w in window_sizes if w < len(signal)]
        
        if not window_sizes:
            return 0.5
            
        rs_values = []
        for w in window_sizes:
            # Split signal into windows
            n_windows = len(signal) // w
            if n_windows == 0:
                continue
                
            rs_window = []
            for i in range(n_windows):
                window = signal[i*w:(i+1)*w]
                
                # Calculate range and standard deviation
                window_range = np.max(window) - np.min(window)
                window_std = np.std(window)
                
                if window_std > 0:
                    rs_window.append(window_range / window_std)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) > 1 and len(window_sizes) > 1:
            # Estimate Hurst exponent as log-log slope
            x = np.log(window_sizes[:len(rs_values)])
            y = np.log(rs_values)
            
            # Simple linear regression
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator = np.sum((x - mean_x)**2)
            
            if denominator > 0:
                hurst = numerator / denominator
                # Normalize to [0, 1] range (H is typically in [0, 1])
                return float(max(0.0, min(1.0, hurst)))
        
        return 0.5

    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize a pattern for similarity comparison."""
        min_val = np.min(pattern)
        max_val = np.max(pattern)
        
        if max_val > min_val:
            return (pattern - min_val) / (max_val - min_val)
        return np.zeros_like(pattern)
    
    def _calculate_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        # Ensure equal length
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        # Normalize patterns
        p1_norm = self._normalize_pattern(p1)
        p2_norm = self._normalize_pattern(p2)
        
        # Calculate distance metrics
        distance = np.sqrt(np.sum((p1_norm - p2_norm)**2))
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(p1_norm, p2_norm)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        except:
            correlation = 0
        
        # Combined similarity score
        similarity = 1.0 / (1.0 + distance) * (0.5 + 0.5 * correlation)
        
        return float(similarity)


    # ----- Cross-Scale Analysis Methods -----
    
    def analyze_cross_scale(self, data: np.ndarray, wavelet: Optional[str] = None,
                         measure: Optional[str] = None) -> CrossScaleResult:
        """
        Analyze interactions between different scales/frequencies.
        
        Args:
            data: Input time series
            wavelet: Wavelet to use (default from config)
            measure: Cross-scale measure (default from config)
            
        Returns:
            Cross-scale interaction result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze cross-scale interactions")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        measure = measure or self.config["cross_scale_measure"]
        
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, measure)
        
        # Check cache
        cached_result = self._get_cached_result(self._cross_scale_cache, cache_key)
        if cached_result is not None:
            return cached_result
            
        # Perform MRA decomposition
        decomp = self.adaptive_decompose(data, wavelet=wavelet)
        
        if decomp is None:
            self.logger.error("Failed to decompose signal for cross-scale analysis")
            return None
            
        # Initialize result containers
        interactions = {}
        information_flow = {}
        causality = {}
        synchronization = {}
        phase_locking = {}
        
        # Get available levels
        levels = list(decomp.details.keys())
        
        # Calculate cross-scale interactions
        for i, level1 in enumerate(levels):
            for j, level2 in enumerate(levels[i+1:], i+1):
                # Get detail signals
                signal1 = decomp.details[level1]
                signal2 = decomp.details[level2]
                
                # Calculate interaction based on measure
                if measure.lower() == "correlation":
                    # Pearson correlation
                    if len(signal1) == len(signal2):
                        if self.has_scipy:
                            from scipy.stats import pearsonr
                            correlation, _ = pearsonr(signal1, signal2)
                            interactions[(level1, level2)] = correlation
                        else:
                            # Manual correlation
                            s1_norm = signal1 - np.mean(signal1)
                            s2_norm = signal2 - np.mean(signal2)
                            correlation = np.sum(s1_norm * s2_norm) / (
                                np.sqrt(np.sum(s1_norm**2) * np.sum(s2_norm**2)) + 1e-10
                            )
                            interactions[(level1, level2)] = correlation
                            
                elif measure.lower() == "mutual_info":
                    # Mutual information
                    if self.has_scipy:
                        from sklearn.feature_selection import mutual_info_regression
                        
                        # Reshape for sklearn
                        s1 = signal1.reshape(-1, 1)
                        s2 = signal2
                        
                        # Calculate mutual information
                        mi = mutual_info_regression(s1, s2)[0]
                        interactions[(level1, level2)] = mi
                    else:
                        # Fallback to correlation
                        self.logger.warning("SciPy not available for mutual information, using correlation")
                        s1_norm = signal1 - np.mean(signal1)
                        s2_norm = signal2 - np.mean(signal2)
                        correlation = np.sum(s1_norm * s2_norm) / (
                            np.sqrt(np.sum(s1_norm**2) * np.sum(s2_norm**2)) + 1e-10
                        )
                        interactions[(level1, level2)] = correlation
                        
                else:
                    # Default to correlation
                    if self.has_scipy:
                        from scipy.stats import pearsonr
                        correlation, _ = pearsonr(signal1, signal2)
                        interactions[(level1, level2)] = correlation
                    else:
                        # Manual correlation
                        s1_norm = signal1 - np.mean(signal1)
                        s2_norm = signal2 - np.mean(signal2)
                        correlation = np.sum(s1_norm * s2_norm) / (
                            np.sqrt(np.sum(s1_norm**2) * np.sum(s2_norm**2)) + 1e-10
                        )
                        interactions[(level1, level2)] = correlation
                        
        # Calculate directed information flow if possible
        if self.config["direction_measure"] == "granger" and self.has_statsmodels:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            for i, level1 in enumerate(levels):
                for j, level2 in enumerate(levels):
                    if i != j:
                        # Get detail signals
                        signal1 = decomp.details[level1]
                        signal2 = decomp.details[level2]
                        
                        # Ensure equal length
                        if len(signal1) == len(signal2):
                            # Prepare data for Granger test
                            data_matrix = np.column_stack([signal1, signal2])
                            
                            try:
                                # Test if level1 Granger-causes level2
                                results = grangercausalitytests(
                                    data_matrix, 
                                    maxlag=self.config["causality_lag"],
                                    verbose=False
                                )
                                
                                # Get p-value of F-test for the first lag
                                p_value = results[1][0]['ssr_ftest'][1]
                                
                                # Store result (p-value)
                                causality[(level1, level2)] = p_value
                                
                                # Convert to information flow measure (1 - p_value)
                                information_flow[(level1, level2)] = 1.0 - p_value
                            except Exception as e:
                                self.logger.error(f"Error in Granger causality test: {e}")
                                
        # Calculate synchronization and phase locking
        if self.has_scipy and self.config["synchronization_method"] == "phase":
            from scipy.signal import hilbert
            
            for level in levels:
                signal = decomp.details[level]
                
                try:
                    # Apply Hilbert transform for phase extraction
                    analytic_signal = hilbert(signal)
                    phase = np.angle(analytic_signal)
                    
                    # Calculate phase coherence (mean phase vector length)
                    phase_vector = np.exp(1j * phase)
                    synchronization[level] = abs(np.mean(phase_vector))
                except Exception as e:
                    self.logger.error(f"Error calculating synchronization for level {level}: {e}")
                    
            # Calculate phase locking between levels
            for i, level1 in enumerate(levels):
                for j, level2 in enumerate(levels[i+1:], i+1):
                    try:
                        # Get phase signals
                        signal1 = decomp.details[level1]
                        signal2 = decomp.details[level2]
                        
                        # Apply Hilbert transform for phase extraction
                        analytic_signal1 = hilbert(signal1)
                        analytic_signal2 = hilbert(signal2)
                        
                        phase1 = np.angle(analytic_signal1)
                        phase2 = np.angle(analytic_signal2)
                        
                        # Calculate phase difference
                        phase_diff = phase1 - phase2
                        
                        # Calculate phase locking value
                        plv = abs(np.mean(np.exp(1j * phase_diff)))
                        phase_locking[(level1, level2)] = plv
                    except Exception as e:
                        self.logger.error(f"Error calculating phase locking: {e}")
                        
        # Create result
        result = CrossScaleResult(
            interactions=interactions,
            information_flow=information_flow,
            causality=causality,
            synchronization=synchronization,
            phase_locking=phase_locking,
            metadata={
                "data_length": len(data),
                "wavelet": wavelet,
                "measure": measure,
                "levels": levels,
                "timestamp": time.time()
            }
        )
        
        # Cache result
        self._cache_result(self._cross_scale_cache, cache_key, result)
        
        return result
    
    # ----- Market-Specific Analysis Methods -----
    
    def analyze_market_cycles(self, data: np.ndarray, wavelet: Optional[str] = None,
                           min_period: Optional[int] = None,
                           max_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze market cycles using MRA.
        
        Args:
            data: Price or return time series
            wavelet: Wavelet to use (default from config)
            min_period: Minimum cycle period to detect
            max_period: Maximum cycle period to detect
            
        Returns:
            Dictionary with cycle analysis results
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze market cycles")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        # Let the wavelet processor handle the cycle detection
        return self.wavelet_processor.detect_cycles(
            data, 
            min_period=min_period,
            max_period=max_period,
            wavelet=wavelet
        )
    
    def find_similar_patterns(self, data: Union[pd.DataFrame, np.ndarray, pd.Series], 
                               lookback_window: int = 20, top_n: int = 3) -> List[Dict]:
        """
        Find historical patterns similar to the current market structure.
        
        Args:
            data: Input data
            lookback_window: Window size for pattern comparison
            top_n: Number of top similar patterns to return
            
        Returns:
            List of dictionaries with similar pattern information
        """
        # Preprocess data
        series = self.preprocess_data(data)
        
        if len(series) < 2 * lookback_window:
            self.logger.warning(f"Data length {len(series)} is insufficient for pattern detection")
            return []
        
        # Get current pattern (last lookback_window points)
        current_pattern = series[-lookback_window:]
        
        # Normalize current pattern
        current_min, current_max = np.min(current_pattern), np.max(current_pattern)
        if current_max > current_min:
            current_norm = (current_pattern - current_min) / (current_max - current_min)
        else:
            current_norm = np.zeros_like(current_pattern)
        
        # Scan historical data for similar patterns
        similarity_scores = []
        
        for i in range(len(series) - 2 * lookback_window):
            # Get historical pattern
            hist_pattern = series[i:i+lookback_window]
            
            # Normalize historical pattern
            hist_min, hist_max = np.min(hist_pattern), np.max(hist_pattern)
            if hist_max > hist_min:
                hist_norm = (hist_pattern - hist_min) / (hist_max - hist_min)
            else:
                hist_norm = np.zeros_like(hist_pattern)
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((current_norm - hist_norm)**2))
            
            # Calculate correlation
            correlation = np.corrcoef(current_norm, hist_norm)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # Calculate overall similarity score (higher is better)
            similarity = 1.0 / (1.0 + distance) * (0.5 + 0.5 * correlation)
            
            # Store pattern info
            similarity_scores.append({
                "start_idx": i,
                "end_idx": i + lookback_window - 1,
                "similarity": float(similarity),
                "distance": float(distance),
                "correlation": float(correlation),
                "outcome": series[i+lookback_window:i+2*lookback_window].tolist() 
                           if i+2*lookback_window <= len(series) else []
            })
        
        # Sort by similarity (descending)
        similarity_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N matches
        return similarity_scores[:top_n]

    
    def analyze_market_structure(self, data: Union[pd.DataFrame, np.ndarray, pd.Series]) -> Dict:
        """
        Analyze market structure based on wavelet decomposition.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with market structure analysis
        """
        # Perform MRA if not already done
        if self._last_decomposition is None:
            self.perform_mra(data)
            
        if self._last_decomposition is None:
            self.logger.error("Failed to perform MRA decomposition")
            return {}
        
        # Access decomposition components
        approximation = self._last_decomposition["approximation"]
        details = self._last_decomposition["details"]
        original = self._last_decomposition["original"]
        
        # Initialize result dictionary
        result = {}
        
        # 1. Calculate trend characteristics
        if len(approximation) > 1:
            # Trend direction
            trend_diff = np.diff(approximation)
            avg_trend = np.mean(trend_diff)
            
            result["trend"] = {
                "direction": "up" if avg_trend > 0 else "down" if avg_trend < 0 else "flat",
                "strength": float(self._calculate_trend_strength(approximation)),
                "volatility": float(np.std(approximation)),
                "momentum": float(self._calculate_momentum(approximation)),
                "consistency": float(np.abs(np.sum(np.sign(trend_diff))) / (len(trend_diff) + 1e-10))
            }
        
        # 2. Calculate cycle characteristics
        if self._cycle_periods:
            result["cycles"] = {
                "dominant_period": int(self._cycle_periods[0]) if self._cycle_periods else 0,
                "secondary_period": int(self._cycle_periods[1]) if len(self._cycle_periods) > 1 else 0,
                "all_periods": [int(p) for p in self._cycle_periods],
                "regularity": float(1.0 - (np.std(self._cycle_periods) / (np.mean(self._cycle_periods) + 1e-10)) 
                                if len(self._cycle_periods) > 1 else 0.5)
            }
        
        # 3. Calculate volatility structure and 4. Market complexity are omitted for brevity
        
        # 5. Current regime information
        if self._regime_history:
            current_regime = self._regime_history[-1]
            
            # Calculate regime stability
            if len(self._regime_history) >= 5:
                recent_regimes = self._regime_history[-5:]
                regime_stability = recent_regimes.count(current_regime) / len(recent_regimes)
            else:
                regime_stability = 1.0
            
            result["regime"] = {
                "current": current_regime,
                "stability": float(regime_stability),
                "duration": len([r for r in reversed(self._regime_history) 
                                if r == current_regime and r == self._regime_history[-1]])
            }
        
        return result

    def get_signal_predictions(self, mra_result: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Generate trading signals based on wavelet analysis.
        
        Args:
            mra_result: Result from perform_mra (optional, uses last computation if None)
            
        Returns:
            Dictionary of generated signals
        """
        # Use provided result or last computation
        if mra_result is None:
            if self._last_decomposition is None:
                self.logger.error("No MRA decomposition available")
                return {}
                
            approximation = self._last_decomposition["approximation"]
            details = self._last_decomposition["details"]
            original = self._last_decomposition["original"]
        else:
            approximation = mra_result["approximation"]
            details = mra_result["details"]
            original = mra_result["original"]
        
        # Initialize signals dictionary
        signals = {}
        
        # Generate trend signal from approximation
        if len(approximation) > 1:
            # Normalize approximation trend
            trend = (approximation - np.min(approximation)) / (np.max(approximation) - np.min(approximation) + 1e-10)
            signals["wavelet_trend"] = trend
            
            # Generate trend direction signal (-1 to 1)
            trend_diff = np.diff(approximation)
            trend_diff = np.append(trend_diff, trend_diff[-1])  # Pad to match original length
            
            # Normalize trend direction
            max_diff = np.max(np.abs(trend_diff))
            if max_diff > 0:
                trend_direction = trend_diff / max_diff
            else:
                trend_direction = np.zeros_like(trend_diff)
                
            signals["wavelet_trend_direction"] = trend_direction
        
        # Generate oscillator signal from details
        if details and len(next(iter(details.values()))) > 0:
            # Combine detail levels with decreasing weights
            oscillator = np.zeros_like(original)
            
            detail_levels = sorted(details.keys(), key=lambda x: int(x.split('_')[1]))
            for i, level in enumerate(detail_levels):
                weight = 1.0 / (i + 1)  # Higher weight to lower frequency details
                detail = details[level]
                if len(detail) == len(oscillator):
                    oscillator += weight * detail
            
            # Normalize oscillator
            max_osc = np.max(np.abs(oscillator))
            if max_osc > 0:
                oscillator = oscillator / max_osc
                
            signals["wavelet_oscillator"] = oscillator
            signals["wavelet_overbought"] = 0.5 + 0.5 * oscillator
            signals["wavelet_mean_reversion"] = -oscillator
        
        # Generate combined signal
        if signals:
            # Combine trend, oscillator signals with appropriate weights
            combined = np.zeros_like(original)
            weights = {
                "wavelet_trend_direction": 0.4,
                "wavelet_oscillator": 0.3
            }
            
            for signal_name, weight in weights.items():
                if signal_name in signals:
                    combined += weight * signals[signal_name]
            
            # Normalize to [-1, 1] range
            max_combined = np.max(np.abs(combined))
            if max_combined > 0:
                combined = combined / max_combined
                
            signals["wavelet_combined"] = combined
            
            # Generate trading signal (discretized to -1, 0, 1)
            threshold = 0.3
            trading_signal = np.zeros_like(combined)
            trading_signal[combined > threshold] = 1.0
            trading_signal[combined < -threshold] = -1.0
            
            signals["wavelet_trading_signal"] = trading_signal
        
        return signals

    def preprocess_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
       """
       Preprocess input data for wavelet analysis.
       
       Args:
           data: Input data as DataFrame or array. If DataFrame, 'close' column is used.
                Otherwise, the input array is used directly.
       
       Returns:
           Preprocessed data as numpy array
       """
       # Extract time series
       if isinstance(data, pd.DataFrame):
           if 'close' in data.columns:
               series = data['close'].values
           else:
               series = data.iloc[:, 0].values
       else:
           series = np.asarray(data)
       
       # Check data length
       min_data_points = getattr(self.config, 'min_data_points', 100)
       if len(series) < min_data_points:
           self.logger.warning(f"Data length {len(series)} is less than minimum required {min_data_points}. Results may be unreliable.")
           return False
       
       # Handle NaN values
       nan_mask = np.isnan(series)
       if np.any(nan_mask):
           self.logger.warning(f"Found {np.sum(nan_mask)} NaN values in input data. Interpolating.")
           valid_indices = np.where(~nan_mask)[0]
           invalid_indices = np.where(nan_mask)[0]
           
           if len(valid_indices) > 0:
               # Use linear interpolation
               interp_values = np.interp(invalid_indices, valid_indices, series[valid_indices])
               series[invalid_indices] = interp_values
           else:
               # All values are NaN
               self.logger.error("All values are NaN. Cannot process data.")
               return np.zeros_like(series)
       
       # Return preprocessed data
       return series

    def perform_mra(self, data: Union[pd.DataFrame, np.ndarray, pd.Series], 
                    denoise: bool = True) -> Dict:
        """
        Perform Multi-Resolution Analysis on the input data.
        
        Args:
            data: Input data
            denoise: Whether to apply wavelet denoising
        
        Returns:
            Dictionary with MRA components:
                - original: Original data
                - approximation: Approximation (low frequency component)
                - details: Dictionary of detail components
                - coefficients: Wavelet coefficients
                - regime: Detected market regime
                - cycles: Detected cycles
        """
        # Preprocess data
        series = self.preprocess_data(data)
        
        # Determine appropriate decomposition level
        data_len = len(series)
        wavelet_family = getattr(self.config, 'wavelet_family', 'db4')
        wavelet_level = getattr(self.config, 'wavelet_level', 5)
        
        max_level = pywt.dwt_max_level(data_len, wavelet_family)
        level = min(wavelet_level, max_level)

        if level < wavelet_level:
            self.logger.warning(f"Requested level {wavelet_level} exceeds maximum possible level {max_level}. Using level {level} instead.")

        coeffs = pywt.wavedec(series, wavelet_family, level=level)
        
        # Store approximation and details
        approx = coeffs[0]
        details = coeffs[1:]
        
        # Denoise if requested
        if denoise:
            details = self._denoise_details(details)
        
        # Reconstruct components
        reconstructed = {}
        wavelet_family = getattr(self.config, 'wavelet_family', 'db4')
        reconstructed["approximation"] = pywt.upcoef('a', approx, wavelet_family, level=level, take=len(series))
        
        detail_components = {}
        for i, detail in enumerate(details, 1):
            # Reconstruct each detail level to original signal length
            wavelet_family = getattr(self.config, 'wavelet_family', 'db4')
            detail_components[f"level_{i}"] = pywt.upcoef('d', detail, wavelet_family, level=i, take=len(series))
        
        # Store last decomposition for further analysis
        self._last_decomposition = {
            "approximation": reconstructed["approximation"],
            "details": detail_components,
            "coefficients": coeffs,
            "original": series
        }
        
        # Perform regime detection if enabled
        regime = None
        if getattr(self.config, 'use_regime_detection', True):
            regime = self._detect_regime(reconstructed["approximation"], detail_components)
            self._regime_history.append(regime)
            
            # Keep history limited
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]
        
        # Perform cycle detection if enabled
        cycles = None
        if getattr(self.config, 'use_cycle_detection', True):
            cycles = self._detect_cycles(detail_components)
            self._cycle_periods = cycles["periods"]
        
        # Create result dictionary
        result = {
            "original": series,
            "approximation": reconstructed["approximation"],
            "details": detail_components,
            "coefficients": coeffs,
            "regime": regime,
            "cycles": cycles
        }
        
        return result

    def _denoise_details(self, details: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply wavelet denoising to detail coefficients.
        
        Args:
            details: List of detail coefficients
            
        Returns:
            Denoised detail coefficients
        """
        denoised_details = []
        threshold = getattr(self.config, 'denoise_threshold', 2.0)
        
        for detail in details:
            # Calculate adaptive threshold
            sigma = np.median(np.abs(detail)) / 0.6745
            adaptive_threshold = sigma * threshold
            
            # Apply soft thresholding
            denoised = pywt.threshold(detail, adaptive_threshold, mode='soft')
            denoised_details.append(denoised)
        
        return denoised_details
  
    def _detect_regime(self, approximation: np.ndarray, details: Dict[str, np.ndarray]) -> str:
        """
        Detect current market regime based on wavelet decomposition.
        
        Args:
            approximation: Approximation component
            details: Dictionary of detail components
            
        Returns:
            Detected regime: 'growth', 'conservation', 'release', or 'reorganization'
        """
        # Calculate regime indicators
        indicators = {}
        
        # Calculate trend strength
        trend_diff = np.diff(approximation)
        trend_strength = np.abs(np.sum(trend_diff)) / (np.sum(np.abs(trend_diff)) + 1e-10)
        indicators["trend_strength"] = trend_strength
        
        # Calculate oscillation strength
        detail_sum = np.zeros_like(approximation)
        for detail in details.values():
            if len(detail) == len(detail_sum):
                detail_sum += detail
        
        oscillation_strength = np.var(detail_sum) / (np.var(approximation) + np.var(detail_sum) + 1e-10)
        indicators["oscillation_strength"] = oscillation_strength
        
        # Calculate energy distribution
        total_energy = np.sum(approximation**2)
        for detail in details.values():
            if len(detail) == len(approximation):
                total_energy += np.sum(detail**2)
        
        approx_energy_ratio = np.sum(approximation**2) / (total_energy + 1e-10)
        indicators["approx_energy_ratio"] = approx_energy_ratio
        
        # Calculate volatility ratio
        approx_volatility = np.std(approximation)
        detail_volatility = np.mean([np.std(d) for d in details.values()])
        volatility_ratio = approx_volatility / (detail_volatility + 1e-10)
        indicators["volatility_ratio"] = volatility_ratio
        
        # Calculate stability score
        recent_window = 20
        if len(approximation) > recent_window:
            recent_trend = approximation[-recent_window:]
            stability = 1.0 - (np.std(np.diff(recent_trend)) / (np.mean(np.abs(recent_trend)) + 1e-10))
            indicators["stability_score"] = max(0, min(1, stability))
        else:
            indicators["stability_score"] = 0.5
        
        # Determine regime based on indicators
        regime_scores = {
            "growth": 0,
            "conservation": 0,
            "release": 0,
            "reorganization": 0
        }
        
        # Growth regime indicators
        if trend_strength > self.regime_thresholds["growth"]["trend_strength"]:
            regime_scores["growth"] += 1
        
        if volatility_ratio > self.regime_thresholds["growth"]["volatility_ratio"]:
            regime_scores["growth"] += 1
            
        if approx_energy_ratio > self.regime_thresholds["growth"]["energy_ratio"]:
            regime_scores["growth"] += 1
        
        # Conservation regime indicators
        if oscillation_strength > self.regime_thresholds["conservation"]["oscillation_strength"]:
            regime_scores["conservation"] += 1
            
        if approx_energy_ratio < self.regime_thresholds["conservation"]["energy_upper_limit"]:
            regime_scores["conservation"] += 1
            
        if volatility_ratio < 1.0 / self.regime_thresholds["conservation"]["volatility_deviation"]:
            regime_scores["conservation"] += 1
        
        # Release regime indicators
        if trend_strength > 0.3 and np.mean(np.diff(approximation[-10:])) < 0:
            regime_scores["release"] += 2
            
        if volatility_ratio > 1.2 and oscillation_strength > 0.5:
            regime_scores["release"] += 1
        
        # Reorganization regime indicators
        if indicators["stability_score"] < self.regime_thresholds["transition"]["stability_score"]:
            regime_scores["reorganization"] += 2
            
        # Consider regime history (momentum)
        if len(self._regime_history) >= self.regime_thresholds["transition"]["regime_change_lag"]:
            recent_regimes = self._regime_history[-self.regime_thresholds["transition"]["regime_change_lag"]:]
            most_common = max(set(recent_regimes), key=recent_regimes.count)
            
            # Add momentum to the most common recent regime
            regime_scores[most_common] += 1
        
        # Select regime with highest score
        selected_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        
        # In case of a tie, prefer the previous regime for stability
        max_score = regime_scores[selected_regime]
        tied_regimes = [r for r, s in regime_scores.items() if s == max_score]
        
        if len(tied_regimes) > 1 and len(self._regime_history) > 0:
            prev_regime = self._regime_history[-1]
            if prev_regime in tied_regimes:
                selected_regime = prev_regime
        
        return selected_regime
    
    def _detect_cycles(self, details: Dict[str, np.ndarray]) -> Dict:
        """
        Detect dominant cycles in the data using wavelet details.
        
        Args:
            details: Dictionary of detail components
            
        Returns:
            Dictionary with cycle information:
                - periods: List of detected cycle periods
                - strengths: Relative strengths of each cycle
                - dominant: The dominant cycle period
        """
        # Calculate average period length for each detail level
        detail_levels = sorted(details.keys())
        
        # Calculate the energy in each detail level
        energies = {}
        for level in detail_levels:
            energies[level] = np.sum(details[level]**2)
        
        # Calculate relative energy
        total_energy = sum(energies.values())
        if total_energy > 0:
            relative_energies = {level: energy/total_energy for level, energy in energies.items()}
        else:
            relative_energies = {level: 0 for level in detail_levels}
        
        # Map detail levels to approximate periods
        # For wavelet decomposition, each level j corresponds to frequencies around 2^j
        cycle_periods = {}
        cycle_strengths = {}
        
        for i, level in enumerate(detail_levels):
            # Extract level number from string (e.g., "level_3" -> 3)
            level_num = int(level.split('_')[1])
            
            # Estimate period as 2^level_num
            period = 2**level_num
            
            # Store period and its relative energy
            cycle_periods[level] = period
            cycle_strengths[level] = relative_energies[level]
        
        # Find dominant cycle (highest energy)
        if cycle_strengths:
            dominant_level = max(cycle_strengths.items(), key=lambda x: x[1])[0]
            dominant_period = cycle_periods[dominant_level]
        else:
            dominant_period = None
        
        # Format results
        result = {
            "periods": [cycle_periods[level] for level in detail_levels],
            "strengths": [cycle_strengths[level] for level in detail_levels],
            "dominant": dominant_period
        }
        
        return result

    def generate_trading_recommendations(self, mra_result: Optional[Dict] = None) -> Dict:
        """
        Generate trading recommendations based on MRA analysis.
        
        Args:
            mra_result: Result from perform_mra (optional, uses last computation if None)
            
        Returns:
            Dictionary with trading recommendations and confidence scores
        """
        # Use the last decomposition if no result provided
        if mra_result is None:
            if self._last_decomposition is None:
                self.logger.error("No MRA decomposition available")
                return {"recommendation": "neutral", "confidence": 0.0, "reasons": ["Insufficient data"]}
            
            # Use the last decomposition
            original = self._last_decomposition["original"]
            regime = self._regime_history[-1] if self._regime_history else None
        else:
            original = mra_result["original"]
            regime = mra_result.get("regime")
        
        # Get prediction signals
        signals = self.get_signal_predictions(mra_result)
        
        if not signals or "wavelet_combined" not in signals:
            return {"recommendation": "neutral", "confidence": 0.0, "reasons": ["Failed to generate signals"]}
        
        # Get the combined signal for recommendation
        combined = signals["wavelet_combined"]
        
        # Calculate the current signal value (last point)
        current_signal = combined[-1] if len(combined) > 0 else 0.0
        
        # Determine recommendation direction
        recommendation = "neutral"
        if current_signal > 0.3:
            recommendation = "buy"
        elif current_signal < -0.3:
            recommendation = "sell"
        
        # Calculate confidence based on multiple factors
        signal_confidence = min(1.0, abs(current_signal) * 2)  # Signal strength
        regime_confidence = 0.5  # Default
        trend_strength = 0.5  # Default
        
        # Calculate overall confidence
        confidence = 0.4 * signal_confidence + 0.3 * regime_confidence + 0.3 * trend_strength
        confidence = float(min(1.0, max(0.0, confidence)))
        
        # Prepare reasons for recommendation
        reasons = []
        if abs(current_signal) > 0.5:
            reasons.append(f"Strong {'bullish' if current_signal > 0 else 'bearish'} signal ({current_signal:.2f})")
        elif abs(current_signal) > 0.3:
            reasons.append(f"Moderate {'bullish' if current_signal > 0 else 'bearish'} signal ({current_signal:.2f})")
        else:
            reasons.append(f"Weak directional signal ({current_signal:.2f})")
        
        # Add regime-based reason
        if regime:
            reasons.append(f"Current regime: {regime}")
        
        # Create recommendation dictionary
        result = {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasons": reasons,
            "signals": {k: float(v[-1]) if isinstance(v, np.ndarray) and len(v) > 0 else v 
                        for k, v in signals.items()}
        }
        
        return result
        
    def estimate_trend_strength(self, data: np.ndarray, wavelet: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate trend strength in financial time series.
        
        Args:
            data: Price or return time series
            wavelet: Wavelet to use (default from config)
            
        Returns:
            Dictionary with trend strength metrics
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot estimate trend strength")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        try:
            # Perform MRA decomposition
            decomp = self.adaptive_decompose(data, wavelet=wavelet)
            
            if decomp is None:
                self.logger.error("Failed to decompose signal for trend strength estimation")
                return {}
                
            # Calculate energy in trend component
            trend_energy = np.sum(decomp.trend**2)
            
            # Calculate total energy
            total_energy = np.sum(data**2)
            
            # Calculate trend strength
            trend_strength = trend_energy / total_energy if total_energy > 0 else 0.0
            
            # Calculate directionality (linear trend)
            if self.has_scipy:
                from scipy.stats import linregress
                
                # Calculate linear regression on trend component
                x = np.arange(len(decomp.trend))
                slope, intercept, r_value, p_value, std_err = linregress(x, decomp.trend)
                
                # Normalized slope
                if np.mean(decomp.trend) != 0:
                    norm_slope = slope * len(decomp.trend) / np.abs(np.mean(decomp.trend))
                else:
                    norm_slope = 0.0
                    
                # Trend directionality (positive or negative)
                trend_direction = np.sign(slope)
                
                # Trend linearity (R²)
                trend_linearity = r_value**2
            else:
                # Simple slope calculation
                x = np.arange(len(decomp.trend))
                if len(decomp.trend) > 1:
                    slope = (decomp.trend[-1] - decomp.trend[0]) / (len(decomp.trend) - 1)
                    
                    # Normalized slope
                    if np.mean(decomp.trend) != 0:
                        norm_slope = slope * len(decomp.trend) / np.abs(np.mean(decomp.trend))
                    else:
                        norm_slope = 0.0
                        
                    # Trend directionality
                    trend_direction = np.sign(slope)
                    
                    # Simple linearity measure
                    trend_diffs = np.diff(decomp.trend)
                    trend_linearity = 1.0 - np.std(trend_diffs) / (np.abs(np.mean(trend_diffs)) + 1e-10)
                else:
                    norm_slope = 0.0
                    trend_direction = 0.0
                    trend_linearity = 0.0
                    
            # Create result
            result = {
                "trend_strength": trend_strength,
                "trend_direction": float(trend_direction),
                "trend_slope": float(norm_slope),
                "trend_linearity": float(trend_linearity)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error estimating trend strength: {e}")
            return {}
    
    def analyze_volatility_regimes(self, data: np.ndarray, wavelet: Optional[str] = None,
                               window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze volatility regimes using MRA.
        
        Args:
            data: Price or return time series
            wavelet: Wavelet to use (default from config)
            window_size: Window size for rolling analysis
            
        Returns:
            Dictionary with volatility regime analysis
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze volatility regimes")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        window_size = window_size or self.config["window_size"]
        
        try:
            # Calculate returns if input is price data
            returns = np.diff(data) / data[:-1]
            returns = np.append(0, returns)  # Add zero for first point
            
            # Perform MRA decomposition
            decomp = self.adaptive_decompose(returns, wavelet=wavelet)
            
            if decomp is None:
                self.logger.error("Failed to decompose signal for volatility regime analysis")
                return {}
                
            # Extract noise component (high-frequency details)
            noise = decomp.noise
            
            # Calculate rolling volatility
            n = len(noise)
            rolling_vol = np.zeros(n)
            
            for i in range(n):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(n, i + window_size // 2)
                
                # Calculate standard deviation in window
                rolling_vol[i] = np.std(noise[start_idx:end_idx])
                
            # Identify regimes based on volatility levels
            # Calculate volatility quantiles
            low_threshold = np.percentile(rolling_vol, 33)
            high_threshold = np.percentile(rolling_vol, 67)
            
            # Classify regimes
            vol_regimes = np.zeros(n, dtype=int)
            vol_regimes[rolling_vol <= low_threshold] = 1  # Low volatility
            vol_regimes[(rolling_vol > low_threshold) & (rolling_vol < high_threshold)] = 2  # Medium
            vol_regimes[rolling_vol >= high_threshold] = 3  # High volatility
            
            # Convert to regime names
            regime_names = {
                1: "low_volatility",
                2: "medium_volatility",
                3: "high_volatility"
            }
            
            named_regimes = [regime_names.get(r, "unknown") for r in vol_regimes]
            
            # Detect regime changes
            change_points = [0]
            
            for i in range(1, n):
                if vol_regimes[i] != vol_regimes[i-1]:
                    change_points.append(i)
                    
            # Add end point
            if change_points[-1] < n - 1:
                change_points.append(n - 1)
                
            # Calculate regime statistics
            regime_stats = {}
            
            for regime_id, regime_name in regime_names.items():
                mask = vol_regimes == regime_id
                if np.any(mask):
                    regime_stats[regime_name] = {
                        "count": int(np.sum(mask)),
                        "mean_volatility": float(np.mean(rolling_vol[mask])),
                        "percentage": float(np.sum(mask) / n * 100)
                    }
                    
            # Create result
            result = {
                "regimes": named_regimes,
                "volatility": rolling_vol.tolist(),
                "change_points": change_points,
                "thresholds": {
                    "low": float(low_threshold),
                    "high": float(high_threshold)
                },
                "regime_statistics": regime_stats
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regimes: {e}")
            return {}
    
    def detect_market_anomalies(self, data: np.ndarray, wavelet: Optional[str] = None,
                             threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect anomalies in financial time series using MRA.
        
        Args:
            data: Price or return time series
            wavelet: Wavelet to use (default from config)
            threshold: Detection threshold (standard deviations)
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot detect anomalies")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        threshold = threshold or 3.0  # Default: 3 standard deviations
        
        try:
            # Perform MRA decomposition
            decomp = self.adaptive_decompose(data, wavelet=wavelet)
            
            if decomp is None:
                self.logger.error("Failed to decompose signal for anomaly detection")
                return {}
                
            # Calculate residuals (original - (trend + cycle))
            residuals = decomp.original - (decomp.trend + decomp.cycle)
            
            # Calculate residual statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            # Detect anomalies (points beyond threshold * std)
            anomaly_mask = np.abs(residuals - mean_residual) > threshold * std_residual
            
            # Get anomaly indices
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            # Calculate anomaly scores
            anomaly_scores = np.abs(residuals - mean_residual) / std_residual
            
            # Create result
            result = {
                "anomaly_indices": anomaly_indices,
                "anomaly_scores": anomaly_scores.tolist(),
                "residuals": residuals.tolist(),
                "threshold": float(threshold),
                "residual_stats": {
                    "mean": float(mean_residual),
                    "std": float(std_residual),
                    "kurtosis": float(kurtosis(residuals)) if self.has_scipy else None
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {}

    def extract_features(self, data: Union[pd.DataFrame, np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Extract wavelet-based features from the input data.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of extracted features
        """
        # Perform MRA if not already done
        if self._last_decomposition is None:
            self.perform_mra(data)
            
        if self._last_decomposition is None:
            self.logger.error("Failed to perform MRA decomposition")
            return {}
        
        # Access decomposition components
        approximation = self._last_decomposition["approximation"]
        details = self._last_decomposition["details"]
        original = self._last_decomposition["original"]
        
        # Initialize features dictionary
        features = {}
        
        # Basic features (always extracted)
        features["wavelet_trend_strength"] = self._calculate_trend_strength(approximation)
        features["wavelet_oscillator_strength"] = self._calculate_oscillator_strength(details)
        features["wavelet_energy_ratio"] = self._calculate_energy_ratio(approximation, details)
        
        # Standard feature set
        features["wavelet_momentum"] = self._calculate_momentum(approximation)
        features["wavelet_volatility_ratio"] = self._calculate_volatility_ratio(approximation, details)
            
        # Add cycle-based features
        if self._cycle_periods:
            features["wavelet_dominant_cycle"] = self._cycle_periods[0] if len(self._cycle_periods) > 0 else 0
            features["wavelet_cycle_ratio"] = self._calculate_cycle_ratio()
        
        # Advanced statistical features (if scipy available)
        try:
            from scipy.stats import skew, kurtosis
            features["wavelet_approx_skewness"] = float(skew(approximation))
            features["wavelet_approx_kurtosis"] = float(kurtosis(approximation))
            features["wavelet_entropy"] = self._calculate_wavelet_entropy()
        except ImportError:
            pass
        
        return features
    
    def export_features(self, data: Union[pd.DataFrame, np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Export all computed wavelet features as a DataFrame for machine learning purposes.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with all computed features
        """
        # Ensure we have the MRA decomposition
        self.perform_mra(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Get signals
        signals = self.get_signal_predictions()
        
        # Get market structure
        structure = self.analyze_market_structure(data)
        
        # Flatten nested dictionaries
        all_features = {}
        
        # Add basic features
        for k, v in features.items():
            all_features[k] = v
        
        # Add signal features
        if signals:
            for k, v in signals.items():
                if isinstance(v, np.ndarray) and len(v) > 0:
                    all_features[f"signal_{k}"] = v[-1]
        
        # Add structure features (flattening the nested structure)
        if structure:
            for category, values in structure.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        if not isinstance(v, dict) and not isinstance(v, list):
                            all_features[f"structure_{category}_{k}"] = v
        
        # Convert to DataFrame
        result = pd.DataFrame([all_features])
        
        return result
    
    def run_all_analysis(self, data: Union[pd.DataFrame, np.ndarray, pd.Series]) -> Dict:
        """
        Run all analyses and return comprehensive results.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with all analysis results
        """
        # Perform MRA decomposition
        mra_result = self.perform_mra(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Get signals
        signals = self.get_signal_predictions(mra_result)
        
        # Get trading recommendations
        recommendations = self.generate_trading_recommendations(mra_result)
        
        # Analyze market structure
        structure = self.analyze_market_structure(data)
        
        # Find similar patterns
        patterns = self.find_similar_patterns(data)
        
        # Combine all results
        result = {
            "mra": mra_result,
            "features": features,
            "signals": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in signals.items()},
            "recommendations": recommendations,
            "market_structure": structure,
            "similar_patterns": patterns
        }
        
        return result

    def __del__(self):
        """Cleanup resources."""
        # Safe cleanup of multiprocessing pool if it exists
        if hasattr(self, '_pool') and self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception as e:
                # Just log the error but don't raise it during garbage collection
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Error cleaning up resources: {e}")
            
            