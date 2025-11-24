#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet Processor for CDFA Extensions

Provides comprehensive wavelet-based processing capabilities for financial data:
- Signal denoising and smoothing
- Multi-resolution analysis for regime detection
- Feature extraction for pattern recognition
- Continuous wavelet transform for cycle detection
- Wavelet scattering for robust feature representation

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

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

# ---- Optional dependencies with graceful fallbacks ----

# PyWavelets for wavelet transform
try:
    import pywt
    from pywt import wavedec, waverec, dwt_max_level
    from pywt import cwt as pywt_cwt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet processing will be limited.", DeprecationWarning, DeprecationWarning)
    
    # Create dummy functions
    def wavedec(*args, **kwargs):
        return None
        
    def waverec(*args, **kwargs):
        return None
        
    def dwt_max_level(*args, **kwargs):
        return 1
        
    def pywt_cwt(*args, **kwargs):
        return None, None

# PyTorch for accelerated wavelet transforms
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Accelerated wavelet processing will be limited.", DeprecationWarning, DeprecationWarning)

# Scikit-learn for wavelet scattering
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
    
    # Check for Kymatio (specialized wavelet scattering library)
    try:
        import kymatio
        from kymatio.numpy import Scattering1D as KymatioScattering1D
        KYMATIO_AVAILABLE = True
    except ImportError:
        KYMATIO_AVAILABLE = False
except ImportError:
    SKLEARN_AVAILABLE = False
    KYMATIO_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Wavelet scattering will be limited.", DeprecationWarning, DeprecationWarning)

# SciPy for signal processing
try:
    from scipy import signal
    from scipy.stats import kurtosis, skew
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced signal processing will be limited.", DeprecationWarning, DeprecationWarning)

# Numba for acceleration
try:
    import numba as nb
    from numba import njit, prange, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. CPU acceleration will be limited.", DeprecationWarning, DeprecationWarning)
    
    # Define dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range
    float64 = int64 = lambda x: x

class WaveletFamily(Enum):
    """Wavelet family types."""
    DAUBECHIES = auto()  # db
    SYMLETS = auto()     # sym
    COIFLETS = auto()    # coif
    BIORTHOGONAL = auto() # bior
    DISCRETE_MEYER = auto() # dmey
    GAUSSIAN = auto()    # gaus
    MEXICAN_HAT = auto() # mexh
    MORLET = auto()      # morl
    HAAR = auto()        # haar
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'WaveletFamily':
        """Create enum from string representation."""
        s_lower = s.lower()
        
        # Handle prefixes/shorthand
        if s_lower.startswith('db'):
            return cls.DAUBECHIES
        elif s_lower.startswith('sym'):
            return cls.SYMLETS
        elif s_lower.startswith('coif'):
            return cls.COIFLETS
        elif s_lower.startswith('bior'):
            return cls.BIORTHOGONAL
        elif s_lower == 'dmey':
            return cls.DISCRETE_MEYER
        elif s_lower.startswith('gaus'):
            return cls.GAUSSIAN
        elif s_lower in ('mexh', 'mexican hat'):
            return cls.MEXICAN_HAT
        elif s_lower in ('morl', 'morlet'):
            return cls.MORLET
        elif s_lower == 'haar':
            return cls.HAAR
            
        # Try exact name match
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
                
        # Try prefix match
        for item in cls:
            if item.name.startswith(s_upper):
                return item
                
        raise ValueError(f"Unknown WaveletFamily: {s}")
    
    def get_pywt_name(self, level: Optional[int] = None) -> str:
        """Get the PyWavelets name for this family."""
        if self == WaveletFamily.DAUBECHIES:
            return f"db{level or 4}"
        elif self == WaveletFamily.SYMLETS:
            return f"sym{level or 4}"
        elif self == WaveletFamily.COIFLETS:
            return f"coif{level or 1}"
        elif self == WaveletFamily.BIORTHOGONAL:
            return f"bior{level or '1.3'}"
        elif self == WaveletFamily.DISCRETE_MEYER:
            return "dmey"
        elif self == WaveletFamily.GAUSSIAN:
            return f"gaus{level or 1}"
        elif self == WaveletFamily.MEXICAN_HAT:
            return "mexh"
        elif self == WaveletFamily.MORLET:
            return "morl"
        elif self == WaveletFamily.HAAR:
            return "haar"
        else:
            return "db4"  # Default

class DenoiseMethod(Enum):
    """Denoising methods for wavelet processing."""
    SOFT = auto()       # Soft thresholding
    HARD = auto()       # Hard thresholding
    GARROTE = auto()    # Garrote thresholding
    UNIVERSAL = auto()  # Universal threshold selection
    BAYES = auto()      # Bayesian threshold selection
    SURE = auto()       # SURE threshold selection
    MINIMAX = auto()    # Minimax threshold selection
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'DenoiseMethod':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown DenoiseMethod: {s}")

@dataclass
class WaveletDecompResult:
    """Result of wavelet decomposition."""
    approximation: np.ndarray  # Approximation coefficients
    details: List[np.ndarray]  # Detail coefficients
    wavelet: str               # Wavelet used
    data_length: int           # Original data length
    level: int                 # Decomposition level
    mode: str                  # Border extension mode
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WaveletDenoiseResult:
    """Result of wavelet denoising."""
    original: np.ndarray      # Original signal
    denoised: np.ndarray      # Denoised signal
    noise: np.ndarray         # Extracted noise
    threshold: float          # Threshold used
    method: str               # Denoising method
    wavelet: str              # Wavelet used
    level: int                # Decomposition level
    energies: Dict[str, float] = field(default_factory=dict)  # Energy by level
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WaveletAnalysisResult:
    """Result of comprehensive wavelet analysis."""
    energy_distribution: Dict[str, float]  # Energy by level
    entropy: Dict[str, float]              # Entropy by level
    regime_indicators: Dict[str, float]    # Regime indicators
    dominant_scales: List[float]           # Dominant scales
    trend_strength: float                  # Trend strength
    cycle_periods: List[float]             # Detected cycle periods
    noise_level: float                     # Estimated noise level
    smoothness: float                      # Signal smoothness
    singularity: float                     # Singularity indicator
    self_similarity: float                 # Self-similarity measure
    features: Dict[str, float] = field(default_factory=dict)  # Extracted features
    metadata: Dict[str, Any] = field(default_factory=dict)

class WaveletProcessor:
    """
    Comprehensive wavelet processor for financial data analysis.
    
    Provides advanced signal processing capabilities using wavelet transforms,
    including denoising, multi-resolution analysis, and feature extraction.
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the wavelet processor.
        
        Args:
            hw_accelerator: Optional hardware accelerator
            config: Configuration parameters
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Initialize hardware accelerator
        self.hw_accelerator = hw_accelerator if hw_accelerator is not None else HardwareAccelerator()
        
        # Default configuration
        self.default_config = {
            # Wavelet parameters
            "default_wavelet": "db4",
            "default_level": 4,
            "default_mode": "symmetric",
            "default_transform": "dwt",
            
            # Denoising parameters
            "default_denoise_method": "soft",
            "threshold_factor": 2.0,  # Scaling factor for universal threshold
            "noise_estimation": "detail_level1",  # "mad" (median absolute deviation) or "detail_level1"
            "denoise_rescale": True,
            
            # Feature extraction
            "feature_energy_norm": True,
            "feature_norm_mode": "global",  # "global", "level", or "none"
            "extract_statistics": True,
            "extract_energy": True,
            "extract_entropy": True,
            
            # MRA parameters
            "mra_trend_level": 4,  # Level for trend extraction
            "mra_fluctuation_levels": [1, 2, 3],  # Levels for fluctuation extraction
            "mra_normalize": True,
            
            # CWT parameters
            "cwt_scales": 32,  # Number of scales for CWT
            "cwt_omega": 6.0,  # Central frequency (Morlet wavelet)
            "cwt_use_fft": True,
            "cwt_precision": "double",
            
            # Cycle detection
            "cycle_min_period": 2,
            "cycle_max_period": 256,
            "cycle_peak_threshold": 0.5,  # Relative threshold for peak detection
            "cycle_significance": 0.05,  # Significance level
            
            # Acceleration and performance
            "use_numba": True,
            "use_torch": True,
            "use_fft": True,
            "dwt_precision": "float32",
            "parallel_threshold": 1000,  # Threshold for parallel processing
            "cache_results": True,
            "cache_ttl": 3600,  # 1 hour
            "adaptive_level": True  # Automatically determine optimal level
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._lock = threading.RLock()
        self._decomp_cache = {}  # key -> (decomp, timestamp)
        self._denoise_cache = {}  # key -> (result, timestamp)
        self._feature_cache = {}  # key -> (features, timestamp)
        self._cwt_cache = {}  # key -> (cwt, frequencies, timestamp)
        
        # Check available backends
        self.has_pywavelets = PYWAVELETS_AVAILABLE
        self.has_torch = TORCH_AVAILABLE
        self.has_scipy = SCIPY_AVAILABLE
        self.has_numba = NUMBA_AVAILABLE
        self.has_kymatio = KYMATIO_AVAILABLE
        
        if not self.has_pywavelets:
            self.logger.warning("PyWavelets not available. Most features will be disabled.")
        else:
            self.logger.info("Wavelet processor initialized with PyWavelets backend")
            
            if self.has_torch and self.config["use_torch"]:
                self.logger.info("Using PyTorch acceleration for wavelet processing")
                
            if self.has_numba and self.config["use_numba"]:
                self.logger.info("Using Numba acceleration for wavelet processing")
                
    def _get_cached_decomp(self, key: Any) -> Optional[WaveletDecompResult]:
        """
        Get cached wavelet decomposition if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached decomposition or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        with self._lock:
            # Check if decomposition is in cache
            cache_entry = self._decomp_cache.get(key)
            
            if cache_entry is None:
                return None
                
            decomp, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._decomp_cache.pop(key, None)
                return None
                
            return decomp
            
    def _cache_decomp(self, key: Any, decomp: WaveletDecompResult):
        """
        Cache wavelet decomposition for future use.
        
        Args:
            key: Cache key
            decomp: Decomposition result
        """
        if not self.config["cache_results"]:
            return
            
        with self._lock:
            self._decomp_cache[key] = (decomp, time.time())
            
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
    
    # ----- Wavelet Transform Methods -----
    
    def decompose_signal(self, data: np.ndarray, wavelet: Optional[str] = None,
                      level: Optional[int] = None, mode: Optional[str] = None) -> WaveletDecompResult:
        """
        Decompose signal using discrete wavelet transform.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            level: Decomposition level (default from config)
            mode: Border extension mode (default from config)
            
        Returns:
            Wavelet decomposition result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot decompose signal")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        mode = mode or self.config["default_mode"]
        
        # Determine maximum decomposition level based on data length and wavelet
        max_level = pywt.dwt_max_level(len(data), wavelet)
        
        # Use default level if not provided, but ensure it's valid
        if level is None:
            if self.config["adaptive_level"]:
                # Automatically select a reasonable level based on data length
                level = min(self.config["default_level"], max_level)
            else:
                level = self.config["default_level"]
        
        # Ensure level is valid
        level = min(level, max_level)
        
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, level, mode)
        
        # Check cache
        cached_decomp = self._get_cached_decomp(cache_key)
        if cached_decomp is not None:
            return cached_decomp
            
        # Choose implementation based on hardware and data size
        if self.has_torch and self.config["use_torch"] and len(data) > self.config["parallel_threshold"]:
            decomp = self._decompose_signal_torch(data, wavelet, level, mode)
        else:
            decomp = self._decompose_signal_pywt(data, wavelet, level, mode)
            
        # Cache result
        self._cache_decomp(cache_key, decomp)
        
        return decomp
    
    def _decompose_signal_pywt(self, data: np.ndarray, wavelet: str,
                             level: int, mode: str) -> WaveletDecompResult:
        """
        Decompose signal using PyWavelets.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use
            level: Decomposition level
            mode: Border extension mode
            
        Returns:
            Wavelet decomposition result
        """
        try:
            # Perform multilevel decomposition
            coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
            
            # Extract approximation and detail coefficients
            approximation = coeffs[0]
            details = coeffs[1:]
            
            # Create result
            decomp = WaveletDecompResult(
                approximation=approximation,
                details=details,
                wavelet=wavelet,
                data_length=len(data),
                level=level,
                mode=mode,
                metadata={
                    "energy_approx": np.sum(approximation**2),
                    "energy_details": [np.sum(d**2) for d in details],
                    "timestamp": time.time()
                }
            )
            
            return decomp
            
        except Exception as e:
            self.logger.error(f"Error decomposing signal with PyWavelets: {e}")
            return None
    
    def _decompose_signal_torch(self, data: np.ndarray, wavelet: str,
                              level: int, mode: str) -> WaveletDecompResult:
        """
        Decompose signal using PyTorch for acceleration.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use
            level: Decomposition level
            mode: Border extension mode
            
        Returns:
            Wavelet decomposition result
        """
        if not self.has_torch:
            return self._decompose_signal_pywt(data, wavelet, level, mode)
            
        try:
            import torch
            import pywt
            
            # Get wavelet filters
            wavelet_filters = pywt.Wavelet(wavelet)
            
            # Convert to torch tensor
            device = self.hw_accelerator.get_torch_device()
            x = torch.tensor(data, dtype=torch.float32, device=device)
            
            # Get decomposition filters
            dec_lo = torch.tensor(wavelet_filters.dec_lo, dtype=torch.float32, device=device)
            dec_hi = torch.tensor(wavelet_filters.dec_hi, dtype=torch.float32, device=device)
            
            # Initialize coefficient lists
            approximation = None
            details = []
            
            # Current signal at each level
            current = x
            
            # Apply multi-level decomposition
            for i in range(level):
                # Get padding size (to match PyWavelets symmetric mode)
                padding = len(dec_lo) - 1
                pad_left = padding // 2
                pad_right = padding - pad_left
                
                # Apply padding based on mode
                if mode == 'symmetric':
                    # Symmetric padding
                    current_pad = torch.nn.functional.pad(
                        current.unsqueeze(0).unsqueeze(0),
                        (pad_left, pad_right),
                        mode='reflect'
                    ).squeeze(0).squeeze(0)
                else:
                    # Zero padding (default fallback)
                    current_pad = torch.nn.functional.pad(
                        current.unsqueeze(0).unsqueeze(0),
                        (pad_left, pad_right),
                        mode='constant'
                    ).squeeze(0).squeeze(0)
                
                # Apply lowpass filter (approximation)
                approx = torch.nn.functional.conv1d(
                    current_pad.unsqueeze(0).unsqueeze(0),
                    dec_lo.view(1, 1, -1),
                    stride=2
                ).squeeze(0).squeeze(0)
                
                # Apply highpass filter (detail)
                detail = torch.nn.functional.conv1d(
                    current_pad.unsqueeze(0).unsqueeze(0),
                    dec_hi.view(1, 1, -1),
                    stride=2
                ).squeeze(0).squeeze(0)
                
                # Store detail coefficients
                details.append(detail.cpu().numpy())
                
                # Update current signal for next level
                current = approx
                
            # Final approximation
            approximation = current.cpu().numpy()
            
            # Reverse order of details to match PyWavelets (highest level first)
            details = details[::-1]
            
            # Create result
            decomp = WaveletDecompResult(
                approximation=approximation,
                details=details,
                wavelet=wavelet,
                data_length=len(data),
                level=level,
                mode=mode,
                metadata={
                    "energy_approx": float(np.sum(approximation**2)),
                    "energy_details": [float(np.sum(d**2)) for d in details],
                    "timestamp": time.time(),
                    "implementation": "torch"
                }
            )
            
            return decomp
            
        except Exception as e:
            self.logger.error(f"Error decomposing signal with PyTorch: {e}")
            # Fallback to PyWavelets
            return self._decompose_signal_pywt(data, wavelet, level, mode)
    
    def reconstruct_signal(self, decomp: WaveletDecompResult, 
                          levels: Optional[List[int]] = None) -> np.ndarray:
        """
        Reconstruct signal from wavelet decomposition.
        
        Args:
            decomp: Wavelet decomposition result
            levels: Specific levels to include (None for all)
            
        Returns:
            Reconstructed signal
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot reconstruct signal")
            return None
            
        try:
            # Create coefficient list for reconstruction
            coeffs = [decomp.approximation] + decomp.details
            
            # Use direct PyWavelets decomposition to ensure we get the correct structure
            # This is safer than manipulating coefficients directly
            original_coeffs = pywt.wavedec(np.zeros(decomp.data_length), decomp.wavelet, 
                                        mode=decomp.mode, level=decomp.level)
            
            # Apply level filtering if specified and copy values to original structure
            filtered_coeffs = []
            
            # Handle approximation (level 0)
            if levels is None or 0 in levels:
                # Use original approximation
                approx = original_coeffs[0].copy()
                # Copy values from decomp approximation
                min_shape = np.minimum(approx.shape, decomp.approximation.shape)
                if len(min_shape) == 0:  # scalar
                    approx = decomp.approximation
                else:
                    # Create slice objects for proper indexing
                    slice_objs = tuple(slice(0, s) for s in min_shape)
                    approx[slice_objs] = decomp.approximation[slice_objs]
            else:
                # Zero out approximation
                approx = np.zeros_like(original_coeffs[0])
                
            filtered_coeffs.append(approx)
            
            # Handle detail coefficients
            for i, (orig_detail, decomp_detail) in enumerate(zip(original_coeffs[1:], decomp.details)):
                level_idx = i + 1  # Detail levels start at 1
                
                if levels is None or level_idx in levels:
                    # Use filtered detail
                    detail = orig_detail.copy()
                    # Copy values from decomp detail
                    min_shape = np.minimum(detail.shape, decomp_detail.shape)
                    if len(min_shape) == 0:  # scalar
                        detail = decomp_detail
                    else:
                        # Create slice objects for proper indexing
                        slice_objs = tuple(slice(0, s) for s in min_shape)
                        detail[slice_objs] = decomp_detail[slice_objs]
                else:
                    # Zero out detail
                    detail = np.zeros_like(orig_detail)
                    
                filtered_coeffs.append(detail)
                
            # Add remaining original coefficients if needed
            while len(filtered_coeffs) < len(original_coeffs):
                filtered_coeffs.append(np.zeros_like(original_coeffs[len(filtered_coeffs)]))
                
            # Reconstruct signal with guaranteed compatible coefficients
            reconstructed = pywt.waverec(filtered_coeffs, decomp.wavelet, mode=decomp.mode)
            
            # Trim to original length if needed
            if len(reconstructed) > decomp.data_length:
                reconstructed = reconstructed[:decomp.data_length]
                
            return reconstructed
            
        except Exception as e:
            self.logger.error(f"Error reconstructing signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


    
    def denoise_signal(self, data: np.ndarray, wavelet: Optional[str] = None,
                     level: Optional[int] = None, method: Optional[str] = None,
                     threshold: Optional[float] = None) -> WaveletDenoiseResult:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            level: Decomposition level (default from config)
            method: Denoising method (default from config)
            threshold: Threshold value (None for automatic)
            
        Returns:
            Denoising result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot denoise signal")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        method = method or self.config["default_denoise_method"]
        
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, level, method, threshold)
        
        # Check cache
        cached_result = self._get_cached_result(self._denoise_cache, cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Determine maximum level
            if level is None:
                level = pywt.dwt_max_level(len(data), wavelet)
                level = min(level, self.config["default_level"])
            
            # Decompose signal directly with PyWavelets to ensure we get the correct structure
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            # Set automatic threshold if not provided
            if threshold is None:
                # Determine noise level from first detail level coefficients (highest frequency noise)
                detail_coeffs = coeffs[1]  # First detail level
                
                if self.config["noise_estimation"] == "mad":
                    # Median Absolute Deviation (robust estimator)
                    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
                    noise_sigma = mad / 0.6745  # Scaling factor for Gaussian noise
                else:
                    # Standard deviation of first level details
                    noise_sigma = np.std(detail_coeffs)
                    
                # Universal threshold with scaling factor
                n = len(data)
                threshold = self.config["threshold_factor"] * noise_sigma * np.sqrt(2 * np.log(n))
            
            # Map our method names to PyWavelets method names
            if method.lower() in ('soft', 'soft_threshold'):
                pywt_method = 'soft'
            elif method.lower() in ('hard', 'hard_threshold'):
                pywt_method = 'hard'
            elif method.lower() in ('garrote', 'non_negative_garrote'):
                pywt_method = 'garotte'  # PyWavelets spelling
            else:
                pywt_method = 'soft'  # Default
                
            # Apply thresholding to coefficients
            # Start from index 1 to keep approximation coefficients unchanged
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold, pywt_method)
                
            # Reconstruct denoised signal
            denoised = pywt.waverec(coeffs, wavelet)
            
            # Trim to original length if needed
            if len(denoised) > len(data):
                denoised = denoised[:len(data)]
                
            # Calculate extracted noise
            noise = data - denoised
            
            # Calculate energy by level
            energy_approx = np.sum(coeffs[0]**2)
            energy_details = [np.sum(coeffs[i+1]**2) for i in range(level)]
            total_energy = energy_approx + sum(energy_details)
            
            # Relative energy distribution
            energies = {
                "approximation": energy_approx / total_energy if total_energy > 0 else 0.0
            }
            
            for i in range(level):
                energies[f"detail_{i+1}"] = energy_details[i] / total_energy if total_energy > 0 else 0.0
                
            # Create result
            result = WaveletDenoiseResult(
                original=data,
                denoised=denoised,
                noise=noise,
                threshold=threshold,
                method=method,
                wavelet=wavelet,
                level=level,
                energies=energies,
                metadata={
                    "noise_sigma": noise_sigma if 'noise_sigma' in locals() else None,
                    "noise_energy": np.sum(noise**2),
                    "denoised_energy": np.sum(denoised**2),
                    "original_energy": np.sum(data**2),
                    "timestamp": time.time()
                }
            )
            
            # Cache result
            self._cache_result(self._denoise_cache, cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in denoise_signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
 
    def extract_wavelet_features(self, data: np.ndarray, wavelet: Optional[str] = None,
                              level: Optional[int] = None) -> Dict[str, float]:
        """
        Extract features from wavelet decomposition.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            level: Decomposition level (default from config)
            
        Returns:
            Dictionary of features
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot extract features")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        # Create cache key
        cache_key = (hash(data.tobytes()), wavelet, level)
        
        # Check cache
        cached_result = self._get_cached_result(self._feature_cache, cache_key)
        if cached_result is not None:
            return cached_result
            
        # Decompose signal
        decomp = self.decompose_signal(data, wavelet, level)
        
        if decomp is None:
            self.logger.error("Failed to decompose signal for feature extraction")
            return {}
            
        # Initialize features dictionary
        features = {}
        
        # Calculate energy features
        if self.config["extract_energy"]:
            # Energy of approximation
            energy_approx = np.sum(decomp.approximation**2)
            features["energy_approx"] = energy_approx
            
            # Energy of details by level
            for i, detail in enumerate(decomp.details):
                features[f"energy_detail_{i+1}"] = np.sum(detail**2)
                
            # Total energy
            total_energy = energy_approx + sum(features[f"energy_detail_{i+1}"] for i in range(len(decomp.details)))
            features["total_energy"] = total_energy
            
            # Relative energy distribution
            if self.config["feature_energy_norm"] and total_energy > 0:
                features["rel_energy_approx"] = energy_approx / total_energy
                
                for i in range(len(decomp.details)):
                    features[f"rel_energy_detail_{i+1}"] = features[f"energy_detail_{i+1}"] / total_energy
                    
        # Calculate entropy features
        if self.config["extract_entropy"] and self.has_scipy:
            from scipy.stats import entropy
            
            # Entropy of approximation coefficients
            approx_abs = np.abs(decomp.approximation)
            if np.sum(approx_abs) > 0:
                p_approx = approx_abs / np.sum(approx_abs)
                features["entropy_approx"] = entropy(p_approx)
            else:
                features["entropy_approx"] = 0.0
                
            # Entropy of detail coefficients by level
            for i, detail in enumerate(decomp.details):
                detail_abs = np.abs(detail)
                if np.sum(detail_abs) > 0:
                    p_detail = detail_abs / np.sum(detail_abs)
                    features[f"entropy_detail_{i+1}"] = entropy(p_detail)
                else:
                    features[f"entropy_detail_{i+1}"] = 0.0
                    
            # Shannon entropy diversity
            entropy_values = [features["entropy_approx"]] + [features[f"entropy_detail_{i+1}"] for i in range(len(decomp.details))]
            features["entropy_diversity"] = np.std(entropy_values) / np.mean(entropy_values) if np.mean(entropy_values) > 0 else 0.0
            
        # Calculate statistical features
        if self.config["extract_statistics"]:
            # Statistics for approximation
            features["mean_approx"] = np.mean(decomp.approximation)
            features["std_approx"] = np.std(decomp.approximation)
            
            if self.has_scipy:
                features["skew_approx"] = skew(decomp.approximation)
                features["kurtosis_approx"] = kurtosis(decomp.approximation)
                
            # Statistics for details by level
            for i, detail in enumerate(decomp.details):
                features[f"mean_detail_{i+1}"] = np.mean(detail)
                features[f"std_detail_{i+1}"] = np.std(detail)
                
                if self.has_scipy:
                    features[f"skew_detail_{i+1}"] = skew(detail)
                    features[f"kurtosis_detail_{i+1}"] = kurtosis(detail)
                    
            # Ratio of standard deviations between levels
            for i in range(len(decomp.details) - 1):
                if features[f"std_detail_{i+2}"] > 0:
                    features[f"std_ratio_{i+1}_{i+2}"] = features[f"std_detail_{i+1}"] / features[f"std_detail_{i+2}"]
                else:
                    features[f"std_ratio_{i+1}_{i+2}"] = 0.0
                    
        # Cross-scale features (correlations between levels)
        for i in range(len(decomp.details) - 1):
            # Ensure the vectors have the same length for correlation
            min_len = min(len(decomp.details[i]), len(decomp.details[i+1]))
            d1 = decomp.details[i][:min_len]
            d2 = decomp.details[i+1][:min_len]
            
            # Calculate cross-correlation
            if self.has_scipy and len(d1) > 1 and len(d2) > 1:
                from scipy.signal import correlate
                
                # Normalize vectors
                d1_norm = (d1 - np.mean(d1)) / (np.std(d1) + 1e-8)
                d2_norm = (d2 - np.mean(d2)) / (np.std(d2) + 1e-8)
                
                # Calculate cross-correlation at zero lag
                correlation = np.correlate(d1_norm, d2_norm, mode='valid')[0] / min_len
                features[f"cross_corr_{i+1}_{i+2}"] = correlation
            else:
                features[f"cross_corr_{i+1}_{i+2}"] = 0.0
                
        # Feature normalization if enabled
        if self.config["feature_norm_mode"] == "global":
            # Global min-max normalization
            feature_values = list(features.values())
            if feature_values:
                value_min = min(feature_values)
                value_max = max(feature_values)
                
                if value_max > value_min:
                    # Apply min-max scaling
                    norm_features = {
                        k: (v - value_min) / (value_max - value_min)
                        for k, v in features.items()
                    }
                    features = norm_features
                    
        elif self.config["feature_norm_mode"] == "level":
            # Normalize within feature types
            for feature_type in ["energy", "entropy", "mean", "std", "skew", "kurtosis", "cross_corr"]:
                # Get features of this type
                type_features = {k: v for k, v in features.items() if k.startswith(feature_type)}
                
                if type_features:
                    type_values = list(type_features.values())
                    type_min = min(type_values)
                    type_max = max(type_values)
                    
                    if type_max > type_min:
                        # Apply min-max scaling within this feature type
                        for k, v in type_features.items():
                            features[k] = (v - type_min) / (type_max - type_min)
                            
        # Cache result
        self._cache_result(self._feature_cache, cache_key, features)
        
        return features
    
    def continuous_wavelet_transform(self, data: np.ndarray, scales: Optional[Union[int, np.ndarray]] = None,
                                   wavelet: Optional[str] = None, sampling_period: float = 1.0,
                                   use_fft: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform continuous wavelet transform.
        
        Args:
            data: Input signal
            scales: Scales for CWT (int for automatic generation, or array)
            wavelet: Wavelet to use (default from config)
            sampling_period: Sampling period of the signal
            use_fft: Whether to use FFT (default from config)
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot perform CWT")
            return None, None
            
        # Get defaults from config if not provided
        wavelet = wavelet or "morl"  # Morlet wavelet is better for CWT
        use_fft = use_fft if use_fft is not None else self.config["cwt_use_fft"]
        
        # Handle scales
        if scales is None:
            # Generate scales (power of 2 for better performance)
            scales = 2 ** np.linspace(1, np.log2(min(len(data) // 2, self.config["cwt_scales"])), 
                                   self.config["cwt_scales"])
        elif isinstance(scales, int):
            # Generate 'scales' number of scales
            scales = 2 ** np.linspace(1, np.log2(min(len(data) // 2, scales)), scales)
            
        # Create cache key
        cache_key = (hash(data.tobytes()), str(scales), wavelet, sampling_period, use_fft)
        
        # Check cache
        cached_result = self._get_cached_result(self._cwt_cache, cache_key)
        if cached_result is not None:
            return cached_result
            
        try:
            # Choose implementation based on hardware and data size
            if self.has_torch and self.config["use_torch"] and len(data) > self.config["parallel_threshold"]:
                coefs, frequencies = self._cwt_torch(data, scales, wavelet, sampling_period)
            else:
                # Use PyWavelets implementation with compatibility handling
                coefs, frequencies = self._cwt_compatible(data, scales, wavelet, sampling_period, use_fft)
                    
            # Cache result
            self._cache_result(self._cwt_cache, cache_key, (coefs, frequencies))
            
            return coefs, frequencies
            
        except Exception as e:
            self.logger.error(f"Error performing CWT: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def _cwt_compatible(self, data: np.ndarray, scales: np.ndarray, wavelet: str,
                       sampling_period: float, use_fft: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modern PyWavelets CWT implementation using proper ContinuousWavelet objects.
        This fixes the complex_cwt attribute error in PyWavelets 1.8.0+ by using
        the correct API: ContinuousWavelet objects instead of string names.
        Also handles discrete vs continuous wavelet compatibility.
        """
        try:
            # Check if wavelet is suitable for CWT (must be continuous)
            if isinstance(wavelet, str):
                # Map discrete wavelets to appropriate continuous wavelets for CWT
                continuous_wavelet_map = {
                    'db4': 'morl',      # Daubechies -> Morlet
                    'db6': 'morl',      # Daubechies -> Morlet
                    'db8': 'morl',      # Daubechies -> Morlet
                    'haar': 'mexh',     # Haar -> Mexican Hat
                    'sym4': 'morl',     # Symlet -> Morlet
                    'sym6': 'morl',     # Symlet -> Morlet
                    'coif2': 'morl',    # Coiflet -> Morlet
                    'coif4': 'morl',    # Coiflet -> Morlet
                    'bior2.2': 'morl',  # Biorthogonal -> Morlet
                    'bior4.4': 'morl'   # Biorthogonal -> Morlet
                }
                
                # Check if we need to map discrete to continuous
                if wavelet in continuous_wavelet_map:
                    mapped_wavelet = continuous_wavelet_map[wavelet]
                    self.logger.debug(f"Mapped discrete wavelet '{wavelet}' to continuous wavelet '{mapped_wavelet}' for CWT")
                    wavelet = mapped_wavelet
                
                # Try to create ContinuousWavelet object
                try:
                    wavelet_obj = pywt.ContinuousWavelet(wavelet)
                except ValueError as e:
                    if "is not a continuous wavelet" in str(e) or "Invalid wavelet name" in str(e):
                        # Fallback to Morlet for any unmapped wavelets
                        self.logger.warning(f"Wavelet '{wavelet}' not suitable for CWT, using Morlet instead")
                        wavelet_obj = pywt.ContinuousWavelet('morl')
                    else:
                        raise
            else:
                wavelet_obj = wavelet
            
            # Use modern PyWavelets 1.8+ API with proper method parameter
            method = 'fft' if use_fft else 'conv'
            
            # Call CWT with proper ContinuousWavelet object and method parameter
            coefs, frequencies = pywt.cwt(data, scales, wavelet_obj, sampling_period, method=method)
            
            return coefs, frequencies
            
        except Exception as e:
            # If there's still an error, log it and re-raise for proper debugging
            self.logger.error(f"Modern CWT implementation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _cwt_torch(self, data: np.ndarray, scales: np.ndarray, wavelet: str,
                  sampling_period: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform CWT using PyTorch for acceleration.
        
        Args:
            data: Input signal
            scales: Scales for CWT
            wavelet: Wavelet to use
            sampling_period: Sampling period of the signal
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        # First, ensure we have torch available at the method level
        if not self.has_torch:
            # Fallback to numpy implementation if torch is not available
            return self._cwt_numpy(data, scales, wavelet, sampling_period)
        
        try:
            # Import torch locally in a way that ensures it's available throughout the method
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.fft import rfft, irfft
            
            # Get device from hardware accelerator
            device = self.hw_accelerator.get_torch_device()
            
            # Get parameters for central frequency based on wavelet type
            central_frequency = None
            if wavelet == 'morl':
                # Morlet wavelet central frequency (default PyWavelets value)
                central_frequency = 5.0
            elif wavelet == 'mexh':
                # Mexican hat central frequency
                central_frequency = 2.0
            elif wavelet.startswith('cmor'):
                # Complex Morlet - extract parameters
                try:
                    # Parse complex Morlet parameters (bandwidth-center)
                    parts = wavelet.split('-')
                    if len(parts) >= 3:
                        center = float(parts[2])
                        central_frequency = center
                    else:
                        central_frequency = 5.0
                except:
                    central_frequency = 5.0
            elif wavelet.startswith('gaus'):
                # Gaussian - extract order
                try:
                    # Parse Gaussian parameters
                    parts = wavelet.split('-')
                    if len(parts) >= 2:
                        order = int(parts[1])
                        central_frequency = 1.0  # Approximation for Gaussian
                    else:
                        central_frequency = 1.0
                except:
                    central_frequency = 1.0
            else:
                # Default for unknown wavelets
                central_frequency = self.config.get("cwt_omega", 6.0)
                
            # Convert to torch tensors
            if isinstance(data, torch.Tensor):
                data_tensor = data.to(device)
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
                
            scales_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
            
            # Get signal length
            n = len(data_tensor)
            
            # Power of 2 for efficiency in FFT
            n_power = 2 ** int(np.ceil(np.log2(n)))
            
            # Pad signal (zero padding for FFT)
            padded_data = torch.zeros(n_power, dtype=torch.float32, device=device)
            padded_data[:n] = data_tensor
            
            # Compute Fourier transform of the signal
            fft_data = rfft(padded_data)
            
            # Create frequency array (positive frequencies only due to rfft)
            frequencies = torch.fft.rfftfreq(n_power, sampling_period, device=device)
            
            # Calculate Fourier transform for each scale
            scale_wavelets = []
            
            for scale in scales_tensor:
                # Create wavelet in frequency domain for this scale
                if wavelet == 'morl':
                    # Morlet wavelet in frequency domain
                    omega = central_frequency
                    wavelet_f = (torch.sqrt(torch.tensor(2 * np.pi * scale, device=device)) * 
                              torch.exp(-(scale * 2*np.pi*frequencies - omega)**2 / 2))
                elif wavelet == 'mexh':
                    # Mexican hat wavelet in frequency domain (second derivative of Gaussian)
                    norm = 2 * np.sqrt(3) * np.pi**0.25 / np.sqrt(2)
                    wavelet_f = norm * (2*np.pi*frequencies*scale)**2 * torch.exp(-(2*np.pi*frequencies*scale)**2 / 2)
                else:
                    # Default: Morlet wavelet
                    omega = central_frequency
                    wavelet_f = (torch.sqrt(torch.tensor(2 * np.pi * scale, device=device)) * 
                              torch.exp(-(scale * 2*np.pi*frequencies - omega)**2 / 2))
                    
                scale_wavelets.append(wavelet_f)
                
            # Stack wavelets for batch processing
            wavelet_f_batch = torch.stack(scale_wavelets)
            
            # Compute convolution in frequency domain
            convolution_f = wavelet_f_batch * fft_data.unsqueeze(0)
            
            # Initialize output for inverse transform
            coefs_real = torch.zeros((len(scales), n_power), dtype=torch.float32, device=device)
            
            # Inverse transform row by row to get CWT coefficients
            for i in range(len(scales)):
                coefs_real[i] = irfft(convolution_f[i], n=n_power)
            
            # Extract relevant part (original signal length)
            coefs = coefs_real[:, :n].cpu().numpy()
            
            # Calculate actual frequencies
            frequencies_out = (central_frequency / (scales * sampling_period))
            if isinstance(frequencies_out, torch.Tensor):
                frequencies_out = frequencies_out.cpu().numpy()
            
            return coefs, frequencies_out
            
        except ImportError as ie:
            self.logger.error(f"Import error in PyTorch CWT implementation: {ie}")
            # Fall back to NumPy implementation
            return self._cwt_numpy(data, scales, wavelet, sampling_period)
        except Exception as e:
            self.logger.error(f"Error in PyTorch CWT implementation: {e}")
            # Fall back to NumPy implementation
            converted_data = data.cpu().numpy() if hasattr(data, 'cpu') else data
            return self._cwt_numpy(converted_data, scales, wavelet, sampling_period)

    def _cwt_numpy(self, data: np.ndarray, widths: np.ndarray, 
                  wavelet: str = 'morl', sampling_period: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Continuous Wavelet Transform using NumPy/SciPy.
        
        This is the fallback implementation when PyTorch isn't available.
        It uses NumPy/SciPy for the computation, which runs on CPU only.
        
        Args:
            data: Input signal (1D array)
            widths: Array of scale values for the CWT
            wavelet: Wavelet to use (default: 'morl' - Morlet)
            sampling_period: Sampling period of the signal
            
        Returns:
            Tuple of (wavelet_matrix, frequencies) where:
            - wavelet_matrix: Complex-valued CWT coefficients [scales x time]
            - frequencies: Corresponding frequencies for each scale
        """
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is 1D
        if data.ndim > 1:
            data = data.squeeze()
        
        if data.ndim != 1:
            raise ValueError(f"Expected 1D input, got {data.ndim}D array")
        
        # Try to use scipy's cwt implementation
        try:
            from scipy import signal
            
            # Create wavelet function
            if wavelet.lower() == 'morl':
                # Morlet wavelet
                w = 6.0  # Parameter for Morlet wavelet
                wavelet_func = signal.morlet2(widths[0] * 10, s=widths[0], w=w)
                
                # Perform CWT
                cwt_matrix, frequencies = signal.cwt(data, signal.morlet2, widths, w=w)
            else:
                # Try other wavelets if available
                cwt_matrix, frequencies = signal.cwt(data, getattr(signal, wavelet), widths)
                
        except (ImportError, AttributeError):
            # Fallback to manual implementation
            self.logger.warning("SciPy not available, using manual CWT implementation")
            
            n_widths = len(widths)
            n_samples = len(data)
            cwt_matrix = np.zeros((n_widths, n_samples), dtype=np.complex128)
            
            # Use FFT for efficiency
            pad_len = n_samples + int(widths.max() * 10)
            data_fft = np.fft.fft(data, n=pad_len)
            
            for i, width in enumerate(widths):
                # Create time vector for wavelet
                t = np.arange(-width * 5, width * 5 + 1)
                
                # Generate Morlet wavelet
                omega0 = 6.0
                wavelet_func = np.exp(1j * omega0 * t / width) * np.exp(-0.5 * (t / width) ** 2)
                wavelet_func = wavelet_func / np.sqrt(width)
                
                # Pad wavelet
                wavelet_pad = np.zeros(pad_len, dtype=np.complex128)
                center = pad_len // 2
                w_len = len(wavelet_func)
                start = center - w_len // 2
                wavelet_pad[start:start + w_len] = wavelet_func
                
                # FFT of wavelet
                wavelet_fft = np.fft.fft(wavelet_pad)
                
                # Convolution
                conv_result = np.fft.ifft(data_fft * np.conj(wavelet_fft))
                cwt_matrix[i, :] = conv_result[:n_samples]
            
            # Calculate frequencies
            frequencies = 0.8125 / (widths * sampling_period)  # Approximate for Morlet
        
        return cwt_matrix, frequencies
                
    def cwt(self, data: Union[np.ndarray, torch.Tensor], widths: Optional[np.ndarray] = None,
            wavelet: str = 'morl', sampling_period: float = 1.0, 
            use_gpu: bool = None) -> Dict[str, np.ndarray]:
        """
        Perform Continuous Wavelet Transform with automatic GPU acceleration.
        
        This method automatically selects between GPU-accelerated (PyTorch) or 
        CPU-based (NumPy/SciPy) implementation based on availability and preference.
        
        Args:
            data: Input signal
            widths: Array of scale values (default: automatic selection)
            wavelet: Wavelet to use (default: 'morl')
            sampling_period: Sampling period
            use_gpu: Force GPU/CPU usage (default: auto-detect)
            
        Returns:
            Dictionary containing:
            - 'coefficients': CWT coefficients
            - 'frequencies': Corresponding frequencies
            - 'widths': Scale values used
            - 'wavelet': Wavelet used
        """
        try:
            # Determine whether to use GPU
            if use_gpu is None:
                use_gpu = self.config.get('use_gpu', True) and self._check_gpu_available()
            
            # Set default widths if not provided
            if widths is None:
                n_samples = len(data)
                widths = self._get_optimal_widths(n_samples)
            
            # Choose implementation
            if use_gpu:
                try:
                    cwt_matrix, frequencies = self._cwt_torch(data, widths, wavelet, sampling_period)
                    if isinstance(cwt_matrix, torch.Tensor):
                        cwt_matrix = cwt_matrix.detach().cpu().numpy()
                except (ImportError, RuntimeError) as e:
                    self.logger.warning(f"Error in PyTorch CWT implementation: {e}")
                    self.logger.info("Falling back to NumPy implementation")
                    cwt_matrix, frequencies = self._cwt_numpy(data, widths, wavelet, sampling_period)
            else:
                cwt_matrix, frequencies = self._cwt_numpy(data, widths, wavelet, sampling_period)
            
            return {
                'coefficients': cwt_matrix,
                'frequencies': frequencies,
                'widths': widths,
                'wavelet': wavelet,
                'sampling_period': sampling_period
            }
            
        except Exception as e:
            self.logger.error(f"Error performing CWT: {e}")
            raise
        
    
    def detect_cycles(self, data: np.ndarray, min_period: Optional[int] = None,
                   max_period: Optional[int] = None, wavelet: Optional[str] = None,
                   peak_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect cycles in time series using wavelet transform.
        
        Args:
            data: Input signal
            min_period: Minimum period to detect (default from config)
            max_period: Maximum period to detect (default from config)
            wavelet: Wavelet to use (default: 'morl')
            peak_threshold: Relative threshold for peak detection (default from config)
            
        Returns:
            Dictionary with cycle detection results
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot detect cycles")
            return {}
            
        # Get defaults from config if not provided
        min_period = min_period or self.config["cycle_min_period"]
        max_period = max_period or self.config["cycle_max_period"]
        wavelet = wavelet or "morl"  # Morlet wavelet is better for cycle detection
        peak_threshold = peak_threshold or self.config["cycle_peak_threshold"]
        
        try:
            # Create scales based on periods
            # For Morlet wavelet with central frequency 5, scale ≈ period/5
            central_freq = 5.0
            scales = np.arange(min_period, min(max_period, len(data) // 2)) / central_freq
            
            # Perform CWT
            coeffs, freq = self.continuous_wavelet_transform(data, scales, wavelet)
            
            if coeffs is None:
                self.logger.error("Failed to perform CWT for cycle detection")
                return {}
                
            # Calculate power spectrum (absolute value squared)
            power = np.abs(coeffs)**2
            
            # Get periods from frequencies
            periods = 1 / freq
            
            # Average power across time
            avg_power = np.mean(power, axis=1)
            
            # Find peaks in average power
            if self.has_scipy:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(avg_power, height=peak_threshold * np.max(avg_power))
            else:
                # Simple peak detection
                peaks = []
                for i in range(1, len(avg_power) - 1):
                    if avg_power[i] > avg_power[i-1] and avg_power[i] > avg_power[i+1]:
                        if avg_power[i] > peak_threshold * np.max(avg_power):
                            peaks.append(i)
                            
            # Extract cycle periods
            cycle_periods = periods[peaks]
            cycle_powers = avg_power[peaks]
            
            # Sort by power (strongest first)
            idx = np.argsort(cycle_powers)[::-1]
            cycle_periods = cycle_periods[idx]
            cycle_powers = cycle_powers[idx]
            
            # Create result
            result = {
                "periods": cycle_periods.tolist(),
                "powers": cycle_powers.tolist(),
                "dominant_period": float(cycle_periods[0]) if len(cycle_periods) > 0 else None,
                "dominant_power": float(cycle_powers[0]) if len(cycle_powers) > 0 else None,
                "power_spectrum": {
                    "periods": periods.tolist(),
                    "power": avg_power.tolist()
                },
                "wavelet": wavelet,
                "wavelet_coeffs": {
                    "periods": periods.tolist(),
                    "time": list(range(len(data))),
                    "power": power.tolist()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting cycles: {e}")
            return {}
    
    # ----- Multi-Resolution Analysis (MRA) Methods -----
    

    def multi_resolution_analysis(self, data: np.ndarray, wavelet: Optional[str] = None,
                               max_level: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform multi-resolution analysis on time series.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            max_level: Maximum decomposition level
            
        Returns:
            Dictionary with MRA components
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot perform MRA")
            return {}
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        try:
            # Determine maximum level
            if max_level is None:
                max_level = pywt.dwt_max_level(len(data), wavelet)
                
            # Perform wavelet decomposition using direct PyWavelets approach
            coeffs = pywt.wavedec(data, wavelet, level=max_level)
            
            # Initialize result dictionary
            mra = {
                "original": data,
                "approximation": None,
                "details": {}
            }
            
            # Extract approximation (smooth component)
            # Create a copy of coefficients with zeros for details
            approx_coeffs = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
            mra["approximation"] = pywt.waverec(approx_coeffs, wavelet)[:len(data)]
            
            # Extract detail (fluctuation) components at each level
            for i in range(len(coeffs) - 1):
                level = i + 1
                
                # Create coefficient list with zeros except for this level
                detail_coeffs = [np.zeros_like(coeffs[0])]
                for j in range(len(coeffs) - 1):
                    if j == i:
                        detail_coeffs.append(coeffs[j+1])
                    else:
                        detail_coeffs.append(np.zeros_like(coeffs[j+1]))
                        
                # Reconstruct detail component
                detail = pywt.waverec(detail_coeffs, wavelet)[:len(data)]
                mra["details"][f"level_{level}"] = detail
                
            return mra
            
        except Exception as e:
            self.logger.error(f"Error performing MRA: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def extract_trend(self, data: np.ndarray, level: Optional[int] = None,
                   wavelet: Optional[str] = None) -> np.ndarray:
        """
        Extract trend component from time series.
        
        Args:
            data: Input signal
            level: Decomposition level for trend (default from config)
            wavelet: Wavelet to use (default from config)
            
        Returns:
            Trend component
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot extract trend")
            return None
            
        # Get defaults from config if not provided
        level = level or self.config["mra_trend_level"]
        wavelet = wavelet or self.config["default_wavelet"]
        
        try:
            # Directly use PyWavelets for decomposition and reconstruction
            # This ensures coefficient compatibility
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            # Extract trend using approximation coefficients only
            trend_coeffs = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
            
            # Reconstruct trend
            trend = pywt.waverec(trend_coeffs, wavelet)
            
            # Trim to original length if needed
            if len(trend) > len(data):
                trend = trend[:len(data)]
                
            return trend
            
        except Exception as e:
            self.logger.error(f"Error extracting trend: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_fluctuations(self, data: np.ndarray, levels: Optional[List[int]] = None,
                          wavelet: Optional[str] = None) -> np.ndarray:
        """
        Extract fluctuation components from time series.
        
        Args:
            data: Input signal
            levels: Decomposition levels for fluctuations (default from config)
            wavelet: Wavelet to use (default from config)
            
        Returns:
            Fluctuation component
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot extract fluctuations")
            return None
            
        # Get defaults from config if not provided
        levels = levels or self.config["mra_fluctuation_levels"]
        wavelet = wavelet or self.config["default_wavelet"]
        
        try:
            # Get maximum possible level
            max_level = pywt.dwt_max_level(len(data), wavelet)
            
            # Ensure levels are valid
            actual_levels = [l for l in levels if l <= max_level]
            
            if not actual_levels:
                self.logger.error(f"No valid levels provided (max possible: {max_level})")
                return None
                
            # Decompose signal to maximum requested level
            max_requested = max(actual_levels)
            coeffs = pywt.wavedec(data, wavelet, level=max_requested)
            
            # Create coefficients for reconstruction (zero out approximation and unwanted details)
            fluct_coeffs = [np.zeros_like(coeffs[0])]
            
            for i in range(len(coeffs) - 1):
                level_idx = i + 1  # Detail levels start at 1
                if level_idx in actual_levels:
                    fluct_coeffs.append(coeffs[i+1])
                else:
                    fluct_coeffs.append(np.zeros_like(coeffs[i+1]))
                    
            # Reconstruct fluctuations
            fluctuations = pywt.waverec(fluct_coeffs, wavelet)
            
            # Trim to original length if needed
            if len(fluctuations) > len(data):
                fluctuations = fluctuations[:len(data)]
                
            return fluctuations
            
        except Exception as e:
            self.logger.error(f"Error extracting fluctuations: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    

    
    def analyze_market_regime(self, data: np.ndarray, wavelet: Optional[str] = None,
                           level: Optional[int] = None) -> WaveletAnalysisResult:
        """
        Analyze market regime using wavelet-based indicators.
        
        Args:
            data: Price or return time series
            wavelet: Wavelet to use (default from config)
            level: Decomposition level (default from config)
            
        Returns:
            Market regime analysis result
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze market regime")
            return None
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["default_wavelet"]
        
        try:
            # Extract wavelet features
            features = self.extract_wavelet_features(data, wavelet, level)
            
            if not features:
                self.logger.error("Failed to extract wavelet features for regime analysis")
                return None
                
            # Calculate wavelet-based regime indicators
            
            # 1. Energy distribution across scales
            energy_distribution = {k: v for k, v in features.items() if k.startswith("rel_energy_")}
            
            # 2. Entropy across scales
            entropy = {k: v for k, v in features.items() if k.startswith("entropy_")}
            
            # 3. Determine dominant scales
            energy_levels = [(int(k.split('_')[-1]), v) for k, v in energy_distribution.items() 
                          if k.startswith("rel_energy_detail_")]
            
            # Sort by energy (highest first)
            energy_levels.sort(key=lambda x: x[1], reverse=True)
            dominant_scales = [level for level, _ in energy_levels[:3]]
            
            # 4. Trend strength - energy in approximation vs. details
            trend_strength = features.get("rel_energy_approx", 0.0)
            
            # 5. Detect cycles using CWT
            cycles = self.detect_cycles(data, wavelet=wavelet)
            cycle_periods = cycles.get("periods", [])
            
            # 6. Estimate noise level using highest frequency detail coefficients
            noise_level = features.get("rel_energy_detail_1", 0.0)
            
            # 7. Estimate smoothness using kurtosis of detail coefficients
            kurtosis_values = [v for k, v in features.items() if k.startswith("kurtosis_detail_")]
            smoothness = 1.0 / (1.0 + np.mean(kurtosis_values)) if kurtosis_values else 0.0
            
            # 8. Singularity detection using local maxima of wavelet coefficients
            # High singularity indicates potential regime change points
            coefs, _ = self.continuous_wavelet_transform(data, wavelet=wavelet)
            if coefs is not None:
                singularity = np.max(np.abs(coefs[-1, :])) / np.mean(np.abs(coefs[-1, :]))
            else:
                singularity = 0.0
                
            # 9. Estimate self-similarity (Hurst exponent)
            # Using log-log plot of variance of details vs. scale
            log_variance = [np.log(features.get(f"std_detail_{i+1}", 1.0)**2) for i in range(len(kurtosis_values))]
            log_scale = [np.log(2**(i+1)) for i in range(len(log_variance))]
            
            if len(log_variance) > 1 and len(log_scale) > 1:
                # Linear regression slope
                coef = np.polyfit(log_scale, log_variance, 1)[0]
                
                # Hurst = (slope + 1) / 2
                self_similarity = (coef + 1) / 2
            else:
                self_similarity = 0.5  # Default: random walk
                
            # Create regime indicators
            trend_indicator = features.get("rel_energy_approx", 0.0)
            mean_period = np.mean(cycle_periods) if cycle_periods else 0.0
            
            regime_indicators = {
                "trend": trend_indicator,
                "noise": noise_level,
                "smoothness": smoothness,
                "cyclical": 1.0 - (noise_level + trend_indicator),
                "mean_period": mean_period,
                "self_similarity": self_similarity,
                "singularity": singularity
            }
            
            # Create analysis result
            result = WaveletAnalysisResult(
                energy_distribution=energy_distribution,
                entropy=entropy,
                regime_indicators=regime_indicators,
                dominant_scales=dominant_scales,
                trend_strength=trend_strength,
                cycle_periods=cycle_periods,
                noise_level=noise_level,
                smoothness=smoothness,
                singularity=singularity,
                self_similarity=self_similarity,
                features=features,
                metadata={
                    "wavelet": wavelet,
                    "data_length": len(data),
                    "timestamp": time.time()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return None

# ----- Financial  Methods -----
    
    def analyze_financial_series(self, data: Union[np.ndarray, pd.DataFrame], include_volume: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive financial analysis using wavelets.
        
        Args:
            data: Price data (array or DataFrame with 'close' column)
            include_volume: Whether to include volume-based analysis
            
        Returns:
            Dict with financial analysis results
        """
        # Preprocess data
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrame must contain 'close' column")
            series = data['close'].values
            volume = data['volume'].values if include_volume and 'volume' in data.columns else None
        else:
            series = data
            volume = None
        
        # Perform MRA
        mra_result = self.perform_mra(series)
        
        # Detect market regime
        energy_dist = self._calculate_energy_distribution(mra_result["coefficients"])
        regime, regime_numeric = self._detect_regime_from_energy(energy_dist)
        
        # Analyze cycles
        cycles = self._analyze_cycles_with_cwt(series)
        
        # Calculate wavelet variance
        variance = self._calculate_wavelet_variance(series)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(series, mra_result["approximation"])
        
        # Calculate momentum
        momentum = self._calculate_wavelet_momentum(series)
        
        # Add volume correlation if available
        volume_correlation = {}
        if volume is not None:
            volume_correlation = self._calculate_wavelet_correlation(series, volume)
        
        # Combine results
        results = {
            "mra": mra_result,
            "regime": regime,
            "cycles": cycles,
            "variance": variance,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "volume_correlation": volume_correlation
        }
        
        return results
    
    def _detect_regime_from_energy(self, energy_distribution):
        """
        Detect market regime based on wavelet energy distribution.
        
        Args:
            energy_distribution: Energy distribution across levels
            
        Returns:
            tuple: (regime_label, regime_numeric)
        """
        # Extract key metrics
        approx_energy = energy_distribution.get('approximation', 0)
        high_freq_energy = sum(energy_distribution.get(f'detail_{i+1}', 0) 
                              for i in range(2))  # First 2 detail levels
        mid_freq_energy = sum(energy_distribution.get(f'detail_{i+1}', 0) 
                             for i in range(2, 4))  # Detail levels 3-4
        low_freq_energy = sum(energy_distribution.get(f'detail_{i+1}', 0) 
                             for i in range(4, len(energy_distribution)-1))  # Remaining detail levels
        
        # Simple regime classification rules - fixed comparison
        if approx_energy > 0.7:
            regime = 'strong_trend'
            regime_value = 1.0
        elif high_freq_energy > 0.4:
            regime = 'choppy'
            regime_value = 0.0
        elif mid_freq_energy > 0.5:
            regime = 'cyclical'
            regime_value = 0.5
        elif low_freq_energy > 0.3:
            regime = 'ranging'
            regime_value = 0.25
        else:
            regime = 'mixed'
            regime_value = 0.3
        
        # Create numeric representation (simple constant value for now)
        # In a more advanced implementation, this would be a time series
        regime_numeric = np.ones(100) * regime_value
        
        return regime, regime_numeric
        
    
    def _analyze_cycles_with_cwt(self, series, min_scale=1, max_scale=128):
        """
        Use Continuous Wavelet Transform to identify cycles.
        
        Args:
            series: Input time series
            min_scale: Minimum scale
            max_scale: Maximum scale
            
        Returns:
            dict: Cycle analysis results
        """
        try:
            # Define scales for analysis (logarithmically spaced)
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 40)
            
            # Perform CWT with Morlet wavelet
            coef, freqs = pywt.cwt(series, scales, 'morl')
            
            # Calculate power spectrum
            power = np.abs(coef)**2
            
            # Find dominant scales
            scale_power = np.sum(power, axis=1)
            dominant_scale_idx = np.argmax(scale_power)
            dominant_scale = scales[dominant_scale_idx]
            dominant_period = 1.0 / freqs[dominant_scale_idx]
            
            # Find temporal locations of peaks at dominant scale
            power_at_dominant = power[dominant_scale_idx, :]
            peaks = []
            for i in range(1, len(power_at_dominant)-1):
                if (power_at_dominant[i] > power_at_dominant[i-1] and 
                    power_at_dominant[i] > power_at_dominant[i+1]):
                    if power_at_dominant[i] > 0.5 * np.max(power_at_dominant):
                        peaks.append(i)
            
            # Calculate cyclicality score
            if len(peaks) >= 2:
                peak_intervals = np.diff(peaks)
                cyclicality = 1.0 - np.std(peak_intervals) / np.mean(peak_intervals)
                cyclicality = max(0, min(1, cyclicality))
            else:
                cyclicality = 0.0
            
            return {
                'dominant_scale': dominant_scale,
                'dominant_period': dominant_period,
                'cycle_peaks': peaks,
                'cyclicality': cyclicality,
                'scale_power': scale_power.tolist(),
                'frequency_power': np.sum(power, axis=0).tolist()
            }
        except Exception as e:
            self.logger.error(f"CWT cycle analysis error: {e}")
            return {
                'dominant_scale': 1.0,
                'dominant_period': 1.0,
                'cycle_peaks': [],
                'cyclicality': 0.0,
                'scale_power': [],
                'frequency_power': []
            }
        
    def extract_features(self, dataframe):
        """
        Extract wavelet-based features from financial data.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        try:
            # Get close prices
            if 'close' not in dataframe.columns:
                self.logger.warning("Close prices not found in dataframe")
                return features
            
            close = dataframe['close'].values
            
            # Continuous wavelet transform for cyclic features
            cycles = self._analyze_cycles_with_cwt(close)
            
            # Extract key cycle features
            features['dominant_cycle'] = np.ones(len(close)) * cycles['dominant_period']
            
            # Wavelet variance at different scales
            wavelet_vars = self._calculate_wavelet_variance(close)
            for scale, values in wavelet_vars.items():
                features[f'wavelet_var_{scale}'] = values
            
            # Wavelet correlation between price and volume
            if 'volume' in dataframe.columns:
                volume = dataframe['volume'].values
                wcorr = self._calculate_wavelet_correlation(close, volume)
                for scale, values in wcorr.items():
                    features[f'vol_price_wcorr_{scale}'] = values
            
            # Trend strength based on approximation
            mra = self.perform_mra(close)
            trend = mra['approximation']
            features['trend_strength'] = self._calculate_trend_strength(close, trend)
            
            # Momentum features from wavelet decomposition
            features['wavelet_momentum'] = self._calculate_wavelet_momentum(close)
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting wavelet features: {e}")
            return features
        
    
    def _calculate_wavelet_variance(self, series, scales=None):
        """
        Calculate wavelet variance at different scales.
        
        Args:
            series: Input time series
            scales: Scales for analysis (default: [2, 4, 8, 16, 32])
            
        Returns:
            dict: Wavelet variance at each scale
        """
        if scales is None:
            scales = [2, 4, 8, 16, 32]
        
        wavelet_var = {}
        
        try:
            # Get data length
            n = len(series)
            
            for scale in scales:
                if scale >= n:
                    continue
                
                # Create array for variance
                var_series = np.zeros(n)
                
                # Calculate rolling wavelet variance
                for i in range(scale, n):
                    window = series[i-scale:i]
                    # Simple MODWT-like approach
                    diffs = window - np.mean(window)
                    var_series[i] = np.sum(diffs**2) / scale
                
                # Fill initial values
                var_series[:scale] = var_series[scale]
                
                # Normalize
                if np.max(var_series) > 0:
                    var_series = var_series / np.max(var_series)
                
                wavelet_var[f'scale_{scale}'] = var_series
                
            return wavelet_var
        except Exception as e:
            self.logger.error(f"Wavelet variance calculation error: {e}")
            return {f'scale_{s}': np.zeros(len(series)) for s in scales}
        
    
    def _calculate_wavelet_correlation(self, series1, series2, scales=None):
        """
        Calculate wavelet correlation between two series at different scales.
        
        Args:
            series1: First time series
            series2: Second time series
            scales: Scales for analysis (default: [4, 8, 16, 32])
            
        Returns:
            dict: Wavelet correlation at each scale
        """
        if scales is None:
            scales = [4, 8, 16, 32]
        
        wcorr = {}
        
        try:
            # Ensure series are the same length
            n = min(len(series1), len(series2))
            s1 = series1[:n]
            s2 = series2[:n]
            
            for scale in scales:
                if scale >= n:
                    continue
                
                # Create array for correlation
                corr_series = np.zeros(n)
                
                # Calculate rolling wavelet correlation
                for i in range(scale, n):
                    win1 = s1[i-scale:i]
                    win2 = s2[i-scale:i]
                    
                    # Simple correlation using MRA-like approach
                    coeffs1 = pywt.wavedec(win1, self.wavelet_family, level=1)
                    coeffs2 = pywt.wavedec(win2, self.wavelet_family, level=1)
                    
                    # Use detail coefficients for correlation
                    d1 = coeffs1[1]
                    d2 = coeffs2[1]
                    
                    # Calculate correlation
                    if len(d1) > 1 and len(d2) > 1:
                        cov = np.mean((d1 - np.mean(d1)) * (d2 - np.mean(d2)))
                        std1 = np.std(d1)
                        std2 = np.std(d2)
                        
                        if std1 > 0 and std2 > 0:
                            corr_series[i] = cov / (std1 * std2)
                        else:
                            corr_series[i] = 0
                    else:
                        corr_series[i] = 0
                
                # Fill initial values
                corr_series[:scale] = corr_series[scale]
                
                # Ensure values are within [-1, 1]
                corr_series = np.clip(corr_series, -1, 1)
                
                wcorr[f'scale_{scale}'] = corr_series
            
            return wcorr
        except Exception as e:
            self.logger.error(f"Wavelet correlation calculation error: {e}")
            return {f'scale_{s}': np.zeros(len(series1)) for s in scales}
        
    def _calculate_trend_strength(self, original, trend):
        """
        Calculate trend strength based on original series and trend component.
        
        Args:
            original: Original time series
            trend: Trend component (approximation from MRA)
            
        Returns:
            array: Trend strength measure
        """
        try:
            # Ensure same length
            n = min(len(original), len(trend))
            original = original[:n]
            trend = trend[:n]
            
            # Calculate rolling trend strength
            window = min(20, n // 2)  # Adaptive window size
            
            trend_strength = np.zeros(n)
            
            for i in range(window, n):
                # Extract windows
                orig_win = original[i-window:i]
                trend_win = trend[i-window:i]
                
                # Calculate variance ratio
                var_orig = np.var(orig_win)
                var_diff = np.var(orig_win - trend_win)
                
                if var_orig > 0:
                    # 1 - (variance of residuals / variance of original)
                    strength = 1 - var_diff / var_orig
                    trend_strength[i] = max(0, min(1, strength))
                else:
                    trend_strength[i] = 0
            
            # Fill initial values
            trend_strength[:window] = trend_strength[window]
            
            return trend_strength
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return np.zeros(len(original))
        
    def _calculate_wavelet_momentum(self, series, window=10):
        """
        Calculate momentum feature based on wavelet decomposition.
        
        Args:
            series: Input time series
            window: Window size for calculation
            
        Returns:
            array: Wavelet momentum indicator
        """
        try:
            n = len(series)
            momentum = np.zeros(n)
            
            # Perform MRA
            mra = self.perform_mra(series)
            
            # Use approximation and first detail level
            approx = mra['approximation']
            detail1 = mra['level_signals'].get('detail_1', np.zeros(n))
            
            # Calculate momentum based on trend direction and detail energy
            for i in range(window, n):
                # Trend direction from approximation slope
                trend_diff = approx[i] - approx[i-window]
                trend_direction = np.sign(trend_diff)
                
                # Detail energy (momentum strength)
                detail_energy = np.sum(detail1[i-window:i]**2) / window
                
                # Scale to [0, 1]
                if np.max(np.abs(detail1)) > 0:
                    detail_energy /= np.max(np.abs(detail1))**2
                
                # Combine direction and strength
                momentum[i] = trend_direction * detail_energy
            
            # Fill initial values
            momentum[:window] = momentum[window]
            
            # Scale to [-1, 1]
            max_abs = np.max(np.abs(momentum))
            if max_abs > 0:
                momentum = momentum / max_abs
            
            return momentum
        except Exception as e:
            self.logger.error(f"Wavelet momentum calculation error: {e}")
            return np.zeros(len(series))
        


    # ----- Wavelet Scattering Transform Methods -----
    
    @staticmethod
    @njit(cache=True)
    def _numba_scattering_1d(data: np.ndarray, filter_bank: np.ndarray, 
                           scales: np.ndarray, order: int = 2) -> np.ndarray:
        """
        Numba implementation of 1D scattering transform.
        
        Args:
            data: Input signal
            filter_bank: Wavelet filters
            scales: Wavelet scales
            order: Scattering order
            
        Returns:
            Scattering coefficients
        """
        n = len(data)
        n_scales = len(scales)
        
        # First-order scattering
        s1 = np.zeros((n_scales, n))
        
        for i in range(n_scales):
            scale = scales[i]
            # Simplified convolution
            for j in range(n):
                for k in range(len(filter_bank)):
                    idx = (j - k) % n
                    s1[i, j] += data[idx] * filter_bank[k * scale]
                    
            # Apply modulus
            s1[i] = np.abs(s1[i])
            
        # Second-order scattering
        s2 = np.zeros((n_scales, n_scales, n))
        
        if order >= 2:
            for i in range(n_scales):
                for j in range(n_scales):
                    scale = scales[j]
                    # Convolve modulus of first order with wavelets
                    for k in range(n):
                        for l in range(len(filter_bank)):
                            idx = (k - l) % n
                            s2[i, j, k] += s1[i, idx] * filter_bank[l * scale]
                            
                    # Apply modulus
                    s2[i, j] = np.abs(s2[i, j])
                    
        # Extract scattering coefficients
        # Average over time for invariant representation
        s1_coeffs = np.mean(s1, axis=1)
        s2_coeffs = np.mean(s2, axis=2)
        
        # Concatenate coefficients
        coeffs = np.concatenate([np.mean(data).reshape(1), s1_coeffs, s2_coeffs.flatten()])
        
        return coeffs
    
    def wavelet_scattering_transform(self, data: np.ndarray, order: int = 2,
                                  scales: Optional[np.ndarray] = None,
                                  wavelet: Optional[str] = None) -> np.ndarray:
        """
        Perform wavelet scattering transform on signal.
        
        Args:
            data: Input signal
            order: Scattering order (1 or 2)
            scales: Wavelet scales (default: dyadic scales)
            wavelet: Wavelet to use (default from config)
            
        Returns:
            Scattering coefficients
        """
        # Check if Kymatio is available for optimized implementation
        if self.has_kymatio:
            return self._kymatio_scattering_1d(data, order, scales, wavelet)
            
        # Check if PyTorch is available for GPU acceleration
        elif self.has_torch and self.config["use_torch"] and len(data) > self.config["parallel_threshold"]:
            return self._torch_scattering_1d(data, order, scales, wavelet)
            
        # Check if PyWavelets and Numba are available for CPU acceleration
        elif self.has_pywavelets and self.has_numba and self.config["use_numba"]:
            return self._numba_scattering_1d_wrapper(data, order, scales, wavelet)
            
        # Check if PyWavelets is available for basic implementation
        elif self.has_pywavelets:
            return self._basic_scattering_1d(data, order, scales, wavelet)
            
        else:
            self.logger.error("No wavelet library available for scattering transform")
            return None
    
    def _kymatio_scattering_1d(self, data: np.ndarray, order: int = 2,
                            scales: Optional[np.ndarray] = None,
                            wavelet: Optional[str] = None) -> np.ndarray:
        """
        Perform wavelet scattering transform using Kymatio.
        
        Args:
            data: Input signal
            order: Scattering order
            scales: Wavelet scales
            wavelet: Wavelet to use
            
        Returns:
            Scattering coefficients
        """
        if not self.has_kymatio:
            self.logger.error("Kymatio not available for scattering transform")
            return None
            
        try:
            # Get parameters
            J = int(np.log2(len(data))) - 2  # Max scale
            if J < 1:
                J = 1
                
            # Create scattering object
            scattering = KymatioScattering1D(J=J, shape=len(data), max_order=order)
            
            # Prepare input
            x = data.astype(np.float32)
            x = x.reshape(1, -1)  # Add batch dimension
            
            # Compute scattering coefficients
            coeffs = scattering(x)
            
            # Remove batch dimension
            coeffs = coeffs.squeeze(0)
            
            # Flatten coefficients
            coeffs = coeffs.flatten()
            
            return coeffs
            
        except Exception as e:
            self.logger.error(f"Error in Kymatio scattering transform: {e}")
            return None
    
    def _torch_scattering_1d(self, data: np.ndarray, order: int = 2,
                          scales: Optional[np.ndarray] = None,
                          wavelet: Optional[str] = None) -> np.ndarray:
        """
        Perform wavelet scattering transform using PyTorch.
        
        Args:
            data: Input signal
            order: Scattering order
            scales: Wavelet scales
            wavelet: Wavelet to use
            
        Returns:
            Scattering coefficients
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available for scattering transform")
            return None
            
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Get device
            device = self.hw_accelerator.get_torch_device()
            
            # Get parameters
            if scales is None:
                # Dyadic scales
                J = int(np.log2(len(data))) - 2
                if J < 1:
                    J = 1
                scales = 2 ** np.arange(1, J + 1)
                
            wavelet = wavelet or self.config["default_wavelet"]
            
            # Create wavelet filters
            filters = []
            
            # Use Morlet wavelets (better for scattering)
            for scale in scales:
                # Create Morlet filter centered at 0
                t = torch.arange(-4 * scale, 4 * scale + 1, dtype=torch.float32, device=device)
                
                # Morlet wavelet: complex exponential modulated by Gaussian
                filter_real = torch.exp(-t**2 / (2 * scale**2)) * torch.cos(5 * t / scale)
                filter_imag = torch.exp(-t**2 / (2 * scale**2)) * torch.sin(5 * t / scale)
                
                # Normalize
                filter_real = filter_real / torch.norm(filter_real)
                filter_imag = filter_imag / torch.norm(filter_imag)
                
                filters.append((filter_real, filter_imag))
                
            # Prepare input
            x = torch.tensor(data, dtype=torch.float32, device=device)
            n = len(x)
            
            # First-order scattering
            s1 = []
            u1 = []
            
            for filter_real, filter_imag in filters:
                # Complex convolution
                y_real = F.conv1d(x.view(1, 1, -1), filter_real.view(1, 1, -1), padding='same').view(-1)
                y_imag = F.conv1d(x.view(1, 1, -1), filter_imag.view(1, 1, -1), padding='same').view(-1)
                
                # Modulus
                y_mod = torch.sqrt(y_real**2 + y_imag**2 + 1e-10)
                
                # Average (low-pass filtering)
                y_avg = F.avg_pool1d(y_mod.view(1, 1, -1), 4, stride=1, padding=2).view(-1)
                
                # Store coefficients
                s1.append(y_avg)
                
                # Store modulus for second order
                u1.append(y_mod)
                
            # Second-order scattering (if requested)
            s2 = []
            
            if order >= 2:
                for i, u in enumerate(u1):
                    for j, (filter_real, filter_imag) in enumerate(filters):
                        # Skip same-scale filtering
                        if scales[j] >= scales[i]:
                            continue
                            
                        # Complex convolution
                        y_real = F.conv1d(u.view(1, 1, -1), filter_real.view(1, 1, -1), padding='same').view(-1)
                        y_imag = F.conv1d(u.view(1, 1, -1), filter_imag.view(1, 1, -1), padding='same').view(-1)
                        
                        # Modulus
                        y_mod = torch.sqrt(y_real**2 + y_imag**2 + 1e-10)
                        
                        # Average (low-pass filtering)
                        y_avg = F.avg_pool1d(y_mod.view(1, 1, -1), 4, stride=1, padding=2).view(-1)
                        
                        # Store coefficients
                        s2.append(y_avg)
                        
            # Combine coefficients
            all_coeffs = [torch.mean(x)]  # Zero-th order coefficient (DC component)
            
            # Add first-order coefficients
            for coef in s1:
                all_coeffs.append(torch.mean(coef))
                
            # Add second-order coefficients
            for coef in s2:
                all_coeffs.append(torch.mean(coef))
                
            # Combine and convert to numpy
            combined = torch.stack(all_coeffs).cpu().numpy()
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error in PyTorch scattering transform: {e}")
            return None
    
    def _numba_scattering_1d_wrapper(self, data: np.ndarray, order: int = 2,
                                  scales: Optional[np.ndarray] = None,
                                  wavelet: Optional[str] = None) -> np.ndarray:
        """
        Wrapper for Numba implementation of scattering transform.
        
        Args:
            data: Input signal
            order: Scattering order
            scales: Wavelet scales
            wavelet: Wavelet to use
            
        Returns:
            Scattering coefficients
        """
        if not self.has_numba or not self.has_pywavelets:
            self.logger.error("Numba or PyWavelets not available for scattering transform")
            return None
            
        try:
            # Get parameters
            if scales is None:
                # Dyadic scales
                J = int(np.log2(len(data))) - 2
                if J < 1:
                    J = 1
                scales = 2 ** np.arange(1, J + 1)
                
            wavelet = wavelet or 'morl'  # Morlet is better for scattering
            
            # Create wavelet filters
            filter_bank = []
            
            # Use Morlet wavelet for scattering
            t = np.linspace(-8, 8, 256)
            morlet = pywt.Wavelet('morl').wavefun(level=8)[0]
            
            # Pad or truncate to desired length
            n_filter = 256
            
            if len(morlet) > n_filter:
                morlet = morlet[:n_filter]
            elif len(morlet) < n_filter:
                morlet = np.pad(morlet, (0, n_filter - len(morlet)))
                
            # Normalize
            morlet = morlet / np.linalg.norm(morlet)
            
            # Call Numba implementation
            coeffs = self._numba_scattering_1d(data, morlet, scales, order)
            
            return coeffs
            
        except Exception as e:
            self.logger.error(f"Error in Numba scattering transform: {e}")
            return None
    
    def _basic_scattering_1d(self, data: np.ndarray, order: int = 2,
                          scales: Optional[np.ndarray] = None,
                          wavelet: Optional[str] = None) -> np.ndarray:
        """
        Basic implementation of scattering transform using PyWavelets.
        
        Args:
            data: Input signal
            order: Scattering order
            scales: Wavelet scales
            wavelet: Wavelet to use
            
        Returns:
            Scattering coefficients
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available for scattering transform")
            return None
            
        try:
            # Get parameters
            if scales is None:
                # Dyadic scales
                J = int(np.log2(len(data))) - 2
                if J < 1:
                    J = 1
                scales = 2 ** np.arange(1, J + 1)
                
            wavelet = wavelet or self.config["default_wavelet"]
            
            # First-order scattering
            s1_coeffs = []
            first_order = []
            
            for scale in scales:
                # Continuous wavelet transform at this scale
                coef, _ = pywt.cwt(data, [scale], wavelet)
                coef = coef[0]  # Remove scale dimension
                
                # Modulus
                mod = np.abs(coef)
                
                # Store for second-order
                first_order.append(mod)
                
                # Average for coefficient
                s1_coeffs.append(np.mean(mod))
                
            # Second-order scattering
            s2_coeffs = []
            
            if order >= 2:
                for i, mod1 in enumerate(first_order):
                    for j, scale in enumerate(scales):
                        # Skip same-scale filtering
                        if scales[j] >= scales[i]:
                            continue
                            
                        # Continuous wavelet transform of first-order modulus
                        coef, _ = pywt.cwt(mod1, [scale], wavelet)
                        coef = coef[0]  # Remove scale dimension
                        
                        # Modulus
                        mod2 = np.abs(coef)
                        
                        # Average for coefficient
                        s2_coeffs.append(np.mean(mod2))
                        
            # Combine coefficients
            all_coeffs = np.concatenate([[np.mean(data)], s1_coeffs, s2_coeffs])
            
            return all_coeffs
            
        except Exception as e:
            self.logger.error(f"Error in basic scattering transform: {e}")
            return None
    
    # ----- Utility and Analysis Methods -----
    
    def analyze_signal_complexity(self, data: np.ndarray, wavelet: Optional[str] = None,
                              level: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze signal complexity using wavelet-based metrics.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            level: Decomposition level (default from config)
            
        Returns:
            Dictionary of complexity metrics
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze complexity")
            return {}
            
        try:
            # Extract wavelet features
            features = self.extract_wavelet_features(data, wavelet, level)
            
            if not features:
                self.logger.error("Failed to extract wavelet features for complexity analysis")
                return {}
                
            # Initialize complexity metrics
            complexity = {}
            
            # 1. Wavelet energy spread (spectral flatness)
            energy_values = [v for k, v in features.items() if k.startswith("rel_energy_")]
            if energy_values:
                # Compute spectral flatness using geometric mean / arithmetic mean
                geo_mean = np.exp(np.mean(np.log(np.array(energy_values) + 1e-10)))
                arith_mean = np.mean(energy_values)
                complexity["spectral_flatness"] = geo_mean / arith_mean if arith_mean > 0 else 0.0
                
                # Shannon entropy of energy distribution
                if self.has_scipy:
                    from scipy.stats import entropy
                    complexity["energy_entropy"] = entropy(energy_values) / np.log(len(energy_values))
                    
            # 2. Multi-scale entropy using wavelet coefficients
            if "entropy_diversity" in features:
                complexity["multi_scale_entropy"] = features["entropy_diversity"]
                
            # 3. Hurst exponent estimation using wavelet variance
            log_variance = []
            log_scale = []
            
            for i in range(1, 6):  # Use first 5 detail levels if available
                var_key = f"std_detail_{i}"
                if var_key in features:
                    log_variance.append(np.log(features[var_key]**2))
                    log_scale.append(np.log(2**i))
                    
            if len(log_variance) > 1:
                # Linear regression slope
                coef = np.polyfit(log_scale, log_variance, 1)[0]
                
                # Hurst = (slope + 1) / 2
                complexity["hurst_exponent"] = (coef + 1) / 2
                
            # 4. Singularity spectrum width (multifractal analysis)
            # This is a simplified approximation
            if "kurtosis_approx" in features:
                detail_kurtosis = [features.get(f"kurtosis_detail_{i}", 0) for i in range(1, 6)]
                if detail_kurtosis:
                    # Higher range of kurtosis values suggests wider singularity spectrum
                    complexity["singularity_width"] = np.max(detail_kurtosis) - np.min(detail_kurtosis)
                    
            # 5. Wavelet denoised signal complexity
            denoised = self.denoise_signal(data, wavelet=wavelet)
            if denoised is not None:
                # Calculate SNR
                signal_power = np.sum(denoised.denoised**2)
                noise_power = np.sum(denoised.noise**2)
                
                if noise_power > 0:
                    complexity["snr"] = 10 * np.log10(signal_power / noise_power)
                    
                # Measure of non-Gaussianity in noise
                if self.has_scipy:
                    complexity["noise_kurtosis"] = kurtosis(denoised.noise)
                    
            # 6. Dominant cycle frequency
            cycles = self.detect_cycles(data, wavelet=wavelet)
            if cycles and "periods" in cycles and cycles["periods"]:
                complexity["dominant_period"] = cycles["periods"][0]
                
                # Strength of cyclical component
                if "powers" in cycles and cycles["powers"]:
                    complexity["cycle_strength"] = cycles["powers"][0] / np.sum(cycles["powers"])
                    
            return complexity
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal complexity: {e}")
            return {}
    
    def analyze(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        General analysis method for compatibility with CDFA server interface
        
        Args:
            data: Input signal data (typically price array or DataFrame)
            
        Returns:
            Dictionary with analysis results including regime and coefficients
        """
        try:
            # Convert DataFrame to array if needed
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    signal_data = data['close'].values
                else:
                    signal_data = data.iloc[:, 0].values
            else:
                signal_data = data
                
            # Ensure we have enough data points
            if len(signal_data) < 10:
                return {
                    'regime': 'insufficient_data',
                    'trend_strength': 0.5,
                    'volatility': 0.0,
                    'coefficients': []
                }
                
            # Perform market regime analysis as the primary analysis
            regime_result = self.analyze_market_regime(signal_data)
            
            # Convert WaveletAnalysisResult to dict if needed
            if regime_result is not None:
                if hasattr(regime_result, '__dict__'):
                    # Convert result object to dictionary
                    result_dict = {
                        'regime': getattr(regime_result, 'regime', 'unknown'),
                        'trend_strength': getattr(regime_result, 'trend_strength', 0.5),
                        'volatility': getattr(regime_result, 'volatility', 0.0),
                        'energy_distribution': getattr(regime_result, 'energy_distribution', {}),
                        'coefficients': []
                    }
                else:
                    result_dict = regime_result
            else:
                # Fallback analysis if regime analysis fails
                result_dict = {
                    'regime': 'analysis_failed',
                    'trend_strength': 0.5,
                    'volatility': 0.0,
                    'coefficients': []
                }
            
            # Add wavelet coefficients for compatibility
            if self.has_pywavelets and len(signal_data) > 0:
                try:
                    wavelet = self.config.get('default_wavelet', 'sym8')
                    max_level = min(self.config.get('max_level', 4), int(np.log2(len(signal_data))))
                    coeffs = pywt.wavedec(signal_data, wavelet, level=max_level)
                    
                    # Convert coefficients to serializable format
                    result_dict['coefficients'] = [coeff.tolist() if hasattr(coeff, 'tolist') else coeff for coeff in coeffs]
                    result_dict['wavelet_used'] = wavelet
                    result_dict['decomposition_level'] = len(coeffs) - 1
                    
                except Exception as e:
                    self.logger.debug(f"Failed to add coefficients: {e}")
                    result_dict['coefficients'] = []
            else:
                result_dict['coefficients'] = []
                
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in analyze method: {e}")
            return {
                'regime': 'error',
                'trend_strength': 0.5,
                'volatility': 0.0,
                'coefficients': [],
                'error': str(e)
            }
    
    def analyze_regime_transitions(self, data: np.ndarray, window_size: int = 50,
                               step_size: int = 10, wavelet: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze regime transitions using rolling wavelet analysis.
        
        Args:
            data: Input signal
            window_size: Analysis window size
            step_size: Step size for sliding window
            wavelet: Wavelet to use (default from config)
            
        Returns:
            Dictionary with transition analysis results
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot analyze regime transitions")
            return {}
            
        try:
            # Get wavelet
            wavelet = wavelet or self.config["default_wavelet"]
            
            # Initialize result arrays
            n_windows = max(1, (len(data) - window_size) // step_size + 1)
            window_indices = [i * step_size for i in range(n_windows)]
            
            trend_strength = np.zeros(n_windows)
            noise_level = np.zeros(n_windows)
            dominant_scales = np.zeros((n_windows, 3), dtype=int)
            entropy_values = np.zeros(n_windows)
            
            # Analyze each window
            for i, start_idx in enumerate(window_indices):
                end_idx = min(start_idx + window_size, len(data))
                window_data = data[start_idx:end_idx]
                
                # Analyze regime
                analysis = self.analyze_market_regime(window_data, wavelet=wavelet)
                
                if analysis is not None:
                    # Store metrics
                    trend_strength[i] = analysis.trend_strength
                    noise_level[i] = analysis.noise_level
                    
                    # Store dominant scales (up to 3)
                    for j, scale in enumerate(analysis.dominant_scales[:3]):
                        if j < 3:
                            dominant_scales[i, j] = scale
                            
                    # Store entropy
                    if analysis.entropy:
                        entropy_values[i] = np.mean(list(analysis.entropy.values()))
                        
            # Detect transitions using changepoints in metrics
            transitions = []
            
            # 1. Trend strength transitions (significant changes)
            if self.has_scipy:
                # Use scipy.signal.find_peaks to detect significant changes
                from scipy.signal import find_peaks
                
                # Compute absolute differences
                trend_diff = np.abs(np.diff(trend_strength))
                
                # Find peaks in differences (changepoints)
                peaks, _ = find_peaks(trend_diff, height=np.std(trend_diff))
                
                for peak in peaks:
                    transitions.append({
                        "index": window_indices[peak],
                        "metric": "trend_strength",
                        "from": trend_strength[peak],
                        "to": trend_strength[peak + 1],
                        "magnitude": trend_diff[peak]
                    })
                    
                # Detect noise level transitions
                noise_diff = np.abs(np.diff(noise_level))
                peaks, _ = find_peaks(noise_diff, height=np.std(noise_diff))
                
                for peak in peaks:
                    transitions.append({
                        "index": window_indices[peak],
                        "metric": "noise_level",
                        "from": noise_level[peak],
                        "to": noise_level[peak + 1],
                        "magnitude": noise_diff[peak]
                    })
                    
                # Detect entropy transitions
                entropy_diff = np.abs(np.diff(entropy_values))
                peaks, _ = find_peaks(entropy_diff, height=np.std(entropy_diff))
                
                for peak in peaks:
                    transitions.append({
                        "index": window_indices[peak],
                        "metric": "entropy",
                        "from": entropy_values[peak],
                        "to": entropy_values[peak + 1],
                        "magnitude": entropy_diff[peak]
                    })
                    
            # Sort transitions by index
            transitions.sort(key=lambda x: x["index"])
            
            # Group nearby transitions (within 2*step_size)
            grouped_transitions = []
            current_group = []
            
            for trans in transitions:
                if not current_group or abs(trans["index"] - current_group[-1]["index"]) <= 2 * step_size:
                    current_group.append(trans)
                else:
                    # Find strongest transition in group
                    strongest = max(current_group, key=lambda x: x["magnitude"])
                    grouped_transitions.append(strongest)
                    
                    # Start new group
                    current_group = [trans]
                    
            # Add last group
            if current_group:
                strongest = max(current_group, key=lambda x: x["magnitude"])
                grouped_transitions.append(strongest)
                
            # Create result
            result = {
                "transitions": grouped_transitions,
                "metrics": {
                    "trend_strength": trend_strength.tolist(),
                    "noise_level": noise_level.tolist(),
                    "entropy": entropy_values.tolist()
                },
                "window_indices": window_indices,
                "window_size": window_size,
                "step_size": step_size
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return {}

    # Add to cdfa_extensions/wavelet_processor.py
    
    def analyze_yfinance_data(self, symbols: Union[str, List[str]], 
                            period: str = '1y', 
                            interval: str = '1d',
                            analysis_type: str = 'wavelet_features') -> Dict[str, Any]:
        """
        Fetch data from Yahoo Finance and perform wavelet analysis.
        
        Args:
            symbols: Symbol or list of symbols to analyze
            period: Time period to analyze
            interval: Data interval
            analysis_type: Type of analysis ('wavelet_features', 'denoise', 'cycles', 'regime')
            
        Returns:
            Analysis results
        """
        # Create data fetcher if not already available
        if not hasattr(self, '_data_fetcher'):
            from .data_fetcher import MarketDataFetcher
            self._data_fetcher = MarketDataFetcher()
            
        # Fetch data
        data_dict = self._data_fetcher.fetch_data_for_cdfa(
            symbols, 
            source='yahoo', 
            period=period, 
            interval=interval
        )
        
        if not data_dict:
            self.logger.error("Failed to fetch data from Yahoo Finance")
            return {}
            
        # Process each symbol
        results = {}
        for symbol, df in data_dict.items():
            # Extract price data (use close prices by default)
            if 'close' in df.columns:
                price_data = df['close'].values
                
                # Perform requested analysis
                if analysis_type == 'wavelet_features':
                    result = self.extract_wavelet_features(price_data)
                elif analysis_type == 'denoise':
                    result = self.denoise_signal(price_data)
                elif analysis_type == 'cycles':
                    result = self.detect_cycles(price_data)
                elif analysis_type == 'regime':
                    result = self.analyze_market_regime(price_data)
                else:
                    self.logger.error(f"Unsupported analysis type: {analysis_type}")
                    continue
                    
                results[symbol] = result
            else:
                self.logger.warning(f"Missing close price data for {symbol}")
                
        return results

    
    def find_singularities(self, data: np.ndarray, wavelet: Optional[str] = None,
                        threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find singularities in time series using wavelet transform modulus maxima.
        
        Args:
            data: Input signal
            wavelet: Wavelet to use (default from config)
            threshold: Detection threshold (default: automatic)
            
        Returns:
            List of detected singularities with properties
        """
        if not self.has_pywavelets:
            self.logger.error("PyWavelets not available, cannot find singularities")
            return []
            
        try:
            # Get wavelet
            wavelet = wavelet or "gaus1"  # Gaussian derivative is better for singularity detection
            
            # Perform CWT at multiple scales
            scales = 2 ** np.linspace(1, 6, 32)  # Multiple scales for multi-resolution detection
            coeffs, _ = self.continuous_wavelet_transform(data, scales, wavelet)
            
            if coeffs is None:
                self.logger.error("Failed to perform CWT for singularity detection")
                return []
                
            # Calculate modulus of coefficients
            modulus = np.abs(coeffs)
            
            # Find maxima at each scale
            maxima = []
            
            for i, scale in enumerate(scales):
                # Find local maxima
                if self.has_scipy:
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(modulus[i], prominence=0.1)
                else:
                    # Simple peak detection
                    peaks = []
                    for j in range(1, len(modulus[i]) - 1):
                        if modulus[i, j] > modulus[i, j-1] and modulus[i, j] > modulus[i, j+1]:
                            peaks.append(j)
                            
                # Store maxima with scale
                for peak in peaks:
                    maxima.append({
                        "scale": scale,
                        "position": peak,
                        "value": modulus[i, peak]
                    })
                    
            # Group maxima by position (singularity locations)
            position_groups = {}
            
            for maximum in maxima:
                pos = maximum["position"]
                scale = maximum["scale"]
                value = maximum["value"]
                
                # Find nearby group
                found_group = False
                window = int(scale / 2)  # Search window depends on scale
                
                for center in list(position_groups.keys()):
                    if abs(center - pos) <= window:
                        # Add to existing group
                        position_groups[center].append({
                            "scale": scale,
                            "value": value
                        })
                        found_group = True
                        break
                        
                if not found_group:
                    # Create new group
                    position_groups[pos] = [{
                        "scale": scale,
                        "value": value
                    }]
                    
            # Filter groups with maxima across multiple scales (true singularities)
            min_scales = 3  # Require presence at multiple scales
            singularities = []
            
            for position, maxima_list in position_groups.items():
                if len(maxima_list) >= min_scales:
                    # Extract scale-value pairs for log-log regression
                    scales_array = np.array([m["scale"] for m in maxima_list])
                    values_array = np.array([m["value"] for m in maxima_list])
                    
                    # Sort by scale
                    sort_idx = np.argsort(scales_array)
                    scales_array = scales_array[sort_idx]
                    values_array = values_array[sort_idx]
                    
                    # Linear regression in log-log space
                    log_scales = np.log(scales_array)
                    log_values = np.log(values_array)
                    
                    if len(log_scales) > 1:
                        # Calculate slope (Hölder exponent + 0.5)
                        coef = np.polyfit(log_scales, log_values, 1)[0]
                        holder_exponent = coef - 0.5
                        
                        # Calculate strength (maximum value)
                        strength = np.max(values_array)
                        
                        # Automatic threshold based on statistics
                        if threshold is None:
                            threshold = np.mean(holder_exponent) + 0.5 * np.std(holder_exponent)
                            
                        # Add if significant
                        if strength > threshold:
                            singularities.append({
                                "position": int(position),
                                "strength": float(strength),
                                "holder_exponent": float(holder_exponent),
                                "num_scales": len(maxima_list)
                            })
                            
            # Sort by position
            singularities.sort(key=lambda x: x["position"])
            
            return singularities
            
        except Exception as e:
            self.logger.error(f"Error finding singularities: {e}")
            return []