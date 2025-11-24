#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Acceleration Utilities for CDFA Extensions

Provides hardware-agnostic acceleration for computationally intensive operations,
supporting NVIDIA GPUs via CUDA, AMD GPUs via ROCm/HIP, Apple Silicon via Metal
Performance Shaders (MPS), and CPU-based acceleration with Numba.

Author: Created on May 6, 2025
"""

import logging
import numpy as np
import warnings
from functools import wraps, partial
from typing import Optional, Callable, Dict, Any, List, Tuple, Union, Set
import os
import sys
import platform
import math
import time
import threading
import subprocess
import re
from enum import Enum, auto
import json
import cProfile
import pstats
from io import StringIO
from collections import defaultdict
import functools

# Initialize flags for available backends
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
ROCM_AVAILABLE = False
HIP_AVAILABLE = False
MPS_AVAILABLE = False
IPEX_AVAILABLE = False


# Configure AMD GPU environment variables
def configure_amd_gpu_env():
    """Configure environment variables for AMD GPU with ROCm."""
    if platform.system() == "Linux":
        # These settings are specific to RX 6800 XT (gfx1030 architecture)
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['GPU_MAX_HEAP_SIZE'] = '100'  # % of GPU memory to use
        os.environ['GPU_MAX_ALLOC_PERCENT'] = '100'
        return True
    return False

# Set AMD GPU environment variables before importing PyTorch
configure_amd_gpu_env()

# Try PyTorch import with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    # Define Tensor type for type hints
    TorchTensor = torch.Tensor
    
    # Check if CUDA is available (works for both NVIDIA and AMD via ROCm)
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # Check for ROCm on AMD GPUs
    if platform.system() == "Linux":
        # Check via PyTorch's CUDA API (ROCm uses this in PyTorch)
        if CUDA_AVAILABLE:
            try:
                device_name = torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name or "gfx" in device_name:
                    ROCM_AVAILABLE = True
                    HIP_AVAILABLE = True
            except Exception:
                pass
                
        # Try direct HIP attribute check
        if hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available'):
            try:
                if torch.hip.is_available():
                    ROCM_AVAILABLE = True
                    HIP_AVAILABLE = True
            except Exception:
                pass
                
        # Check environment variables
        if 'HIP_VISIBLE_DEVICES' in os.environ or 'ROCR_VISIBLE_DEVICES' in os.environ:
            ROCM_AVAILABLE = True
            HIP_AVAILABLE = True
    
    # Detect MPS (Apple Metal) availability for macOS
    if not CUDA_AVAILABLE and not ROCM_AVAILABLE and platform.system() == "Darwin":
        MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Detect IPEX for Intel GPUs
    try:
        import intel_extension_for_pytorch as ipex
        IPEX_AVAILABLE = True
    except ImportError:
        IPEX_AVAILABLE = False
    

        
except ImportError as e:
    # Log warning if PyTorch not available
    error_msg = f"PyTorch is required but not available: {e}. Please install PyTorch."
    warnings.warn(error_msg, DeprecationWarning, DeprecationWarning)
    # Create a placeholder for torch.Tensor type hints
    from typing import TypeVar
    TorchTensor = TypeVar('TorchTensor')
    # Create dummy torch module for type checking
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    ROCM_AVAILABLE = False
    MPS_AVAILABLE = False
    IPEX_AVAILABLE = False
    HIP_AVAILABLE = False
    class torch:
        class Tensor: pass
        class nn:
            class Module: pass
            class functional: pass
        class cuda:
            @staticmethod
            def is_available():
                return False
        class jit:
            class ScriptModule: pass
            @staticmethod
            def trace(*args, **kwargs):
                return None
            @staticmethod
            def script(*args, **kwargs):
                return None

# Import Numba with conditional imports to handle environments where it might not be available
try:
    import numba as nb
    from numba import njit, prange, cuda, vectorize, guvectorize, float64, int64, boolean
    from numba.typed import Dict as NumbaDict
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
    
    # Check if CUDA is available through Numba
    try:
        NUMBA_CUDA_AVAILABLE = cuda.is_available()
    except:
        NUMBA_CUDA_AVAILABLE = False
        
except ImportError:
    NUMBA_AVAILABLE = False
    # Create fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args, **kwargs):
        return range(*args)
    
    vectorize = njit
    guvectorize = njit
    
    # Dummy type definitions
    class DummyNumbaType:
        def __getitem__(self, *args):
            return lambda x: x
    
    float64 = DummyNumbaType()
    int64 = DummyNumbaType()
    boolean = DummyNumbaType()
    
    # Dummy container classes
    class NumbaDict(dict):
        pass
    
    class NumbaList(list):
        pass
# SciPy for signal processing and statistics
try:
    from scipy import signal
    from scipy.stats import kurtosis, skew
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced MRA capabilities will be limited.", DeprecationWarning, DeprecationWarning)


    
# Check for ROCm on AMD GPUs
ROCM_AVAILABLE = configure_amd_gpu_env()
HIP_AVAILABLE = ROCM_AVAILABLE
AMD_GPU_CONFIGURED = ROCM_AVAILABLE

# Detect MPS (Apple Metal) availability for macOS
MPS_AVAILABLE = False
if not CUDA_AVAILABLE and not ROCM_AVAILABLE and platform.system() == "Darwin":
    try:
        # Check PyTorch MPS support
        MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        MPS_AVAILABLE = False

    

# RAPIDS for CUDA-accelerated DataFrame operations
try:
    import cudf
    import cupy
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

# TVM for hardware-specific code generation
try:
    import tvm
    from tvm import relay
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

# OpenCL for cross-platform GPU compute
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

class AcceleratorType(Enum):
    """Types of hardware accelerators."""
    CPU = auto()
    CUDA = auto()    # NVIDIA GPUs
    ROCM = auto()    # AMD GPUs via ROCm
    MPS = auto()     # Apple Silicon via Metal
    VULKAN = auto()  # Cross-platform graphics API
    OPENCL = auto()  # Open Computing Language
    IPEX = auto()    # Intel GPUs
    XLA = auto()     # Google TPUs
    HABANA = auto()  # Habana Gaudi accelerators
    OTHER = auto()   # Other accelerators
    
    def __str__(self) -> str:
        return self.name.lower()

class MemoryMode(Enum):
    """Memory handling modes."""
    AUTO = auto()        # Automatic memory management
    CONSERVATIVE = auto() # Conservative memory usage
    AGGRESSIVE = auto()   # Aggressive memory utilization
    DYNAMIC = auto()      # Dynamic scaling based on workload

class HardwareInfo:
    """Information about detected hardware accelerators."""
    def __init__(self):
        self.devices = []
        self.active_device = None
        self.device_memory = {}
        self.device_capabilities = {}
        self.initialized = False
        
    def add_device(self, name, type, index, memory, compute_capability=None):
        device = {
            "name": name,
            "type": type,
            "index": index,
            "memory": memory,
            "compute_capability": compute_capability
        }
        self.devices.append(device)
        self.device_memory[name] = memory
        
        if compute_capability:
            self.device_capabilities[name] = compute_capability
            
    def get_device_by_name(self, name):
        for device in self.devices:
            if device["name"] == name:
                return device
        return None
        
    def get_device_by_index(self, type, index):
        for device in self.devices:
            if device["type"] == type and device["index"] == index:
                return device
        return None
        
    def get_devices_by_type(self, type):
        return [d for d in self.devices if d["type"] == type]
        
    def __str__(self):
        result = "HardwareInfo:\n"
        for device in self.devices:
            result += f"  - {device['name']} ({device['type']}): {device['memory']} MB"
            if device['compute_capability']:
                result += f", CC: {device['compute_capability']}"
            result += "\n"
        return result

    
# ----- Financial Time Series Operations -----

@njit(float64[:](float64[:], int64), fastmath=True, parallel=True)
def _calculate_returns_numba(prices: np.ndarray, mode: int = 0) -> np.ndarray:
    """
    Calculate returns from price series using Numba acceleration.
    
    Args:
        prices: Price series
        mode: Return calculation mode (0=percent, 1=log, 2=diff)
        
    Returns:
        Returns series
    """
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    
    # Skip first element
    for i in prange(1, n):
        if mode == 0:  # Percent returns
            if prices[i-1] != 0:
                returns[i] = (prices[i] / prices[i-1]) - 1.0
        elif mode == 1:  # Log returns
            if prices[i-1] > 0 and prices[i] > 0:
                returns[i] = np.log(prices[i] / prices[i-1])
        else:  # Simple difference
            returns[i] = prices[i] - prices[i-1]
            
    return returns


@njit(float64[:](float64[:], int64), fastmath=True)
def _calculate_rolling_std_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling standard deviation using Numba acceleration.
    
    Args:
        data: Input data
        window: Window size
        
    Returns:
        Rolling standard deviation
    """
    n = len(data)
    result = np.zeros(n, dtype=np.float64)
    
    # Need at least window samples
    if n < window:
        return result
        
    # Calculate first window
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    count = 0
    
    for i in range(window):
        if np.isfinite(data[i]):
            rolling_sum += data[i]
            rolling_sum_sq += data[i] * data[i]
            count += 1
            
    if count > 1:
        mean = rolling_sum / count
        var = (rolling_sum_sq / count) - (mean * mean)
        result[window-1] = np.sqrt(max(0.0, var))
        
    # Calculate remaining windows
    for i in range(window, n):
        # Remove oldest value
        if np.isfinite(data[i-window]):
            rolling_sum -= data[i-window]
            rolling_sum_sq -= data[i-window] * data[i-window]
            count -= 1
            
        # Add newest value
        if np.isfinite(data[i]):
            rolling_sum += data[i]
            rolling_sum_sq += data[i] * data[i]
            count += 1
            
        if count > 1:
            mean = rolling_sum / count
            var = (rolling_sum_sq / count) - (mean * mean)
            result[i] = np.sqrt(max(0.0, var))
            
    return result


@njit(float64[:](float64[:], int64, float64), fastmath=True)
def _calculate_rolling_volatility_numba(returns: np.ndarray, window: int,
                                     annualization_factor: float = 252.0) -> np.ndarray:
    """
    Calculate rolling volatility using Numba acceleration.
    
    Args:
        returns: Return series
        window: Window size
        annualization_factor: Annualization factor (252=daily, 52=weekly, 12=monthly)
        
    Returns:
        Rolling volatility (annualized)
    """
    # Calculate rolling standard deviation (inlined version of _calculate_rolling_std_numba)
    n = len(returns)
    rolling_std = np.zeros(n, dtype=np.float64)
    
    if n < window:
        return rolling_std * np.sqrt(annualization_factor)
        
    # Calculate first window
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    count = 0
    
    for i in range(window):
        if np.isfinite(returns[i]):
            rolling_sum += returns[i]
            rolling_sum_sq += returns[i] * returns[i]
            count += 1
            
    if count > 1:
        mean = rolling_sum / count
        var = (rolling_sum_sq / count) - (mean * mean)
        rolling_std[window-1] = np.sqrt(max(0.0, var))
        
    # Calculate remaining windows
    for i in range(window, n):
        # Remove oldest value
        if np.isfinite(returns[i-window]):
            rolling_sum -= returns[i-window]
            rolling_sum_sq -= returns[i-window] * returns[i-window]
            count -= 1
            
        # Add newest value
        if np.isfinite(returns[i]):
            rolling_sum += returns[i]
            rolling_sum_sq += returns[i] * returns[i]
            count += 1
            
        if count > 1:
            mean = rolling_sum / count
            var = (rolling_sum_sq / count) - (mean * mean)
            rolling_std[i] = np.sqrt(max(0.0, var))
    
    # Annualize
    return rolling_std * np.sqrt(annualization_factor)


@njit(float64[:](float64[:], int64), fastmath=True)
def _calculate_rsi_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate RSI using Numba acceleration.
    
    Args:
        prices: Price series
        window: Window size
        
    Returns:
        RSI values
    """
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n <= window:
        return rsi
        
    # Calculate price changes
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = prices[i] - prices[i-1]
        
    # Separate gains and losses
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if diff[i] > 0:
            gain[i] = diff[i]
        elif diff[i] < 0:
            loss[i] = -diff[i]
            
    # Calculate first average gain and loss
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(1, window+1):
        avg_gain += gain[i]
        avg_loss += loss[i]
        
    avg_gain /= window
    avg_loss /= window
    
    # Calculate RSI
    if avg_loss == 0:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))
        
    # Calculate remaining values
    for i in range(window+1, n):
        avg_gain = ((avg_gain * (window-1)) + gain[i]) / window
        avg_loss = ((avg_loss * (window-1)) + loss[i]) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi


@njit(float64[:, :](float64[:], float64[:], float64[:], float64[:], int64), fastmath=True)
def _calculate_bollinger_bands_numba(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                                  volume: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Bollinger Bands using Numba acceleration.
    
    Args:
        close: Close price series
        high: High price series
        low: Low price series
        volume: Volume series
        window: Window size
        
    Returns:
        Array with [upper, middle, lower, bandwidth] bands
    """
    n = len(close)
    bands = np.zeros((n, 4), dtype=np.float64)  # [upper, middle, lower, bandwidth]
    
    if n < window:
        return bands
        
    # Calculate SMA
    sma = np.zeros(n, dtype=np.float64)
    for i in range(window-1, n):
        sum_vals = 0.0
        count = 0
        
        for j in range(i-window+1, i+1):
            if np.isfinite(close[j]):
                sum_vals += close[j]
                count += 1
                
        if count > 0:
            sma[i] = sum_vals / count
            
    # Calculate standard deviation
    std = np.zeros(n, dtype=np.float64)
    for i in range(window-1, n):
        sum_sq_diff = 0.0
        count = 0
        
        for j in range(i-window+1, i+1):
            if np.isfinite(close[j]):
                diff = close[j] - sma[i]
                sum_sq_diff += diff * diff
                count += 1
                
        if count > 1:
            std[i] = np.sqrt(sum_sq_diff / (count - 1))
            
    # Calculate bands
    for i in range(window-1, n):
        bands[i, 0] = sma[i] + (2.0 * std[i])  # Upper
        bands[i, 1] = sma[i]                   # Middle
        bands[i, 2] = sma[i] - (2.0 * std[i])  # Lower
        
        # Bandwidth
        if sma[i] > 0:
            bands[i, 3] = (bands[i, 0] - bands[i, 2]) / sma[i]
            
    return bands
    

    
@njit(float64[:, :](float64[:, :], float64[:, :], float64), fastmath=True, parallel=True)
def _fast_correlation_matrix_numba(series1: np.ndarray, series2: np.ndarray, 
                                 min_periods: int) -> np.ndarray:
    """
    Calculate correlation matrix between two sets of time series using Numba.
    
    Args:
        series1: First set of time series [n_samples, n_series1]
        series2: Second set of time series [n_samples, n_series2]
        min_periods: Minimum number of valid periods
        
    Returns:
        Correlation matrix [n_series1, n_series2]
    """
    n_samples, n_series1 = series1.shape
    _, n_series2 = series2.shape
    
    # Initialize correlation matrix
    corr_matrix = np.zeros((n_series1, n_series2), dtype=np.float64)
    
    # Calculate correlation for each pair
    for i in prange(n_series1):
        for j in range(n_series2):
            # Extract series
            x = series1[:, i]
            y = series2[:, j]
            
            # Count valid periods
            valid_count = 0
            for k in range(n_samples):
                if np.isfinite(x[k]) and np.isfinite(y[k]):
                    valid_count += 1
                    
            # Skip if not enough valid periods
            if valid_count < min_periods:
                corr_matrix[i, j] = np.nan
                continue
                
            # Calculate means
            sum_x = 0.0
            sum_y = 0.0
            valid_count = 0
            
            for k in range(n_samples):
                if np.isfinite(x[k]) and np.isfinite(y[k]):
                    sum_x += x[k]
                    sum_y += y[k]
                    valid_count += 1
                    
            if valid_count == 0:
                corr_matrix[i, j] = np.nan
                continue
                
            mean_x = sum_x / valid_count
            mean_y = sum_y / valid_count
            
            # Calculate covariance and variances
            sum_xy = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0
            
            for k in range(n_samples):
                if np.isfinite(x[k]) and np.isfinite(y[k]):
                    x_diff = x[k] - mean_x
                    y_diff = y[k] - mean_y
                    sum_xy += x_diff * y_diff
                    sum_x2 += x_diff * x_diff
                    sum_y2 += y_diff * y_diff
                    
            # Calculate correlation
            if sum_x2 > 0 and sum_y2 > 0:
                corr_matrix[i, j] = sum_xy / np.sqrt(sum_x2 * sum_y2)
            else:
                corr_matrix[i, j] = np.nan
                
    return corr_matrix
    
    
@staticmethod
@cuda.jit
def _fast_kendall_tau_kernel(ranks_a, ranks_b, result):
    """CUDA kernel for fast Kendall Tau calculation"""
    i = cuda.grid(1)
    
    if i < ranks_a.shape[0]:
        discord_count = 0
        for j in range(ranks_a.shape[0]):
            if i != j:
                if ((ranks_a[i] < ranks_a[j]) and (ranks_b[i] > ranks_b[j])) or \
                   ((ranks_a[i] > ranks_a[j]) and (ranks_b[i] < ranks_b[j])):
                    discord_count += 1
        
        result[i] = discord_count

def fast_kendall_tau_cuda(self, ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
    """Fast Kendall Tau calculation using CUDA"""
    n = len(ranks_a)
    
    if n <= 1 or n != len(ranks_b):
        return 0.0
    
    # Allocate device memory
    d_ranks_a = cuda.to_device(ranks_a)
    d_ranks_b = cuda.to_device(ranks_b)
    d_result = cuda.device_array(n, dtype=np.int32)
    
    # Set up grid for kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    self._fast_kendall_tau_kernel[blocks_per_grid, threads_per_block](
        d_ranks_a, d_ranks_b, d_result
    )
    
    # Retrieve result
    result = d_result.copy_to_host()
    
    # Calculate final result
    discordant = np.sum(result) // 2  # Divide by 2 as each discordant pair is counted twice
    n_pairs = n * (n - 1) // 2
    
    return discordant / n_pairs

    
def _apply_shots_to_qnode(qnode, shots):
    """
    Apply shots configuration to a QNode call.
    
    This function provides a way to apply shots configuration to QNode calls,
    respecting the PennyLane API where shots are passed as kwargs to QNode calls.
    
    Args:
        qnode: The QNode function to call with shots configuration
        shots: Number of shots to use for the QNode call
        
    Returns:
        A wrapped QNode function that automatically applies the shots configuration
    """
    import functools
    
    @functools.wraps(qnode)
    def wrapped(*args, **kwargs):
        # Only add shots if not already in kwargs
        if 'shots' not in kwargs and shots is not None:
            kwargs['shots'] = shots
        return qnode(*args, **kwargs)
    return wrapped


def quantum_accelerated(use_hw_accel=True, hw_batch_size=None, device_shots=None, device_type=None):
    """
    Decorator for hardware-accelerated quantum functions.
    
    This decorator enhances quantum functions with hardware acceleration capabilities
    by configuring the quantum device with appropriate batch size and shot settings.
    It also provides fallback mechanisms if acceleration fails.
    
    Args:
        use_hw_accel (bool): Whether to use hardware acceleration if available
        hw_batch_size (int, optional): Batch size for hardware-accelerated quantum processing
        device_shots (int, optional): Number of shots for the quantum device
        device_type (str, optional): Specific device type to use (e.g., 'lightning.gpu', 'lightning.kokkos')
        
    Returns:
        Decorated function with hardware acceleration support
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if hardware acceleration is available and requested
            accelerate = hasattr(self, 'hw_accelerator') and self.hw_accelerator is not None and use_hw_accel
            
            # If acceleration is requested but not available, log a warning
            if use_hw_accel and not accelerate and hasattr(self, 'logger'):
                self.logger.debug(f"Hardware acceleration requested for {func.__name__} but not available")
            
            # Ensure memory pattern storage is initialized
            if not hasattr(self, 'memory_patterns'):
                self.memory_patterns = []
            
            if not hasattr(self, 'memory_metadata'):
                self.memory_metadata = []
                
            # If device is not available, run without acceleration
            if not hasattr(self, 'device') or self.device is None:
                return func(self, *args, **kwargs)
                
            # Store shots configuration for the QNode wrapper function
            # This allows decorated QNodes to automatically receive the shots parameter
            if device_shots is not None:
                # Store the shots value for QNode calls
                self._qnode_shots = device_shots
                
                # Add guidance for using the _apply_shots_to_qnode helper
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Setting shots={device_shots} for QNode calls in {func.__name__}")
                    self.logger.debug("To use with QNodes: qnode = _apply_shots_to_qnode(qnode, self._qnode_shots)")
                
            # Apply hardware acceleration settings if available
            if accelerate:
                original_batch_size = None
                
                # Store original settings
                if hasattr(self.device, 'batch_size'):
                    original_batch_size = self.device.batch_size
                    if hw_batch_size is not None:
                        self.device.batch_size = hw_batch_size
                
                # No need to set shots on device directly as it's not supported by PennyLane anymore
                
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Running {func.__name__} with hardware acceleration")
                
                try:
                    # Run the function with acceleration
                    result = func(self, *args, **kwargs)
                    
                    # Restore original settings
                    if original_batch_size is not None:
                        self.device.batch_size = original_batch_size
                    
                    # Clean up the shots configuration
                    if hasattr(self, '_qnode_shots'):
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"Cleaning up shots configuration after successful execution")
                        delattr(self, '_qnode_shots')
                        
                    return result
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Hardware acceleration failed for {func.__name__}: {e}")
                    # Restore original settings on error
                    if original_batch_size is not None:
                        self.device.batch_size = original_batch_size
                    
                    # Clean up the shots configuration on error
                    if hasattr(self, '_qnode_shots'):
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"Cleaning up shots configuration after error")
                        delattr(self, '_qnode_shots')
                    
                    # Fall back to standard execution
                    return func(self, *args, **kwargs)
            else:
                # Run without acceleration
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class HardwareAccelerator:
    """
    Provides hardware-agnostic acceleration for computationally intensive operations.
    
    This class automatically detects available hardware acceleration options and 
    provides optimized implementations for common operations used in CDFA.
    Supports NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS), and
    CPU-based acceleration (Numba).
    """
    def __init__(self, enable_gpu: bool = True, prefer_cuda: bool = True, 
                 device: Optional[str] = None, log_level: int = logging.INFO,
                 memory_mode: MemoryMode = MemoryMode.AUTO, 
                 optimization_level: str = "balanced",
                 enable_profiling: bool = False):
        """
        Initialize the hardware accelerator.
        
        Args:
            enable_gpu: Whether to use GPU acceleration if available
            prefer_cuda: Whether to prefer CUDA over other GPU backends
            device: Specific device to use (e.g., 'cpu', 'cuda', 'rocm', 'mps')
            log_level: Logging level
            memory_mode: Memory management mode (AUTO, CONSERVATIVE, AGGRESSIVE, DYNAMIC)
            optimization_level: Optimization level ("performance", "balanced", "memory")
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling

        self.logger.info("Detecting hardware acceleration capabilities...")
        
        self._lock = threading.Lock()
        self._torch_script_cache = {}  # Type annotation removed to avoid errors
        self._kernel_cache = {}  # Type annotation removed to avoid errors
        self.hardware_info = HardwareInfo()
        self.memory_mode = memory_mode

        self.logger.info("Detecting hardware acceleration capabilities...")
        
        # Initialize flags for available backends (will be set by _detect_capabilities)
        self._can_use_numba = False
        self._can_use_torch = False
        self._can_use_cuda = False # PyTorch CUDA
        self._can_use_rocm = False
        self._can_use_mps = False
        # Assuming _detect_capabilities also sets these or they are derived from module-level flags
        self._can_use_ipex = IPEX_AVAILABLE 
        self._can_use_rapids = RAPIDS_AVAILABLE
        self._can_use_tvm = TVM_AVAILABLE
        self._can_use_opencl = OPENCL_AVAILABLE
        
        self._device_info = {}  # Type annotation removed to avoid errors
        self._torch_device = None  # Will be set by _init_torch

        # Store preferences
        self._enable_gpu = enable_gpu
        self._prefer_cuda = prefer_cuda
        
        # Initial default device state (will be updated by switch_device)
        self._device = "cpu" 
        self._accelerator_type = AcceleratorType.CPU

        # 2. Detect available libraries and capabilities
        self._detect_capabilities() # Sets self._can_use_numba, self._can_use_torch, self._can_use_cuda, etc.
                                    # and populates self._device_info dictionary.
        
        # 3. Detect detailed hardware information
        self._detect_hardware() # Populates self.hardware_info with specific device details.

        # 4. Determine the target device string
        if device is not None:
            target_device_str = device
            self.logger.info(f"User specified device: {target_device_str}")
        else:
            self.logger.info("Auto-selecting device...")
            # Auto-selection logic with your specified priority order
            if not self._enable_gpu:
                target_device_str = "cpu"
            elif self._can_use_rocm:  # 1. AMD GPUs via ROCm (highest priority)
                target_device_str = "rocm:0"
                self.logger.info(f"Selected AMD GPU via ROCm (highest priority)")
            elif self._can_use_cuda:  # 2. NVIDIA GPUs via CUDA (second priority)
                # Check if this is actually an NVIDIA GPU (not AMD via CUDA API)
                is_nvidia = False
                try:
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        if "NVIDIA" in device_name or "GeForce" in device_name or "RTX" in device_name or "GTX" in device_name:
                            is_nvidia = True
                except Exception:
                    pass
                    
                if is_nvidia:
                    target_device_str = "cuda:0"
                    self.logger.info(f"Selected NVIDIA GPU via CUDA (second priority)")
                else:
                    # If CUDA is available but not NVIDIA, probably means ROCm wasn't detected
                    # Use CUDA API anyway since it might be AMD through CUDA compatibility
                    target_device_str = "cuda:0"
                    self.logger.info(f"Selected GPU via CUDA API (ROCm not explicitly detected)")
            elif self._can_use_mps:  # 3. Apple Silicon via MPS (third priority)
                target_device_str = "mps:0"
                self.logger.info(f"Selected Apple Silicon via MPS (third priority)")
            elif self._can_use_ipex:  # 4. Intel GPUs via IPEX (fourth priority)
                target_device_str = "ipex:0"
                self.logger.info(f"Selected Intel GPU via IPEX (fourth priority)")
            else:  # 5. CPU (final fallback)
                target_device_str = "cpu"
                self.logger.info(f"No GPU detected, using CPU fallback")
            
            self.logger.info(f"Auto-selected device: {target_device_str}")
            
        # 5. Switch to the determined device.
        # self._device is "cpu" initially. switch_device uses it as old_device.
        if not self.switch_device(target_device_str):
            self.logger.warning(
                f"Failed to switch to target device '{target_device_str}'. Attempting to initialize with CPU."
            )
            # If the initial switch failed, self._device might be the old_device ("cpu") or the failed target.
            # Force a switch to "cpu" if not already on it or if the previous attempt failed for "cpu".
            if self._device != "cpu" or target_device_str == "cpu": # check if already CPU or if CPU was the one that failed
                 if not self.switch_device("cpu"):
                    self.logger.critical("CRITICAL: Failed to initialize accelerator even on CPU. Accelerator may be unstable.")
                    # Manually set state to CPU as a last resort
                    self._device = "cpu"
                    self._accelerator_type = self._get_accelerator_type()
                    if self._can_use_torch:
                        try:
                            self._init_torch() # Attempt to initialize PyTorch for CPU
                        except Exception as e_torch_cpu:
                            self.logger.error(f"Failed to initialize PyTorch on CPU after fallback: {e_torch_cpu}")
                    # Other critical initializations for CPU might be needed here.
                 else:
                    self.logger.info("Successfully initialized with CPU after fallback.")
            else: # If self._device is already "cpu" due to how switch_device handles failure
                self.logger.info(f"Continuing with device '{self._device}' after failed switch attempt.")

        # switch_device calls _init_torch(), _init_opencl() (if applicable), _init_platform_specific()
        
        # 6. Initialize Numba (depends on self._accelerator_type set by switch_device)
        self._init_numba()
        
        # Note: The original _configure_libraries() call is removed as its functionality
        # seems covered by the various _init_* methods called directly or via switch_device.

        # 7. Log final initialization state
        self.logger.info(f"Hardware Accelerator initialized with device: {self._device} (Type: {self._accelerator_type})")
        self._log_capabilities()
        self._set_optimization_parameters()
        
        # 8. Initialize profiling if enabled
        if self.enable_profiling:
            self._init_profiling()
        
        
    def _set_optimization_parameters(self):
        """Set optimization parameters based on level."""
        if self.optimization_level == "performance":
            self.params = {
                "precision": "float32",
                "cache_size": "large",
                "threads_per_block": 256,
                "max_memory_usage": 0.8,  # 80% of available memory
                "use_mixed_precision": True
            }
        elif self.optimization_level == "memory":
            self.params = {
                "precision": "float32",
                "cache_size": "small",
                "threads_per_block": 128,
                "max_memory_usage": 0.5,  # 50% of available memory
                "use_mixed_precision": False
            }
        else:  # balanced
            self.params = {
                "precision": "float32",
                "cache_size": "medium",
                "threads_per_block": 192,
                "max_memory_usage": 0.7,  # 70% of available memory
                "use_mixed_precision": True
            }
        
        # Apply these parameters
        self._apply_optimization_parameters()

    def _apply_optimization_parameters(self):
        """Apply optimization parameters to the selected device."""
        # Now we can safely use self._accelerator_type because it's already set
        if self._accelerator_type in (AcceleratorType.CUDA, AcceleratorType.ROCM):
            try:
                # Set memory limits based on optimization level
                if TORCH_AVAILABLE and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(self.params["max_memory_usage"])
                    self.logger.info(f"Set GPU memory usage limit to {self.params['max_memory_usage']*100:.0f}%")
                    
                # Configure thread blocks for CUDA/ROCm kernels
                self._threads_per_block = self.params["threads_per_block"]
            except Exception as e:
                self.logger.warning(f"Failed to apply GPU optimization parameters: {e}")
        
        elif self._accelerator_type == AcceleratorType.CPU:
            # Set thread count for CPU operations
            try:
                if TORCH_AVAILABLE:
                    thread_count = min(os.cpu_count() or 4, 16)  # Reasonable default
                    torch.set_num_threads(thread_count)
                    self.logger.info(f"Set CPU threads to {thread_count}")
            except Exception as e:
                self.logger.warning(f"Failed to set CPU thread count: {e}")
                
    def _detect_capabilities(self):
        """Detect available hardware acceleration capabilities."""
        # Detect NumPy
        self._can_use_numpy = True
        self._device_info["numpy"] = True
        
        # Detect Numba
        try:
            import numba
            self._can_use_numba = True
            self._device_info["numba"] = True
            
            # Get Numba configuration
            self._numba_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
            self._device_info["numba_threads"] = self._numba_threads
            
            # Log Numba detection
            self.logger.info(f"Numba configured with {self._numba_threads} threads")
        except ImportError:
            self._can_use_numba = False
            self._device_info["numba"] = False
        
        # Detect PyTorch
        try:
            # We use the global TORCH_AVAILABLE flag
            self._can_use_torch = TORCH_AVAILABLE
            self._device_info["torch"] = TORCH_AVAILABLE
            
            if TORCH_AVAILABLE:
                self._torch_version = torch.__version__
                
                # Detect CUDA (works for both NVIDIA and AMD via ROCm)
                self._can_use_cuda = CUDA_AVAILABLE
                self._device_info["cuda"] = CUDA_AVAILABLE
                
                if self._can_use_cuda:
                    self._cuda_device_count = torch.cuda.device_count()
                    self._device_info["cuda_device_count"] = self._cuda_device_count
                    if hasattr(torch.version, 'cuda'):
                        self._device_info["cuda_version"] = torch.version.cuda
                    
                    # Detect GPU info
                    device_names = []
                    device_memory = []
                    
                    for i in range(self._cuda_device_count):
                        device_name = torch.cuda.get_device_name(i)
                        device_names.append(device_name)
                        
                        # Get memory info
                        with torch.cuda.device(i):
                            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)  # MB
                            device_memory.append(int(memory_total))
                            
                    self._device_info["cuda_device_names"] = device_names
                    self._device_info["cuda_device_memory"] = device_memory
                    
                    # Log GPU detection
                    if len(device_names) > 0:
                        self.logger.info(f"Detected CUDA device: {device_names[0]} with {device_memory[0]} MB memory")
                
                # Check AMD GPU specifically
                self._can_use_rocm = ROCM_AVAILABLE
                self._device_info["rocm"] = ROCM_AVAILABLE
                
                if self._can_use_rocm:
                    self.logger.info("ROCm: Available (AMD GPU detected)")
                else:
                    self.logger.info("ROCm: Not available")
                
                # Detect MPS (Apple Silicon)
                self._can_use_mps = MPS_AVAILABLE
                self._device_info["mps"] = MPS_AVAILABLE
                
                # Detect supported precision
                self._supported_precisions = ["FP32"]
                
                if self._can_use_cuda or self._can_use_rocm:
                    self._supported_precisions.append("FP16")
                    
                    # Check for Tensor Cores (FP16/BF16)
                    if hasattr(torch.cuda, 'get_device_capability') and torch.cuda.get_device_capability()[0] >= 7:
                        self._supported_precisions.append("BF16")
                        
                # FP64 is always supported but may be slow
                self._supported_precisions.append("FP64")
                
                self._device_info["supported_precisions"] = self._supported_precisions
                
                # Log precision support
                self.logger.info(f"Supported precisions: {', '.join(self._supported_precisions)}")
        except Exception as e:
            self.logger.warning(f"Error detecting PyTorch capabilities: {e}")
            self._can_use_torch = False
            self._can_use_cuda = False
            self._can_use_rocm = False
            self._can_use_mps = False
            self._device_info["torch"] = False
            self._device_info["cuda"] = False
            self._device_info["rocm"] = False
            self._device_info["mps"] = False

    def _detect_hardware(self):
        """Detect available hardware acceleration capabilities."""
        self.logger.info("Detecting hardware acceleration capabilities...")
        
        # Detect CPU capabilities
        cpu_count = os.cpu_count() or 1
        self.hardware_info.add_device("cpu", "CPU", 0, cpu_count * 4096, None)  # Estimate 4GB per core
        
        # Detect CUDA devices
        if self._can_use_cuda:
            try:
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    props = torch.cuda.get_device_properties(i)
                    name = props.name
                    memory = props.total_memory / (1024 * 1024)  # Convert to MB
                    compute_capability = f"{props.major}.{props.minor}"
                    
                    self.hardware_info.add_device(
                        f"cuda:{i}", "CUDA", i, memory, compute_capability
                    )
                    
                    self.logger.info(f"Detected CUDA device: {name} with {memory:.0f} MB memory")
            except Exception as e:
                self.logger.warning(f"Error detecting CUDA devices: {e}")
        
        # Detect ROCm/HIP devices for AMD GPUs
        if self._can_use_rocm:
            try:
                # PyTorch ROCm detection
                if hasattr(torch, 'hip') and hasattr(torch.hip, 'device_count'):
                    rocm_count = torch.hip.device_count()
                else:
                    # Manual detection from environment
                    devices_str = os.environ.get('HIP_VISIBLE_DEVICES', '')
                    if devices_str:
                        devices = devices_str.split(',')
                        rocm_count = len(devices)
                    else:
                        # Try to detect using rocm-smi command
                        try:
                            output = subprocess.check_output(['rocm-smi', '--showmeminfo', 'vram']).decode('utf-8')
                            rocm_count = output.count("GPU")
                        except:
                            rocm_count = 0
                
                for i in range(rocm_count):
                    # Try to get device properties
                    try:
                        current_device = torch.device(f"cuda:{i}")  # ROCm uses cuda namespace in PyTorch
                        torch.cuda.set_device(current_device)
                        props = torch.cuda.get_device_properties(i)
                        name = props.name
                        memory = props.total_memory / (1024 * 1024)  # Convert to MB
                    except:
                        # Fallback to generic info if properties not available
                        name = f"AMD GPU {i}"
                        memory = 8192  # Assume 8GB if unknown
                    
                    self.hardware_info.add_device(
                        f"rocm:{i}", "ROCM", i, memory, None
                    )
                    
                    self.logger.info(f"Detected ROCm device: {name} with {memory:.0f} MB memory")
            except Exception as e:
                self.logger.warning(f"Error detecting ROCm devices: {e}")
        
        # Detect MPS for Apple Silicon
        if self._can_use_mps:
            try:
                if torch.backends.mps.is_available():
                    # macOS system profile to get GPU info
                    try:
                        output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType']).decode('utf-8')
                        
                        # Parse for GPU name
                        name_match = re.search(r"Chipset Model: (.+)", output)
                        name = name_match.group(1) if name_match else "Apple Silicon GPU"
                        
                        # Parse for VRAM
                        vram_match = re.search(r"VRAM \(Dynamic, Max\): (\d+) ([MG])B", output)
                        if vram_match:
                            vram = int(vram_match.group(1))
                            if vram_match.group(2) == 'G':
                                vram *= 1024  # Convert GB to MB
                        else:
                            # Estimate based on common configurations
                            vram = 4096  # Assume 4GB if unknown
                    except:
                        name = "Apple Silicon GPU"
                        vram = 4096  # Assume 4GB if unknown
                    
                    self.hardware_info.add_device(
                        "mps:0", "MPS", 0, vram, None
                    )
                    
                    self.logger.info(f"Detected MPS device: {name} with approximately {vram} MB available memory")
            except Exception as e:
                self.logger.warning(f"Error detecting MPS devices: {e}")
        
        # Detect Intel GPUs with IPEX
        if self._can_use_ipex:
            try:
                # Check for Intel GPUs via SYCL or Level Zero
                intel_gpu_count = 0
                
                # Try to get count using ipex
                try:
                    if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'device_count'):
                        intel_gpu_count = ipex.xpu.device_count()
                except:
                    intel_gpu_count = 0
                
                # Fallback to system tools for detection
                if intel_gpu_count == 0:
                    try:
                        # Check on Linux
                        if platform.system() == "Linux":
                            output = subprocess.check_output(['lspci']).decode('utf-8')
                            if re.search(r"Intel.*Graphics", output):
                                intel_gpu_count = 1
                        # Check on Windows
                        elif platform.system() == "Windows":
                            output = subprocess.check_output(['wmic', 'path', 'win32_VideoController', 'get', 'name']).decode('utf-8')
                            if re.search(r"Intel.*Graphics", output):
                                intel_gpu_count = 1
                    except:
                        pass
                
                for i in range(intel_gpu_count):
                    self.hardware_info.add_device(
                        f"ipex:{i}", "IPEX", i, 4096, None  # Assume 4GB if unknown
                    )
                    
                    self.logger.info(f"Detected Intel XPU device {i}")
            except Exception as e:
                self.logger.warning(f"Error detecting Intel GPUs: {e}")
        
        # Flag hardware info as initialized
        self.hardware_info.initialized = True
        
    def _select_device(self, device: Optional[str]) -> str:
        """
        Select the appropriate computation device.
        
        Args:
            device: Manually specified device
            
        Returns:
            Selected device string
        """
        if device is not None:
            return device
            
        if not self.enable_gpu:
            return "cpu"
            
        # Auto-detection logic
        if self._can_use_cuda and (self.prefer_cuda or not self._can_use_rocm):
            cuda_devices = self.hardware_info.get_devices_by_type("CUDA")
            if cuda_devices:
                return cuda_devices[0]["name"]  # Use first CUDA device
        
        if self._can_use_rocm:
            rocm_devices = self.hardware_info.get_devices_by_type("ROCM")
            if rocm_devices:
                return rocm_devices[0]["name"]  # Use first ROCm device
        
        if self._can_use_mps:
            mps_devices = self.hardware_info.get_devices_by_type("MPS")
            if mps_devices:
                return mps_devices[0]["name"]  # Use MPS device
        
        if self._can_use_ipex:
            ipex_devices = self.hardware_info.get_devices_by_type("IPEX")
            if ipex_devices:
                return ipex_devices[0]["name"]  # Use first Intel GPU
        
        return "cpu"
    
    def _get_accelerator_type(self) -> AcceleratorType:
        """Determine the accelerator type based on selected device."""
        if self._device.startswith("cuda"):
            return AcceleratorType.CUDA
        elif self._device.startswith("rocm") or self._device.startswith("hip"):
            return AcceleratorType.ROCM
        elif self._device.startswith("mps"):
            return AcceleratorType.MPS
        elif self._device.startswith("ipex") or self._device.startswith("xpu"):
            return AcceleratorType.IPEX
        elif self._device.startswith("opencl"):
            return AcceleratorType.OPENCL
        elif self._device.startswith("vulkan"):
            return AcceleratorType.VULKAN
        elif self._device.startswith("tpu") or self._device.startswith("xla"):
            return AcceleratorType.XLA
        elif self._device.startswith("habana"):
            return AcceleratorType.HABANA
        elif self._device == "cpu":
            return AcceleratorType.CPU
        else:
            return AcceleratorType.OTHER
    
    def _detect_rocm_capabilities(self):
        """Detect AMD GPU capabilities through ROCm/HIP."""
        if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
            return False
            
        rocm_available = False
        
        try:
            # Get device name through CUDA API (ROCm uses CUDA namespace in PyTorch)
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"CUDA device detected: {device_name}")
            
            # Check for AMD identifiers in device name
            if any(id in device_name for id in ["AMD", "Radeon", "gfx"]):
                rocm_available = True
                self.logger.info(f"AMD GPU detected via ROCm: {device_name}")
                
                # Get compute capability if available
                try:
                    compute_capability = "unknown"
                    # ROCm might report architecture via CUDA API
                    if hasattr(torch.cuda, 'get_device_capability'):
                        cc = torch.cuda.get_device_capability(0)
                        compute_capability = f"{cc[0]}.{cc[1]}"
                    
                    # Extract additional AMD GPU info
                    props = torch.cuda.get_device_properties(0)
                    gpu_memory = props.total_memory / (1024**3)  # Convert to GB
                    
                    self.logger.info(f"AMD GPU specs: {compute_capability}, {gpu_memory:.2f} GB memory")
                except Exception as e:
                    self.logger.warning(f"Error getting AMD GPU details: {e}")
        except Exception as e:
            self.logger.warning(f"Error detecting GPU via CUDA API: {e}")
        
        # Additional checks for ROCm/HIP environment
        if not rocm_available and platform.system() == "Linux":
            # Check ROCm environment variables
            rocm_env_vars = ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']
            if any(var in os.environ for var in rocm_env_vars):
                rocm_available = True
                self.logger.info("AMD GPU detected via environment variables")
            
            # Check system commands
            try:
                # Try rocm-smi command
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                if "GPU" in result.stdout and "AMD" in result.stdout:
                    rocm_available = True
                    self.logger.info("AMD GPU detected via rocm-smi")
            except Exception:
                pass
        
        return rocm_available
    
    def _init_numba(self):
        """Initialize Numba acceleration if available."""
        if not self._can_use_numba:
            self.logger.info("Numba not available. CPU acceleration will be limited.")
            return
            
        # Configure thread count for parallel execution
        try:
            thread_count = os.cpu_count() or 4
            nb.set_num_threads(thread_count)
            self.logger.info(f"Numba configured with {thread_count} threads")
            
            # Enable Numba CUDA if available
            if self._accelerator_type == AcceleratorType.CUDA and NUMBA_CUDA_AVAILABLE:
                self.logger.info("Numba CUDA acceleration enabled")
        except Exception as e:
            self.logger.warning(f"Failed to configure Numba: {e}")
    
    def _init_torch(self):
        """Initialize PyTorch and configure device."""
        if not self._can_use_torch:
            self.logger.info("PyTorch not available. GPU acceleration will be disabled.")
            return
            
        try:
            # Prioritize ROCm for AMD GPUs
            if self._device.startswith("rocm") and self._can_use_rocm:
                # ROCm uses cuda namespace in PyTorch
                if ":" in self._device:
                    index = int(self._device.split(":")[-1])
                    self.torch_device = torch.device(f"cuda:{index}")
                else:
                    self.torch_device = torch.device("cuda")
                
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(self.torch_device)
                    self.logger.info(f"Using AMD ROCm device: {gpu_name}")
                    
                    # Configure memory usage based on mode
                    if self.memory_mode == MemoryMode.CONSERVATIVE:
                        # Limit memory growth
                        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                            torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available memory
                    elif self.memory_mode == MemoryMode.AGGRESSIVE:
                        # Allow memory growth
                        torch.cuda.empty_cache()
            
            # NVIDIA CUDA GPUs (second priority)
            elif self._device.startswith("cuda") and self._can_use_cuda:
                # Extract device index if present
                if ":" in self._device:
                    index = int(self._device.split(":")[-1])
                    self.torch_device = torch.device(f"cuda:{index}")
                else:
                    self.torch_device = torch.device("cuda")
                
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(self.torch_device)
                    self.logger.info(f"Using CUDA device: {gpu_name}")
                    
                    # Configure memory usage based on mode
                    if self.memory_mode == MemoryMode.CONSERVATIVE:
                        # Limit memory growth
                        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available memory
                    elif self.memory_mode == MemoryMode.AGGRESSIVE:
                        # Allow memory growth
                        torch.cuda.empty_cache()
                    
                    # Enable TF32 for newer GPUs
                    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        self.logger.info("TF32 enabled for applicable operations")
                    
                    # Enable cuDNN benchmarking for performance
                    torch.backends.cudnn.benchmark = True
            
            # Apple Silicon via MPS (third priority)
            elif self._device.startswith("mps") and self._can_use_mps:
                self.torch_device = torch.device("mps")
                self.logger.info("Using Apple Metal Performance Shaders (MPS) for acceleration")
            
            # Intel GPUs via IPEX (fourth priority)
            elif self._device.startswith("ipex") and self._can_use_ipex:
                # Initialize Intel GPU via IPEX
                if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'device'):
                    if ":" in self._device:
                        index = int(self._device.split(":")[-1])
                        self.torch_device = ipex.xpu.device(f"xpu:{index}")
                    else:
                        self.torch_device = ipex.xpu.device("xpu:0")
                    
                    self.logger.info("Using Intel GPU via IPEX for acceleration")
                else:
                    self.torch_device = torch.device("cpu")
                    self.logger.warning("Intel GPU requested but not fully available, falling back to CPU")
            
            # CPU fallback (lowest priority)
            else:
                self.torch_device = torch.device("cpu")
                self.logger.info("Using CPU for PyTorch operations")
            
            # Set default tensor type for better performance
            if self.torch_device.type != "cpu":
                torch.set_default_tensor_type(torch.FloatTensor)
            
            # Initialize precision configuration
            self._init_precision_config()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize PyTorch device: {e}")
            self.torch_device = torch.device("cpu")
        
        
    def _init_precision_config(self):
        """Initialize precision configuration for PyTorch."""
        if not self._can_use_torch:
            return
        
        # Set precision configuration based on device capabilities
        self.supports_half = False
        self.supports_bfloat16 = False
        self.supports_fp64 = False
        
        try:
            if self.torch_device.type == "cuda":
                # Check for half precision (FP16) support
                self.supports_half = True
                
                # Check for bfloat16 support (CUDA compute capability 8.0+)
                if torch.cuda.get_device_capability(self.torch_device.index if self.torch_device.index is not None else 0) >= (8, 0):
                    self.supports_bfloat16 = True
                
                # All CUDA devices support FP64, but with different performance
                self.supports_fp64 = True
                
            elif self.torch_device.type == "mps":
                # MPS supports half precision
                self.supports_half = True
                
                # MPS does not support bfloat16 as of 2023
                self.supports_bfloat16 = False
                
                # MPS supports fp64
                self.supports_fp64 = True
                
            elif self.torch_device.type == "xpu":
                # Intel GPUs support half precision
                self.supports_half = True
                
                # Some Intel GPUs support bfloat16
                try:
                    # Test bfloat16 support
                    x = torch.tensor([1.0], device=self.torch_device).to(torch.bfloat16)
                    self.supports_bfloat16 = True
                except:
                    self.supports_bfloat16 = False
                
                # Intel GPUs support fp64
                self.supports_fp64 = True
            
            # CPU supports all precisions
            elif self.torch_device.type == "cpu":
                self.supports_half = True
                self.supports_bfloat16 = True
                self.supports_fp64 = True
            
            # Log supported precisions
            precision_msg = f"Supported precisions: FP32"
            if self.supports_half:
                precision_msg += ", FP16"
            if self.supports_bfloat16:
                precision_msg += ", BF16"
            if self.supports_fp64:
                precision_msg += ", FP64"
            
            self.logger.info(precision_msg)
            
        except Exception as e:
            self.logger.warning(f"Error determining precision support: {e}")
            # Default to conservative settings
            self.supports_half = False
            self.supports_bfloat16 = False
            self.supports_fp64 = True
    
    def _init_opencl(self):
        """Initialize OpenCL if available."""
        if not self._can_use_opencl:
            return
        
        try:
            # Get OpenCL platforms
            platforms = cl.get_platforms()
            if not platforms:
                self.logger.warning("No OpenCL platforms found")
                self._can_use_opencl = False
                return
            
            # Create context and command queue for first available device
            for platform in platforms:
                try:
                    # Try to get GPU devices first
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if not devices:
                        # Fall back to any device
                        devices = platform.get_devices()
                    
                    if devices:
                        self.opencl_ctx = cl.Context(devices=[devices[0]])
                        self.opencl_queue = cl.CommandQueue(self.opencl_ctx)
                        self.opencl_device = devices[0]
                        
                        # Add device to hardware info
                        device_name = devices[0].name
                        device_memory = devices[0].global_mem_size / (1024 * 1024)  # Convert to MB
                        
                        self.hardware_info.add_device(
                            f"opencl:{platform.name}:{device_name}", 
                            "OPENCL", 
                            0, 
                            device_memory, 
                            None
                        )
                        
                        self.logger.info(f"Initialized OpenCL device: {device_name} on platform {platform.name}")
                        break
                except cl.Error as e:
                    self.logger.warning(f"Error initializing OpenCL device on platform {platform.name}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenCL: {e}")
            self._can_use_opencl = False
    
    def _init_platform_specific(self):
        """Initialize platform-specific optimizations."""
        # NVIDIA-specific optimizations
        if self._accelerator_type == AcceleratorType.CUDA:
            self._init_cuda_specific()
        
        # AMD-specific optimizations
        elif self._accelerator_type == AcceleratorType.ROCM:
            self._init_rocm_specific()
        
        # Apple-specific optimizations
        elif self._accelerator_type == AcceleratorType.MPS:
            self._init_mps_specific()
        
        # Intel-specific optimizations
        elif self._accelerator_type == AcceleratorType.IPEX:
            self._init_intel_specific()
    
    def _init_cuda_specific(self):
        """Initialize NVIDIA CUDA-specific optimizations."""
        if not self._can_use_cuda:
            return
        
        try:
            # Set up CUDA device properties
            cuda_index = 0
            if ":" in self._device:
                cuda_index = int(self._device.split(":")[-1])
            
            # Get compute capability
            compute_capability = torch.cuda.get_device_capability(cuda_index)
            compute_capability_str = f"{compute_capability[0]}.{compute_capability[1]}"
            
            # Store in class for reference
            self.cuda_compute_capability = compute_capability
            self.cuda_compute_capability_str = compute_capability_str
            
            # Enable TensorCores if available (compute capability 7.0+)
            if compute_capability >= (7, 0):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                self.logger.info(f"TensorCores enabled on device with compute capability {compute_capability_str}")
            
            # Get memory information
            total_memory = torch.cuda.get_device_properties(cuda_index).total_memory
            self.total_cuda_memory = total_memory
            
            self.logger.info(f"CUDA device has {total_memory / (1024**3):.2f} GB total memory")
            
        except Exception as e:
            self.logger.warning(f"Error initializing CUDA-specific optimizations: {e}")
    
    def _init_rocm_specific(self):
        """Initialize AMD ROCm-specific optimizations."""
        if not self._can_use_rocm:
            return
        
        try:
            # ROCm uses CUDA backends in PyTorch
            rocm_index = 0
            if ":" in self._device:
                rocm_index = int(self._device.split(":")[-1])
            
            # Get memory information if possible
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(rocm_index)
                total_memory = props.total_memory
                self.total_rocm_memory = total_memory
                
                self.logger.info(f"ROCm device has {total_memory / (1024**3):.2f} GB total memory")
            
            # Check for mixed precision support
            has_fp16 = True  # Most AMD GPUs support FP16
            self.rocm_has_fp16 = has_fp16
            
            # Enable ROCm-specific optimizations
            if has_fp16:
                self.logger.info("ROCm FP16 (half precision) is available")
            
        except Exception as e:
            self.logger.warning(f"Error initializing ROCm-specific optimizations: {e}")
    
    def _init_mps_specific(self):
        """Initialize Apple Metal-specific optimizations."""
        if not self._can_use_mps:
            return
        
        try:
            # Check if PyTorch is built with MPS
            if torch.backends.mps.is_available():
                # Enable MPS-specific optimizations
                self.logger.info("Metal Performance Shaders (MPS) acceleration enabled")
                
                # Note: unlike CUDA, Metal doesn't allow direct memory queries
                # so we can't get memory information programmatically
            
        except Exception as e:
            self.logger.warning(f"Error initializing MPS-specific optimizations: {e}")
    
    def _init_intel_specific(self):
        """Initialize Intel-specific optimizations."""
        if not self._can_use_ipex:
            return
        
        try:
            # Enable Intel IPEX optimizations
            if IPEX_AVAILABLE:
                # Initialize Intel extension for PyTorch
                torch.xpu.set_device(0)  # Set default device
                
                # Enable BFloat16 if available (some Intel GPUs support BF16)
                self.intel_has_bf16 = True  # Assumed for newer Intel GPUs
                
                self.logger.info("Intel XPU acceleration enabled via IPEX")
            
        except Exception as e:
            self.logger.warning(f"Error initializing Intel-specific optimizations: {e}")
    
    def _log_capabilities(self):
        """Log hardware capabilities."""
        self.logger.info(f"Numba JIT: {'Available' if self._can_use_numba else 'Not available'}")
        self.logger.info(f"PyTorch: {'Available' if self._can_use_torch else 'Not available'}")
        self.logger.info(f"CUDA: {'Available' if self._can_use_cuda else 'Not available'}")
        self.logger.info(f"ROCm: {'Available' if self._can_use_rocm else 'Not available'}")
        self.logger.info(f"MPS: {'Available' if self._can_use_mps else 'Not available'}")
        self.logger.info(f"IPEX: {'Available' if self._can_use_ipex else 'Not available'}")
        self.logger.info(f"RAPIDS: {'Available' if self._can_use_rapids else 'Not available'}")
        self.logger.info(f"TVM: {'Available' if self._can_use_tvm else 'Not available'}")
        self.logger.info(f"OpenCL: {'Available' if self._can_use_opencl else 'Not available'}")
        
    def _init_profiling(self):
        """Initialize profiling for performance monitoring."""
        self.profiling_data = {
            "function_calls": {},
            "execution_times": {},
            "memory_usage": {},
            "device_utilization": {},
            "call_stack": [],
            "start_time": time.time()
        }
        
        self.logger.info("Performance profiling enabled")
        
    def profile_function(self, func_name: str):
        """Decorator for profiling function execution times."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                    
                start_time = time.time()
                
                # Track memory before execution
                memory_before = self.get_memory_usage() if hasattr(self, 'get_memory_usage') else None
                
                # Execute the function
                self.profiling_data["call_stack"].append(func_name)
                result = func(*args, **kwargs)
                self.profiling_data["call_stack"].pop()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Update profiling data
                if func_name not in self.profiling_data["function_calls"]:
                    self.profiling_data["function_calls"][func_name] = 0
                    self.profiling_data["execution_times"][func_name] = {
                        "total": 0.0,
                        "min": float('inf'),
                        "max": 0.0,
                        "avg": 0.0
                    }
                
                self.profiling_data["function_calls"][func_name] += 1
                self.profiling_data["execution_times"][func_name]["total"] += execution_time
                self.profiling_data["execution_times"][func_name]["min"] = min(
                    self.profiling_data["execution_times"][func_name]["min"], 
                    execution_time
                )
                self.profiling_data["execution_times"][func_name]["max"] = max(
                    self.profiling_data["execution_times"][func_name]["max"], 
                    execution_time
                )
                self.profiling_data["execution_times"][func_name]["avg"] = (
                    self.profiling_data["execution_times"][func_name]["total"] / 
                    self.profiling_data["function_calls"][func_name]
                )
                
                # Track memory after execution
                if memory_before is not None:
                    memory_after = self.get_memory_usage()
                    memory_diff = {k: memory_after.get(k, 0) - memory_before.get(k, 0) 
                                 for k in set(memory_after) | set(memory_before)}
                    
                    if func_name not in self.profiling_data["memory_usage"]:
                        self.profiling_data["memory_usage"][func_name] = []
                    
                    self.profiling_data["memory_usage"][func_name].append(memory_diff)
                
                return result
            return wrapper
        return decorator
        
    def get_profiling_data(self):
        """Get profiling data in a formatted dictionary."""
        if not self.enable_profiling:
            return {"profiling_enabled": False}
            
        return {
            "profiling_enabled": True,
            "total_runtime": time.time() - self.profiling_data["start_time"],
            "function_calls": self.profiling_data["function_calls"],
            "execution_times": self.profiling_data["execution_times"],
            "memory_usage": self.profiling_data["memory_usage"],
            "device_utilization": self.profiling_data["device_utilization"],
            "device_type": str(self._accelerator_type)
        }
    
    def get_device(self) -> str:
        """Get the current computation device."""
        return self._device
    
    def get_torch_device(self) -> Any:
        """Get the PyTorch device object."""
        if not self._can_use_torch:
            raise RuntimeError("PyTorch is not available")
        return self.torch_device
    
    def get_accelerator_type(self) -> AcceleratorType:
        """Get the current accelerator type."""
        return self._accelerator_type
    
    def get_hardware_info(self) -> HardwareInfo:
        """Get hardware information."""
        return self.hardware_info
    
    def switch_device(self, device: str) -> bool:
        """
        Switch to a different computation device.
        
        Args:
            device: New device to use
            
        Returns:
            Success flag
        """
        with self._lock:
            old_device = self._device
            
            # Validate device
            if device.startswith("cuda") and not self._can_use_cuda:
                self.logger.error("CUDA requested but not available")
                return False
            
            if device.startswith("rocm") and not self._can_use_rocm:
                self.logger.error("ROCm requested but not available")
                return False
            
            if device.startswith("mps") and not self._can_use_mps:
                self.logger.error("MPS requested but not available")
                return False
            
            if device.startswith("ipex") and not self._can_use_ipex:
                self.logger.error("IPEX requested but not available")
                return False
            
            # Update device and type
            self._device = device
            self._accelerator_type = self._get_accelerator_type()
            
            # Update PyTorch device if available
            if self._can_use_torch:
                self._init_torch()
            
            # Update OpenCL if applicable
            if self._can_use_opencl and device.startswith("opencl"):
                self._init_opencl()
            
            # Update platform-specific optimizations
            self._init_platform_specific()
            
            self.logger.info(f"Switched device from {old_device} to {self._device}")
            return True
    
    def to_device(self, data: Any) -> Any:
        """
        Move data to the appropriate device for computation.
        
        Args:
            data: Input data (PyTorch tensor, numpy array, or other)
            
        Returns:
            Data on the target device
        """
        if not self._can_use_torch:
            return data
            
        try:
            # Convert numpy arrays to PyTorch tensors
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
                return tensor.to(self.torch_device)
            # Move tensors to device
            elif isinstance(data, torch.Tensor):
                return data.to(self.torch_device)
            # Handle lists of tensors
            elif isinstance(data, list) and all(isinstance(x, torch.Tensor) for x in data):
                return [x.to(self.torch_device) for x in data]
            # Handle dictionaries with tensor values
            elif isinstance(data, dict) and all(isinstance(v, torch.Tensor) for v in data.values()):
                return {k: v.to(self.torch_device) for k, v in data.items()}
            # Handle mixed dictionaries
            elif isinstance(data, dict):
                return {k: self.to_device(v) if isinstance(v, (torch.Tensor, np.ndarray, list, dict)) else v 
                        for k, v in data.items()}
            else:
                return data
        except Exception as e:
            self.logger.warning(f"Failed to move data to device: {e}")
            return data
    
    def to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert data to numpy array.
        
        Args:
            data: Input data (PyTorch tensor, list, or other)
            
        Returns:
            Numpy array
        """
        if isinstance(data, torch.Tensor):
            # Move to CPU first if needed
            if data.device != torch.device("cpu"):
                data = data.cpu()
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            raise TypeError(f"Cannot convert {type(data)} to numpy array")

    
    def apply_mixed_precision(self, model: Any) -> Any:
        """
        Apply mixed precision to PyTorch model based on device capabilities.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with mixed precision applied if supported
        """
        if not self._can_use_torch:
            return model
        
        try:
            # Check if model is a PyTorch model
            if not isinstance(model, nn.Module):
                self.logger.warning(f"Model is not a PyTorch model: {type(model)}")
                return model
            
            # Move model to device first
            model = model.to(self.torch_device)
            
            # Apply mixed precision based on device capabilities
            if self._accelerator_type == AcceleratorType.CUDA and self.supports_half:
                # For NVIDIA GPUs
                from torch.cuda.amp import autocast
                
                # Wrap model's forward method with autocast
                original_forward = model.forward
                
                @wraps(original_forward)
                def forward_with_autocast(*args, **kwargs):
                    with autocast():
                        return original_forward(*args, **kwargs)
                
                model.forward = forward_with_autocast
                self.logger.info("Applied mixed precision (FP16) to model")
                
            elif self._accelerator_type == AcceleratorType.ROCM and self.rocm_has_fp16:
                # For AMD GPUs (similar to CUDA approach)
                from torch.cuda.amp import autocast
                
                # Wrap model's forward method with autocast
                original_forward = model.forward
                
                @wraps(original_forward)
                def forward_with_autocast(*args, **kwargs):
                    with autocast():
                        return original_forward(*args, **kwargs)
                
                model.forward = forward_with_autocast
                self.logger.info("Applied mixed precision (FP16) to model")
                
            elif self._accelerator_type == AcceleratorType.MPS and self.supports_half:
                # For Apple Silicon
                # As of 2023, PyTorch MPS doesn't have a dedicated autocast
                # but we can manually convert to float16
                model = model.half()
                self.logger.info("Applied half precision (FP16) to model")
                
            elif self._accelerator_type == AcceleratorType.IPEX and IPEX_AVAILABLE:
                # For Intel GPUs
                if self.supports_bfloat16:
                    # Apply IPEX optimization with BF16
                    model = ipex.optimize(model, dtype=torch.bfloat16)
                    self.logger.info("Applied IPEX optimization with BF16 to model")
                else:
                    # Apply IPEX optimization with FP32
                    model = ipex.optimize(model)
                    self.logger.info("Applied IPEX optimization to model")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to apply mixed precision: {e}")
            return model
    
    # ----- Accelerated Algorithm Implementations -----
    
    @staticmethod
    @njit(float64[:](float64[:]), cache=True, fastmath=True)
    def _normalize_scores_numba(scores: np.ndarray) -> np.ndarray:
        """
        Accelerated score normalization using Numba.
        
        Args:
            scores: Input array of scores
            
        Returns:
            Normalized scores
        """
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
    
    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _normalize_matrix_numba(matrix: np.ndarray) -> np.ndarray:
        """
        Accelerated matrix normalization using Numba with parallel processing.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Normalized matrix
        """
        result = np.zeros_like(matrix)
        rows, cols = matrix.shape
        
        # Process each row in parallel
        for i in prange(rows):
            # Find min/max for this row
            row_min = np.inf
            row_max = -np.inf
            has_finite = False
            
            for j in range(cols):
                if np.isfinite(matrix[i, j]):
                    has_finite = True
                    if matrix[i, j] < row_min:
                        row_min = matrix[i, j]
                    if matrix[i, j] > row_max:
                        row_max = matrix[i, j]
            
            # Normalize this row
            if has_finite and row_max - row_min > 1e-9:
                row_range = row_max - row_min
                for j in range(cols):
                    if np.isfinite(matrix[i, j]):
                        result[i, j] = (matrix[i, j] - row_min) / row_range
                    else:
                        result[i, j] = 0.5
            else:
                # Set neutral values if no range
                for j in range(cols):
                    result[i, j] = 0.5
                    
        return result
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _calculate_distance_matrix_numba(data: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise Euclidean distance matrix with Numba acceleration.
        
        Args:
            data: Input data matrix (n_samples, n_features)
            
        Returns:
            Distance matrix (n_samples, n_samples)
        """
        n_samples = data.shape[0]
        distances = np.zeros((n_samples, n_samples), dtype=np.float64)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = 0.0
                for k in range(data.shape[1]):
                    diff = data[i, k] - data[j, k]
                    dist += diff * diff
                dist = math.sqrt(dist)
                distances[i, j] = distances[j, i] = dist
                
        return distances
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _dtw_distance_numba(s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        """
        Numba-accelerated DTW distance calculation.
        
        Args:
            s1: First time series
            s2: Second time series
            window: Warping window size
            
        Returns:
            DTW distance
        """
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

    
    @staticmethod
    def _normalize_scores_cpu(scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0,1] range (CPU implementation).
        
        Args:
            scores: Input array of scores
            
        Returns:
            Normalized scores
        """
        if len(scores) == 0:
            return np.array([])
            
        # Create mask for finite values
        finite_mask = np.isfinite(scores)
        
        if not np.any(finite_mask):
            return np.full_like(scores, 0.5)
            
        # Get min/max of finite values
        finite_values = scores[finite_mask]
        min_s = np.min(finite_values)
        max_s = np.max(finite_values)
        s_range = max_s - min_s
        
        if s_range < 1e-9:
            return np.full_like(scores, 0.5)
            
        # Normalize using vectorized operations
        normalized = np.full_like(scores, 0.5)
        normalized[finite_mask] = np.clip((scores[finite_mask] - min_s) / s_range, 0.0, 1.0)
        
        return normalized
    
    def normalize_scores_torch(self, scores):
        """
        Normalize scores to [0,1] range with PyTorch (GPU-enabled).
        
        Args:
            scores: Input tensor of scores
            
        Returns:
            Normalized scores tensor
        """
        if not TORCH_AVAILABLE:
            # Fall back to numpy
            numpy_scores = self.to_numpy(scores) if hasattr(scores, 'numpy') else scores
            return self._normalize_scores_cpu(numpy_scores)
            
        # Ensure tensor is on the correct device
        scores = scores.to(self.torch_device)
        
        # Create mask for finite values
        finite_mask = torch.isfinite(scores)
        
        if not torch.any(finite_mask):
            return torch.full_like(scores, 0.5)
            
        # Get min/max of finite values only
        finite_values = scores[finite_mask]
        min_s = torch.min(finite_values)
        max_s = torch.max(finite_values)
        s_range = max_s - min_s
        
        if s_range < 1e-9:
            return torch.full_like(scores, 0.5)
            
        # Normalize using vectorized operations
        normalized = torch.full_like(scores, 0.5)
        normalized[finite_mask] = torch.clamp((scores[finite_mask] - min_s) / s_range, 0.0, 1.0)
        
        return normalized

    def normalize_scores(self, scores):
        """
        Normalize scores to [0,1] range.
        
        Args:
            scores: Input scores
            
        Returns:
            Normalized scores
        """
        if self._can_use_torch and isinstance(scores, torch.Tensor):
            return self.normalize_scores_torch(scores)
        else:
            # Convert to numpy if needed
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            return self._normalize_scores_cpu(scores)
    
    def normalize_matrix(self, matrix: Union[np.ndarray, 'TorchTensor'], 
                       axis: Optional[int] = None) -> Union[np.ndarray, 'TorchTensor']:
        """
        Normalize matrix values along specified axis.
        
        Args:
            matrix: Input matrix
            axis: Axis for normalization (None for global)
            
        Returns:
            Normalized matrix
        """
        is_torch_tensor = isinstance(matrix, torch.Tensor)
        
        if is_torch_tensor:
            if not self._can_use_torch:
                # Fall back to numpy
                numpy_matrix = self.to_numpy(matrix)
                result = self._normalize_matrix_numpy(numpy_matrix, axis)
                return torch.from_numpy(result).to(matrix.device)
                
            # Process with PyTorch
            return self._normalize_matrix_torch(matrix, axis)
        else:
            # Process with Numba or NumPy
            if self._can_use_numba and matrix.shape[0] > 10 and axis in (0, 1) and matrix.ndim == 2:
                if axis == 0:
                    # Normalize columns
                    return self._normalize_matrix_numba(matrix.T).T
                else:
                    # Normalize rows
                    return self._normalize_matrix_numba(matrix)
            else:
                # Fall back to numpy
                return self._normalize_matrix_numpy(matrix, axis)
    
    def _normalize_matrix_torch(self, matrix: 'TorchTensor', axis: Optional[int] = None) -> 'TorchTensor':
        """
        Normalize matrix using PyTorch.
        
        Args:
            matrix: Input matrix tensor
            axis: Axis for normalization (None for global)
            
        Returns:
            Normalized matrix tensor
        """
        # Ensure tensor is on the correct device
        if not self._can_use_torch:
            return self._normalize_matrix_numpy(self.to_numpy(matrix))
        matrix = matrix.to(self.torch_device)
        
        if axis is None:
            # Global normalization
            finite_mask = torch.isfinite(matrix)
            if not torch.any(finite_mask):
                return torch.full_like(matrix, 0.5)
                
            # Get min/max of finite values
            finite_values = matrix[finite_mask]
            min_val = torch.min(finite_values)
            max_val = torch.max(finite_values)
            value_range = max_val - min_val
            
            if value_range < 1e-9:
                return torch.full_like(matrix, 0.5)
                
            # Normalize using vectorized operations
            normalized = torch.full_like(matrix, 0.5)
            normalized[finite_mask] = torch.clamp((matrix[finite_mask] - min_val) / value_range, 0.0, 1.0)
            
            return normalized
        else:
            # Axis-wise normalization
            dim_size = matrix.shape[axis]
            result = torch.full_like(matrix, 0.5)
            
            # Process each slice along the specified axis
            for i in range(dim_size):
                if axis == 0:
                    # Normalize each column
                    col = matrix[:, i]
                    result[:, i] = self.normalize_scores_torch(col)
                elif axis == 1:
                    # Normalize each row
                    row = matrix[i, :]
                    result[i, :] = self.normalize_scores_torch(row)
                else:
                    # Handle higher dimensions
                    # This is a simplified approach that doesn't fully leverage GPU parallelism
                    # For production, consider using torch.apply_along_axis or custom kernels
                    slices = [slice(None)] * matrix.ndim
                    slices[axis] = i
                    result[tuple(slices)] = self.normalize_scores_torch(matrix[tuple(slices)])
                    
            return result
    
    def _normalize_matrix_numpy(self, matrix: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Normalize matrix using NumPy.
        
        Args:
            matrix: Input matrix
            axis: Axis for normalization (None for global)
            
        Returns:
            Normalized matrix
        """
        if axis is None:
            # Global normalization
            return self._normalize_scores_cpu(matrix.flatten()).reshape(matrix.shape)
        else:
            # Axis-wise normalization
            result = np.full_like(matrix, 0.5)
            
            # Apply normalization along specified axis
            for i in range(matrix.shape[axis]):
                # Extract slice
                slices = [slice(None)] * matrix.ndim
                slices[axis] = i
                
                # Normalize slice
                result[tuple(slices)] = self._normalize_scores_cpu(matrix[tuple(slices)])
                
            return result
    
    def calculate_distance_matrix(self, data: Union[np.ndarray, 'TorchTensor']) -> Union[np.ndarray, 'TorchTensor']:
        """
        Calculate pairwise Euclidean distance matrix with hardware acceleration.
        
        Args:
            data: Input data matrix (n_samples, n_features)
            
        Returns:
            Distance matrix (n_samples, n_samples)
        """
        is_torch_tensor = isinstance(data, torch.Tensor)
        
        # PyTorch implementation for GPU acceleration
        if is_torch_tensor and self._can_use_torch:
            # Ensure data is on the correct device
            data = data.to(self.torch_device)
            
            # Calculate squared Euclidean distances efficiently
            norm = torch.sum(data * data, dim=1, keepdim=True)
            dist = norm + norm.transpose(0, 1) - 2.0 * torch.mm(data, data.transpose(0, 1))
            
            # Fix potential numerical issues
            dist = torch.clamp(dist, min=0.0)
            
            # Take square root for Euclidean distance
            return torch.sqrt(dist)
            
        # Numba implementation for CPU acceleration
        else:
            if is_torch_tensor:
                numpy_data = self.to_numpy(data)
            else:
                numpy_data = data
                
            if self._can_use_numba and len(numpy_data) > 10:
                result = self._calculate_distance_matrix_numba(numpy_data)
            else:
                # Vectorized numpy implementation
                norm = np.sum(numpy_data * numpy_data, axis=1)
                result = np.sqrt(np.maximum(0, norm[:, np.newaxis] + norm[np.newaxis, :] - 2 * np.dot(numpy_data, numpy_data.T)))
                
            # Return in the same format as input
            if is_torch_tensor:
                return torch.from_numpy(result).to(data.device)
            else:
                return result
            
    def calculate_returns(self, prices: np.ndarray, mode: str = 'percent') -> np.ndarray:
        """
        Calculate returns from price series with hardware acceleration.
        
        Args:
            prices: Price series
            mode: Return calculation mode ('percent', 'log', or 'diff')
            
        Returns:
            Returns series
        """
        # Map mode to numeric value
        mode_map = {'percent': 0, 'log': 1, 'diff': 2}
        mode_num = mode_map.get(mode.lower(), 0)
        
        # Use Numba if available and prices large enough
        if NUMBA_AVAILABLE and len(prices) > 100:
            return self._calculate_returns_numba(prices, mode_num)
        
        # Fallback to numpy implementation
        n = len(prices)
        returns = np.zeros(n)
        
        if mode == 'log':
            # Log returns
            returns[1:] = np.log(prices[1:] / prices[:-1])
        elif mode == 'diff':
            # Simple difference
            returns[1:] = prices[1:] - prices[:-1]
        else:
            # Percent returns (default)
            returns[1:] = (prices[1:] / prices[:-1]) - 1.0
            
        return returns
    
    def calculate_volatility(self, returns: np.ndarray, window: int = 20,
                          annualization_factor: float = 252.0) -> np.ndarray:
        """
        Calculate rolling volatility with hardware acceleration.
        
        Args:
            returns: Return series
            window: Window size
            annualization_factor: Annualization factor (252=daily, 52=weekly, 12=monthly)
            
        Returns:
            Rolling volatility (annualized)
        """
        # Use Numba if available and returns large enough
        if NUMBA_AVAILABLE and len(returns) > 100:
            return self._calculate_rolling_volatility_numba(returns, window, annualization_factor)
        
        # Fallback to numpy implementation
        n = len(returns)
        volatility = np.zeros(n)
        
        for i in range(window, n):
            window_returns = returns[i-window:i]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                std = np.std(valid_returns, ddof=1)
                volatility[i] = std * np.sqrt(annualization_factor)
                
        return volatility
    
    def calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """
        Calculate RSI with hardware acceleration.
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            RSI values
        """
        # Use Numba if available and prices large enough
        if NUMBA_AVAILABLE and len(prices) > 100:
            return self._calculate_rsi_numba(prices, window)
        
        # Fallback to numpy implementation
        n = len(prices)
        rsi = np.zeros(n)
        
        # Calculate price changes
        diff = np.zeros(n)
        diff[1:] = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gain = np.zeros(n)
        loss = np.zeros(n)
        
        for i in range(1, n):
            if diff[i] > 0:
                gain[i] = diff[i]
            elif diff[i] < 0:
                loss[i] = -diff[i]
                
        # Calculate average gain and loss
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        # First average
        if n > window:
            avg_gain[window] = np.mean(gain[1:window+1])
            avg_loss[window] = np.mean(loss[1:window+1])
            
            # Calculate remaining values
            for i in range(window+1, n):
                avg_gain[i] = ((avg_gain[i-1] * (window-1)) + gain[i]) / window
                avg_loss[i] = ((avg_loss[i-1] * (window-1)) + loss[i]) / window
                
            # Calculate RSI
            rs = np.divide(avg_gain[window:], avg_loss[window:], 
                         out=np.zeros_like(avg_gain[window:]), 
                         where=avg_loss[window:] != 0)
            
            rsi[window:] = 100.0 - (100.0 / (1.0 + rs))
            
            # Set RSI to 100 where avg_loss is zero
            rsi[window:] = np.where(avg_loss[window:] == 0, 100.0, rsi[window:])
            
        return rsi
    
    def calculate_bollinger_bands(self, ohlcv_data: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Calculate Bollinger Bands with hardware acceleration.
        
        Args:
            ohlcv_data: OHLCV data [open, high, low, close, volume]
            window: Window size
            
        Returns:
            Array with [upper, middle, lower, bandwidth] bands
        """
        # Extract OHLCV components
        if ohlcv_data.ndim == 2 and ohlcv_data.shape[1] >= 5:
            close = ohlcv_data[:, 3]
            high = ohlcv_data[:, 1]
            low = ohlcv_data[:, 2]
            volume = ohlcv_data[:, 4]
        else:
            # Assume input is just close prices
            close = ohlcv_data
            high = close
            low = close
            volume = np.ones_like(close)
            
        # Use Numba if available and data large enough
        if NUMBA_AVAILABLE and len(close) > 100:
            return self._calculate_bollinger_bands_numba(close, high, low, volume, window)
        
        # Fallback to numpy implementation
        n = len(close)
        bands = np.zeros((n, 4))  # [upper, middle, lower, bandwidth]
        
        # Calculate SMA and standard deviation
        sma = np.zeros(n)
        std = np.zeros(n)
        
        for i in range(window-1, n):
            window_data = close[i-window+1:i+1]
            valid_data = window_data[np.isfinite(window_data)]
            
            if len(valid_data) > 0:
                sma[i] = np.mean(valid_data)
                
                if len(valid_data) > 1:
                    std[i] = np.std(valid_data, ddof=1)
                    
        # Calculate bands
        bands[:, 1] = sma
        bands[:, 0] = sma + (2.0 * std)
        bands[:, 2] = sma - (2.0 * std)
        
        # Calculate bandwidth
        with np.errstate(divide='ignore', invalid='ignore'):
            bands[:, 3] = np.where(sma > 0, (bands[:, 0] - bands[:, 2]) / sma, 0)
            
        return bands

    def calculate_correlation_matrix(self, series1: np.ndarray, series2: Optional[np.ndarray] = None,
                                  method: str = 'pearson',
                                  min_periods: int = 1) -> np.ndarray:
        """
        Calculate correlation matrix with hardware acceleration.
        
        Args:
            series1: First set of time series [n_samples, n_series1]
            series2: Second set of time series [n_samples, n_series2] (optional)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum number of valid periods
            
        Returns:
            Correlation matrix
        """
        # If series2 is None, calculate correlations within series1
        if series2 is None:
            series2 = series1
            
        # Check dimensions
        if series1.ndim == 1:
            series1 = series1.reshape(-1, 1)
        if series2.ndim == 1:
            series2 = series2.reshape(-1, 1)
            
        # Use PyTorch if available and data is large
        if self.has_torch and len(series1) > 1000 and method == 'pearson':
            return self._torch_correlation_matrix(series1, series2, min_periods)
            
        # Use Numba if available and method is Pearson
        if NUMBA_AVAILABLE and method == 'pearson':
            return self._fast_correlation_matrix_numba(series1, series2, min_periods)
            
        # Fallback to numpy implementation
        n_series1 = series1.shape[1]
        n_series2 = series2.shape[1]
        
        # Initialize correlation matrix
        corr_matrix = np.zeros((n_series1, n_series2))
        
        if method == 'pearson':
            # Vectorized Pearson correlation
            for i in range(n_series1):
                for j in range(n_series2):
                    # Extract series
                    x = series1[:, i]
                    y = series2[:, j]
                    
                    # Create mask for valid values
                    mask = np.isfinite(x) & np.isfinite(y)
                    
                    if np.sum(mask) >= min_periods:
                        # Calculate correlation
                        x_valid = x[mask]
                        y_valid = y[mask]
                        
                        corr = np.corrcoef(x_valid, y_valid)[0, 1]
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = np.nan
        elif method == 'spearman':
            # Spearman rank correlation
            for i in range(n_series1):
                for j in range(n_series2):
                    # Extract series
                    x = series1[:, i]
                    y = series2[:, j]
                    
                    # Create mask for valid values
                    mask = np.isfinite(x) & np.isfinite(y)
                    
                    if np.sum(mask) >= min_periods:
                        # Calculate ranks
                        x_valid = x[mask]
                        y_valid = y[mask]
                        
                        x_rank = np.argsort(np.argsort(x_valid))
                        y_rank = np.argsort(np.argsort(y_valid))
                        
                        # Calculate correlation of ranks
                        corr = np.corrcoef(x_rank, y_rank)[0, 1]
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = np.nan
        elif method == 'kendall':
            if SCIPY_AVAILABLE:
                from scipy.stats import kendalltau
                
                # Kendall's tau correlation
                for i in range(n_series1):
                    for j in range(n_series2):
                        # Extract series
                        x = series1[:, i]
                        y = series2[:, j]
                        
                        # Create mask for valid values
                        mask = np.isfinite(x) & np.isfinite(y)
                        
                        if np.sum(mask) >= min_periods:
                            # Calculate correlation
                            x_valid = x[mask]
                            y_valid = y[mask]
                            
                            tau, _ = kendalltau(x_valid, y_valid)
                            corr_matrix[i, j] = tau
                        else:
                            corr_matrix[i, j] = np.nan
            else:
                self.logger.warning("SciPy not available, falling back to Pearson correlation")
                # Fall back to Pearson
                return self.calculate_correlation_matrix(series1, series2, 'pearson', min_periods)
        else:
            self.logger.warning(f"Unknown correlation method: {method}, falling back to Pearson")
            # Fall back to Pearson
            return self.calculate_correlation_matrix(series1, series2, 'pearson', min_periods)
            
        return corr_matrix

    def calculate_diversity(self, system_a, system_b, method=None):
        """
        Hardware-accelerated calculation of cognitive diversity.
        
        Args:
            system_a: Scores from system A
            system_b: Scores from system B
            method: Diversity calculation method ('kendall', 'euclidean', 'dtw')
                
        Returns:
            float: Diversity metric
        """
        # Check if we have CUDA via Numba
        numba_cuda_available = NUMBA_AVAILABLE and hasattr(nb, 'cuda') and hasattr(nb.cuda, 'is_available') and nb.cuda.is_available()
        
        if method is None or method == "kendall":
            # Kendall tau distance
            if self._accelerator_type in (AcceleratorType.CUDA, AcceleratorType.ROCM) and numba_cuda_available:
                # GPU implementation with CUDA/ROCm if available
                try:
                    return self._calculate_kendall_distance_cuda(system_a, system_b)
                except Exception as e:
                    self.logger.warning(f"GPU Kendall distance failed: {e}. Falling back to CPU.")
            
            # CPU implementation with Numba
            if NUMBA_AVAILABLE:
                return self._calculate_kendall_distance_numba(system_a, system_b)
            else:
                # Pure Python fallback
                return self._calculate_kendall_distance_python(system_a, system_b)
        
        elif method == "euclidean":
            # Euclidean distance implementation
            if TORCH_AVAILABLE:
                a_tensor = self.to_device(system_a)
                b_tensor = self.to_device(system_b)
                
                # Calculate squared differences
                diff = a_tensor - b_tensor
                squared_diff = diff * diff
                sum_squared_diff = torch.sum(squared_diff).item()
                
                # Normalize by length
                distance = np.sqrt(sum_squared_diff) / len(system_a)
                return min(1.0, distance)
            else:
                # Numpy fallback
                a_np = np.array(system_a)
                b_np = np.array(system_b)
                diff = a_np - b_np
                distance = np.sqrt(np.sum(diff * diff)) / len(system_a)
                return min(1.0, distance)
        
        elif method == "dtw":
            # Dynamic Time Warping implementation
            if NUMBA_AVAILABLE:
                return self._calculate_dtw_distance_numba(system_a, system_b)
            else:
                # Pure Python fallback
                return self._calculate_dtw_distance_python(system_a, system_b)
        
        else:
            # Default method
            self.logger.warning(f"Unknown diversity method: {method}. Using Kendall tau.")
            if NUMBA_AVAILABLE:
                return self._calculate_kendall_distance_numba(system_a, system_b)
            else:
                return self._calculate_kendall_distance_python(system_a, system_b)
        
    def _calculate_kendall_distance_python(self, a, b):
        """
        Pure Python implementation of Kendall tau distance.
        
        Args:
            a: First score array
            b: Second score array
            
        Returns:
            float: Normalized Kendall tau distance
        """
        n = len(a)
        if n <= 1:
            return 0.0
        
        # Count discordant pairs
        discordant = 0
        total_pairs = n * (n - 1) // 2
        
        for i in range(n):
            for j in range(i + 1, n):
                if (a[i] > a[j] and b[i] < b[j]) or (a[i] < a[j] and b[i] > b[j]):
                    discordant += 1
        
        # Normalize to [0, 1]
        return discordant / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_kendall_distance_numba(self, a, b):
        """
        Numba-accelerated implementation of Kendall tau distance.
        
        Args:
            a: First score array
            b: Second score array
            
        Returns:
            float: Normalized Kendall tau distance
        """
        # Convert inputs to numpy arrays
        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        
        # Define the Numba implementation
        @njit(fastmath=True)
        def _kendall_distance_numba_impl(a, b):
            n = len(a)
            if n <= 1:
                return 0.0
            
            discordant = 0
            total_pairs = n * (n - 1) // 2
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (a[i] > a[j] and b[i] < b[j]) or (a[i] < a[j] and b[i] > b[j]):
                        discordant += 1
            
            return discordant / total_pairs if total_pairs > 0 else 0.0
        
        # Run the Numba-optimized implementation
        return _kendall_distance_numba_impl(a_np, b_np)
    
    def _calculate_kendall_distance_cuda(self, a, b):
        """
        CUDA-accelerated implementation of Kendall tau distance.
        
        Args:
            a: First score array
            b: Second score array
            
        Returns:
            float: Normalized Kendall tau distance
        """
        # This is a placeholder for the CUDA implementation
        # For now, fall back to the Numba implementation
        self.logger.info("CUDA Kendall implementation not yet available, using Numba CPU implementation")
        return self._calculate_kendall_distance_numba(a, b)
    
    def _calculate_dtw_distance_python(self, a, b):
        """
        Pure Python implementation of Dynamic Time Warping distance.
        
        Args:
            a: First time series
            b: Second time series
            
        Returns:
            float: Normalized DTW distance
        """
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0
        
        # Create cost matrix
        dtw = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            dtw[i, 0] = np.inf
        for j in range(1, m + 1):
            dtw[0, j] = np.inf
        dtw[0, 0] = 0
        
        # Fill the cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i-1] - b[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        # Normalize by path length (approximately n+m)
        return dtw[n, m] / (n + m)
    
    def _calculate_dtw_distance_numba(self, a, b):
        """
        Numba-accelerated implementation of Dynamic Time Warping distance.
        
        Args:
            a: First time series
            b: Second time series
            
        Returns:
            float: Normalized DTW distance
        """
        # Convert inputs to numpy arrays
        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        
        # Define the Numba implementation
        @njit(fastmath=True)
        def _dtw_distance_numba_impl(a, b):
            n, m = len(a), len(b)
            if n == 0 or m == 0:
                return 0.0
            
            # Create cost matrix
            dtw = np.zeros((n + 1, m + 1))
            for i in range(1, n + 1):
                dtw[i, 0] = np.inf
            for j in range(1, m + 1):
                dtw[0, j] = np.inf
            dtw[0, 0] = 0
            
            # Fill the cost matrix
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(a[i-1] - b[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            
            # Normalize by path length
            return dtw[n, m] / (n + m)
        
        # Run the Numba-optimized implementation
        return _dtw_distance_numba_impl(a_np, b_np)
    
    def fast_kendall_tau_cuda(self, ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
        """Fast Kendall Tau calculation using CUDA"""
        n = len(ranks_a)
        
        if n <= 1 or n != len(ranks_b):
            return 0.0
        
        # Allocate device memory
        d_ranks_a = cuda.to_device(ranks_a)
        d_ranks_b = cuda.to_device(ranks_b)
        d_result = cuda.device_array(n, dtype=np.int32)
        
        # Set up grid for kernel
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        self._fast_kendall_tau_kernel[blocks_per_grid, threads_per_block](
            d_ranks_a, d_ranks_b, d_result
        )
        
        # Retrieve result
        result = d_result.copy_to_host()
        
        # Calculate final result
        discordant = np.sum(result) // 2  # Divide by 2 as each discordant pair is counted twice
        n_pairs = n * (n - 1) // 2
        
        return discordant / n_pairs
    
    def _torch_correlation_matrix(self, series1: np.ndarray, series2: np.ndarray,
                                min_periods: int) -> np.ndarray:
        """
        Calculate correlation matrix using PyTorch for acceleration.
        
        Args:
            series1: First set of time series [n_samples, n_series1]
            series2: Second set of time series [n_samples, n_series2]
            min_periods: Minimum number of valid periods
            
        Returns:
            Correlation matrix
        """
        if not self.has_torch:
            # Fallback to numpy implementation
            return self.calculate_correlation_matrix(series1, series2, 'pearson', min_periods)
            
        import torch
        
        # Convert to tensors
        device = self.torch_device
        x = torch.tensor(series1, dtype=torch.float32, device=device)
        y = torch.tensor(series2, dtype=torch.float32, device=device)
        
        # Create masks for valid values
        x_mask = torch.isfinite(x)
        y_mask = torch.isfinite(y)
        
        # Initialize correlation matrix
        n_series1 = x.shape[1]
        n_series2 = y.shape[1]
        corr_matrix = torch.zeros((n_series1, n_series2), dtype=torch.float32, device=device)
        
        # Calculate mean and standard deviation
        x_mean = torch.zeros((n_series1,), dtype=torch.float32, device=device)
        y_mean = torch.zeros((n_series2,), dtype=torch.float32, device=device)
        x_std = torch.zeros((n_series1,), dtype=torch.float32, device=device)
        y_std = torch.zeros((n_series2,), dtype=torch.float32, device=device)
        
        for i in range(n_series1):
            valid = x_mask[:, i]
            if valid.sum() >= min_periods:
                x_mean[i] = x[valid, i].mean()
                x_std[i] = x[valid, i].std()
                
        for j in range(n_series2):
            valid = y_mask[:, j]
            if valid.sum() >= min_periods:
                y_mean[j] = y[valid, j].mean()
                y_std[j] = y[valid, j].std()
                
        # Calculate correlation
        for i in range(n_series1):
            for j in range(n_series2):
                # Create combined mask
                valid = x_mask[:, i] & y_mask[:, j]
                
                if valid.sum() >= min_periods and x_std[i] > 0 and y_std[j] > 0:
                    # Compute correlation
                    x_centered = x[valid, i] - x_mean[i]
                    y_centered = y[valid, j] - y_mean[j]
                    
                    corr = (x_centered * y_centered).sum() / (x_std[i] * y_std[j] * valid.sum())
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = float('nan')
                    
        # Convert back to numpy
        return corr_matrix.cpu().numpy()
    
    def dtw_distance(self, s1: Union[np.ndarray, List[float], 'TorchTensor'], 
                   s2: Union[np.ndarray, List[float], 'TorchTensor'], 
                   window: int = 10) -> float:
        """
        Calculate Dynamic Time Warping distance with hardware acceleration.
        
        Args:
            s1: First time series
            s2: Second time series
            window: Warping window size
            
        Returns:
            DTW distance
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(s1, list):
            s1 = np.array(s1, dtype=np.float64)
        elif isinstance(s1, torch.Tensor):
            s1 = self.to_numpy(s1)
            
        if isinstance(s2, list):
            s2 = np.array(s2, dtype=np.float64)
        elif isinstance(s2, torch.Tensor):
            s2 = self.to_numpy(s2)
            
        # Check if we can use GPU implementation
        if self._can_use_torch and self._accelerator_type in (AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.MPS) and len(s1) > 100:
            return self._dtw_distance_torch(s1, s2, window)
        
        # Use Numba if available
        if self._can_use_numba:
            return self._dtw_distance_numba(s1, s2, window)
        else:
            # Fallback to pure Python implementation
            n, m = len(s1), len(s2)
            if n == 0 or m == 0:
                return float('inf')
                
            # Create DTW matrix
            dtw_matrix = np.full((n+1, m+1), np.inf)
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
            return float(dtw_matrix[n, m] / (n + m))
    
    def _dtw_distance_torch(self, s1: np.ndarray, s2: np.ndarray, window: int = 10) -> float:
        """
        Calculate DTW distance using PyTorch (GPU implementation).
        
        Args:
            s1: First time series
            s2: Second time series
            window: Warping window size
            
        Returns:
            DTW distance
        """
        if not self._can_use_torch:
            return self._dtw_distance_numba(s1, s2, window) if self._can_use_numba else float('inf')
        
        try:
            # Convert to torch tensors and move to device
            s1_tensor = torch.tensor(s1, dtype=torch.float32, device=self.torch_device)
            s2_tensor = torch.tensor(s2, dtype=torch.float32, device=self.torch_device)
            
            n, m = len(s1), len(s2)
            if n == 0 or m == 0:
                return float('inf')
            
            # Adjust window size
            w = max(window, abs(n - m))
            
            # Create cost matrix
            dtw_matrix = torch.full((n+1, m+1), float('inf'), dtype=torch.float32, device=self.torch_device)
            dtw_matrix[0, 0] = 0.0
            
            # Fill the matrix using dynamic programming
            for i in range(1, n+1):
                # Apply window constraint
                start_j = max(1, i-w)
                end_j = min(m+1, i+w+1)
                
                # Extract values
                cur_v = s1_tensor[i-1]
                
                for j in range(start_j, end_j):
                    cost = abs(cur_v - s2_tensor[j-1])
                    
                    # Calculate min of three options
                    vals = torch.tensor([
                        dtw_matrix[i-1, j], 
                        dtw_matrix[i, j-1], 
                        dtw_matrix[i-1, j-1]
                    ], device=self.torch_device)
                    
                    last_min = torch.min(vals)
                    dtw_matrix[i, j] = cost + last_min
            
            # Get final distance
            result = float(dtw_matrix[n, m].item() / (n + m))
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error in PyTorch DTW calculation, falling back to CPU: {e}")
            return self._dtw_distance_numba(s1, s2, window) if self._can_use_numba else float('inf')
    
    def correlation_matrix(self, data: Union[np.ndarray, 'TorchTensor'], method: str = 'pearson') -> Union[np.ndarray, 'TorchTensor']:
        """
        Calculate correlation matrix with hardware acceleration.
        
        Args:
            data: Input data matrix (n_samples, n_features)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Correlation matrix (n_features, n_features)
        """
        is_torch_tensor = isinstance(data, torch.Tensor)
        
        # PyTorch implementation for GPU acceleration
        if is_torch_tensor and self._can_use_torch and method == 'pearson':
            # Ensure data is on the correct device
            data = data.to(self.torch_device)
            
            # Center the data
            centered_data = data - torch.mean(data, dim=0, keepdim=True)
            
            # Calculate correlation matrix
            std = torch.std(data, dim=0, unbiased=True, keepdim=True)
            std_product = torch.matmul(std.t(), std)
            
            # Avoid division by zero
            std_product = torch.clamp(std_product, min=1e-8)
            
            # Calculate correlation
            correlation = torch.matmul(centered_data.t(), centered_data) / (data.shape[0] - 1)
            correlation = correlation / std_product
            
            # Ensure diagonal is exactly 1.0
            eye = torch.eye(correlation.shape[0], device=self.torch_device)
            correlation = correlation * (1 - eye) + eye
            
            return correlation
            
        # Numpy implementation for CPU
        else:
            if is_torch_tensor:
                numpy_data = self.to_numpy(data)
            else:
                numpy_data = data
                
            # Calculate correlation based on method
            if method == 'pearson':
                correlation = np.corrcoef(numpy_data.T)
            elif method == 'spearman':
                from scipy.stats import spearmanr
                correlation, _ = spearmanr(numpy_data, axis=0)
                # Handle case with only one feature
                if not isinstance(correlation, np.ndarray):
                    correlation = np.array([[1.0]])
            elif method == 'kendall':
                from scipy.stats import kendalltau
                n_features = numpy_data.shape[1]
                correlation = np.ones((n_features, n_features))
                
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        tau, _ = kendalltau(numpy_data[:, i], numpy_data[:, j])
                        correlation[i, j] = correlation[j, i] = tau if not np.isnan(tau) else 0.0
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
                
            # Return in the same format as input
            if is_torch_tensor:
                return torch.from_numpy(correlation).to(data.device)
            else:
                return correlation
    
    # ----- TorchScript Model Acceleration -----
    
    def create_torchscript_model(self, model, example_input,
                              optimization_level: int = 3, dynamic: bool = False,
                              quantize: bool = False, strict: bool = True,
                              freeze: bool = True):
        """
        Create an optimized TorchScript model for hardware-agnostic acceleration.
        
        Args:
            model: PyTorch model (nn.Module)
            example_input: Example input tensor for tracing
            optimization_level: TorchScript optimization level (0-3)
            dynamic: Whether to support dynamic input shapes
            quantize: Whether to apply quantization
            strict: Whether to use strict tracing
            freeze: Whether to freeze model parameters
            
        Returns:
            Optimized TorchScript model
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        try:
            # Clone model to avoid modifying the original
            if hasattr(model, '__init_args__'):
                model_copy = type(model)(*model.__init_args__, **model.__init_kwargs__)
            else:
                model_copy = model
            model_copy.load_state_dict(model.state_dict())
            
            # Prepare model for export
            model_copy.eval()
            
            # Freeze parameters if requested
            if freeze:
                for param in model_copy.parameters():
                    param.requires_grad_(False)
                    
            # Move model and input to the appropriate device
            model_copy = model_copy.to(self.torch_device)
            example_input = example_input.to(self.torch_device)
            
            # Try scripting first (works with control flow)
            try:
                if dynamic:
                    # For dynamic shapes, use scripting
                    scripted_model = torch.jit.script(model_copy)
                    self.logger.info("Created TorchScript model via scripting with dynamic shapes")
                    
                    # Apply optimizations
                    if optimization_level > 0:
                        scripted_model = torch.jit.optimize_for_inference(scripted_model)
                        self.logger.info("Applied inference optimizations to TorchScript model")
                        
                    jit_model = scripted_model
                else:
                    # For static shapes, try script and trace
                    scripted_model = torch.jit.script(model_copy)
                    self.logger.info("Created TorchScript model via scripting")
                    
                    # Apply optimizations
                    if optimization_level > 0:
                        scripted_model = torch.jit.optimize_for_inference(scripted_model)
                        
                    jit_model = scripted_model
            except Exception as e:
                self.logger.warning(f"Scripting failed: {e}, falling back to tracing")
                
                # Fall back to tracing
                with torch.no_grad():
                    traced_model = torch.jit.trace(model_copy, example_input, check_trace=strict)
                    
                    # Apply optimizations
                    if optimization_level > 0:
                        traced_model = torch.jit.optimize_for_inference(traced_model)
                        
                    jit_model = traced_model
                    
                self.logger.info("Created TorchScript model via tracing")
                
            # Apply quantization if requested
            if quantize:
                try:
                    # Import quantization modules
                    from torch.quantization import quantize_dynamic, per_channel_dynamic_qconfig
                    
                    # Create quantization configuration
                    qconfig_dict = {"": per_channel_dynamic_qconfig}
                    
                    # Apply dynamic quantization
                    quantized_model = quantize_dynamic(
                        jit_model,
                        qconfig_dict=qconfig_dict,
                        dtype=torch.qint8
                    )
                    
                    self.logger.info("Applied quantization to TorchScript model")
                    return quantized_model
                except Exception as e:
                    self.logger.warning(f"Quantization failed: {e}, returning unquantized model")
                    
            # Save the model to optimize via JIT passes
            if optimization_level > 1:
                # Apply JIT optimization passes
                jit_model = self._apply_jit_passes(jit_model, optimization_level)
                
            return jit_model
            
        except Exception as e:
            self.logger.error(f"Failed to create TorchScript model: {e}")
            raise
        
    def _apply_jit_passes(self, jit_model: Any, 
                        optimization_level: int) -> torch.jit.ScriptModule:
        """
        Apply TorchScript optimization passes.
        
        Args:
            jit_model: TorchScript model
            optimization_level: Optimization level (0-3)
            
        Returns:
            Optimized TorchScript model
        """
        try:
            # Import JIT pass modules
            import torch._C as _C
            from torch._C import _jit_pass_dce, _jit_pass_constant_propagation
            from torch._C import _jit_pass_peephole, _jit_pass_fuse_linear
            
            # Apply optimization passes based on level
            if optimization_level >= 1:
                # Basic optimizations
                _jit_pass_dce(jit_model.graph)
                _jit_pass_constant_propagation(jit_model.graph)
                _jit_pass_peephole(jit_model.graph)
                
            if optimization_level >= 2:
                # Intermediate optimizations
                _jit_pass_fuse_linear(jit_model.graph)
                
                # Try to fuse adjacent ops
                try:
                    from torch._C import _jit_pass_fuse_addmm
                    _jit_pass_fuse_addmm(jit_model.graph)
                except:
                    pass
                    
            if optimization_level >= 3:
                # Advanced optimizations
                try:
                    # These might not be available in all PyTorch versions
                    from torch._C import _jit_pass_fold_convbn
                    _jit_pass_fold_convbn(jit_model.graph)
                except:
                    pass
                    
                try:
                    from torch._C import _jit_pass_remove_inplace_ops
                    _jit_pass_remove_inplace_ops(jit_model.graph)
                except:
                    pass
                    
            self.logger.info(f"Applied level {optimization_level} JIT optimization passes")
            return jit_model
            
        except Exception as e:
            self.logger.warning(f"Failed to apply JIT optimization passes: {e}")
            return jit_model
    
    def export_torchscript_model(self, model: torch.jit.ScriptModule, 
                              filename: str, use_zipfile: bool = True) -> str:
        """
        Export TorchScript model to file.
        
        Args:
            model: TorchScript model
            filename: Output filename
            use_zipfile: Whether to use zipfile format (smaller but slower)
            
        Returns:
            Path to saved model
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save model
            if use_zipfile:
                model.save(filename)
            else:
                # Save without zipfile for faster loading
                model._save_for_lite_interpreter(filename)
                
            self.logger.info(f"Exported TorchScript model to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to export TorchScript model: {e}")
            raise
    
    def load_torchscript_model(self, filename: str, device: Optional[str] = None) -> torch.jit.ScriptModule:
        """
        Load TorchScript model from file.
        
        Args:
            filename: Path to saved model
            device: Target device (default: current device)
            
        Returns:
            Loaded TorchScript model
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        try:
            # Determine device
            if device is None:
                device = self.torch_device
            else:
                device = torch.device(device)
                
            # Load model
            model = torch.jit.load(filename, map_location=device)
            
            self.logger.info(f"Loaded TorchScript model from {filename}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load TorchScript model: {e}")
            raise
    
    def run_torchscript_model(self, model: torch.jit.ScriptModule, input_data: Any,
                           optimize_execution: bool = True, warm_up: bool = True,
                           batch_size: Optional[int] = None) -> Any:
        """
        Run a TorchScript model with automatic input conversion and optimization.
        
        Args:
            model: TorchScript model
            input_data: Input data (can be tensor, numpy array, or list)
            optimize_execution: Whether to optimize execution
            warm_up: Whether to warm up the model
            batch_size: Optional batch size for processing large inputs
            
        Returns:
            Model output
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        try:
            # Convert input to tensor
            if not isinstance(input_data, torch.Tensor):
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data)
                elif isinstance(input_data, list):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                else:
                    raise TypeError(f"Unsupported input type: {type(input_data)}")
            else:
                input_tensor = input_data
                
            # Move to device
            input_tensor = input_tensor.to(self.torch_device)
            
            # Optimize execution
            if optimize_execution:
                # Set eval mode
                model.eval()
                
                # Set compile mode for PyTorch 2.0+
                if hasattr(torch, 'compile') and callable(getattr(torch, 'compile')):
                    try:
                        model = torch.compile(model)
                        self.logger.debug("Using torch.compile for model execution")
                    except Exception as e:
                        self.logger.warning(f"Failed to use torch.compile: {e}")
                        
            # Warm up if requested
            if warm_up:
                with torch.no_grad():
                    # Run one forward pass to warm up
                    if isinstance(input_tensor, (list, tuple)):
                        model(*input_tensor)
                    else:
                        model(input_tensor)
                        
            # Run model
            with torch.no_grad():
                if batch_size is not None and len(input_tensor) > batch_size:
                    # Process in batches
                    outputs = []
                    for i in range(0, len(input_tensor), batch_size):
                        batch = input_tensor[i:i+batch_size]
                        batch_output = model(batch)
                        outputs.append(batch_output)
                        
                    # Concatenate outputs
                    if isinstance(outputs[0], torch.Tensor):
                        output = torch.cat(outputs, dim=0)
                    else:
                        output = outputs
                else:
                    # Process all at once
                    if isinstance(input_tensor, (list, tuple)):
                        output = model(*input_tensor)
                    else:
                        output = model(input_tensor)
                
            return output
            
        except Exception as e:
            self.logger.error(f"Error running TorchScript model: {e}")
            raise
    
    def compile_for_platform(self, model: Any, example_input: 'TorchTensor') -> Any:
        """
        Compile model with platform-specific optimizations.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor for tracing
            
        Returns:
            Optimized model for current platform
        """
        if not self._can_use_torch:
            return model
        
        # First, create TorchScript model
        model_ts = self.create_torchscript_model(model, example_input)
        
        # Apply platform-specific optimizations
        if self._accelerator_type == AcceleratorType.CUDA:
            # NVIDIA GPU optimizations
            try:
                import torch_tensorrt
                
                if hasattr(torch_tensorrt, 'compile'):
                    # Get device properties
                    if not hasattr(self, 'cuda_compute_capability'):
                        self._init_cuda_specific()
                    
                    enabled_precisions = {torch.float32}
                    if self.supports_half:
                        enabled_precisions.add(torch.float16)
                    if self.supports_bfloat16:
                        enabled_precisions.add(torch.bfloat16)
                    
                    # Compile with TensorRT
                    optimized = torch_tensorrt.compile(
                        model_ts,
                        inputs=[example_input],
                        enabled_precisions=enabled_precisions
                    )
                    
                    self.logger.info("Model compiled with TensorRT optimizations")
                    return optimized
            except (ImportError, Exception) as e:
                self.logger.warning(f"TensorRT compilation not available: {e}")
        
        elif self._accelerator_type == AcceleratorType.ROCM:
            # AMD GPU optimizations
            try:
                # No specialized ROCm compiler yet, return TorchScript model
                self.logger.info("Using TorchScript optimization for ROCm")
                return model_ts
            except Exception as e:
                self.logger.warning(f"ROCm optimization failed: {e}")
        
        elif self._accelerator_type == AcceleratorType.MPS:
            # Apple Silicon optimizations
            try:
                # Apple Silicon uses regular TorchScript
                self.logger.info("Using TorchScript optimization for Apple Silicon")
                return model_ts
            except Exception as e:
                self.logger.warning(f"MPS optimization failed: {e}")
        
        elif self._accelerator_type == AcceleratorType.IPEX:
            # Intel GPU optimizations
            try:
                import intel_extension_for_pytorch as ipex
                
                # Convert to IPEX optimized model
                model_ipex = ipex.optimize(model)
                
                # Apply mixed precision if supported
                if self.supports_bfloat16:
                    model_ipex = ipex.optimize(model, dtype=torch.bfloat16)
                
                self.logger.info("Model optimized with IPEX")
                return model_ipex
            except (ImportError, Exception) as e:
                self.logger.warning(f"IPEX optimization not available: {e}")
        
        # If no platform-specific optimizations applied, return TorchScript model
        return model_ts
    
    # ----- OpenCL-based Acceleration -----
    
    def create_opencl_kernel(self, kernel_name: str, kernel_src: str) -> Any:
        """
        Create OpenCL kernel for cross-platform GPU acceleration.
        
        Args:
            kernel_name: Name of the kernel function
            kernel_src: OpenCL kernel source code
            
        Returns:
            Compiled OpenCL kernel
        """
        if not self._can_use_opencl:
            raise RuntimeError("OpenCL not available")
            
        try:
            # Check if kernel already in cache
            cache_key = (kernel_name, hash(kernel_src))
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]
                
            # Create program
            program = cl.Program(self.opencl_ctx, kernel_src).build()
            
            # Get kernel
            kernel = getattr(program, kernel_name)
            
            # Cache kernel
            self._kernel_cache[cache_key] = kernel
            
            return kernel
            
        except Exception as e:
            self.logger.error(f"Error creating OpenCL kernel: {e}")
            raise
    
    def run_opencl_kernel(self, kernel: Any, global_size: Tuple[int, ...], 
                        *args, local_size: Optional[Tuple[int, ...]] = None) -> Any:
        """
        Run OpenCL kernel with given arguments.
        
        Args:
            kernel: OpenCL kernel
            global_size: Global work size
            *args: Kernel arguments
            local_size: Local work size
            
        Returns:
            Execution event
        """
        if not self._can_use_opencl:
            raise RuntimeError("OpenCL not available")
            
        try:
            # Run kernel
            event = kernel(self.opencl_queue, global_size, local_size, *args)
            
            # Return event
            return event
            
        except Exception as e:
            self.logger.error(f"Error running OpenCL kernel: {e}")
            raise
    
    # ----- Accelerated Data Transfer -----
    
    def copy_to_gpu(self, data: np.ndarray) -> Any:
        """
        Efficiently copy data to GPU memory using available backend.
        
        Args:
            data: NumPy array to transfer
            
        Returns:
            GPU memory object (backend-specific)
        """
        # PyTorch backend
        if self._can_use_torch:
            return torch.from_numpy(data).to(self.torch_device)
            
        # RAPIDS backend (NVIDIA only)
        elif self._can_use_rapids and self._accelerator_type == AcceleratorType.CUDA:
            import cupy
            return cupy.array(data)
            
        # OpenCL backend
        elif self._can_use_opencl:
            import pyopencl as cl
            
            # Create OpenCL buffer
            mem_flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
            buffer = cl.Buffer(self.opencl_ctx, mem_flags, hostbuf=data)
            
            return buffer
            
        # No GPU backend available
        else:
            return data
    
    def copy_from_gpu(self, gpu_data: Any) -> np.ndarray:
        """
        Copy data from GPU memory to CPU.
        
        Args:
            gpu_data: GPU memory object
            
        Returns:
            NumPy array
        """
        # PyTorch tensor
        if isinstance(gpu_data, torch.Tensor):
            return gpu_data.cpu().numpy()
            
        # RAPIDS/CuPy array
        elif self._can_use_rapids and hasattr(gpu_data, 'get'):
            return gpu_data.get()
            
        # OpenCL buffer
        elif self._can_use_opencl and isinstance(gpu_data, cl.Buffer):
            # Create output array
            result = np.empty(gpu_data.size // np.dtype(np.float32).itemsize, dtype=np.float32)
            
            # Copy data
            cl.enqueue_copy(self.opencl_queue, result, gpu_data)
            
            return result
            
        # Already a NumPy array or unsupported type
        else:
            return np.array(gpu_data)
    
    # ----- Parallel Computing Utilities -----
    
    def parallel_for(self, function: Callable, data: List[Any], chunksize: Optional[int] = None) -> List[Any]:
        """
        Execute function in parallel across multiple inputs.
        
        Args:
            function: Function to execute
            data: List of inputs
            chunksize: Chunk size for parallelization
            
        Returns:
            List of results
        """
        # Use concurrent.futures if inputs are large or function is computationally intensive
        if len(data) > 5 or chunksize is not None:
            import concurrent.futures
            
            # Determine processes based on system
            max_workers = os.cpu_count()
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Execute in parallel
                if chunksize is not None:
                    results = list(executor.map(function, data, chunksize=chunksize))
                else:
                    results = list(executor.map(function, data))
                    
            return results
            
        # For small inputs, use synchronous execution
        else:
            return [function(item) for item in data]
    
    def execute_in_batches(self, function: Callable, data: np.ndarray, batch_size: int = 32) -> List[Any]:
        """
        Execute function in batches to manage memory usage.
        
        Args:
            function: Function to execute on each batch
            data: Input data
            batch_size: Batch size
            
        Returns:
            List of results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_result = function(batch)
            results.append(batch_result)
            
        return results
    
    # ----- Hardware Monitoring -----
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage of accelerator.
        
        Returns:
            Dictionary with memory usage information
        """
        memory_info = {}
        
        # CUDA memory usage
        if self._accelerator_type == AcceleratorType.CUDA and self._can_use_torch:
            try:
                device_idx = 0
                if ":" in self._device:
                    device_idx = int(self._device.split(":")[-1])
                
                allocated = torch.cuda.memory_allocated(device_idx)
                reserved = torch.cuda.memory_reserved(device_idx)
                max_memory = torch.cuda.max_memory_allocated(device_idx)
                
                memory_info["allocated"] = allocated / (1024**2)  # MB
                memory_info["reserved"] = reserved / (1024**2)    # MB
                memory_info["max_allocated"] = max_memory / (1024**2)  # MB
                memory_info["utilization"] = allocated / self.total_cuda_memory if hasattr(self, 'total_cuda_memory') else 0.0
            except Exception as e:
                self.logger.warning(f"Error getting CUDA memory usage: {e}")
        
        # ROCm memory usage
        elif self._accelerator_type == AcceleratorType.ROCM and self._can_use_torch:
            try:
                # ROCm uses CUDA API in PyTorch
                device_idx = 0
                if ":" in self._device:
                    device_idx = int(self._device.split(":")[-1])
                
                allocated = torch.cuda.memory_allocated(device_idx)
                reserved = torch.cuda.memory_reserved(device_idx)
                
                memory_info["allocated"] = allocated / (1024**2)  # MB
                memory_info["reserved"] = reserved / (1024**2)    # MB
                memory_info["utilization"] = allocated / self.total_rocm_memory if hasattr(self, 'total_rocm_memory') else 0.0
            except Exception as e:
                self.logger.warning(f"Error getting ROCm memory usage: {e}")
        
        # MPS memory usage not available programmatically
        elif self._accelerator_type == AcceleratorType.MPS:
            memory_info["note"] = "Memory usage statistics not available for Apple MPS"
        
        # Intel XPU memory
        elif self._accelerator_type == AcceleratorType.IPEX and self._can_use_ipex:
            try:
                # IPEX memory API if available
                if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'memory_allocated'):
                    allocated = ipex.xpu.memory_allocated()
                    reserved = ipex.xpu.memory_reserved() if hasattr(ipex.xpu, 'memory_reserved') else 0
                    
                    memory_info["allocated"] = allocated / (1024**2)  # MB
                    memory_info["reserved"] = reserved / (1024**2)    # MB
                else:
                    memory_info["note"] = "Detailed memory statistics not available for IPEX"
            except Exception as e:
                self.logger.warning(f"Error getting IPEX memory usage: {e}")
        
        # Add CPU memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss
            
            memory_info["cpu_allocated"] = cpu_memory / (1024**2)  # MB
        except ImportError:
            pass
        
        return memory_info
    
    def clear_memory(self) -> bool:
        """
        Clear unused memory in accelerator.
        
        Returns:
            Success flag
        """
        success = False
        
        # CUDA memory clearing
        if self._accelerator_type == AcceleratorType.CUDA and self._can_use_torch:
            try:
                torch.cuda.empty_cache()
                success = True
            except Exception as e:
                self.logger.warning(f"Error clearing CUDA memory: {e}")
        
        # ROCm memory clearing
        elif self._accelerator_type == AcceleratorType.ROCM and self._can_use_torch:
            try:
                # ROCm uses CUDA API in PyTorch
                torch.cuda.empty_cache()
                success = True
            except Exception as e:
                self.logger.warning(f"Error clearing ROCm memory: {e}")
        
        # MPS memory management
        elif self._accelerator_type == AcceleratorType.MPS and self._can_use_torch:
            try:
                # Current PyTorch doesn't have explicit MPS memory management
                # Trigger garbage collection instead
                import gc
                gc.collect()
                torch.mps.empty_cache()
                success = True
            except Exception as e:
                self.logger.warning(f"Error clearing MPS memory: {e}")
        
        # Intel XPU memory clearing
        elif self._accelerator_type == AcceleratorType.IPEX and self._can_use_ipex:
            try:
                if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'empty_cache'):
                    ipex.xpu.empty_cache()
                    success = True
            except Exception as e:
                self.logger.warning(f"Error clearing IPEX memory: {e}")
        
        # Always trigger Python garbage collection
        import gc
        gc.collect()
        
        return success
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current device.
        
        Returns:
            Dictionary with device information
        """
        device_info = {
            "device": self._device,
            "type": str(self._accelerator_type),
            "memory": self.get_memory_usage()
        }
        
        # Add CUDA-specific info
        if self._accelerator_type == AcceleratorType.CUDA and self._can_use_torch:
            try:
                device_idx = 0
                if ":" in self._device:
                    device_idx = int(self._device.split(":")[-1])
                
                props = torch.cuda.get_device_properties(device_idx)
                
                device_info.update({
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory": props.total_memory / (1024**3),  # GB
                    "multi_processor_count": props.multi_processor_count,
                    "clock_rate": props.clock_rate / 1000,  # MHz
                    "memory_clock_rate": getattr(props, 'memory_clock_rate', 0) / 1000,  # MHz
                    "pci_bus_id": getattr(props, 'pci_bus_id', 'unknown'),
                    "pci_device_id": getattr(props, 'pci_device_id', 'unknown')
                })
            except Exception as e:
                self.logger.warning(f"Error getting CUDA device info: {e}")
        
        # Add ROCm-specific info
        elif self._accelerator_type == AcceleratorType.ROCM and self._can_use_torch:
            try:
                device_idx = 0
                if ":" in self._device:
                    device_idx = int(self._device.split(":")[-1])
                
                # ROCm uses CUDA API in PyTorch
                props = torch.cuda.get_device_properties(device_idx)
                
                device_info.update({
                    "name": props.name,
                    "total_memory": props.total_memory / (1024**3),  # GB
                    "multi_processor_count": props.multi_processor_count,
                    "clock_rate": props.clock_rate / 1000  # MHz
                })
            except Exception as e:
                self.logger.warning(f"Error getting ROCm device info: {e}")
        
        # Add MPS-specific info
        elif self._accelerator_type == AcceleratorType.MPS and self._can_use_torch:
            try:
                import platform
                
                # Get macOS version
                mac_version = platform.mac_ver()[0]
                
                # Get device info from system_profiler if available
                gpu_name = "Apple Silicon GPU"
                try:
                    import subprocess
                    output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType']).decode('utf-8')
                    
                    # Parse for GPU name
                    name_match = re.search(r"Chipset Model: (.+)", output)
                    if name_match:
                        gpu_name = name_match.group(1)
                except:
                    pass
                
                device_info.update({
                    "name": gpu_name,
                    "platform": "macOS",
                    "version": mac_version
                })
            except Exception as e:
                self.logger.warning(f"Error getting MPS device info: {e}")
        
        # Add Intel XPU info
        elif self._accelerator_type == AcceleratorType.IPEX and self._can_use_ipex:
            try:
                if hasattr(ipex, 'xpu') and hasattr(ipex.xpu, 'get_device_name'):
                    device_name = ipex.xpu.get_device_name(0)
                    device_info["name"] = device_name
            except Exception as e:
                self.logger.warning(f"Error getting IPEX device info: {e}")
        
        return device_info

def test_hw_acceleration():
    """Test hardware acceleration capabilities."""
    print("\n===== Hardware Acceleration Test =====")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    print(f"\nPyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch path: {torch.__file__}")
        print(f"CUDA available: {CUDA_AVAILABLE}")
        print(f"ROCm available: {ROCM_AVAILABLE}")
        
        if CUDA_AVAILABLE:
            try:
                print(f"GPU device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"Error getting GPU details: {e}")
    
    # Create accelerator with logging to see detailed output
    print("\nCreating HardwareAccelerator...")
    logging.basicConfig(level=logging.INFO)
    accelerator = HardwareAccelerator()
    print(f"Accelerator device: {accelerator._device}")
    print(f"Accelerator type: {accelerator._accelerator_type}")
    
    # Print detected devices
    print("\nDetected devices:")
    for device in accelerator.hardware_info.devices:
        print(f"  - {device['name']} ({device['type']}): {device['memory']} MB")
        
    return accelerator

if __name__ == "__main__":
    test_hw_acceleration()