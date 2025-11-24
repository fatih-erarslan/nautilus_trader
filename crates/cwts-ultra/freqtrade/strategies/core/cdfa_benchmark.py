#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Hardware Acceleration Benchmark Suite

Provides comprehensive benchmarking for hardware-accelerated components:
- CPU vs GPU performance comparisons
- Memory usage profiling
- Scaling tests for large datasets
- Numba vs PyTorch vs TorchScript comparisons
- Cross-platform performance (CUDA vs ROCm vs MPS)
- Wavelet transform optimizations
- Accelerated data preprocessing pipeline

Author: Created on May 6, 2025
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import importlib
import logging
import gc
import traceback
import sys
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
            
# Import test framework
from test_framework import CDFABenchmarkTest, fetch_test_data, memory_profiling, format_execution_time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cdfa_becnhmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure logging
logger = logging.getLogger('cdfa_benchmark_hardware')

# Try to import CDFA components
try:
    from cdfa_extensions.hw_acceleration import HardwareAccelerator
    HW_ACCEL_AVAILABLE = True
except ImportError:
    logger.warning("Hardware acceleration module not available")
    HW_ACCEL_AVAILABLE = False

try:
    from cdfa_extensions.wavelet_processor import WaveletProcessor
    WAVELET_AVAILABLE = True
except ImportError:
    logger.warning("Wavelet processor not available")
    WAVELET_AVAILABLE = False

try:
    from cdfa_extensions.torchscript_fusion import TorchScriptFusion
    TORCHSCRIPT_AVAILABLE = True
except ImportError:
    logger.warning("TorchScript fusion not available")
    TORCHSCRIPT_AVAILABLE = False

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

class HardwareAccelerationBenchmark(CDFABenchmarkTest):
    """Benchmark suite for hardware acceleration components."""
    
    def __init__(self, iterations: int = 5, warmup: int = 2, hardware: str = "cpu"):
        """Initialize benchmark test."""
        super().__init__(iterations, warmup, hardware)
        
        # Skip if hardware acceleration not available
        if not HW_ACCEL_AVAILABLE:
            logger.error("Hardware acceleration not available, skipping benchmark")
            return
        
        # Initialize hardware accelerator based on requested hardware
        self.hw_accelerator = HardwareAccelerator(
            enable_gpu=(hardware != "cpu"),
            prefer_cuda=(hardware == "cuda"),
            device=hardware
        )
        
        # Get device information
        self.device_info = {
            "device": self.hw_accelerator.get_device(),
            "numba_available": self.hw_accelerator._can_use_numba,
            "torch_available": self.hw_accelerator._can_use_torch,
            "cuda_available": self.hw_accelerator._can_use_cuda,
            "rocm_available": self.hw_accelerator._can_use_rocm,
            "mps_available": self.hw_accelerator._can_use_mps,
        }
        
        logger.info(f"Hardware acceleration benchmark initialized with device: {self.device_info['device']}")
        
        
        # Initialize test data
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data for benchmarks."""
        data = {}
        
        # Various dataset sizes
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            # Random data
            data[f"random_{size}"] = np.random.randn(size)
            
            # Sine wave with noise
            x = np.linspace(0, 10, size)
            data[f"sine_{size}"] = np.sin(x) + 0.1 * np.random.randn(size)
            
            # Step function with noise
            step = np.zeros(size)
            step[size//3:2*size//3] = 1.0
            data[f"step_{size}"] = step + 0.1 * np.random.randn(size)
            
            # Matrix data
            data[f"matrix_{size}"] = np.random.randn(int(np.sqrt(size)), int(np.sqrt(size)))
        
        # Add market data if available
        try:
            market_data = fetch_test_data(["SPY"], period="1y", interval="1d")
            if market_data and "SPY" in market_data:
                spy_df = market_data["SPY"]
                if "close" in spy_df.columns:
                    data["market_spy"] = spy_df["close"].values
        except Exception as e:
            logger.warning(f"Failed to fetch market data: {e}")
        
        return data
    
    def run(self):
        """Run all hardware acceleration benchmarks."""
        if not HW_ACCEL_AVAILABLE:
            logger.error("Hardware acceleration not available, skipping benchmark")
            return
        
        logger.info("Running hardware acceleration benchmarks...")
        
        # Basic operations benchmarks
        self._benchmark_basic_operations()
        
        # Array operations benchmarks
        self._benchmark_array_operations()
        
        # Scaling tests
        self._benchmark_scaling()
        
        # Device transfer benchmarks
        if TORCH_AVAILABLE and self.hw_accelerator._can_use_torch:
            self._benchmark_device_transfers()
        
        # Specialized algorithms
        self._benchmark_specialized_algorithms()
        
        # Library-specific benchmarks
        if TORCH_AVAILABLE and self.hw_accelerator._can_use_torch:
            self._benchmark_pytorch_operations()
        
        if NUMBA_AVAILABLE and self.hw_accelerator._can_use_numba:
            self._benchmark_numba_operations()
        
        if TORCHSCRIPT_AVAILABLE:
            self._benchmark_torchscript()
        
        logger.info("Hardware acceleration benchmarks completed")
    
    def _benchmark_basic_operations(self):
        """Benchmark basic operations."""
        logger.info("Benchmarking basic operations...")
        
        # Test normalize_scores implementation
        data = self.test_data["random_10000"]
        
        # Define implementations
        implementations = {
            "CPU Implementation": self._normalize_scores_cpu,
            "Hardware Accelerated": self.hw_accelerator.normalize_scores
        }
        
        if NUMBA_AVAILABLE:
            implementations["Numba Implementation"] = self._normalize_scores_numba
        
        if TORCH_AVAILABLE and self.hw_accelerator._can_use_torch:
            implementations["PyTorch Implementation"] = self._normalize_scores_torch
        
        # Run benchmark
        self.compare_implementations(implementations, data, component="hw_acceleration")
        
        # Test distance matrix calculation
        data = self.test_data["random_1000"]
        data_2d = data.reshape(100, 10)  # Reshape to 2D for distance matrix
        
        # Define implementations
        implementations = {
            "CPU Implementation": self._calculate_distance_matrix_cpu,
            "Hardware Accelerated": self.hw_accelerator.calculate_distance_matrix
        }
        
        if NUMBA_AVAILABLE:
            implementations["Numba Implementation"] = self._calculate_distance_matrix_numba
        
        if TORCH_AVAILABLE and self.hw_accelerator._can_use_torch:
            implementations["PyTorch Implementation"] = self._calculate_distance_matrix_torch
        
        # Run benchmark
        self.compare_implementations(implementations, data_2d, component="hw_acceleration")
    
    def _benchmark_array_operations(self):
        """Benchmark array operations."""
        logger.info("Benchmarking array operations...")
        
        # Test correlation matrix calculation
        data = self.test_data["matrix_1000"]
        
        # Define implementations
        implementations = {
            "Numpy Implementation": np.corrcoef,
            "Hardware Accelerated": self.hw_accelerator.correlation_matrix
        }
        
        # Run benchmark
        self.compare_implementations(implementations, data, component="hw_acceleration")
        
        # Test DTW distance calculation
        s1 = self.test_data["sine_1000"]
        s2 = self.test_data["sine_1000"] + 0.2 * np.random.randn(1000)
        
        # Define implementations
        implementations = {
            "CPU Implementation": lambda a, b: self._dtw_distance_cpu(a, b, window=10),
            "Hardware Accelerated": lambda a, b: self.hw_accelerator.dtw_distance(a, b, window=10)
        }
        
        if NUMBA_AVAILABLE:
            implementations["Numba Implementation"] = lambda a, b: self._dtw_distance_numba(a, b, window=10)
        
        # Run benchmark
        self.compare_implementations(implementations, s1, s2, component="hw_acceleration")
    
    def _benchmark_scaling(self):
        """Benchmark scaling with different data sizes."""
        logger.info("Benchmarking scaling with different data sizes...")
        
        # Data sizes to test
        sizes = [100, 1000, 10000, 100000]
        
        # Test normalize_scores scaling
        for size in sizes:
            data = self.test_data[f"random_{size}"]
            
            # Run benchmark
            self.benchmark_function(
                self.hw_accelerator.normalize_scores,
                data,
                name=f"normalize_scores_{size}",
                component="hw_acceleration_scaling"
            )
        
        # Test distance matrix scaling (for smaller sizes)
        for size in [100, 1000]:
            data = self.test_data[f"matrix_{size}"]
            
            # Run benchmark
            self.benchmark_function(
                self.hw_accelerator.calculate_distance_matrix,
                data,
                name=f"distance_matrix_{int(np.sqrt(size))}",
                component="hw_acceleration_scaling"
            )
    
    def _benchmark_device_transfers(self):
        """Benchmark data transfers between CPU and device."""
        if not TORCH_AVAILABLE or not self.hw_accelerator._can_use_torch:
            logger.warning("PyTorch not available, skipping device transfer benchmarks")
            return
        
        logger.info("Benchmarking device transfers...")
        
        device = self.hw_accelerator.get_torch_device()
        
        # Different data sizes
        sizes = [100, 1000, 10000, 100000, 1000000]
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float32)
            tensor = torch.from_numpy(data)
            
            # CPU to device benchmark
            self.benchmark_function(
                self.hw_accelerator.to_device,
                tensor,
                name=f"cpu_to_device_{size}",
                component="device_transfer"
            )
            
            # Only test device to CPU for data on device
            if device.type != "cpu":
                device_tensor = tensor.to(device)
                
                # Device to CPU benchmark
                self.benchmark_function(
                    self.hw_accelerator.to_numpy,
                    device_tensor,
                    name=f"device_to_cpu_{size}",
                    component="device_transfer"
                )
    
    def _benchmark_specialized_algorithms(self):
        """Benchmark specialized algorithms."""
        logger.info("Benchmarking specialized algorithms...")
        
        # Test matrix multiplication implementations
        if TORCH_AVAILABLE:
            data1 = self.test_data["matrix_1000"]
            data2 = self.test_data["matrix_1000"]
            
            implementations = {
                "Numpy matmul": lambda a, b: np.matmul(a, b)
            }
            
            if self.hw_accelerator._can_use_torch:
                # PyTorch implementation
                implementations["PyTorch matmul"] = lambda a, b: torch.matmul(
                    self.hw_accelerator.to_device(torch.from_numpy(a.astype(np.float32))),
                    self.hw_accelerator.to_device(torch.from_numpy(b.astype(np.float32)))
                ).cpu().numpy()
            
            # Run benchmark
            self.compare_implementations(implementations, data1, data2, component="specialized_algorithms")
        
        # Test eigenvalue decomposition
        data = self.test_data["matrix_1000"]
        
        implementations = {
            "Numpy eig": lambda x: np.linalg.eig(x)
        }
        
        if TORCH_AVAILABLE and self.hw_accelerator._can_use_torch:
            # PyTorch implementation
            implementations["PyTorch eig"] = lambda x: torch.linalg.eig(
                self.hw_accelerator.to_device(torch.from_numpy(x.astype(np.float32)))
            )[0].cpu().numpy()
        
        # Run benchmark
        self.compare_implementations(implementations, data, component="specialized_algorithms")
    
    def _benchmark_pytorch_operations(self):
        """Benchmark PyTorch-specific operations."""
        if not TORCH_AVAILABLE or not self.hw_accelerator._can_use_torch:
            logger.warning("PyTorch not available, skipping PyTorch benchmarks")
            return
        
        logger.info("Benchmarking PyTorch operations...")
        
        device = self.hw_accelerator.get_torch_device()
        
        # Test convolution operation
        data = self.test_data["random_10000"]
        kernel = np.ones(5) / 5  # Simple moving average kernel
        
        implementations = {
            "Numpy convolve": lambda x, k: np.convolve(x, k, mode='same')
        }
        
        # PyTorch implementation
        implementations["PyTorch conv1d"] = lambda x, k: torch.nn.functional.conv1d(
            self.hw_accelerator.to_device(torch.from_numpy(x.astype(np.float32)).reshape(1, 1, -1)),
            self.hw_accelerator.to_device(torch.from_numpy(k.astype(np.float32)).reshape(1, 1, -1)),
            padding=len(k)//2
        ).squeeze().cpu().numpy()
        
        # Run benchmark
        self.compare_implementations(implementations, data, kernel, component="pytorch_operations")
        
        # Test FFT
        data = self.test_data["sine_10000"]
        
        implementations = {
            "Numpy FFT": lambda x: np.fft.fft(x)
        }
        
        # PyTorch implementation
        implementations["PyTorch FFT"] = lambda x: torch.fft.fft(
            self.hw_accelerator.to_device(torch.from_numpy(x.astype(np.float32)))
        ).cpu().numpy()
        
        # Run benchmark
        self.compare_implementations(implementations, data, component="pytorch_operations")
    
    def _benchmark_numba_operations(self):
        """Benchmark Numba-specific operations."""
        if not NUMBA_AVAILABLE:
            logger.warning("Numba not available, skipping Numba benchmarks")
            return
        
        logger.info("Benchmarking Numba operations...")
        
        # Test running sum implementation
        data = self.test_data["random_100000"]
        
        implementations = {
            "Numpy cumsum": np.cumsum,
            "Numba implementation": self._cumsum_numba
        }
        
        # Run benchmark
        self.compare_implementations(implementations, data, component="numba_operations")
        
        # Test moving average implementation
        data = self.test_data["sine_100000"]
        window = 20
        
        implementations = {
            "Numpy rolling mean": lambda x, w: pd.Series(x).rolling(window=w).mean().values,
            "Numba implementation": lambda x, w: self._moving_average_numba(x, w)
        }
        
        # Run benchmark
        self.compare_implementations(implementations, data, window, component="numba_operations")
    
    def _benchmark_torchscript(self):
        """Benchmark TorchScript models."""
        if not TORCHSCRIPT_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("TorchScript not available, skipping TorchScript benchmarks")
            return
        
        logger.info("Benchmarking TorchScript models...")
        
        # Skip if fusion module is not available
        try:
            fusion = TorchScriptFusion()
        except:
            logger.error("Failed to initialize TorchScriptFusion")
            return
        
        # Create simple neural network model
        if TORCH_AVAILABLE:
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(10, 20)
                    self.relu = torch.nn.ReLU()
                    self.fc2 = torch.nn.Linear(20, 1)
                
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
            
            model = SimpleModel()
            
            # Create example input
            input_data = torch.randn(100, 10)
            
            # Benchmark PyTorch model
            self.benchmark_function(
                lambda m, x: m(x),
                model,
                input_data,
                name="PyTorch model",
                component="torchscript"
            )
            
            # Create TorchScript model
            scripted_model = torch.jit.script(model)
            
            # Benchmark TorchScript model
            self.benchmark_function(
                lambda m, x: m(x),
                scripted_model,
                input_data,
                name="TorchScript model",
                component="torchscript"
            )
            
            # Benchmark TorchScript fusion wrapper
            try:
                wrapped_model = fusion.create_optimized_model(model, input_data)
                
                self.benchmark_function(
                    lambda m, x: fusion.run_model(m, x),
                    wrapped_model,
                    input_data.numpy(),
                    name="TorchScript fusion wrapper",
                    component="torchscript"
                )
            except Exception as e:
                logger.error(f"Failed to benchmark TorchScript fusion wrapper: {e}")
                traceback.print_exc()
    
    # ---- Implementation Methods ----
    
    def _normalize_scores_cpu(self, scores: np.ndarray) -> np.ndarray:
        """CPU implementation of normalize_scores."""
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
    
    def _normalize_scores_torch(self, scores: np.ndarray) -> np.ndarray:
        """PyTorch implementation of normalize_scores."""
        if not TORCH_AVAILABLE:
            return self._normalize_scores_cpu(scores)
        
        try:
            device = self.hw_accelerator.get_torch_device()
            
            # Convert to tensor
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
            
            # Create mask for finite values
            finite_mask = torch.isfinite(scores_tensor)
            
            if not torch.any(finite_mask):
                return torch.full_like(scores_tensor, 0.5).cpu().numpy()
                
            # Get min/max of finite values
            finite_values = scores_tensor[finite_mask]
            min_s = torch.min(finite_values)
            max_s = torch.max(finite_values)
            s_range = max_s - min_s
            
            if s_range < 1e-9:
                return torch.full_like(scores_tensor, 0.5).cpu().numpy()
                
            # Normalize using vectorized operations
            normalized = torch.full_like(scores_tensor, 0.5)
            normalized[finite_mask] = torch.clamp(
                (scores_tensor[finite_mask] - min_s) / s_range, 0.0, 1.0
            )
            
            return normalized.cpu().numpy()
        except Exception as e:
            logger.error(f"Error in PyTorch normalize_scores: {e}")
            return self._normalize_scores_cpu(scores)
    
    def _normalize_scores_numba(self, scores: np.ndarray) -> np.ndarray:
        """Numba implementation of normalize_scores."""
        if not NUMBA_AVAILABLE:
            return self._normalize_scores_cpu(scores)
        
        try:
            import numba
            
            @numba.jit(nopython=True)
            def _normalize(scores):
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
            
            return _normalize(scores)
        except Exception as e:
            logger.error(f"Error in Numba normalize_scores: {e}")
            return self._normalize_scores_cpu(scores)
    
    def _calculate_distance_matrix_cpu(self, data: np.ndarray) -> np.ndarray:
        """CPU implementation of distance matrix calculation."""
        n_samples = data.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((data[i] - data[j])**2))
                distances[i, j] = distances[j, i] = dist
                
        return distances
    
    def _calculate_distance_matrix_numba(self, data: np.ndarray) -> np.ndarray:
        """Numba implementation of distance matrix calculation."""
        if not NUMBA_AVAILABLE:
            return self._calculate_distance_matrix_cpu(data)
        
        try:
            import numba
            
            @numba.jit(nopython=True)
            def _calculate_distance(data):
                n_samples, n_features = data.shape
                distances = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        dist = 0.0
                        for k in range(n_features):
                            diff = data[i, k] - data[j, k]
                            dist += diff * diff
                        dist = np.sqrt(dist)
                        distances[i, j] = distances[j, i] = dist
                        
                return distances
            
            return _calculate_distance(data)
        except Exception as e:
            logger.error(f"Error in Numba distance matrix: {e}")
            return self._calculate_distance_matrix_cpu(data)
    
    def _calculate_distance_matrix_torch(self, data: np.ndarray) -> np.ndarray:
        """PyTorch implementation of distance matrix calculation."""
        if not TORCH_AVAILABLE:
            return self._calculate_distance_matrix_cpu(data)
        
        try:
            device = self.hw_accelerator.get_torch_device()
            
            # Convert to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            
            # Calculate squared Euclidean distances efficiently
            norm = torch.sum(data_tensor * data_tensor, dim=1, keepdim=True)
            dist = norm + norm.transpose(0, 1) - 2.0 * torch.mm(data_tensor, data_tensor.transpose(0, 1))
            
            # Fix potential numerical issues
            dist = torch.clamp(dist, min=0.0)
            
            # Take square root for Euclidean distance
            return torch.sqrt(dist).cpu().numpy()
        except Exception as e:
            logger.error(f"Error in PyTorch distance matrix: {e}")
            return self._calculate_distance_matrix_cpu(data)
    
    def _dtw_distance_cpu(self, s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        """CPU implementation of DTW distance."""
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
    
    def _dtw_distance_numba(self, s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        """Numba implementation of DTW distance."""
        if not NUMBA_AVAILABLE:
            return self._dtw_distance_cpu(s1, s2, window)
        
        try:
            import numba
            
            @numba.jit(nopython=True)
            def _dtw(s1, s2, window):
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
            
            return float(_dtw(s1, s2, window))
        except Exception as e:
            logger.error(f"Error in Numba DTW distance: {e}")
            return self._dtw_distance_cpu(s1, s2, window)
    
    def _cumsum_numba(self, data: np.ndarray) -> np.ndarray:
        """Numba implementation of cumulative sum."""
        if not NUMBA_AVAILABLE:
            return np.cumsum(data)
        
        try:
            import numba
            
            @numba.jit(nopython=True)
            def _cumsum(data):
                result = np.zeros_like(data)
                cumsum = 0.0
                for i in range(len(data)):
                    cumsum += data[i]
                    result[i] = cumsum
                return result
            
            return _cumsum(data)
        except Exception as e:
            logger.error(f"Error in Numba cumsum: {e}")
            return np.cumsum(data)
    
    def _moving_average_numba(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numba implementation of moving average."""
        if not NUMBA_AVAILABLE:
            return pd.Series(data).rolling(window=window).mean().values
        
        try:
            import numba
            
            @numba.jit(nopython=True)
            def _moving_avg(data, window):
                n = len(data)
                result = np.zeros_like(data)
                
                for i in range(n):
                    if i < window-1:
                        # Not enough data yet
                        result[i] = np.nan
                    else:
                        # Calculate mean of window
                        sum_val = 0.0
                        for j in range(window):
                            sum_val += data[i - j]
                        result[i] = sum_val / window
                        
                return result
            
            return _moving_avg(data, window)
        except Exception as e:
            logger.error(f"Error in Numba moving average: {e}")
            return pd.Series(data).rolling(window=window).mean().values
# Define these classes at module level, not inside a function

class BenchmarkMLP(nn.Module):
    """MLP model for benchmarking."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1):
        super(BenchmarkMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class BenchmarkCNN(nn.Module):
    """CNN model for benchmarking."""
    def __init__(self, input_channels, output_size):
        super(BenchmarkCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

class BenchmarkFusionModel(nn.Module):
    """Fusion model combining multiple inputs."""
    def __init__(self, input_sizes, hidden_size, output_size):
        super(BenchmarkFusionModel, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2)
            ) for size in input_sizes
        ])
        
        total_encoded_size = len(input_sizes) * (hidden_size // 2)
        self.fusion = nn.Sequential(
            nn.Linear(total_encoded_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, inputs):
        encoded = [encoder(inp) for encoder, inp in zip(self.encoders, inputs)]
        fused = torch.cat(encoded, dim=1)
        return self.fusion(fused)

class BenchmarkLSTM(nn.Module):
    """LSTM model for sequence processing."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BenchmarkLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # Use the last output
        out = self.fc(out[:, -1, :])
        return out

# Now update the test creation function to use these module-level classes
def _create_test_models(self):
    """Create test models for benchmarking."""
    self.models = {}
    
    # Create MLP models of different sizes
    self.models['mlp_small'] = BenchmarkMLP(10, [32, 16], 1)
    self.models['mlp_medium'] = BenchmarkMLP(50, [128, 64, 32], 1)
    self.models['mlp_large'] = BenchmarkMLP(100, [512, 256, 128, 64], 1)
    
    # Create CNN for 1D signal processing
    self.models['cnn'] = BenchmarkCNN(1, 1)
    
    # Create fusion model
    self.models['fusion'] = BenchmarkFusionModel([10, 20, 30], 64, 1)
    
    # Create LSTM model
    self.models['lstm'] = BenchmarkLSTM(10, 32, 2, 1)
    
    # Create example inputs for each model
    self.example_inputs = {
        'mlp_small': torch.randn(32, 10),
        'mlp_medium': torch.randn(32, 50),
        'mlp_large': torch.randn(32, 100),
        'cnn': torch.randn(32, 1, 100),  # (batch, channels, length)
        'fusion': [torch.randn(32, 10), torch.randn(32, 20), torch.randn(32, 30)],
        'lstm': torch.randn(32, 20, 10)  # (batch, seq_len, input_size)
    }
    
    return self.models, self.example_inputs

class WaveletProcessingBenchmark(CDFABenchmarkTest):
    """Benchmark suite for wavelet processing components."""
    
    def __init__(self, iterations: int = 5, warmup: int = 2, hardware: str = "cpu"):
        """Initialize benchmark test."""
        super().__init__(iterations, warmup, hardware)
        
        # Skip if wavelet processor not available
        if not WAVELET_AVAILABLE:
            logger.error("Wavelet processor not available, skipping benchmark")
            return
        
        # Initialize wavelet processor
        try:
            self.wavelet_processor = WaveletProcessor(
                hw_accelerator=HardwareAccelerator(
                    enable_gpu=(hardware != "cpu"),
                    prefer_cuda=(hardware == "cuda"),
                    device=hardware
                )
            )
            
            # Get hardware info
            self.device_info = {
                "device": hardware,
                "pywavelets_available": self.wavelet_processor.has_pywavelets,
                "torch_available": self.wavelet_processor.has_torch,
                "scipy_available": self.wavelet_processor.has_scipy,
                "numba_available": self.wavelet_processor.has_numba,
            }
            
            logger.info(f"Wavelet benchmarks initialized with device: {hardware}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wavelet processor: {e}")
            self.wavelet_processor = None
        
        # Initialize test data
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data for wavelet benchmarks."""
        data = {}
        
        # Various dataset sizes
        sizes = [128, 512, 1024, 4096, 16384]
        
        for size in sizes:
            # Sine wave with noise
            t = np.linspace(0, 10, size)
            data[f"sine_{size}"] = np.sin(t) + 0.1 * np.random.randn(size)
            
            # Square wave with noise
            square = np.zeros(size)
            square[size//4:(3*size)//4] = 1.0
            data[f"square_{size}"] = square + 0.1 * np.random.randn(size)
            
            # Chirp signal
            chirp = np.sin(t * t)
            data[f"chirp_{size}"] = chirp + 0.1 * np.random.randn(size)
            
            # Random walk
            walk = np.cumsum(np.random.randn(size))
            data[f"walk_{size}"] = walk
        
        # Add market data if available
        try:
            market_data = fetch_test_data(["SPY"], period="1y", interval="1d")
            if market_data and "SPY" in market_data:
                spy_df = market_data["SPY"]
                if "close" in spy_df.columns:
                    data["market_spy"] = spy_df["close"].values
        except Exception as e:
            logger.warning(f"Failed to fetch market data: {e}")
        
        # Add multiscale signal
        size = 4096
        t = np.linspace(0, 10, size)
        multiscale = np.sin(2 * np.pi * 1 * t) + \
                   0.5 * np.sin(2 * np.pi * 4 * t) + \
                   0.25 * np.sin(2 * np.pi * 16 * t) + \
                   0.1 * np.random.randn(size)
        data["multiscale"] = multiscale
        
        return data
    
    def run(self):
        """Run all wavelet processing benchmarks."""
        if not WAVELET_AVAILABLE or self.wavelet_processor is None:
            logger.error("Wavelet processor not available, skipping benchmark")
            return
        
        logger.info("Running wavelet processing benchmarks...")
        
        # Wavelet transform benchmarks
        self._benchmark_wavelet_transform()
        
        # Denoising benchmarks
        self._benchmark_denoising()
        
        # Feature extraction benchmarks
        self._benchmark_feature_extraction()
        
        # MRA benchmarks
        self._benchmark_mra()
        
        # CWT benchmarks
        self._benchmark_cwt()
        
        # Cycle detection benchmarks
        self._benchmark_cycle_detection()
        
        # Scaling benchmarks
        self._benchmark_scaling()
        
        # Compare implementations
        self._benchmark_implementation_comparison()
        
        logger.info("Wavelet benchmarks completed")
    
    def _benchmark_wavelet_transform(self):
        """Benchmark wavelet transform operations."""
        logger.info("Benchmarking wavelet transform operations...")
        
        # Test decompose_signal with different wavelets
        wavelets = ["db4", "sym4", "coif3", "haar"]
        signal = self.test_data["sine_4096"]
        
        for wavelet in wavelets:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.decompose_signal,
                signal,
                wavelet=wavelet,
                level=4,
                name=f"decompose_signal_{wavelet}",
                component="wavelet_transform"
            )
        
        # Test reconstruct_signal
        # First decompose a signal
        signal = self.test_data["sine_4096"]
        decomp = self.wavelet_processor.decompose_signal(signal, "db4", 4)
        
        # Benchmark reconstruction
        self.benchmark_function(
            self.wavelet_processor.reconstruct_signal,
            decomp,
            name="reconstruct_signal",
            component="wavelet_transform"
        )
        
        # Benchmark reconstruction of specific levels
        self.benchmark_function(
            self.wavelet_processor.reconstruct_signal,
            decomp,
            levels=[0, 2],  # Approximate coefficients and level 2 details
            name="reconstruct_signal_partial",
            component="wavelet_transform"
        )
    
    def _benchmark_denoising(self):
        """Benchmark wavelet denoising operations."""
        logger.info("Benchmarking wavelet denoising operations...")
        
        # Test denoise_signal with different methods
        methods = ["soft", "hard", "garrote"]
        signal = self.test_data["sine_4096"]
        
        for method in methods:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.denoise_signal,
                signal,
                method=method,
                name=f"denoise_signal_{method}",
                component="wavelet_denoising"
            )
        
        # Test denoising with different wavelets
        wavelets = ["db4", "sym4", "coif3", "haar"]
        signal = self.test_data["sine_4096"]
        
        for wavelet in wavelets:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.denoise_signal,
                signal,
                wavelet=wavelet,
                name=f"denoise_signal_wavelet_{wavelet}",
                component="wavelet_denoising"
            )
        
        # Test denoising with different noise levels
        noise_levels = [0.05, 0.1, 0.2, 0.5]
        base_signal = np.sin(np.linspace(0, 10, 4096))
        
        for noise in noise_levels:
            signal = base_signal + noise * np.random.randn(len(base_signal))
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.denoise_signal,
                signal,
                name=f"denoise_signal_noise_{noise}",
                component="wavelet_denoising"
            )
    
    def _benchmark_feature_extraction(self):
        """Benchmark wavelet feature extraction."""
        logger.info("Benchmarking wavelet feature extraction...")
        
        # Test extract_wavelet_features with different signals
        signals = ["sine_4096", "square_4096", "chirp_4096", "walk_4096"]
        
        for signal_name in signals:
            signal = self.test_data[signal_name]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.extract_wavelet_features,
                signal,
                name=f"extract_features_{signal_name}",
                component="wavelet_features"
            )
        
        # Test feature extraction with different wavelets
        wavelets = ["db4", "sym4", "coif3", "haar"]
        signal = self.test_data["multiscale"]
        
        for wavelet in wavelets:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.extract_wavelet_features,
                signal,
                wavelet=wavelet,
                name=f"extract_features_wavelet_{wavelet}",
                component="wavelet_features"
            )
    
    def _benchmark_mra(self):
        """Benchmark multi-resolution analysis."""
        logger.info("Benchmarking multi-resolution analysis...")
        
        # Test multi_resolution_analysis with different signals
        signals = ["sine_4096", "multiscale", "chirp_4096"]
        
        for signal_name in signals:
            signal = self.test_data[signal_name]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.multi_resolution_analysis,
                signal,
                name=f"mra_{signal_name}",
                component="wavelet_mra"
            )
        
        # Test trend extraction
        signal = self.test_data["walk_4096"]
        
        # Run benchmark
        self.benchmark_function(
            self.wavelet_processor.extract_trend,
            signal,
            name="extract_trend",
            component="wavelet_mra"
        )
        
        # Test fluctuation extraction
        self.benchmark_function(
            self.wavelet_processor.extract_fluctuations,
            signal,
            name="extract_fluctuations",
            component="wavelet_mra"
        )
    
    def _benchmark_cwt(self):
        """Benchmark continuous wavelet transform."""
        logger.info("Benchmarking continuous wavelet transform...")
        
        # Test CWT with different signals
        signals = ["sine_1024", "chirp_1024", "multiscale"]
        
        if len(signals) > 0 and "multiscale" in self.test_data:
            signals[-1] = self.test_data["multiscale"][:1024]  # Use shorter version
        
        for signal_name in signals:
            if isinstance(signal_name, np.ndarray):
                signal = signal_name
                name = "multiscale_1024"
            else:
                signal = self.test_data[signal_name]
                name = signal_name
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.continuous_wavelet_transform,
                signal,
                name=f"cwt_{name}",
                component="wavelet_cwt"
            )
        
        # Test CWT with different wavelets
        wavelets = ["morl", "mexh", "gaus1"]
        signal = self.test_data["sine_1024"]
        
        for wavelet in wavelets:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.continuous_wavelet_transform,
                signal,
                wavelet=wavelet,
                name=f"cwt_wavelet_{wavelet}",
                component="wavelet_cwt"
            )
        
        # Test CWT with different number of scales
        scales = [8, 16, 32, 64]
        signal = self.test_data["sine_1024"]
        
        for scale in scales:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.continuous_wavelet_transform,
                signal,
                scales=scale,
                name=f"cwt_scales_{scale}",
                component="wavelet_cwt"
            )
    
    def _benchmark_cycle_detection(self):
        """Benchmark cycle detection."""
        logger.info("Benchmarking cycle detection...")
        
        # Test detect_cycles with different signals
        signals = ["sine_4096", "multiscale"]
        
        for signal_name in signals:
            signal = self.test_data[signal_name]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.detect_cycles,
                signal,
                name=f"detect_cycles_{signal_name}",
                component="wavelet_cycles"
            )
        
        # Create signal with multiple cycles
        t = np.linspace(0, 20, 4096)
        multi_cycle = np.sin(2*np.pi*0.5*t) + 0.5*np.sin(2*np.pi*2*t) + 0.25*np.sin(2*np.pi*8*t)
        
        # Run benchmark with different min/max periods
        period_ranges = [(2, 128), (2, 256), (8, 512)]
        
        for min_p, max_p in period_ranges:
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.detect_cycles,
                multi_cycle,
                min_period=min_p,
                max_period=max_p,
                name=f"detect_cycles_period_{min_p}_{max_p}",
                component="wavelet_cycles"
            )
    
    def _benchmark_scaling(self):
        """Benchmark scaling with different data sizes."""
        logger.info("Benchmarking scaling with different data sizes...")
        
        # Test wavelet decomposition scaling
        sizes = [512, 1024, 4096, 16384]  # Remove 2048 which doesn't exist
        
        for size in sizes:
            signal = self.test_data[f"sine_{size}"]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.decompose_signal,
                signal,
                name=f"decompose_signal_{size}",
                component="wavelet_scaling"
            )
        
        # Test denoising scaling
        for size in sizes:
            signal = self.test_data[f"sine_{size}"]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.denoise_signal,
                signal,
                name=f"denoise_signal_{size}",
                component="wavelet_scaling"
            )
        
        # Test CWT scaling (for smaller sizes)
        smaller_sizes = [512, 1024]  # Remove 2048 which doesn't exist
        
        for size in smaller_sizes:
            signal = self.test_data[f"sine_{size}"]
            
            # Run benchmark
            self.benchmark_function(
                self.wavelet_processor.continuous_wavelet_transform,
                signal,
                name=f"cwt_{size}",
                component="wavelet_scaling"
            )
            
    def _benchmark_implementation_comparison(self):
        """Benchmark comparison of different implementations."""
        if not self.wavelet_processor.has_pywavelets:
            return
        
        logger.info("Benchmarking implementation comparison...")
        
        # Compare PyWavelets vs PyTorch implementation of DWT
        if self.wavelet_processor.has_torch:
            signal = self.test_data["sine_4096"]
            
            implementations = {
                "PyWavelets": self._decompose_pywt,
                "WaveletProcessor": self.wavelet_processor.decompose_signal,
            }
            
            # Run benchmark
            self.compare_implementations(implementations, signal, component="implementation_comparison")
        
        # Compare regular vs accelerated CWT
        if self.wavelet_processor.has_torch:
            signal = self.test_data["sine_1024"]
            
            implementations = {
                "CWT standard": lambda x: self.wavelet_processor.continuous_wavelet_transform(x, use_fft=False),
                "CWT with FFT": lambda x: self.wavelet_processor.continuous_wavelet_transform(x, use_fft=True),
            }
            
            # Run benchmark
            self.compare_implementations(implementations, signal, component="implementation_comparison")
    
    # ---- Utility Methods ----
    
    def _decompose_pywt(self, data: np.ndarray, wavelet: str = "db4", level: int = 4, mode: str = "symmetric") -> Any:
        """Direct PyWavelets implementation of wavelet decomposition."""
        if not PYWAVELETS_AVAILABLE:
            return None
        
        try:
            import pywt
            
            # Determine maximum decomposition level based on data length and wavelet
            max_level = pywt.dwt_max_level(len(data), wavelet)
            
            # Ensure level is valid
            level = min(level, max_level)
            
            # Perform multilevel decomposition
            coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
            
            return coeffs
        except Exception as e:
            logger.error(f"Error in PyWavelets decomposition: {e}")
            return None


class TorchScriptFusionBenchmark(CDFABenchmarkTest):
    """Benchmark suite for TorchScript fusion component."""
    
    def __init__(self, iterations: int = 5, warmup: int = 2, hardware: str = "cpu"):
        """Initialize benchmark test."""
        super().__init__(iterations, warmup, hardware)
        
        # Skip if TorchScript fusion not available
        if not TORCHSCRIPT_AVAILABLE:
            logger.error("TorchScript fusion not available, skipping benchmark")
            return
        
        # Set up hardware
        try:
            self.hw_accelerator = HardwareAccelerator(
                enable_gpu=(hardware != "cpu"),
                prefer_cuda=(hardware == "cuda"),
                device=hardware
            )
            
            # Initialize fusion module
            self.fusion = TorchScriptFusion(hw_accelerator=self.hw_accelerator)
            
            logger.info(f"TorchScript fusion benchmark initialized with device: {hardware}")
        except Exception as e:
            logger.error(f"Failed to initialize TorchScript fusion: {e}")
            self.fusion = None
        
        # Initialize test models
        self.test_models = self._create_test_models()
    
    def _create_test_models(self) -> Dict[str, Any]:
        """Create test models for benchmarks."""
        if not TORCH_AVAILABLE:
            return {}
        
        try:
            models = {}
            
            # Create simple MLP
            class MLP(torch.nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_size, hidden_size)
                    self.relu = torch.nn.ReLU()
                    self.fc2 = torch.nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
            
            # Create examples with different sizes
            models["mlp_small"] = (MLP(10, 20, 1), torch.randn(32, 10))
            models["mlp_medium"] = (MLP(50, 100, 10), torch.randn(32, 50))
            models["mlp_large"] = (MLP(100, 500, 10), torch.randn(16, 100))
            
            # Create simple CNN
            class CNN(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
                    self.relu = torch.nn.ReLU()
                    self.pool = torch.nn.MaxPool1d(2)
                    self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
                    self.fc = torch.nn.Linear(32 * 50, 10)
                
                def forward(self, x):
                    # Reshape for 1D convolution
                    batch_size = x.shape[0]
                    x = x.view(batch_size, 1, -1)
                    
                    x = self.conv1(x)
                    x = self.relu(x)
                    x = self.pool(x)
                    x = self.conv2(x)
                    x = self.relu(x)
                    x = x.view(batch_size, -1)
                    x = self.fc(x)
                    return x
            
            # Create time series CNN example
            models["cnn"] = (CNN(), torch.randn(16, 1, 100))
            
            # Create fusion model
            class FusionModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight1 = torch.nn.Parameter(torch.ones(1))
                    self.weight2 = torch.nn.Parameter(torch.ones(1))
                    self.bias = torch.nn.Parameter(torch.zeros(1))
                
                def forward(self, x, y):
                    return self.weight1 * x + self.weight2 * y + self.bias
            
            # Create fusion model example
            models["fusion"] = (FusionModel(), (torch.randn(10), torch.randn(10)))
            
            # Create LSTM model
            class LSTMModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(10, 20, batch_first=True)
                    self.fc = torch.nn.Linear(20, 1)
                
                def forward(self, x):
                    # x: [batch_size, seq_len, input_size]
                    lstm_out, _ = self.lstm(x)
                    # Take the output of the last time step
                    lstm_out = lstm_out[:, -1, :]
                    output = self.fc(lstm_out)
                    return output
            
            # Create LSTM example
            models["lstm"] = (LSTMModel(), torch.randn(8, 20, 10))
            
            return models
        except Exception as e:
            logger.error(f"Error creating test models: {e}")
            return {}
    
    def run(self):
        """Run all TorchScript fusion benchmarks."""
        if not TORCHSCRIPT_AVAILABLE or self.fusion is None:
            logger.error("TorchScript fusion not available, skipping benchmark")
            return
        
        logger.info("Running TorchScript fusion benchmarks...")
        
        # Basic model compilation benchmarks
        self._benchmark_model_compilation()
        
        # Execution benchmarks
        self._benchmark_model_execution()
        
        # Optimization benchmarks
        self._benchmark_optimizations()
        
        # Model size benchmarks
        self._benchmark_model_size()
        
        # Memory usage benchmarks
        self._benchmark_memory_usage()
        
        logger.info("TorchScript fusion benchmarks completed")
    
    def _benchmark_model_compilation(self):
        """Benchmark model compilation times."""
        logger.info("Benchmarking model compilation times...")
        
        for name, (model, example_input) in self.test_models.items():
            # Benchmark model compilation
            self.benchmark_function(
                self.fusion.create_optimized_model,
                model,
                example_input,
                name=f"compile_{name}",
                component="model_compilation"
            )
    
    def _benchmark_model_execution(self):
        """Benchmark model execution times."""
        logger.info("Benchmarking model execution times...")
        
        for name, (model, example_input) in self.test_models.items():
            # First compile the model
            try:
                compiled_model = self.fusion.create_optimized_model(model, example_input)
                
                # Benchmark original model execution
                def run_original_model(model, input_data):
                    if isinstance(input_data, tuple):
                        return model(*input_data)
                    else:
                        return model(input_data)
                
                self.benchmark_function(
                    run_original_model,
                    model,
                    example_input,
                    name=f"execute_original_{name}",
                    component="model_execution"
                )
                
                # Benchmark compiled model execution
                def run_compiled_model(model, input_data):
                    if isinstance(input_data, tuple):
                        inputs = [
                            self.hw_accelerator.to_device(inp) if isinstance(inp, torch.Tensor) else inp
                            for inp in input_data
                        ]
                        return self.fusion.run_model(model, inputs)
                    else:
                        return self.fusion.run_model(model, input_data)
                
                self.benchmark_function(
                    run_compiled_model,
                    compiled_model,
                    example_input,
                    name=f"execute_compiled_{name}",
                    component="model_execution"
                )
            except Exception as e:
                logger.error(f"Error benchmarking model execution for {name}: {e}")
                traceback.print_exc()
    
    def _benchmark_optimizations(self):
        """Benchmark different optimization techniques."""
        logger.info("Benchmarking optimization techniques...")
        
        # Test different fusion optimization levels
        if "mlp_medium" in self.test_models:
            model, example_input = self.test_models["mlp_medium"]
            
            try:
                # Test different optimization configurations
                optimization_configs = {
                    "Default": {},
                    "Aggressive": {"optimization_level": "aggressive"},
                    "Conservative": {"optimization_level": "conservative"},
                    "Trace": {"use_tracing": True},
                    "Inference": {"inference_mode": True},
                }
                
                for name, config in optimization_configs.items():
                    # Create optimized model with this config
                    self.benchmark_function(
                        lambda m, inp, cfg: self.fusion.create_optimized_model(m, inp, **cfg),
                        model,
                        example_input,
                        config,
                        name=f"optimize_{name}",
                        component="optimization_techniques"
                    )
            except Exception as e:
                logger.error(f"Error benchmarking optimization techniques: {e}")
                traceback.print_exc()
    
    def _benchmark_model_size(self):
        """Benchmark model size."""
        logger.info("Benchmarking model size...")
        
        for name, (model, example_input) in self.test_models.items():
            try:
                # Measure original model size
                original_size = self._measure_model_size(model)
                
                # Compile model
                compiled_model = self.fusion.create_optimized_model(model, example_input)
                
                # Measure compiled model size
                compiled_size = self._measure_model_size(compiled_model)
                
                logger.info(f"Model {name} - Original size: {original_size / 1024:.2f} KB, "
                          f"Compiled size: {compiled_size / 1024:.2f} KB, "
                          f"Ratio: {compiled_size / original_size:.2f}x")
            except Exception as e:
                logger.error(f"Error benchmarking model size for {name}: {e}")
    
    def _benchmark_memory_usage(self):
        """Benchmark memory usage."""
        logger.info("Benchmarking memory usage...")
        
        for name, (model, example_input) in self.test_models.items():
            try:
                # Compile model
                compiled_model = self.fusion.create_optimized_model(model, example_input)
                
                # Benchmark memory usage for original model
                def run_original_many_times(model, input_data, times=100):
                    for _ in range(times):
                        if isinstance(input_data, tuple):
                            model(*input_data)
                        else:
                            model(input_data)
                
                with memory_profiling() as get_mem_used:
                    run_original_many_times(model, example_input)
                    original_memory = get_mem_used()
                
                # Benchmark memory usage for compiled model
                def run_compiled_many_times(model, input_data, times=100):
                    for _ in range(times):
                        if isinstance(input_data, tuple):
                            inputs = [
                                self.hw_accelerator.to_device(inp) if isinstance(inp, torch.Tensor) else inp
                                for inp in input_data
                            ]
                            self.fusion.run_model(model, inputs)
                        else:
                            self.fusion.run_model(model, input_data)
                
                with memory_profiling() as get_mem_used:
                    run_compiled_many_times(compiled_model, example_input)
                    compiled_memory = get_mem_used()
                
                logger.info(f"Model {name} - Original memory: {original_memory:.2f} MiB, "
                          f"Compiled memory: {compiled_memory:.2f} MiB, "
                          f"Ratio: {compiled_memory / (original_memory or 1):.2f}x")
            except Exception as e:
                logger.error(f"Error benchmarking memory usage for {name}: {e}")
    
    def _measure_model_size(self, model: Any) -> int:
        """Measure model size in bytes."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.jit.save(model, tmp.name) if isinstance(model, torch.jit.ScriptModule) else torch.save(model, tmp.name)
            return os.path.getsize(tmp.name)


def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CDFA Hardware Benchmarks')
    parser.add_argument('--hardware', choices=['cpu', 'cuda', 'rocm', 'mps'], default='cpu',
                      help='Hardware to benchmark on')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=2,
                      help='Number of warmup iterations')
    parser.add_argument('--component', choices=['hardware', 'wavelet', 'torchscript', 'all'],
                      default='all', help='Component to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results',
                      help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Configure output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run selected benchmarks
    if args.component in ('hardware', 'all'):
        try:
            hw_bench = HardwareAccelerationBenchmark(
                iterations=args.iterations,
                warmup=args.warmup,
                hardware=args.hardware
            )
            hw_bench.run()
            hw_bench.generate_report(args.output)
        except Exception as e:
            logger.error(f"Error running hardware benchmarks: {e}")
            traceback.print_exc()
    
    if args.component in ('wavelet', 'all'):
        try:
            wavelet_bench = WaveletProcessingBenchmark(
                iterations=args.iterations,
                warmup=args.warmup,
                hardware=args.hardware
            )
            wavelet_bench.run()
            wavelet_bench.generate_report(args.output)
        except Exception as e:
            logger.error(f"Error running wavelet benchmarks: {e}")
            traceback.print_exc()
    
    if args.component in ('torchscript', 'all'):
        try:
            ts_bench = TorchScriptFusionBenchmark(
                iterations=args.iterations,
                warmup=args.warmup,
                hardware=args.hardware
            )
            ts_bench.run()
            ts_bench.generate_report(args.output)
        except Exception as e:
            logger.error(f"Error running TorchScript benchmarks: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
