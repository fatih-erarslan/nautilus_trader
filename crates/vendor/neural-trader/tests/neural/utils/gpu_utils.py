"""
GPU Testing Utilities

Utilities for testing GPU functionality, performance, and compatibility.
"""

import torch
import numpy as np
import time
import psutil
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json


@dataclass
class GPUInfo:
    """GPU information container."""
    available: bool = False
    count: int = 0
    current_device: int = 0
    name: str = ""
    memory_total: int = 0
    memory_available: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    cuda_version: str = ""


class GPUDetector:
    """Detect and report GPU capabilities."""
    
    @staticmethod
    def get_gpu_info() -> GPUInfo:
        """Get comprehensive GPU information."""
        info = GPUInfo()
        
        if not torch.cuda.is_available():
            return info
        
        info.available = True
        info.count = torch.cuda.device_count()
        info.current_device = torch.cuda.current_device()
        info.name = torch.cuda.get_device_name()
        
        # Memory information
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated()
        info.memory_total = memory_total
        info.memory_available = memory_total - memory_allocated
        
        # Compute capability
        props = torch.cuda.get_device_properties(0)
        info.compute_capability = (props.major, props.minor)
        
        # CUDA version
        info.cuda_version = torch.version.cuda or "Unknown"
        
        return info
    
    @staticmethod
    def check_gpu_requirements(min_memory_gb: float = 4.0,
                             min_compute_capability: Tuple[int, int] = (3, 5)) -> bool:
        """Check if GPU meets minimum requirements."""
        info = GPUDetector.get_gpu_info()
        
        if not info.available:
            return False
        
        # Check memory
        memory_gb = info.memory_total / (1024 ** 3)
        if memory_gb < min_memory_gb:
            warnings.warn(f"GPU memory {memory_gb:.1f}GB < required {min_memory_gb}GB")
            return False
        
        # Check compute capability
        if info.compute_capability < min_compute_capability:
            warnings.warn(f"Compute capability {info.compute_capability} < required {min_compute_capability}")
            return False
        
        return True
    
    @staticmethod
    def get_optimal_batch_size(model_size_mb: float,
                             sequence_length: int,
                             feature_dim: int = 1,
                             safety_factor: float = 0.8) -> int:
        """Estimate optimal batch size for GPU memory."""
        if not torch.cuda.is_available():
            return 32  # Default CPU batch size
        
        # Get available memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory_mb = available_memory / (1024 ** 2) * safety_factor
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = sequence_length * feature_dim * 4  # 4 bytes per float32
        memory_per_sample_mb = memory_per_sample / (1024 ** 2)
        
        # Account for model and intermediate activations
        available_for_batch = available_memory_mb - model_size_mb - 500  # Buffer
        
        if available_for_batch <= 0:
            return 1
        
        batch_size = int(available_for_batch / (memory_per_sample_mb * 3))  # 3x for gradients
        return max(1, min(batch_size, 512))  # Cap at 512


class GPUMemoryTracker:
    """Track GPU memory usage during tests."""
    
    def __init__(self):
        self.checkpoints = {}
        self.peak_memory = 0
        self.initial_memory = 0
        
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
            self.peak_memory = self.initial_memory
    
    def checkpoint(self, name: str) -> Dict[str, int]:
        """Create a memory checkpoint."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'cached': 0}
        
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        self.checkpoints[name] = {
            'allocated': allocated,
            'cached': cached,
            'timestamp': time.time()
        }
        
        self.peak_memory = max(self.peak_memory, allocated)
        
        return {'allocated': allocated, 'cached': cached}
    
    def get_memory_usage(self, format_mb: bool = True) -> Dict[str, float]:
        """Get current memory usage."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'cached': 0, 'peak': 0}
        
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        if format_mb:
            return {
                'allocated': allocated / (1024 ** 2),
                'cached': cached / (1024 ** 2),
                'peak': self.peak_memory / (1024 ** 2)
            }
        else:
            return {
                'allocated': allocated,
                'cached': cached,
                'peak': self.peak_memory
            }
    
    def detect_memory_leak(self, tolerance_mb: float = 10.0) -> bool:
        """Detect if there's a memory leak."""
        if not torch.cuda.is_available():
            return False
        
        current_memory = torch.cuda.memory_allocated()
        leak_size = (current_memory - self.initial_memory) / (1024 ** 2)
        
        return leak_size > tolerance_mb
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset(self):
        """Reset memory tracking."""
        self.checkpoints.clear()
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
            self.peak_memory = self.initial_memory


@contextmanager
def gpu_memory_context(name: str = "test", clear_cache: bool = True):
    """Context manager for GPU memory tracking."""
    tracker = GPUMemoryTracker()
    
    try:
        tracker.checkpoint(f"{name}_start")
        yield tracker
    finally:
        tracker.checkpoint(f"{name}_end")
        
        if clear_cache:
            tracker.clear_cache()
        
        # Log memory usage
        usage = tracker.get_memory_usage()
        print(f"GPU Memory Usage for {name}: {usage['allocated']:.1f}MB allocated, "
              f"{usage['peak']:.1f}MB peak")


class GPUPerformanceBenchmark:
    """Benchmark GPU performance for neural operations."""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.results = {}
    
    def benchmark_tensor_operations(self, size: Tuple[int, ...] = (1000, 1000)) -> Dict[str, float]:
        """Benchmark basic tensor operations."""
        if not torch.cuda.is_available():
            return {'matmul': 0, 'add': 0, 'conv1d': 0}
        
        # Create test tensors
        a = torch.randn(size, device=self.device)
        b = torch.randn(size, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark matrix multiplication
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        matmul_time = (time.perf_counter() - start) / 100
        
        # Benchmark element-wise addition
        start = time.perf_counter()
        for _ in range(100):
            _ = a + b
        torch.cuda.synchronize()
        add_time = (time.perf_counter() - start) / 100
        
        # Benchmark 1D convolution
        if len(size) >= 2:
            conv = torch.nn.Conv1d(size[0], size[0], 3, padding=1).to(self.device)
            input_tensor = torch.randn(1, size[0], size[1], device=self.device)
            
            start = time.perf_counter()
            for _ in range(100):
                _ = conv(input_tensor)
            torch.cuda.synchronize()
            conv_time = (time.perf_counter() - start) / 100
        else:
            conv_time = 0
        
        results = {
            'matmul_ms': matmul_time * 1000,
            'add_ms': add_time * 1000,
            'conv1d_ms': conv_time * 1000
        }
        
        self.results['tensor_ops'] = results
        return results
    
    def benchmark_memory_bandwidth(self, size_mb: int = 100) -> Dict[str, float]:
        """Benchmark memory bandwidth."""
        if not torch.cuda.is_available():
            return {'bandwidth_gb_s': 0}
        
        # Calculate tensor size
        elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        tensor = torch.randn(elements, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = tensor.sum()
        
        torch.cuda.synchronize()
        
        # Benchmark memory reads
        start = time.perf_counter()
        for _ in range(100):
            _ = tensor.sum()
        torch.cuda.synchronize()
        read_time = (time.perf_counter() - start) / 100
        
        # Calculate bandwidth
        bandwidth_gb_s = (size_mb / 1024) / read_time
        
        result = {'bandwidth_gb_s': bandwidth_gb_s}
        self.results['memory_bandwidth'] = result
        return result
    
    def benchmark_mixed_precision(self, size: Tuple[int, ...] = (1000, 1000)) -> Dict[str, float]:
        """Benchmark mixed precision performance."""
        if not torch.cuda.is_available():
            return {'fp32_ms': 0, 'fp16_ms': 0, 'speedup': 1.0}
        
        # FP32 benchmark
        a_fp32 = torch.randn(size, device=self.device, dtype=torch.float32)
        b_fp32 = torch.randn(size, device=self.device, dtype=torch.float32)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.perf_counter() - start) / 100
        
        # FP16 benchmark
        a_fp16 = torch.randn(size, device=self.device, dtype=torch.float16)
        b_fp16 = torch.randn(size, device=self.device, dtype=torch.float16)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / 100
        
        speedup = fp32_time / fp16_time if fp16_time > 0 else 1.0
        
        results = {
            'fp32_ms': fp32_time * 1000,
            'fp16_ms': fp16_time * 1000,
            'speedup': speedup
        }
        
        self.results['mixed_precision'] = results
        return results


class GPUCompatibilityTester:
    """Test GPU compatibility and functionality."""
    
    def __init__(self):
        self.test_results = {}
    
    def run_compatibility_tests(self) -> Dict[str, Any]:
        """Run comprehensive GPU compatibility tests."""
        results = {
            'gpu_available': torch.cuda.is_available(),
            'tests': {}
        }
        
        if not torch.cuda.is_available():
            results['message'] = "GPU not available - skipping compatibility tests"
            return results
        
        # Basic tensor operations
        results['tests']['basic_ops'] = self._test_basic_operations()
        
        # Memory allocation
        results['tests']['memory'] = self._test_memory_allocation()
        
        # Data type support
        results['tests']['dtypes'] = self._test_data_types()
        
        # CUDA streams
        results['tests']['streams'] = self._test_cuda_streams()
        
        # Mixed precision
        results['tests']['mixed_precision'] = self._test_mixed_precision()
        
        return results
    
    def _test_basic_operations(self) -> Dict[str, bool]:
        """Test basic GPU operations."""
        try:
            # Create tensors
            a = torch.randn(100, 100, device='cuda')
            b = torch.randn(100, 100, device='cuda')
            
            # Test operations
            c = a + b
            d = torch.matmul(a, b)
            e = torch.sum(a)
            
            # Verify results
            assert c.device.type == 'cuda'
            assert d.device.type == 'cuda'
            assert e.device.type == 'cuda'
            
            return {'passed': True, 'error': None}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_memory_allocation(self) -> Dict[str, Any]:
        """Test GPU memory allocation."""
        try:
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate memory
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, device='cuda')
                tensors.append(tensor)
            
            allocated_memory = torch.cuda.memory_allocated()
            
            # Clear tensors
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            
            return {
                'passed': True,
                'initial_mb': initial_memory / (1024**2),
                'peak_mb': allocated_memory / (1024**2),
                'final_mb': final_memory / (1024**2),
                'error': None
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_data_types(self) -> Dict[str, bool]:
        """Test different data type support."""
        results = {}
        
        dtypes = [
            torch.float32, torch.float16, torch.int32, torch.int64,
            torch.bool, torch.complex64
        ]
        
        for dtype in dtypes:
            try:
                tensor = torch.randn(10, 10, device='cuda', dtype=dtype)
                results[str(dtype)] = True
            except Exception:
                results[str(dtype)] = False
        
        return results
    
    def _test_cuda_streams(self) -> Dict[str, bool]:
        """Test CUDA streams functionality."""
        try:
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
            
            with torch.cuda.stream(stream1):
                a = torch.randn(1000, 1000, device='cuda')
                b = torch.randn(1000, 1000, device='cuda')
                c = a + b
            
            with torch.cuda.stream(stream2):
                d = torch.randn(1000, 1000, device='cuda')
                e = torch.randn(1000, 1000, device='cuda')
                f = d + e
            
            torch.cuda.synchronize()
            
            return {'passed': True, 'error': None}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_mixed_precision(self) -> Dict[str, bool]:
        """Test mixed precision support."""
        try:
            # Test automatic mixed precision
            scaler = torch.cuda.amp.GradScaler()
            
            model = torch.nn.Linear(100, 10).cuda()
            optimizer = torch.optim.Adam(model.parameters())
            
            x = torch.randn(32, 100, device='cuda')
            y = torch.randn(32, 10, device='cuda')
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, y)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            return {'passed': True, 'error': None}
        except Exception as e:
            return {'passed': False, 'error': str(e)}


def skip_if_no_gpu(test_func):
    """Decorator to skip tests if GPU is not available."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("GPU not available")
        return test_func(*args, **kwargs)
    return wrapper


def require_gpu_memory(min_memory_gb: float):
    """Decorator to skip tests if insufficient GPU memory."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                import pytest
                pytest.skip("GPU not available")
            
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_gb = total_memory / (1024 ** 3)
            
            if available_gb < min_memory_gb:
                import pytest
                pytest.skip(f"Insufficient GPU memory: {available_gb:.1f}GB < {min_memory_gb}GB")
            
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


# Context managers for GPU testing
@contextmanager
def gpu_device_context(device_id: int = 0):
    """Context manager for GPU device selection."""
    if not torch.cuda.is_available():
        yield torch.device('cpu')
        return
    
    original_device = torch.cuda.current_device()
    
    try:
        torch.cuda.set_device(device_id)
        yield torch.device(f'cuda:{device_id}')
    finally:
        torch.cuda.set_device(original_device)


@contextmanager
def gpu_memory_limit(limit_gb: float):
    """Context manager to limit GPU memory usage."""
    if not torch.cuda.is_available():
        yield
        return
    
    # Note: PyTorch doesn't have built-in memory limiting
    # This is a placeholder for memory monitoring
    initial_memory = torch.cuda.memory_allocated()
    
    try:
        yield
    finally:
        current_memory = torch.cuda.memory_allocated()
        used_memory_gb = (current_memory - initial_memory) / (1024 ** 3)
        
        if used_memory_gb > limit_gb:
            warnings.warn(f"Memory usage {used_memory_gb:.2f}GB exceeded limit {limit_gb}GB")


# Export utilities
__all__ = [
    'GPUInfo',
    'GPUDetector',
    'GPUMemoryTracker',
    'GPUPerformanceBenchmark',
    'GPUCompatibilityTester',
    'gpu_memory_context',
    'skip_if_no_gpu',
    'require_gpu_memory',
    'gpu_device_context',
    'gpu_memory_limit'
]