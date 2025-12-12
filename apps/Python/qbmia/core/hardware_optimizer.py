"""
QBMIA-specific hardware optimization extending base hardware manager.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable, Union
import psutil
import time
import threading
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

class QBMIAHardwareOptimizer:
    """
    Hardware optimizer specifically tuned for QBMIA workloads.
    """

    def __init__(self, force_cpu: bool = False, enable_profiling: bool = True):
        """
        Initialize QBMIA hardware optimizer.

        Args:
            force_cpu: Force CPU execution even if GPU available
            enable_profiling: Enable performance profiling
        """
        self.force_cpu = force_cpu
        self.enable_profiling = enable_profiling

        # Import base hardware manager if available
        try:
            from hardware_manager import HardwareManager
            self.hw_manager = HardwareManager.get_manager(
                force_cpu=force_cpu,
                use_jit=True,
                multi_gpu=True
            )
            self.hw_manager.initialize_hardware()
        except ImportError:
            logger.warning("Base hardware manager not available, using fallback")
            self.hw_manager = None

        # Device information
        self._device = self._detect_device()
        self._device_type = self._determine_device_type()

        # Performance profiling
        self.profile_data = {}
        self.optimization_cache = {}

        # Memory pools for different workload types
        self.memory_pools = {
            'quantum_state': None,
            'matrix_ops': None,
            'pattern_matching': None
        }

        # Workload-specific optimizations
        self.workload_configs = self._initialize_workload_configs()

        logger.info(f"QBMIA Hardware Optimizer initialized on {self._device_type}")

    def _detect_device(self) -> str:
        """Detect available compute device."""
        if self.hw_manager:
            device_info = self.hw_manager.devices
            if device_info['nvidia_gpu']['available']:
                return 'cuda'
            elif device_info['amd_gpu']['available']:
                return 'rocm'

        # Fallback detection
        if not self.force_cpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
            except ImportError:
                pass

            try:
                import pyopencl as cl
                platforms = cl.get_platforms()
                if platforms:
                    return 'opencl'
            except ImportError:
                pass

        return 'cpu'

    def _determine_device_type(self) -> str:
        """Determine device type string."""
        if 'cuda' in self._device:
            return 'nvidia_gpu'
        elif 'rocm' in self._device or 'opencl' in self._device:
            return 'amd_gpu'
        else:
            return 'cpu'

    def _initialize_workload_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimized configurations for different workloads."""
        configs = {
            'quantum_simulation': {
                'batch_size': 32 if 'gpu' in self._device_type else 8,
                'precision': 'float32' if 'gpu' in self._device_type else 'float64',
                'parallel_circuits': 4 if 'gpu' in self._device_type else 2,
                'memory_fraction': 0.8
            },
            'nash_equilibrium': {
                'max_iterations': 200,
                'convergence_check_interval': 10,
                'use_jit': True,
                'vectorize': True
            },
            'pattern_matching': {
                'chunk_size': 10000,
                'num_threads': psutil.cpu_count() // 2,
                'use_simd': True
            },
            'matrix_operations': {
                'block_size': 512 if 'gpu' in self._device_type else 128,
                'use_cublas': 'cuda' in self._device,
                'use_rocblas': 'rocm' in self._device
            }
        }

        return configs

    def get_optimal_config(self, workload_type: str) -> Dict[str, Any]:
        """
        Get optimal configuration for specific workload type.

        Args:
            workload_type: Type of workload (quantum_simulation, nash_equilibrium, etc.)

        Returns:
            Optimized configuration dictionary
        """
        return self.workload_configs.get(workload_type, {})

    def allocate_memory_pool(self, pool_type: str, size_mb: int) -> bool:
        """
        Allocate memory pool for specific workload type.

        Args:
            pool_type: Type of memory pool
            size_mb: Size in megabytes

        Returns:
            Success status
        """
        try:
            if 'cuda' in self._device:
                import torch
                # Allocate GPU memory pool
                size_bytes = size_mb * 1024 * 1024
                self.memory_pools[pool_type] = torch.cuda.allocate_shared_memory(size_bytes)
            else:
                # CPU memory pool (numpy)
                size_elements = (size_mb * 1024 * 1024) // 8  # 8 bytes per float64
                self.memory_pools[pool_type] = np.zeros(size_elements, dtype=np.float64)

            logger.info(f"Allocated {size_mb}MB memory pool for {pool_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to allocate memory pool: {e}")
            return False

    @lru_cache(maxsize=128)
    def optimize_quantum_circuit_execution(self, num_qubits: int,
                                         circuit_depth: int) -> Dict[str, Any]:
        """
        Optimize quantum circuit execution parameters.

        Args:
            num_qubits: Number of qubits
            circuit_depth: Circuit depth

        Returns:
            Optimized execution parameters
        """
        # Memory requirement estimation
        state_vector_size = 2 ** num_qubits
        memory_required = state_vector_size * 16  # Complex128
        memory_required_mb = memory_required / (1024 * 1024)

        # Determine execution strategy
        if memory_required_mb > 4096:  # 4GB threshold
            strategy = 'distributed'
            chunk_size = min(2 ** 20, state_vector_size // 4)
        elif memory_required_mb > 1024:  # 1GB threshold
            strategy = 'batched'
            chunk_size = min(2 ** 18, state_vector_size // 2)
        else:
            strategy = 'direct'
            chunk_size = state_vector_size

        # Backend-specific optimizations
        if 'cuda' in self._device:
            backend_config = {
                'backend': 'lightning.gpu',
                'batch_obs': True,
                'c_dtype': np.complex64 if memory_required_mb > 2048 else np.complex128,
                'parallel_shots': min(32, circuit_depth // 10)
            }
        elif 'rocm' in self._device:
            backend_config = {
                'backend': 'lightning.kokkos',
                'execution_space': 'hip',
                'parallel_shots': min(16, circuit_depth // 10)
            }
        else:
            backend_config = {
                'backend': 'lightning.qubit',
                'num_threads': psutil.cpu_count(),
                'parallel_shots': min(8, circuit_depth // 10)
            }

        return {
            'strategy': strategy,
            'chunk_size': chunk_size,
            'memory_required_mb': memory_required_mb,
            'backend_config': backend_config,
            'estimated_time_ms': self._estimate_execution_time(num_qubits, circuit_depth)
        }

    def _estimate_execution_time(self, num_qubits: int, circuit_depth: int) -> float:
        """Estimate execution time for quantum circuit."""
        # Base time estimation (milliseconds)
        base_time = circuit_depth * (2 ** num_qubits) / 1e6

        # Device-specific multipliers
        device_multipliers = {
            'nvidia_gpu': 0.1,
            'amd_gpu': 0.15,
            'cpu': 1.0
        }

        multiplier = device_multipliers.get(self._device_type, 1.0)

        return base_time * multiplier

    def accelerate(self, func: Callable, *args, **kwargs) -> Any:
        """
        Accelerate function execution using optimal backend.

        Args:
            func: Function to accelerate
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if self.enable_profiling:
            start_time = time.time()

        # Determine acceleration method
        if hasattr(func, '__name__'):
            func_name = func.__name__

            # Check optimization cache
            cache_key = f"{func_name}_{str(args)[:50]}"
            if cache_key in self.optimization_cache:
                result = self.optimization_cache[cache_key]
                if self.enable_profiling:
                    self.profile_data[func_name] = time.time() - start_time
                return result

        # Execute with appropriate backend
        if 'numba' in str(func):
            # Already Numba-compiled
            result = func(*args, **kwargs)
        elif 'cuda' in self._device and hasattr(func, 'cuda'):
            # CUDA acceleration
            result = func.cuda(*args, **kwargs)
        else:
            # Default execution
            result = func(*args, **kwargs)

        # Cache result if appropriate
        if hasattr(func, '__name__') and len(str(result)) < 10000:
            self.optimization_cache[cache_key] = result

        if self.enable_profiling:
            execution_time = time.time() - start_time
            if hasattr(func, '__name__'):
                self.profile_data[func.__name__] = execution_time

        return result

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        mem_stats = {
            'system_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'system_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'system_used_percent': psutil.virtual_memory().percent,
            'process_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
        }

        if 'cuda' in self._device:
            try:
                import torch
                if torch.cuda.is_available():
                    mem_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                    mem_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                    mem_stats['gpu_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except:
                pass

        return mem_stats

    def optimize_batch_size(self, base_size: int, memory_per_item: float) -> int:
        """
        Optimize batch size based on available memory.

        Args:
            base_size: Base batch size
            memory_per_item: Memory required per batch item (MB)

        Returns:
            Optimized batch size
        """
        mem_stats = self.get_memory_usage()

        if 'gpu' in self._device_type and 'gpu_total_mb' in mem_stats:
            available = mem_stats['gpu_total_mb'] - mem_stats.get('gpu_allocated_mb', 0)
            target_usage = 0.8  # Use 80% of available GPU memory
        else:
            available = mem_stats['system_available_mb']
            target_usage = 0.5  # Use 50% of available system memory

        max_batch_size = int((available * target_usage) / memory_per_item)

        return min(max(1, max_batch_size), base_size)

    def profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.enable_profiling:
            return {'profiling_disabled': True}

        summary = {
            'total_functions_profiled': len(self.profile_data),
            'total_execution_time': sum(self.profile_data.values()),
            'function_times': self.profile_data.copy(),
            'slowest_function': max(self.profile_data.items(),
                                   key=lambda x: x[1])[0] if self.profile_data else None,
            'cache_hits': len(self.optimization_cache)
        }

        return summary

    def cleanup(self):
        """Clean up resources."""
        # Clear memory pools
        self.memory_pools.clear()

        # Clear caches
        self.optimization_cache.clear()

        # GPU cleanup
        if 'cuda' in self._device:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

        logger.info("Hardware optimizer cleanup complete")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device': self._device,
            'type': self._device_type,
            'force_cpu': self.force_cpu,
            'profiling_enabled': self.enable_profiling
        }
