"""
Automatic GPU/CPU Fallback System
Intelligent fallback mechanism for seamless operation when GPU resources are unavailable.
"""

import cupy as cp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Type
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
import functools
import traceback
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from .flyio_gpu_config import get_gpu_config_manager
from .gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for operations."""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    AUTO = "auto"
    HYBRID = "hybrid"


class FallbackReason(Enum):
    """Reasons for falling back to CPU."""
    GPU_UNAVAILABLE = "gpu_unavailable"
    MEMORY_EXHAUSTED = "memory_exhausted"
    CUDA_ERROR = "cuda_error"
    PERFORMANCE_DEGRADED = "performance_degraded"
    THERMAL_THROTTLING = "thermal_throttling"
    POWER_LIMIT = "power_limit"
    USER_OVERRIDE = "user_override"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    default_mode: ExecutionMode = ExecutionMode.AUTO
    
    # GPU health thresholds
    max_gpu_utilization: float = 95.0
    max_memory_utilization: float = 90.0
    max_temperature: float = 80.0
    max_error_rate: float = 0.05
    
    # Performance thresholds
    min_gpu_speedup: float = 2.0  # Minimum speedup to justify GPU usage
    performance_window: int = 10  # Number of operations to evaluate
    
    # Retry behavior
    gpu_retry_interval: float = 30.0  # Seconds to wait before retrying GPU
    max_retry_attempts: int = 3
    
    # CPU optimization
    cpu_thread_count: Optional[int] = None  # None = auto-detect
    enable_multiprocessing: bool = True
    
    # Monitoring
    enable_performance_tracking: bool = True
    log_fallback_events: bool = True


@dataclass
class OperationMetrics:
    """Metrics for operation execution."""
    execution_time: float
    memory_used: float
    success: bool
    error_message: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.AUTO


class CPUImplementations:
    """CPU implementations of GPU operations."""
    
    def __init__(self, thread_count: Optional[int] = None):
        """Initialize CPU implementations."""
        self.thread_count = thread_count or mp.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.thread_count)
        
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """CPU matrix multiplication."""
        return np.matmul(a, b)
        
    def conv2d(self, input_array: np.ndarray, kernel: np.ndarray,
               stride: int = 1, padding: int = 0) -> np.ndarray:
        """CPU 2D convolution."""
        # Simplified implementation - in practice use scipy or similar
        batch_size, in_channels, in_height, in_width = input_array.shape
        out_channels, _, kernel_height, kernel_width = kernel.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1
        
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # Apply padding
        if padding > 0:
            input_padded = np.pad(input_array, 
                                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                                mode='constant')
        else:
            input_padded = input_array
            
        # Convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_height
                        w_end = w_start + kernel_width
                        
                        region = input_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = np.sum(region * kernel[oc])
                        
        return output
        
    def moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """CPU moving average calculation."""
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values
        
    def exponential_moving_average(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """CPU exponential moving average."""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def rsi(self, prices: np.ndarray, window_size: int = 14) -> np.ndarray:
        """CPU RSI calculation."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=window_size, min_periods=1).mean()
        avg_losses = pd.Series(losses).rolling(window=window_size, min_periods=1).mean()
        
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend first value to match input length
        return np.concatenate([[prices[0]], rsi.values])
        
    def bollinger_bands(self, prices: np.ndarray, window_size: int = 20, 
                       num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU Bollinger Bands calculation."""
        sma = self.moving_average(prices, window_size)
        rolling_std = pd.Series(prices).rolling(window=window_size, min_periods=1).std().values
        
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        
        return upper_band, lower_band, sma
        
    def parallel_operation(self, operation: Callable, data: np.ndarray, 
                          chunk_size: Optional[int] = None) -> np.ndarray:
        """Execute operation in parallel on CPU."""
        if chunk_size is None:
            chunk_size = max(1, len(data) // self.thread_count)
            
        # Split data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        futures = [self.executor.submit(operation, chunk) for chunk in chunks]
        results = [future.result() for future in futures]
        
        # Concatenate results
        return np.concatenate(results)


class GPUHealthChecker:
    """Monitors GPU health and determines fallback necessity."""
    
    def __init__(self, config: FallbackConfig):
        """Initialize GPU health checker."""
        self.config = config
        self.gpu_monitor = get_gpu_monitor()
        self.last_check_time = 0
        self.check_interval = 5.0  # Check every 5 seconds
        self.health_history = []
        
    def is_gpu_healthy(self) -> Tuple[bool, Optional[FallbackReason]]:
        """Check if GPU is healthy for operation."""
        current_time = time.time()
        
        # Rate limit health checks
        if current_time - self.last_check_time < self.check_interval:
            return True, None
            
        self.last_check_time = current_time
        
        try:
            # Check if GPU is available
            if not cp.cuda.is_available():
                return False, FallbackReason.GPU_UNAVAILABLE
                
            # Get current metrics
            metrics = self.gpu_monitor.get_current_metrics()
            if not metrics:
                return True, None  # Assume healthy if no metrics
                
            # Check GPU utilization
            if metrics.gpu_utilization > self.config.max_gpu_utilization:
                logger.warning(f"GPU utilization too high: {metrics.gpu_utilization}%")
                return False, FallbackReason.PERFORMANCE_DEGRADED
                
            # Check memory utilization  
            if metrics.memory_utilization > self.config.max_memory_utilization:
                logger.warning(f"GPU memory utilization too high: {metrics.memory_utilization}%")
                return False, FallbackReason.MEMORY_EXHAUSTED
                
            # Check temperature
            if metrics.temperature_c > self.config.max_temperature:
                logger.warning(f"GPU temperature too high: {metrics.temperature_c}°C")
                return False, FallbackReason.THERMAL_THROTTLING
                
            # Check error rate
            if hasattr(metrics, 'error_rate'):
                error_rate = getattr(metrics, 'error_rate', 0.0)
                if error_rate > self.config.max_error_rate:
                    logger.warning(f"GPU error rate too high: {error_rate}")
                    return False, FallbackReason.CUDA_ERROR
                    
            return True, None
            
        except Exception as e:
            logger.error(f"GPU health check failed: {str(e)}")
            return False, FallbackReason.CUDA_ERROR


class PerformanceTracker:
    """Tracks performance to determine optimal execution mode."""
    
    def __init__(self, config: FallbackConfig):
        """Initialize performance tracker."""
        self.config = config
        self.gpu_metrics = []
        self.cpu_metrics = []
        self.operation_history = {}
        
    def record_operation(self, operation_name: str, metrics: OperationMetrics):
        """Record operation metrics."""
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = {'gpu': [], 'cpu': []}
            
        if metrics.execution_mode in [ExecutionMode.GPU_ONLY, ExecutionMode.AUTO]:
            self.operation_history[operation_name]['gpu'].append(metrics)
        else:
            self.operation_history[operation_name]['cpu'].append(metrics)
            
        # Keep only recent metrics
        for mode in self.operation_history[operation_name]:
            if len(self.operation_history[operation_name][mode]) > self.config.performance_window:
                self.operation_history[operation_name][mode].pop(0)
                
    def should_use_gpu(self, operation_name: str) -> bool:
        """Determine if GPU should be used for operation."""
        if operation_name not in self.operation_history:
            return True  # Default to GPU for new operations
            
        gpu_metrics = self.operation_history[operation_name]['gpu']
        cpu_metrics = self.operation_history[operation_name]['cpu']
        
        if not gpu_metrics or not cpu_metrics:
            return True  # Need both for comparison
            
        # Calculate average performance
        avg_gpu_time = np.mean([m.execution_time for m in gpu_metrics[-5:]])
        avg_cpu_time = np.mean([m.execution_time for m in cpu_metrics[-5:]])
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        return speedup >= self.config.min_gpu_speedup
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation_name, metrics in self.operation_history.items():
            gpu_times = [m.execution_time for m in metrics['gpu']]
            cpu_times = [m.execution_time for m in metrics['cpu']]
            
            stats[operation_name] = {
                'gpu_avg_time': np.mean(gpu_times) if gpu_times else 0,
                'cpu_avg_time': np.mean(cpu_times) if cpu_times else 0,
                'gpu_success_rate': np.mean([m.success for m in metrics['gpu']]) if metrics['gpu'] else 0,
                'cpu_success_rate': np.mean([m.success for m in metrics['cpu']]) if metrics['cpu'] else 0,
                'gpu_count': len(gpu_times),
                'cpu_count': len(cpu_times)
            }
            
            if gpu_times and cpu_times:
                stats[operation_name]['speedup'] = np.mean(cpu_times) / np.mean(gpu_times)
                
        return stats


class AutoFallbackManager:
    """Manages automatic GPU/CPU fallback for operations."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        """Initialize auto fallback manager."""
        self.config = config or FallbackConfig()
        self.health_checker = GPUHealthChecker(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.cpu_impl = CPUImplementations(self.config.cpu_thread_count)
        
        # Fallback state
        self.current_mode = self.config.default_mode
        self.fallback_until = 0  # Timestamp until which to stay in fallback
        self.retry_count = 0
        self.lock = threading.Lock()
        
        # Operation mapping
        self.operation_map = {
            'matmul': {'gpu': self._gpu_matmul, 'cpu': self.cpu_impl.matmul},
            'conv2d': {'gpu': self._gpu_conv2d, 'cpu': self.cpu_impl.conv2d},
            'moving_average': {'gpu': self._gpu_moving_average, 'cpu': self.cpu_impl.moving_average},
            'ema': {'gpu': self._gpu_ema, 'cpu': self.cpu_impl.exponential_moving_average},
            'rsi': {'gpu': self._gpu_rsi, 'cpu': self.cpu_impl.rsi},
            'bollinger_bands': {'gpu': self._gpu_bollinger_bands, 'cpu': self.cpu_impl.bollinger_bands}
        }
        
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation with automatic fallback."""
        if operation_name not in self.operation_map:
            raise ValueError(f"Unknown operation: {operation_name}")
            
        with self.lock:
            # Determine execution mode
            execution_mode = self._determine_execution_mode(operation_name)
            
            # Execute operation
            start_time = time.time()
            try:
                if execution_mode in [ExecutionMode.GPU_ONLY, ExecutionMode.AUTO]:
                    result = self._execute_gpu_operation(operation_name, *args, **kwargs)
                    success = True
                    error_msg = None
                else:
                    result = self._execute_cpu_operation(operation_name, *args, **kwargs)
                    success = True
                    error_msg = None
                    
            except Exception as e:
                error_msg = str(e)
                if self.config.log_fallback_events:
                    logger.warning(f"Operation {operation_name} failed in {execution_mode.value} mode: {error_msg}")
                
                # Try fallback if GPU operation failed
                if execution_mode in [ExecutionMode.GPU_ONLY, ExecutionMode.AUTO]:
                    try:
                        result = self._execute_cpu_operation(operation_name, *args, **kwargs)
                        execution_mode = ExecutionMode.CPU_ONLY
                        success = True
                        error_msg = None
                        self._trigger_fallback(FallbackReason.CUDA_ERROR)
                    except Exception as fallback_error:
                        success = False
                        error_msg = f"GPU: {error_msg}, CPU: {str(fallback_error)}"
                        raise fallback_error
                else:
                    success = False
                    raise
                    
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record metrics
            metrics = OperationMetrics(
                execution_time=execution_time,
                memory_used=0.0,  # Would need actual memory tracking
                success=success,
                error_message=error_msg,
                execution_mode=execution_mode
            )
            
            if self.config.enable_performance_tracking:
                self.performance_tracker.record_operation(operation_name, metrics)
                
            return result
            
    def _determine_execution_mode(self, operation_name: str) -> ExecutionMode:
        """Determine the appropriate execution mode."""
        current_time = time.time()
        
        # Check if we're in forced fallback mode
        if current_time < self.fallback_until:
            return ExecutionMode.CPU_ONLY
            
        # Check configured mode
        if self.config.default_mode == ExecutionMode.CPU_ONLY:
            return ExecutionMode.CPU_ONLY
        elif self.config.default_mode == ExecutionMode.GPU_ONLY:
            return ExecutionMode.GPU_ONLY
            
        # Auto mode - check GPU health and performance
        gpu_healthy, fallback_reason = self.health_checker.is_gpu_healthy()
        
        if not gpu_healthy:
            if self.config.log_fallback_events:
                logger.info(f"Falling back to CPU due to: {fallback_reason.value}")
            self._trigger_fallback(fallback_reason)
            return ExecutionMode.CPU_ONLY
            
        # Check performance history
        if (self.config.enable_performance_tracking and 
            not self.performance_tracker.should_use_gpu(operation_name)):
            return ExecutionMode.CPU_ONLY
            
        return ExecutionMode.GPU_ONLY
        
    def _trigger_fallback(self, reason: FallbackReason):
        """Trigger fallback to CPU mode."""
        self.fallback_until = time.time() + self.config.gpu_retry_interval
        self.retry_count += 1
        
        if self.config.log_fallback_events:
            logger.warning(f"GPU fallback triggered: {reason.value} (retry {self.retry_count})")
            
    def _execute_gpu_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation on GPU."""
        gpu_func = self.operation_map[operation_name]['gpu']
        
        # Convert inputs to GPU arrays if needed
        gpu_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                gpu_args.append(cp.asarray(arg))
            else:
                gpu_args.append(arg)
                
        result = gpu_func(*gpu_args, **kwargs)
        
        # Convert result back to CPU if needed
        if isinstance(result, cp.ndarray):
            return cp.asnumpy(result)
        elif isinstance(result, tuple):
            return tuple(cp.asnumpy(r) if isinstance(r, cp.ndarray) else r for r in result)
        else:
            return result
            
    def _execute_cpu_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation on CPU."""
        cpu_func = self.operation_map[operation_name]['cpu']
        
        # Convert inputs to CPU arrays if needed
        cpu_args = []
        for arg in args:
            if isinstance(arg, cp.ndarray):
                cpu_args.append(cp.asnumpy(arg))
            else:
                cpu_args.append(arg)
                
        return cpu_func(*cpu_args, **kwargs)
        
    # GPU operation implementations
    def _gpu_matmul(self, a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
        """GPU matrix multiplication."""
        return cp.matmul(a, b)
        
    def _gpu_conv2d(self, input_array: cp.ndarray, kernel: cp.ndarray,
                   stride: int = 1, padding: int = 0) -> cp.ndarray:
        """GPU 2D convolution (simplified)."""
        # This is a placeholder - would use cuDNN in practice
        return self.cpu_impl.conv2d(cp.asnumpy(input_array), cp.asnumpy(kernel), stride, padding)
        
    def _gpu_moving_average(self, data: cp.ndarray, window_size: int) -> cp.ndarray:
        """GPU moving average."""
        # Simplified implementation using CuPy
        result = cp.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            result[i] = cp.mean(data[start_idx:i+1])
        return result
        
    def _gpu_ema(self, data: cp.ndarray, alpha: float) -> cp.ndarray:
        """GPU exponential moving average."""
        ema = cp.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
        
    def _gpu_rsi(self, prices: cp.ndarray, window_size: int = 14) -> cp.ndarray:
        """GPU RSI calculation."""
        # Simplified implementation
        deltas = cp.diff(prices)
        gains = cp.where(deltas > 0, deltas, 0)
        losses = cp.where(deltas < 0, -deltas, 0)
        
        # Moving averages (simplified)
        avg_gains = cp.zeros(len(gains))
        avg_losses = cp.zeros(len(losses))
        
        for i in range(len(gains)):
            start_idx = max(0, i - window_size + 1)
            avg_gains[i] = cp.mean(gains[start_idx:i+1])
            avg_losses[i] = cp.mean(losses[start_idx:i+1])
            
        rs = avg_gains / cp.where(avg_losses == 0, 1e-8, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        return cp.concatenate([[prices[0]], rsi])
        
    def _gpu_bollinger_bands(self, prices: cp.ndarray, window_size: int = 20,
                           num_std: float = 2.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """GPU Bollinger Bands calculation."""
        sma = self._gpu_moving_average(prices, window_size)
        
        # Calculate rolling standard deviation
        rolling_std = cp.zeros_like(prices)
        for i in range(len(prices)):
            start_idx = max(0, i - window_size + 1)
            rolling_std[i] = cp.std(prices[start_idx:i+1])
            
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        
        return upper_band, lower_band, sma
        
    def get_status(self) -> Dict[str, Any]:
        """Get fallback manager status."""
        gpu_healthy, fallback_reason = self.health_checker.is_gpu_healthy()
        
        return {
            'current_mode': self.current_mode.value,
            'gpu_healthy': gpu_healthy,
            'fallback_reason': fallback_reason.value if fallback_reason else None,
            'fallback_until': self.fallback_until,
            'retry_count': self.retry_count,
            'performance_stats': self.performance_tracker.get_performance_stats()
        }
        
    def force_mode(self, mode: ExecutionMode, duration_seconds: Optional[float] = None):
        """Force specific execution mode."""
        with self.lock:
            self.current_mode = mode
            if duration_seconds:
                if mode == ExecutionMode.CPU_ONLY:
                    self.fallback_until = time.time() + duration_seconds
                    
            logger.info(f"Forced execution mode to {mode.value}")


# Global fallback manager
_global_fallback_manager = None


def get_fallback_manager() -> AutoFallbackManager:
    """Get the global fallback manager."""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = AutoFallbackManager()
    return _global_fallback_manager


def auto_fallback(operation_name: str):
    """Decorator for automatic GPU/CPU fallback."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_fallback_manager()
            return manager.execute_operation(operation_name, *args, **kwargs)
        return wrapper
    return decorator


# Convenience functions
def matmul_auto(a: Union[np.ndarray, cp.ndarray], 
               b: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
    """Matrix multiplication with automatic fallback."""
    manager = get_fallback_manager()
    return manager.execute_operation('matmul', a, b)


def moving_average_auto(data: Union[np.ndarray, cp.ndarray], 
                       window_size: int) -> np.ndarray:
    """Moving average with automatic fallback."""
    manager = get_fallback_manager()
    return manager.execute_operation('moving_average', data, window_size)


def rsi_auto(prices: Union[np.ndarray, cp.ndarray], 
            window_size: int = 14) -> np.ndarray:
    """RSI calculation with automatic fallback."""
    manager = get_fallback_manager()
    return manager.execute_operation('rsi', prices, window_size)


if __name__ == "__main__":
    # Test fallback system
    logger.info("Testing GPU/CPU fallback system...")
    
    # Create test data
    test_data = np.random.rand(1000).astype(np.float32)
    test_matrix_a = np.random.rand(100, 100).astype(np.float32)
    test_matrix_b = np.random.rand(100, 100).astype(np.float32)
    
    # Test operations
    manager = get_fallback_manager()
    
    # Test matrix multiplication
    result = manager.execute_operation('matmul', test_matrix_a, test_matrix_b)
    logger.info(f"Matrix multiplication result shape: {result.shape}")
    
    # Test moving average
    ma_result = manager.execute_operation('moving_average', test_data, 20)
    logger.info(f"Moving average result shape: {ma_result.shape}")
    
    # Test RSI
    rsi_result = manager.execute_operation('rsi', test_data, 14)
    logger.info(f"RSI result shape: {rsi_result.shape}")
    
    # Test convenience functions
    auto_matmul_result = matmul_auto(test_matrix_a, test_matrix_b)
    auto_ma_result = moving_average_auto(test_data, 20)
    auto_rsi_result = rsi_auto(test_data, 14)
    
    logger.info("Convenience functions tested")
    
    # Get status
    status = manager.get_status()
    logger.info(f"Fallback manager status: {status}")
    
    # Test forced CPU mode
    manager.force_mode(ExecutionMode.CPU_ONLY, 10.0)
    cpu_result = manager.execute_operation('matmul', test_matrix_a, test_matrix_b)
    logger.info("Forced CPU mode tested")
    
    print("GPU/CPU fallback system tested successfully!")