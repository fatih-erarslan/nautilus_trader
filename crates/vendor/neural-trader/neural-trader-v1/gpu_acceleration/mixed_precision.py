"""
Mixed Precision Training and Inference System
Optimized FP16/BF16 support for maximum GPU performance on A100/V100/RTX series.
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from contextlib import contextmanager
import time

from .flyio_gpu_config import PrecisionMode, get_gpu_config_manager

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Loss scaling strategies for mixed precision."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    precision_mode: PrecisionMode = PrecisionMode.MIXED
    loss_scale: float = 65536.0
    scaling_strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    min_loss_scale: float = 1.0
    max_loss_scale: float = 2**16
    scale_factor: float = 2.0
    scale_window: int = 2000
    growth_interval: int = 1000
    
    # Tensor Core optimization
    enable_tensor_cores: bool = True
    tensor_core_threshold: int = 8  # Minimum dimension for Tensor Core usage
    
    # Memory optimization
    fp16_storage: bool = True
    fp32_master_weights: bool = True
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Performance monitoring
    track_overflow: bool = True
    profile_memory: bool = False


class MixedPrecisionManager:
    """Manages mixed precision training and inference operations."""
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        """Initialize mixed precision manager."""
        self.config = config or MixedPrecisionConfig()
        self.gpu_config = get_gpu_config_manager().config
        
        # Initialize based on GPU capabilities
        self._initialize_precision_support()
        
        # Loss scaling state
        self.current_loss_scale = self.config.loss_scale
        self.scale_growth_tracker = 0
        self.overflow_tracker = 0
        
        # Performance tracking
        self.performance_stats = {
            'operations_count': 0,
            'overflow_count': 0,
            'scale_adjustments': 0,
            'tensor_core_usage': 0,
            'memory_savings_gb': 0.0
        }
        
    def _initialize_precision_support(self):
        """Initialize precision support based on GPU capabilities."""
        compute_capability = self.gpu_config.compute_capability
        
        # Check FP16 support (requires compute capability 5.3+)
        self.fp16_supported = compute_capability >= (5, 3)
        
        # Check BF16 support (requires compute capability 8.0+, A100)
        self.bf16_supported = compute_capability >= (8, 0)
        
        # Check Tensor Core support
        # Tensor Cores: 7.0+ for FP16, 8.0+ for BF16, 8.6+ for FP16/INT8/INT4
        self.tensor_cores_fp16 = compute_capability >= (7, 0)
        self.tensor_cores_bf16 = compute_capability >= (8, 0)
        
        # Adjust config based on capabilities
        if self.config.precision_mode == PrecisionMode.BF16 and not self.bf16_supported:
            logger.warning("BF16 not supported, falling back to FP16")
            self.config.precision_mode = PrecisionMode.FP16
            
        if self.config.precision_mode == PrecisionMode.FP16 and not self.fp16_supported:
            logger.warning("FP16 not supported, falling back to FP32")
            self.config.precision_mode = PrecisionMode.FP32
            
        if not self.tensor_cores_fp16:
            self.config.enable_tensor_cores = False
            logger.warning("Tensor Cores not supported on this GPU")
            
        logger.info(f"Mixed precision initialized: {self.config.precision_mode.value}")
        logger.info(f"Tensor Cores enabled: {self.config.enable_tensor_cores}")
        
    def get_compute_dtype(self) -> cp.dtype:
        """Get the compute dtype based on precision mode."""
        if self.config.precision_mode == PrecisionMode.FP16:
            return cp.float16
        elif self.config.precision_mode == PrecisionMode.BF16:
            return cp.float16  # CuPy doesn't have native BF16, use custom implementation
        elif self.config.precision_mode == PrecisionMode.MIXED:
            return cp.float16  # Mixed precision uses FP16 for forward pass
        else:
            return cp.float32
            
    def get_storage_dtype(self) -> cp.dtype:
        """Get the storage dtype for parameters."""
        if self.config.fp16_storage and self.config.precision_mode != PrecisionMode.FP32:
            return cp.float16
        return cp.float32
        
    def convert_to_compute_precision(self, tensor: cp.ndarray) -> cp.ndarray:
        """Convert tensor to compute precision."""
        target_dtype = self.get_compute_dtype()
        if tensor.dtype != target_dtype:
            return tensor.astype(target_dtype)
        return tensor
        
    def convert_to_storage_precision(self, tensor: cp.ndarray) -> cp.ndarray:
        """Convert tensor to storage precision."""
        target_dtype = self.get_storage_dtype()
        if tensor.dtype != target_dtype:
            return tensor.astype(target_dtype)
        return tensor
        
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        old_default_dtype = cp.float32
        
        try:
            if self.config.precision_mode in [PrecisionMode.FP16, PrecisionMode.MIXED]:
                # Set default dtype to FP16 for operations
                cp.seterr(over='ignore', under='ignore')  # Ignore underflow/overflow warnings
                
            yield self
            
        finally:
            # Restore original settings
            cp.seterr(over='warn', under='warn')
            
    def scale_loss(self, loss: cp.ndarray) -> cp.ndarray:
        """Scale loss for mixed precision training."""
        if self.config.precision_mode == PrecisionMode.FP32:
            return loss
            
        return loss * self.current_loss_scale
        
    def unscale_gradients(self, gradients: List[cp.ndarray]) -> Tuple[List[cp.ndarray], bool]:
        """Unscale gradients and check for overflow."""
        if self.config.precision_mode == PrecisionMode.FP32:
            return gradients, False
            
        # Check for overflow
        has_overflow = False
        for grad in gradients:
            if grad is not None:
                if cp.any(cp.isinf(grad)) or cp.any(cp.isnan(grad)):
                    has_overflow = True
                    break
                    
        if has_overflow:
            self.overflow_tracker += 1
            self.performance_stats['overflow_count'] += 1
            logger.warning(f"Gradient overflow detected, scale: {self.current_loss_scale}")
            
            # Don't update weights, just adjust scale
            self._adjust_loss_scale(has_overflow)
            return gradients, True
            
        # Unscale gradients
        unscaled_gradients = []
        for grad in gradients:
            if grad is not None:
                unscaled_grad = grad / self.current_loss_scale
                unscaled_gradients.append(unscaled_grad)
            else:
                unscaled_gradients.append(None)
                
        # Adjust loss scale
        self._adjust_loss_scale(has_overflow)
        
        return unscaled_gradients, False
        
    def _adjust_loss_scale(self, has_overflow: bool):
        """Adjust loss scale based on overflow detection."""
        if self.config.scaling_strategy == ScalingStrategy.FIXED:
            return
            
        if has_overflow:
            # Reduce scale on overflow
            self.current_loss_scale = max(
                self.current_loss_scale / self.config.scale_factor,
                self.config.min_loss_scale
            )
            self.scale_growth_tracker = 0
            self.performance_stats['scale_adjustments'] += 1
            logger.debug(f"Loss scale reduced to {self.current_loss_scale}")
            
        else:
            # Increase scale if no overflow for a while
            self.scale_growth_tracker += 1
            
            if self.scale_growth_tracker >= self.config.growth_interval:
                self.current_loss_scale = min(
                    self.current_loss_scale * self.config.scale_factor,
                    self.config.max_loss_scale
                )
                self.scale_growth_tracker = 0
                self.performance_stats['scale_adjustments'] += 1
                logger.debug(f"Loss scale increased to {self.current_loss_scale}")
                
    def clip_gradients(self, gradients: List[cp.ndarray]) -> List[cp.ndarray]:
        """Clip gradients to prevent exploding gradients."""
        if not self.config.gradient_clipping:
            return gradients
            
        # Calculate global norm
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += cp.sum(grad * grad)
                
        total_norm = cp.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.config.max_grad_norm:
            clip_coef = self.config.max_grad_norm / (total_norm + 1e-8)
            clipped_gradients = []
            for grad in gradients:
                if grad is not None:
                    clipped_gradients.append(grad * clip_coef)
                else:
                    clipped_gradients.append(None)
            return clipped_gradients
            
        return gradients
        
    def optimize_tensor_for_tensor_cores(self, tensor: cp.ndarray, 
                                       min_dim: int = None) -> cp.ndarray:
        """Optimize tensor dimensions for Tensor Core usage."""
        if not self.config.enable_tensor_cores:
            return tensor
            
        min_dim = min_dim or self.config.tensor_core_threshold
        
        # Tensor Cores work best with dimensions that are multiples of 8 (FP16) or 16 (INT8)
        optimal_multiple = 8 if self.config.precision_mode in [PrecisionMode.FP16, PrecisionMode.BF16] else 16
        
        # Check if tensor dimensions are already optimal
        if len(tensor.shape) >= 2:
            h, w = tensor.shape[-2], tensor.shape[-1]
            
            if h >= min_dim and w >= min_dim:
                # Pad to optimal dimensions if needed
                pad_h = (optimal_multiple - h % optimal_multiple) % optimal_multiple
                pad_w = (optimal_multiple - w % optimal_multiple) % optimal_multiple
                
                if pad_h > 0 or pad_w > 0:
                    # Pad the tensor
                    pad_width = [(0, 0)] * (len(tensor.shape) - 2) + [(0, pad_h), (0, pad_w)]
                    tensor = cp.pad(tensor, pad_width, mode='constant', constant_values=0)
                    self.performance_stats['tensor_core_usage'] += 1
                    
        return tensor
        
    def matmul_mixed_precision(self, a: cp.ndarray, b: cp.ndarray, 
                             use_tensor_cores: bool = True) -> cp.ndarray:
        """Perform matrix multiplication with mixed precision optimization."""
        # Convert to compute precision
        a_compute = self.convert_to_compute_precision(a)
        b_compute = self.convert_to_compute_precision(b)
        
        # Optimize for Tensor Cores if enabled
        if self.config.enable_tensor_cores and use_tensor_cores:
            a_compute = self.optimize_tensor_for_tensor_cores(a_compute)
            b_compute = self.optimize_tensor_for_tensor_cores(b_compute)
            
        # Perform computation
        with self.autocast():
            result = cp.matmul(a_compute, b_compute)
            
        self.performance_stats['operations_count'] += 1
        
        return result
        
    def conv2d_mixed_precision(self, input_tensor: cp.ndarray, weight: cp.ndarray,
                             bias: Optional[cp.ndarray] = None,
                             stride: int = 1, padding: int = 0) -> cp.ndarray:
        """Perform 2D convolution with mixed precision optimization."""
        # Convert to compute precision
        input_compute = self.convert_to_compute_precision(input_tensor)
        weight_compute = self.convert_to_compute_precision(weight)
        
        # Use CuPy's convolution (simplified implementation)
        with self.autocast():
            # This is a simplified conv2d - in practice you'd use cuDNN
            result = self._simple_conv2d(input_compute, weight_compute, stride, padding)
            
            if bias is not None:
                bias_compute = self.convert_to_compute_precision(bias)
                result = result + bias_compute
                
        self.performance_stats['operations_count'] += 1
        
        return result
        
    def _simple_conv2d(self, input_tensor: cp.ndarray, weight: cp.ndarray,
                      stride: int, padding: int) -> cp.ndarray:
        """Simplified 2D convolution implementation."""
        # This is a basic implementation - in practice use cuDNN
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1
        
        # Simple convolution using matrix multiplication
        # This is not optimized - use cuDNN in production
        output = cp.zeros((batch_size, out_channels, out_height, out_width), 
                         dtype=input_tensor.dtype)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * stride - padding
                        w_start = ow * stride - padding
                        h_end = h_start + kernel_height
                        w_end = w_start + kernel_width
                        
                        if h_start >= 0 and w_start >= 0 and h_end <= in_height and w_end <= in_width:
                            region = input_tensor[b, :, h_start:h_end, w_start:w_end]
                            output[b, oc, oh, ow] = cp.sum(region * weight[oc])
                            
        return output
        
    def calculate_memory_savings(self, tensor_sizes: List[Tuple[int, ...]], 
                               original_dtype: cp.dtype = cp.float32) -> float:
        """Calculate memory savings from mixed precision."""
        if self.config.precision_mode == PrecisionMode.FP32:
            return 0.0
            
        storage_dtype = self.get_storage_dtype()
        
        original_size = 0
        compressed_size = 0
        
        for size in tensor_sizes:
            elements = np.prod(size)
            original_size += elements * original_dtype().itemsize
            compressed_size += elements * storage_dtype().itemsize
            
        savings_bytes = original_size - compressed_size
        savings_gb = savings_bytes / (1024**3)
        
        self.performance_stats['memory_savings_gb'] = savings_gb
        
        return savings_gb
        
    def benchmark_mixed_precision(self, matrix_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Benchmark mixed precision performance."""
        if matrix_sizes is None:
            matrix_sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
            
        results = {}
        
        for size in matrix_sizes:
            h, w = size
            
            # Generate test matrices
            a_fp32 = cp.random.rand(h, w).astype(cp.float32)
            b_fp32 = cp.random.rand(w, h).astype(cp.float32)
            
            # Benchmark FP32
            start_time = time.time()
            for _ in range(10):
                result_fp32 = cp.matmul(a_fp32, b_fp32)
            cp.cuda.Stream.null.synchronize()
            fp32_time = time.time() - start_time
            
            # Benchmark mixed precision
            start_time = time.time()
            for _ in range(10):
                result_mixed = self.matmul_mixed_precision(a_fp32, b_fp32)
            cp.cuda.Stream.null.synchronize()
            mixed_time = time.time() - start_time
            
            # Calculate speedup
            speedup = fp32_time / mixed_time if mixed_time > 0 else 0
            
            # Memory comparison
            memory_savings = self.calculate_memory_savings([(h, w), (w, h)])
            
            results[f"{h}x{w}"] = {
                'fp32_time_sec': fp32_time,
                'mixed_time_sec': mixed_time,
                'speedup': speedup,
                'memory_savings_gb': memory_savings,
                'precision_mode': self.config.precision_mode.value
            }
            
        return results
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        stats.update({
            'current_loss_scale': self.current_loss_scale,
            'overflow_rate': self.overflow_tracker / max(self.performance_stats['operations_count'], 1),
            'precision_mode': self.config.precision_mode.value,
            'tensor_cores_enabled': self.config.enable_tensor_cores,
            'fp16_supported': self.fp16_supported,
            'bf16_supported': self.bf16_supported
        })
        return stats
        
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'operations_count': 0,
            'overflow_count': 0,
            'scale_adjustments': 0,
            'tensor_core_usage': 0,
            'memory_savings_gb': 0.0
        }
        self.overflow_tracker = 0
        self.scale_growth_tracker = 0


class AutoMixedPrecisionWrapper:
    """Automatic mixed precision wrapper for existing models."""
    
    def __init__(self, model_func, precision_manager: MixedPrecisionManager):
        """Initialize AMP wrapper."""
        self.model_func = model_func
        self.precision_manager = precision_manager
        
    def __call__(self, *args, **kwargs):
        """Forward pass with automatic mixed precision."""
        # Convert inputs to compute precision
        converted_args = []
        for arg in args:
            if isinstance(arg, cp.ndarray):
                converted_args.append(self.precision_manager.convert_to_compute_precision(arg))
            else:
                converted_args.append(arg)
                
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, cp.ndarray):
                converted_kwargs[key] = self.precision_manager.convert_to_compute_precision(value)
            else:
                converted_kwargs[key] = value
                
        # Forward pass with autocast
        with self.precision_manager.autocast():
            output = self.model_func(*converted_args, **converted_kwargs)
            
        return output


def create_mixed_precision_manager(precision_mode: Optional[str] = None) -> MixedPrecisionManager:
    """Create mixed precision manager with optimal settings."""
    if precision_mode:
        config = MixedPrecisionConfig(precision_mode=PrecisionMode(precision_mode))
    else:
        # Use GPU config to determine optimal precision
        gpu_config = get_gpu_config_manager().config
        config = MixedPrecisionConfig(precision_mode=gpu_config.precision_mode)
        
    return MixedPrecisionManager(config)


def amp_wrapper(precision_mode: Optional[str] = None):
    """Decorator for automatic mixed precision."""
    def decorator(func):
        precision_manager = create_mixed_precision_manager(precision_mode)
        return AutoMixedPrecisionWrapper(func, precision_manager)
    return decorator


if __name__ == "__main__":
    # Test mixed precision system
    logger.info("Testing mixed precision system...")
    
    # Create manager
    mp_manager = create_mixed_precision_manager()
    
    # Test basic operations
    a = cp.random.rand(1000, 1000).astype(cp.float32)
    b = cp.random.rand(1000, 1000).astype(cp.float32)
    
    # Test matrix multiplication
    result = mp_manager.matmul_mixed_precision(a, b)
    logger.info(f"Matrix multiplication completed with shape: {result.shape}")
    
    # Benchmark
    benchmark_results = mp_manager.benchmark_mixed_precision()
    logger.info(f"Benchmark results: {benchmark_results}")
    
    # Performance stats
    stats = mp_manager.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    print("Mixed precision system tested successfully!")