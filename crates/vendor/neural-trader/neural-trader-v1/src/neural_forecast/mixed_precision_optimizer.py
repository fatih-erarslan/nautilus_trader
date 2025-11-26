"""
Advanced Mixed Precision Optimizer for 2x Performance Boost
Implements FP16/BF16 mixed precision training with automatic loss scaling and Tensor Core optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import warnings

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Precision modes for training and inference."""
    FP32 = "fp32"           # Full precision
    FP16 = "fp16"           # Half precision
    BF16 = "bf16"           # Brain Float 16
    MIXED_FP16 = "mixed_fp16"  # Mixed precision with FP16
    MIXED_BF16 = "mixed_bf16"  # Mixed precision with BF16
    AUTO = "auto"           # Automatic selection based on hardware


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    precision_mode: PrecisionMode = PrecisionMode.AUTO
    enable_autocast: bool = True
    enable_grad_scaling: bool = True
    
    # Loss scaling parameters
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Optimization parameters
    enable_tensor_cores: bool = True
    channels_last: bool = True
    use_fused_ops: bool = True
    
    # Stability parameters
    loss_scale_window: int = 1000
    max_loss_scale: float = 2**24
    min_loss_scale: float = 1.0
    
    # Performance tuning
    cache_enabled: bool = True
    compile_autocast: bool = True


@dataclass
class PrecisionStats:
    """Statistics for mixed precision training."""
    total_steps: int = 0
    overflow_steps: int = 0
    scale_updates: int = 0
    current_scale: float = 0.0
    avg_grad_norm: float = 0.0
    memory_saved_mb: float = 0.0
    speedup_factor: float = 1.0
    tensor_core_utilization: float = 0.0


class TensorCoreOptimizer:
    """Optimizes operations for Tensor Core utilization."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.tensor_core_capable = self._check_tensor_core_capability()
        
    def _check_tensor_core_capability(self) -> bool:
        """Check if device supports Tensor Cores."""
        if self.device.type != 'cuda':
            return False
            
        try:
            props = torch.cuda.get_device_properties(self.device)
            major, minor = props.major, props.minor
            
            # Tensor Cores available on:
            # - Volta (V100): 7.0
            # - Turing (T4, RTX 20xx): 7.5
            # - Ampere (A100, RTX 30xx): 8.0+
            # - Ada Lovelace (RTX 40xx): 8.9
            # - Hopper (H100): 9.0
            
            return major >= 7 or (major == 7 and minor >= 0)
            
        except Exception:
            return False
    
    def optimize_tensor_shape(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[int, ...]:
        """Optimize tensor shape for Tensor Core operations."""
        if not self.tensor_core_capable:
            return shape
        
        # Tensor Cores work best with specific dimension multiples
        if dtype in [torch.float16, torch.bfloat16]:
            optimal_multiples = [8, 16, 32, 64]  # For FP16/BF16
        else:
            optimal_multiples = [16, 32, 64]     # For FP32
        
        optimized_shape = list(shape)
        
        # Optimize last two dimensions (matrix multiplication dimensions)
        for i in [-2, -1]:
            if i < -len(shape):
                continue
                
            dim_size = optimized_shape[i]
            
            # Find best multiple
            best_multiple = optimal_multiples[0]
            for multiple in optimal_multiples:
                if dim_size <= multiple:
                    best_multiple = multiple
                    break
            else:
                # If larger than all multiples, round up to nearest multiple
                best_multiple = ((dim_size + optimal_multiples[-1] - 1) // optimal_multiples[-1]) * optimal_multiples[-1]
            
            optimized_shape[i] = best_multiple
        
        return tuple(optimized_shape)
    
    def create_optimized_tensor(self, shape: Tuple[int, ...], 
                               dtype: torch.dtype = torch.float16,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """Create tensor optimized for Tensor Core operations."""
        device = device or self.device
        optimized_shape = self.optimize_tensor_shape(shape, dtype)
        
        tensor = torch.empty(optimized_shape, dtype=dtype, device=device)
        
        # Use channels_last memory format for conv operations
        if len(optimized_shape) == 4:  # Batch, Channel, Height, Width
            tensor = tensor.to(memory_format=torch.channels_last)
        
        return tensor


class AdvancedGradScaler(GradScaler):
    """Enhanced gradient scaler with advanced features."""
    
    def __init__(self, config: MixedPrecisionConfig):
        super().__init__(
            init_scale=config.init_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval
        )
        
        self.config = config
        self.stats = PrecisionStats()
        self.loss_history = []
        self.scale_history = []
        
        # Adaptive scaling parameters
        self.adaptive_scaling = True
        self.stability_threshold = 0.1
        self.min_stable_steps = 100
        
    def scale(self, outputs):
        """Enhanced scale function with monitoring."""
        self.stats.current_scale = self.get_scale()
        self.scale_history.append(self.stats.current_scale)
        
        # Limit history size
        if len(self.scale_history) > self.config.loss_scale_window:
            self.scale_history.pop(0)
        
        return super().scale(outputs)
    
    def step(self, optimizer, *args, **kwargs):
        """Enhanced step function with statistics tracking."""
        self.stats.total_steps += 1
        
        # Store gradient norms before step
        total_norm = 0.0
        param_count = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(dtype=torch.float32)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.stats.avg_grad_norm = (self.stats.avg_grad_norm * 0.99 + total_norm * 0.01)
        
        # Call parent step
        retval = super().step(optimizer, *args, **kwargs)
        
        # Check for overflow
        if self.get_scale() < self.scale_history[-1] if self.scale_history else False:
            self.stats.overflow_steps += 1
        
        # Adaptive scaling adjustment
        if self.adaptive_scaling:
            self._adjust_scale_adaptively()
        
        return retval
    
    def _adjust_scale_adaptively(self):
        """Adaptively adjust loss scale based on training stability."""
        if len(self.scale_history) < self.min_stable_steps:
            return
        
        recent_scales = self.scale_history[-self.min_stable_steps:]
        scale_variance = np.var(recent_scales)
        scale_mean = np.mean(recent_scales)
        
        if scale_mean > 0:
            coefficient_of_variation = np.sqrt(scale_variance) / scale_mean
            
            # If scale is very stable, try to increase it more aggressively
            if coefficient_of_variation < self.stability_threshold:
                self.set_growth_factor(min(self.config.growth_factor * 1.1, 4.0))
            else:
                # If unstable, be more conservative
                self.set_growth_factor(max(self.config.growth_factor * 0.9, 1.5))
    
    def get_stats(self) -> PrecisionStats:
        """Get comprehensive statistics."""
        self.stats.scale_updates = len(self.scale_history)
        
        if self.stats.total_steps > 0:
            overflow_rate = self.stats.overflow_steps / self.stats.total_steps
        else:
            overflow_rate = 0.0
        
        return self.stats


class MixedPrecisionOptimizer:
    """Advanced mixed precision optimizer with comprehensive features."""
    
    def __init__(self, config: MixedPrecisionConfig = None):
        self.config = config or MixedPrecisionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.scaler = AdvancedGradScaler(self.config) if self.config.enable_grad_scaling else None
        self.tensor_core_optimizer = TensorCoreOptimizer(self.device)
        
        # Performance tracking
        self.memory_baseline = 0
        self.time_baseline = 0
        self.precision_mode = self._determine_optimal_precision()
        
        # Setup device-specific optimizations
        self._setup_device_optimizations()
        
        logger.info(f"Mixed precision optimizer initialized with {self.precision_mode.value}")
    
    def _determine_optimal_precision(self) -> PrecisionMode:
        """Automatically determine optimal precision mode."""
        if self.config.precision_mode != PrecisionMode.AUTO:
            return self.config.precision_mode
        
        if self.device.type != 'cuda':
            return PrecisionMode.FP32
        
        try:
            props = torch.cuda.get_device_properties(self.device)
            major, minor = props.major, props.minor
            
            # H100, A100 with BF16 support
            if major >= 8:
                if torch.cuda.is_bf16_supported():
                    return PrecisionMode.MIXED_BF16
                else:
                    return PrecisionMode.MIXED_FP16
            
            # V100, T4, RTX series with FP16
            elif major >= 7:
                return PrecisionMode.MIXED_FP16
            
            # Older GPUs
            else:
                return PrecisionMode.FP32
                
        except Exception:
            return PrecisionMode.FP32
    
    def _setup_device_optimizations(self):
        """Setup device-specific optimizations."""
        if self.device.type == 'cuda':
            # Enable Tensor Core optimizations
            if self.config.enable_tensor_cores:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable optimized attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)  # Disable math fallback for performance
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except AttributeError:
                pass  # Not available in older PyTorch versions
    
    def create_autocast_context(self, enabled: bool = None):
        """Create autocast context with optimal settings."""
        if enabled is None:
            enabled = self.precision_mode in [PrecisionMode.MIXED_FP16, PrecisionMode.MIXED_BF16]
        
        if not enabled or self.device.type != 'cuda':
            return torch.cuda.amp.autocast(enabled=False)
        
        dtype = torch.bfloat16 if self.precision_mode == PrecisionMode.MIXED_BF16 else torch.float16
        
        return torch.cuda.amp.autocast(
            enabled=True,
            dtype=dtype,
            cache_enabled=self.config.cache_enabled
        )
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for mixed precision training."""
        # Convert to appropriate precision
        if self.precision_mode == PrecisionMode.FP16:
            model = model.half()
        elif self.precision_mode == PrecisionMode.BF16:
            model = model.to(dtype=torch.bfloat16)
        
        # Apply channels_last memory format for conv layers
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Fuse operations if possible
        if self.config.use_fused_ops:
            try:
                model = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(model)
            except Exception as e:
                logger.warning(f"Could not apply JIT optimizations: {e}")
        
        # Compile with PyTorch 2.0 if available
        if self.config.compile_autocast and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("Model compiled with PyTorch 2.0")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def optimize_tensor_for_training(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for mixed precision training."""
        # Optimize shape for Tensor Cores
        if self.tensor_core_optimizer.tensor_core_capable:
            target_dtype = torch.float16 if self.precision_mode == PrecisionMode.MIXED_FP16 else torch.bfloat16
            optimal_shape = self.tensor_core_optimizer.optimize_tensor_shape(
                tensor.shape, target_dtype
            )
            
            if optimal_shape != tensor.shape:
                # Pad tensor to optimal shape
                padding = []
                for i in range(len(tensor.shape)):
                    diff = optimal_shape[i] - tensor.shape[i]
                    padding.extend([0, diff])
                
                tensor = F.pad(tensor, padding[::-1])  # Reverse for F.pad format
        
        # Apply channels_last memory format if applicable
        if self.config.channels_last and tensor.ndim == 4:
            tensor = tensor.to(memory_format=torch.channels_last)
        
        return tensor
    
    def training_step(self, model: nn.Module, loss_fn, optimizer, inputs, targets):
        """Optimized training step with mixed precision."""
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with self.create_autocast_context():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            # Optional gradient clipping
            if hasattr(self.config, 'max_grad_norm'):
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        return loss.item(), outputs
    
    def inference_step(self, model: nn.Module, inputs: torch.Tensor):
        """Optimized inference step with mixed precision."""
        with torch.no_grad():
            with self.create_autocast_context():
                return model(inputs)
    
    def benchmark_precision_modes(self, model: nn.Module, 
                                 sample_input: torch.Tensor,
                                 num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark different precision modes."""
        results = {}
        
        original_mode = self.precision_mode
        original_model_state = model.state_dict()
        
        modes_to_test = [PrecisionMode.FP32, PrecisionMode.MIXED_FP16]
        if torch.cuda.is_bf16_supported():
            modes_to_test.append(PrecisionMode.MIXED_BF16)
        
        for mode in modes_to_test:
            logger.info(f"Benchmarking {mode.value}...")
            
            # Setup mode
            self.precision_mode = mode
            test_model = self.optimize_model(model)
            test_model.load_state_dict(original_model_state)
            test_model.eval()
            
            # Warmup
            for _ in range(10):
                with self.create_autocast_context():
                    _ = test_model(sample_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure memory
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    with self.create_autocast_context():
                        _ = test_model(sample_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            
            if self.device.type == 'cuda':
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                peak_memory_mb = 0
            
            results[mode.value] = {
                'avg_time_ms': avg_time_ms,
                'peak_memory_mb': peak_memory_mb,
                'throughput_per_sec': 1000 / avg_time_ms if avg_time_ms > 0 else 0
            }
            
            logger.info(f"{mode.value}: {avg_time_ms:.2f}ms, {peak_memory_mb:.1f}MB")
        
        # Restore original state
        self.precision_mode = original_mode
        
        # Calculate speedups and memory savings
        fp32_time = results[PrecisionMode.FP32.value]['avg_time_ms']
        fp32_memory = results[PrecisionMode.FP32.value]['peak_memory_mb']
        
        for mode_name, metrics in results.items():
            if mode_name != PrecisionMode.FP32.value:
                metrics['speedup'] = fp32_time / metrics['avg_time_ms'] if metrics['avg_time_ms'] > 0 else 1.0
                metrics['memory_saving'] = (fp32_memory - metrics['peak_memory_mb']) / fp32_memory if fp32_memory > 0 else 0.0
        
        return results
    
    def get_optimization_recommendations(self, 
                                       model: nn.Module,
                                       sample_input: torch.Tensor) -> Dict[str, Any]:
        """Get optimization recommendations based on model and hardware."""
        recommendations = {
            'current_precision': self.precision_mode.value,
            'tensor_core_capable': self.tensor_core_optimizer.tensor_core_capable,
            'recommendations': []
        }
        
        # Hardware-based recommendations
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            
            if props.major >= 8 and self.precision_mode != PrecisionMode.MIXED_BF16:
                recommendations['recommendations'].append(
                    "Consider using BF16 for better numerical stability on Ampere+ GPUs"
                )
            
            if props.major >= 7 and self.precision_mode == PrecisionMode.FP32:
                recommendations['recommendations'].append(
                    "Enable mixed precision for significant speedup on Tensor Core GPUs"
                )
        
        # Model-based recommendations
        param_count = sum(p.numel() for p in model.parameters())
        if param_count > 100_000_000:  # Large model
            recommendations['recommendations'].append(
                "Large model detected - mixed precision will provide significant memory savings"
            )
        
        # Input shape recommendations
        if self.tensor_core_optimizer.tensor_core_capable:
            optimal_shape = self.tensor_core_optimizer.optimize_tensor_shape(
                sample_input.shape, torch.float16
            )
            if optimal_shape != sample_input.shape:
                recommendations['recommendations'].append(
                    f"Consider padding input to {optimal_shape} for optimal Tensor Core utilization"
                )
        
        # Memory recommendations
        if self.device.type == 'cuda':
            total_memory = props.total_memory / 1024**3  # GB
            if total_memory < 8:
                recommendations['recommendations'].append(
                    "Limited GPU memory - mixed precision is highly recommended"
                )
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'precision_mode': self.precision_mode.value,
            'tensor_core_capable': self.tensor_core_optimizer.tensor_core_capable,
            'device': str(self.device)
        }
        
        if self.scaler:
            scaler_stats = self.scaler.get_stats()
            stats.update({
                'gradient_scaling': {
                    'current_scale': scaler_stats.current_scale,
                    'total_steps': scaler_stats.total_steps,
                    'overflow_steps': scaler_stats.overflow_steps,
                    'overflow_rate': scaler_stats.overflow_steps / max(scaler_stats.total_steps, 1),
                    'avg_grad_norm': scaler_stats.avg_grad_norm
                }
            })
        
        return stats


# Convenience functions
def create_mixed_precision_optimizer(precision_mode: str = "auto") -> MixedPrecisionOptimizer:
    """Create mixed precision optimizer with specified mode."""
    config = MixedPrecisionConfig(
        precision_mode=PrecisionMode(precision_mode),
        enable_autocast=True,
        enable_grad_scaling=True,
        enable_tensor_cores=True
    )
    
    return MixedPrecisionOptimizer(config)


def optimize_model_for_inference(model: nn.Module, 
                                precision_mode: str = "auto") -> nn.Module:
    """Optimize model for mixed precision inference."""
    optimizer = create_mixed_precision_optimizer(precision_mode)
    return optimizer.optimize_model(model)


def benchmark_mixed_precision(model: nn.Module, 
                             sample_input: torch.Tensor,
                             num_iterations: int = 100) -> Dict[str, Any]:
    """Quick benchmark of mixed precision performance."""
    optimizer = create_mixed_precision_optimizer()
    return optimizer.benchmark_precision_modes(model, sample_input, num_iterations)