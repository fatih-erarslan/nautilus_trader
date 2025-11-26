"""
GPU Acceleration Support for NHITS Forecasting on Fly.io.

This module provides GPU acceleration capabilities optimized for fly.io deployment,
including memory management, device detection, and fallback mechanisms.
"""

import asyncio
import logging
import os
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUAccelerationManager:
    """
    GPU acceleration manager optimized for fly.io deployment.
    
    Features:
    - Automatic GPU detection and configuration
    - Memory management and optimization
    - Batch processing optimization
    - Fallback to CPU when needed
    - Performance monitoring
    - Resource allocation for multi-tenant environments
    """
    
    def __init__(
        self,
        enable_gpu: bool = True,
        memory_limit_gb: Optional[float] = None,
        batch_size_optimization: bool = True,
        enable_monitoring: bool = True,
        fallback_threshold: float = 0.9  # GPU memory threshold for fallback
    ):
        """
        Initialize GPU acceleration manager.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            memory_limit_gb: GPU memory limit in GB (auto-detect if None)
            batch_size_optimization: Enable automatic batch size optimization
            enable_monitoring: Enable GPU monitoring
            fallback_threshold: Memory threshold for CPU fallback (0-1)
        """
        self.enable_gpu = enable_gpu and TORCH_AVAILABLE
        self.memory_limit_gb = memory_limit_gb
        self.batch_size_optimization = batch_size_optimization
        self.enable_monitoring = enable_monitoring
        self.fallback_threshold = fallback_threshold
        
        # Device configuration
        self.device = None
        self.gpu_available = False
        self.gpu_properties = {}
        self.memory_info = {}
        
        # Performance tracking
        self.performance_metrics = {
            'gpu_utilization': [],
            'memory_usage': [],
            'batch_processing_times': [],
            'fallback_events': []
        }
        
        # Batch size optimization
        self.optimal_batch_sizes = {}
        self.batch_size_history = {}
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize GPU configuration
        self._initialize_gpu()
        
        self.logger.info(f"GPU Acceleration Manager initialized - GPU Available: {self.gpu_available}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_gpu(self):
        """Initialize GPU configuration and detection."""
        try:
            if not self.enable_gpu or not TORCH_AVAILABLE:
                self.logger.info("GPU acceleration disabled or PyTorch not available")
                self.device = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
                return
            
            # Detect GPU availability
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device = torch.device('cuda:0')
                
                # Get GPU properties
                self.gpu_properties = {
                    'name': torch.cuda.get_device_name(0),
                    'compute_capability': torch.cuda.get_device_capability(0),
                    'total_memory': torch.cuda.get_device_properties(0).total_memory,
                    'multi_processor_count': torch.cuda.get_device_properties(0).multi_processor_count
                }
                
                # Initialize memory management
                self._initialize_memory_management()
                
                # Configure optimal settings for fly.io
                self._configure_flyio_optimizations()
                
                self.logger.info(f"GPU initialized: {self.gpu_properties['name']}")
                
            else:
                self.logger.info("CUDA not available, using CPU")
                self.device = torch.device('cpu')
                
        except Exception as e:
            self.logger.error(f"GPU initialization failed: {str(e)}")
            self.device = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
            self.gpu_available = False
    
    def _initialize_memory_management(self):
        """Initialize GPU memory management."""
        try:
            if not self.gpu_available:
                return
            
            # Get memory info
            self.memory_info = self._get_memory_info()
            
            # Set memory limit if specified
            if self.memory_limit_gb:
                memory_limit_bytes = int(self.memory_limit_gb * 1024**3)
                torch.cuda.set_per_process_memory_fraction(
                    memory_limit_bytes / self.gpu_properties['total_memory']
                )
                self.logger.info(f"GPU memory limit set to {self.memory_limit_gb}GB")
            
            # Enable memory caching optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            self.logger.info("GPU memory management initialized")
            
        except Exception as e:
            self.logger.error(f"Memory management initialization failed: {str(e)}")
    
    def _configure_flyio_optimizations(self):
        """Configure optimizations specific to fly.io deployment."""
        try:
            # Set environment variables for optimal performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
            os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.0;8.6'  # Common GPU architectures
            
            # Configure memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Enable tensor core usage if available
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('medium')
            
            self.logger.info("Fly.io GPU optimizations configured")
            
        except Exception as e:
            self.logger.warning(f"Fly.io optimization configuration failed: {str(e)}")
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get current GPU memory information."""
        try:
            if not self.gpu_available:
                return {}
            
            memory_allocated = torch.cuda.memory_allocated()
            memory_cached = torch.cuda.memory_reserved()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'allocated': memory_allocated,
                'cached': memory_cached,
                'total': memory_total,
                'free': memory_total - memory_allocated,
                'utilization': memory_allocated / memory_total
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory info: {str(e)}")
            return {}
    
    async def optimize_batch_size(
        self,
        model_type: str,
        input_shape: Tuple[int, ...],
        target_memory_usage: float = 0.8
    ) -> int:
        """
        Optimize batch size for given model and input shape.
        
        Args:
            model_type: Type of model ('nhits', 'general')
            input_shape: Input tensor shape
            target_memory_usage: Target GPU memory usage (0-1)
            
        Returns:
            Optimal batch size
        """
        if not self.gpu_available or not self.batch_size_optimization:
            return 32  # Default batch size for CPU
        
        cache_key = f"{model_type}_{input_shape}"
        
        # Check cache first
        if cache_key in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[cache_key]
        
        try:
            # Start with a reasonable batch size
            batch_size = 16
            max_batch_size = 256
            
            # Binary search for optimal batch size
            while batch_size <= max_batch_size:
                try:
                    # Test memory usage with this batch size
                    test_memory = await self._estimate_memory_usage(
                        batch_size, input_shape, model_type
                    )
                    
                    memory_info = self._get_memory_info()
                    if memory_info:
                        projected_usage = (memory_info['allocated'] + test_memory) / memory_info['total']
                        
                        if projected_usage > target_memory_usage:
                            break
                    
                    batch_size *= 2
                    
                except torch.cuda.OutOfMemoryError:
                    batch_size //= 2
                    break
                    
            # Use 75% of maximum working batch size for safety
            optimal_batch_size = max(1, int(batch_size * 0.75))
            
            # Cache the result
            self.optimal_batch_sizes[cache_key] = optimal_batch_size
            
            self.logger.info(f"Optimal batch size for {model_type}: {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.error(f"Batch size optimization failed: {str(e)}")
            return 32
    
    async def _estimate_memory_usage(
        self,
        batch_size: int,
        input_shape: Tuple[int, ...],
        model_type: str
    ) -> int:
        """Estimate memory usage for given batch size and input shape."""
        try:
            # Estimate based on model type and input dimensions
            if model_type == 'nhits':
                # NHITS model memory estimation
                sequence_length = input_shape[-1] if input_shape else 24
                hidden_size = 512  # Typical hidden size
                
                # Rough estimation: input + hidden states + gradients
                estimated_memory = (
                    batch_size * sequence_length * 4 +  # Input (float32)
                    batch_size * hidden_size * 4 * 3 +  # Hidden layers
                    batch_size * sequence_length * 4 * 2  # Gradients
                ) * 2  # Safety factor
                
            else:
                # General model estimation
                total_params = 1
                for dim in input_shape:
                    total_params *= dim
                
                estimated_memory = batch_size * total_params * 4 * 3  # Input + forward + backward
            
            return estimated_memory
            
        except Exception:
            return batch_size * 1024 * 1024  # 1MB per batch item fallback
    
    async def check_memory_availability(self, required_memory_gb: float = 1.0) -> bool:
        """
        Check if sufficient GPU memory is available.
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            True if sufficient memory available
        """
        try:
            if not self.gpu_available:
                return False
            
            memory_info = self._get_memory_info()
            if not memory_info:
                return False
            
            required_bytes = required_memory_gb * 1024**3
            available_bytes = memory_info['free']
            
            available = available_bytes >= required_bytes
            
            if not available:
                self.logger.warning(
                    f"Insufficient GPU memory: {available_bytes/1024**3:.2f}GB available, "
                    f"{required_memory_gb:.2f}GB required"
                )
            
            return available
            
        except Exception as e:
            self.logger.error(f"Memory availability check failed: {str(e)}")
            return False
    
    async def manage_memory(self, force_cleanup: bool = False) -> Dict[str, Any]:
        """
        Manage GPU memory and perform cleanup if needed.
        
        Args:
            force_cleanup: Force memory cleanup regardless of usage
            
        Returns:
            Memory management results
        """
        try:
            if not self.gpu_available:
                return {'status': 'gpu_not_available'}
            
            memory_before = self._get_memory_info()
            
            # Check if cleanup is needed
            if force_cleanup or (memory_before.get('utilization', 0) > self.fallback_threshold):
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Synchronize to ensure operations complete
                torch.cuda.synchronize()
                
                memory_after = self._get_memory_info()
                
                freed_memory = memory_before.get('allocated', 0) - memory_after.get('allocated', 0)
                
                self.logger.info(f"GPU memory cleanup: freed {freed_memory / 1024**3:.2f}GB")
                
                return {
                    'status': 'cleanup_performed',
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'freed_memory_gb': freed_memory / 1024**3
                }
            
            return {
                'status': 'no_cleanup_needed',
                'memory_info': memory_before
            }
            
        except Exception as e:
            self.logger.error(f"Memory management failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def should_use_gpu(self, task_complexity: str = 'medium') -> bool:
        """
        Determine if GPU should be used for a given task.
        
        Args:
            task_complexity: Task complexity ('low', 'medium', 'high')
            
        Returns:
            True if GPU should be used
        """
        try:
            if not self.gpu_available:
                return False
            
            # Check memory availability
            memory_info = self._get_memory_info()
            if memory_info.get('utilization', 1.0) > self.fallback_threshold:
                self.logger.warning("GPU memory usage too high, falling back to CPU")
                self._record_fallback_event("memory_threshold_exceeded")
                return False
            
            # Task complexity considerations
            complexity_thresholds = {
                'low': 0.1,     # Use GPU only if very lightly loaded
                'medium': 0.5,  # Use GPU if moderately loaded
                'high': 0.8     # Use GPU even if heavily loaded
            }
            
            threshold = complexity_thresholds.get(task_complexity, 0.5)
            if memory_info.get('utilization', 1.0) > threshold:
                return False
            
            # Check system load
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage < 30 and task_complexity == 'low':
                # If CPU is idle and task is simple, use CPU to preserve GPU
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPU decision check failed: {str(e)}")
            return False
    
    def _record_fallback_event(self, reason: str):
        """Record GPU fallback event for monitoring."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'memory_info': self._get_memory_info()
            }
            
            self.performance_metrics['fallback_events'].append(event)
            
            # Keep only recent events
            if len(self.performance_metrics['fallback_events']) > 100:
                self.performance_metrics['fallback_events'] = self.performance_metrics['fallback_events'][-100:]
                
        except Exception as e:
            self.logger.warning(f"Failed to record fallback event: {str(e)}")
    
    async def benchmark_device_performance(self) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance for typical workloads.
        
        Returns:
            Performance benchmark results
        """
        try:
            results = {
                'gpu_available': self.gpu_available,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            
            # Simple matrix multiplication benchmark
            if TORCH_AVAILABLE:
                size = 1024
                iterations = 10
                
                # CPU benchmark
                torch.set_num_threads(psutil.cpu_count())
                cpu_times = []
                
                for _ in range(iterations):
                    start_time = time.time()
                    a = torch.randn(size, size)
                    b = torch.randn(size, size)
                    c = torch.mm(a, b)
                    cpu_times.append(time.time() - start_time)
                
                results['cpu_avg_time'] = sum(cpu_times) / len(cpu_times)
                
                # GPU benchmark if available
                if self.gpu_available:
                    gpu_times = []
                    
                    for _ in range(iterations):
                        start_time = time.time()
                        a_gpu = torch.randn(size, size, device=self.device)
                        b_gpu = torch.randn(size, size, device=self.device)
                        c_gpu = torch.mm(a_gpu, b_gpu)
                        torch.cuda.synchronize()
                        gpu_times.append(time.time() - start_time)
                    
                    results['gpu_avg_time'] = sum(gpu_times) / len(gpu_times)
                    results['speedup'] = results['cpu_avg_time'] / results['gpu_avg_time']
                    
                    # Clean up GPU memory
                    del a_gpu, b_gpu, c_gpu
                    torch.cuda.empty_cache()
            
            self.logger.info(f"Device benchmark completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Device benchmark failed: {str(e)}")
            return {'error': str(e)}
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Device information dictionary
        """
        try:
            info = {
                'gpu_available': self.gpu_available,
                'torch_available': TORCH_AVAILABLE,
                'cupy_available': CUPY_AVAILABLE,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            
            if self.gpu_available:
                info['gpu_properties'] = self.gpu_properties
                info['memory_info'] = self._get_memory_info()
                info['optimal_batch_sizes'] = self.optimal_batch_sizes
                
                # Add CUDA information
                info['cuda_version'] = torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown'
                info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'unknown'
            
            # Add system information
            info['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'platform': os.uname().sysname if hasattr(os, 'uname') else 'unknown'
            }
            
            # Fly.io specific information
            if 'FLY_APP_NAME' in os.environ:
                info['flyio_info'] = {
                    'app_name': os.environ.get('FLY_APP_NAME'),
                    'region': os.environ.get('FLY_REGION'),
                    'instance_id': os.environ.get('FLY_ALLOC_ID')
                }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get device info: {str(e)}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics and statistics.
        
        Returns:
            Performance metrics
        """
        try:
            metrics = {
                'fallback_events': len(self.performance_metrics['fallback_events']),
                'optimal_batch_sizes_count': len(self.optimal_batch_sizes),
                'memory_management_enabled': self.gpu_available,
                'current_memory_info': self._get_memory_info() if self.gpu_available else {},
                'device_info': {
                    'type': 'GPU' if self.gpu_available else 'CPU',
                    'name': self.gpu_properties.get('name', 'CPU') if self.gpu_available else 'CPU'
                }
            }
            
            # Calculate fallback rate
            if self.performance_metrics['fallback_events']:
                recent_events = [
                    e for e in self.performance_metrics['fallback_events']
                    if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)
                ]
                metrics['fallback_rate_last_hour'] = len(recent_events)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of GPU acceleration.
        
        Returns:
            Health check results
        """
        try:
            health = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'gpu_available': self.gpu_available,
                'torch_available': TORCH_AVAILABLE
            }
            
            if self.gpu_available:
                # Check GPU memory
                memory_info = self._get_memory_info()
                if memory_info.get('utilization', 0) > 0.95:
                    health['status'] = 'warning'
                    health['warnings'] = ['High GPU memory usage']
                
                # Check for recent fallback events
                recent_fallbacks = [
                    e for e in self.performance_metrics['fallback_events']
                    if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(minutes=30)
                ]
                
                if len(recent_fallbacks) > 5:
                    health['status'] = 'warning'
                    health['warnings'] = health.get('warnings', []) + ['High fallback rate']
                
                health['memory_info'] = memory_info
                health['recent_fallbacks'] = len(recent_fallbacks)
            
            # Test basic operations
            try:
                if TORCH_AVAILABLE:
                    test_tensor = torch.randn(10, 10)
                    if self.gpu_available:
                        test_tensor = test_tensor.to(self.device)
                        result = torch.mm(test_tensor, test_tensor)
                        torch.cuda.synchronize() if self.gpu_available else None
                    health['basic_operations'] = 'passed'
            except Exception as e:
                health['status'] = 'error'
                health['basic_operations'] = f'failed: {str(e)}'
            
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup GPU resources on destruction."""
        try:
            if self.gpu_available and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
        except Exception:
            pass