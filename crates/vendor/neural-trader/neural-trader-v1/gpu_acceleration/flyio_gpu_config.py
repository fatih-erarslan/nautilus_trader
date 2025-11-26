"""
Fly.io GPU Configuration and Optimization System
Advanced GPU configuration for A100, V100, RTX series with dynamic optimization.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import cupy as cp
import numpy as np
from numba import cuda
import platform
import subprocess
import psutil

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """Supported GPU types on fly.io."""
    A100_40GB = "a100-40gb"
    A100_80GB = "a100-80gb" 
    V100_16GB = "v100-16gb"
    V100_32GB = "v100-32gb"
    RTX_4090 = "rtx-4090"
    RTX_3090 = "rtx-3090"
    RTX_A6000 = "rtx-a6000"
    UNKNOWN = "unknown"


class PrecisionMode(Enum):
    """Precision modes for mixed precision training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class GPUOptimizationConfig:
    """GPU optimization configuration."""
    gpu_type: GPUType
    memory_gb: float
    compute_capability: Tuple[int, int]
    
    # Memory optimization
    memory_pool_fraction: float = 0.8
    memory_growth: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Compute optimization
    cuda_threads_per_block: int = 256
    cuda_blocks_per_grid: int = 128
    max_concurrent_kernels: int = 16
    
    # Precision settings
    precision_mode: PrecisionMode = PrecisionMode.MIXED
    enable_tensor_cores: bool = True
    
    # Batch processing
    optimal_batch_size: int = 10000
    max_batch_size: int = 100000
    min_batch_size: int = 1000
    batch_size_multiplier: float = 1.5
    
    # Performance targets
    target_gpu_utilization: float = 0.85
    max_memory_utilization: float = 0.9
    min_throughput_ops_sec: int = 50000
    
    # Fly.io specific
    auto_scaling_enabled: bool = True
    cost_optimization_mode: bool = True
    health_check_interval: int = 30
    

class FlyIOGPUDetector:
    """Detects and configures GPU settings for fly.io environment."""
    
    def __init__(self):
        """Initialize GPU detector."""
        self.gpu_info = self._detect_gpu_hardware()
        self.system_info = self._get_system_info()
        
    def _detect_gpu_hardware(self) -> Dict[str, Any]:
        """Detect GPU hardware specifications."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'memory_total': 0,
            'cuda_version': None,
            'driver_version': None
        }
        
        try:
            # Check CUDA availability
            if not cuda.is_available():
                logger.warning("CUDA not available")
                return gpu_info
                
            gpu_info['available'] = True
            gpu_info['count'] = len(cuda.gpus)
            
            # Get CUDA version
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            gpu_info['cuda_version'] = line.split('release')[1].split(',')[0].strip()
                            break
            except:
                pass
            
            # Get driver version
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                       '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info['driver_version'] = result.stdout.strip()
            except:
                pass
            
            # Get device information
            for i in range(gpu_info['count']):
                try:
                    device = cuda.get(i)
                    device_info = {
                        'id': i,
                        'name': device.name.decode('utf-8'),
                        'compute_capability': device.compute_capability,
                        'memory_gb': device.total_memory / (1024**3),
                        'multiprocessor_count': device.MULTIPROCESSOR_COUNT,
                        'max_threads_per_block': device.MAX_THREADS_PER_BLOCK,
                        'max_block_dim_x': device.MAX_BLOCK_DIM_X,
                        'max_grid_dim_x': device.MAX_GRID_DIM_X,
                        'warp_size': device.WARP_SIZE
                    }
                    gpu_info['devices'].append(device_info)
                    gpu_info['memory_total'] += device_info['memory_gb']
                except Exception as e:
                    logger.warning(f"Failed to get info for GPU {i}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to detect GPU hardware: {str(e)}")
            
        return gpu_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version(),
            'environment': os.environ.get('FLY_APP_NAME', 'local')
        }
    
    def detect_gpu_type(self) -> GPUType:
        """Detect the specific GPU type."""
        if not self.gpu_info['available'] or not self.gpu_info['devices']:
            return GPUType.UNKNOWN
            
        device = self.gpu_info['devices'][0]  # Use first GPU
        name = device['name'].lower()
        memory_gb = device['memory_gb']
        
        # A100 detection
        if 'a100' in name:
            if memory_gb > 70:
                return GPUType.A100_80GB
            else:
                return GPUType.A100_40GB
        
        # V100 detection
        elif 'v100' in name:
            if memory_gb > 25:
                return GPUType.V100_32GB
            else:
                return GPUType.V100_16GB
        
        # RTX detection
        elif 'rtx' in name or 'geforce rtx' in name:
            if '4090' in name:
                return GPUType.RTX_4090
            elif '3090' in name:
                return GPUType.RTX_3090
            elif 'a6000' in name:
                return GPUType.RTX_A6000
        
        return GPUType.UNKNOWN
    
    def get_optimal_config(self) -> GPUOptimizationConfig:
        """Get optimal configuration for detected GPU."""
        gpu_type = self.detect_gpu_type()
        
        if not self.gpu_info['available']:
            logger.warning("No GPU detected, using default configuration")
            return self._get_default_config()
        
        device = self.gpu_info['devices'][0]
        compute_capability = device['compute_capability']
        memory_gb = device['memory_gb']
        
        # Base configuration
        config = GPUOptimizationConfig(
            gpu_type=gpu_type,
            memory_gb=memory_gb,
            compute_capability=compute_capability
        )
        
        # Optimize based on GPU type
        if gpu_type in [GPUType.A100_40GB, GPUType.A100_80GB]:
            # A100 optimizations
            config.memory_pool_fraction = 0.85
            config.cuda_threads_per_block = 512
            config.cuda_blocks_per_grid = 256
            config.max_concurrent_kernels = 32
            config.optimal_batch_size = 50000
            config.max_batch_size = 500000
            config.precision_mode = PrecisionMode.BF16  # A100 supports BF16
            config.enable_tensor_cores = True
            config.target_gpu_utilization = 0.9
            config.min_throughput_ops_sec = 200000
            
        elif gpu_type in [GPUType.V100_16GB, GPUType.V100_32GB]:
            # V100 optimizations
            config.memory_pool_fraction = 0.8
            config.cuda_threads_per_block = 256
            config.cuda_blocks_per_grid = 128
            config.max_concurrent_kernels = 16
            config.optimal_batch_size = 25000
            config.max_batch_size = 250000
            config.precision_mode = PrecisionMode.FP16  # V100 supports FP16
            config.enable_tensor_cores = True
            config.target_gpu_utilization = 0.85
            config.min_throughput_ops_sec = 100000
            
        elif gpu_type in [GPUType.RTX_4090, GPUType.RTX_3090, GPUType.RTX_A6000]:
            # RTX optimizations
            config.memory_pool_fraction = 0.75
            config.cuda_threads_per_block = 256
            config.cuda_blocks_per_grid = 64
            config.max_concurrent_kernels = 8
            config.optimal_batch_size = 15000
            config.max_batch_size = 150000
            config.precision_mode = PrecisionMode.FP16
            config.enable_tensor_cores = True
            config.target_gpu_utilization = 0.8
            config.min_throughput_ops_sec = 75000
        
        # Adjust for memory constraints
        if memory_gb < 16:
            config.memory_pool_fraction = 0.7
            config.optimal_batch_size = min(config.optimal_batch_size, 10000)
            config.max_batch_size = min(config.max_batch_size, 50000)
        
        return config
    
    def _get_default_config(self) -> GPUOptimizationConfig:
        """Get default configuration for unknown/no GPU."""
        return GPUOptimizationConfig(
            gpu_type=GPUType.UNKNOWN,
            memory_gb=0,
            compute_capability=(0, 0),
            precision_mode=PrecisionMode.FP32,
            enable_tensor_cores=False,
            optimal_batch_size=1000,
            max_batch_size=10000,
            target_gpu_utilization=0.0,
            min_throughput_ops_sec=1000
        )


class FlyIOGPUConfigManager:
    """Manages GPU configuration for fly.io deployment."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file or "/app/gpu_config.json"
        self.detector = FlyIOGPUDetector()
        self.config = self._load_or_create_config()
        
    def _load_or_create_config(self) -> GPUOptimizationConfig:
        """Load existing config or create new one."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    # Convert to config object
                    config = GPUOptimizationConfig(**data)
                    logger.info(f"Loaded GPU configuration from {self.config_file}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config: {str(e)}, creating new one")
        
        # Create optimal config
        config = self.detector.get_optimal_config()
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[GPUOptimizationConfig] = None):
        """Save configuration to file."""
        config = config or self.config
        
        try:
            # Convert enum values to strings for JSON serialization
            config_dict = asdict(config)
            config_dict['gpu_type'] = config.gpu_type.value
            config_dict['precision_mode'] = config.precision_mode.value
            
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Saved GPU configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config parameter: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        self.save_config()
    
    def get_cuda_launch_config(self) -> Dict[str, int]:
        """Get optimal CUDA launch configuration."""
        return {
            'threads_per_block': self.config.cuda_threads_per_block,
            'blocks_per_grid': self.config.cuda_blocks_per_grid,
            'max_concurrent_kernels': self.config.max_concurrent_kernels
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return {
            'pool_fraction': self.config.memory_pool_fraction,
            'growth': self.config.memory_growth,
            'limit_gb': self.config.memory_limit_gb,
            'total_gb': self.config.memory_gb
        }
    
    def get_batch_config(self) -> Dict[str, int]:
        """Get batch processing configuration."""
        return {
            'optimal_size': self.config.optimal_batch_size,
            'max_size': self.config.max_batch_size,
            'min_size': self.config.min_batch_size,
            'multiplier': self.config.batch_size_multiplier
        }
    
    def get_precision_config(self) -> Dict[str, Any]:
        """Get precision configuration."""
        return {
            'mode': self.config.precision_mode.value,
            'tensor_cores': self.config.enable_tensor_cores,
            'compute_capability': self.config.compute_capability
        }
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets."""
        return {
            'gpu_utilization': self.config.target_gpu_utilization,
            'memory_utilization': self.config.max_memory_utilization,
            'throughput_ops_sec': self.config.min_throughput_ops_sec
        }
    
    def get_flyio_config(self) -> Dict[str, Any]:
        """Get fly.io specific configuration."""
        return {
            'auto_scaling': self.config.auto_scaling_enabled,
            'cost_optimization': self.config.cost_optimization_mode,
            'health_check_interval': self.config.health_check_interval,
            'gpu_type': self.config.gpu_type.value
        }
    
    def generate_fly_toml_config(self) -> str:
        """Generate fly.toml configuration section."""
        gpu_type_map = {
            GPUType.A100_40GB: "a100-40gb",
            GPUType.A100_80GB: "a100-80gb", 
            GPUType.V100_16GB: "v100-16gb",
            GPUType.V100_32GB: "v100-32gb",
            GPUType.RTX_4090: "rtx-4090",
            GPUType.RTX_3090: "rtx-3090",
            GPUType.RTX_A6000: "rtx-a6000",
        }
        
        gpu_kind = gpu_type_map.get(self.config.gpu_type, "a100-40gb")
        memory_gb = max(int(self.config.memory_gb * 1.2), 16)  # Add 20% overhead
        
        return f"""
# GPU-optimized configuration for {self.config.gpu_type.value}
[vm]
  size = "performance-8x"
  memory = "{memory_gb}gb"
  cpus = 8
  gpu_kind = "{gpu_kind}"

[env]
  CUDA_VISIBLE_DEVICES = "0"
  RAPIDS_NO_INITIALIZE = "1"
  CUPY_CACHE_DIR = "/tmp/.cupy"
  GPU_MEMORY_FRACTION = "{self.config.memory_pool_fraction}"
  CUDA_THREADS_PER_BLOCK = "{self.config.cuda_threads_per_block}"
  OPTIMAL_BATCH_SIZE = "{self.config.optimal_batch_size}"
  PRECISION_MODE = "{self.config.precision_mode.value}"
  ENABLE_TENSOR_CORES = "{'true' if self.config.enable_tensor_cores else 'false'}"

[scaling]
  min_instances = 1
  max_instances = {"3" if self.config.auto_scaling_enabled else "1"}
"""
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check GPU availability
        if not self.detector.gpu_info['available']:
            validation_results['errors'].append("No GPU detected")
            validation_results['valid'] = False
        
        # Check memory settings
        if self.config.memory_pool_fraction > 0.95:
            validation_results['warnings'].append("Memory pool fraction very high, may cause OOM")
        
        # Check batch size
        available_memory = self.config.memory_gb * self.config.memory_pool_fraction
        estimated_batch_memory = (self.config.optimal_batch_size * 4) / (1024**3)  # 4 bytes per float32
        
        if estimated_batch_memory > available_memory * 0.5:
            validation_results['warnings'].append("Batch size may be too large for available memory")
            validation_results['recommendations'].append(
                f"Consider reducing batch size to {int(self.config.optimal_batch_size * 0.7)}"
            )
        
        # Check precision mode compatibility
        if self.config.precision_mode == PrecisionMode.BF16:
            if self.config.compute_capability < (8, 0):
                validation_results['errors'].append("BF16 requires compute capability 8.0+")
                validation_results['valid'] = False
                validation_results['recommendations'].append("Use FP16 instead of BF16")
        
        return validation_results


# Global configuration instance
_gpu_config_manager = None


def get_gpu_config_manager() -> FlyIOGPUConfigManager:
    """Get global GPU configuration manager."""
    global _gpu_config_manager
    if _gpu_config_manager is None:
        _gpu_config_manager = FlyIOGPUConfigManager()
    return _gpu_config_manager


def initialize_flyio_gpu_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Initialize fly.io GPU configuration."""
    logger.info("Initializing fly.io GPU configuration")
    
    config_manager = FlyIOGPUConfigManager(config_file)
    validation = config_manager.validate_config()
    
    if not validation['valid']:
        logger.error(f"GPU configuration validation failed: {validation['errors']}")
        return {
            'status': 'failed',
            'validation': validation,
            'config': None
        }
    
    if validation['warnings']:
        logger.warning(f"GPU configuration warnings: {validation['warnings']}")
    
    if validation['recommendations']:
        logger.info(f"GPU configuration recommendations: {validation['recommendations']}")
    
    logger.info(f"GPU configuration initialized successfully for {config_manager.config.gpu_type.value}")
    
    return {
        'status': 'success',
        'validation': validation,
        'config': config_manager.config,
        'gpu_info': config_manager.detector.gpu_info,
        'system_info': config_manager.detector.system_info
    }


if __name__ == "__main__":
    # Test the configuration system
    result = initialize_flyio_gpu_config()
    print(json.dumps({
        'status': result['status'],
        'gpu_type': result['config'].gpu_type.value if result['config'] else 'none',
        'validation': result['validation']
    }, indent=2))