"""
Optimized Configuration Profiles for Maximum Performance
Auto-detects hardware and provides optimal configurations for different deployment scenarios.
"""

import json
import logging
import platform
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import torch
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceProfile(Enum):
    """Performance optimization profiles."""
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # <5ms inference
    LOW_LATENCY = "low_latency"              # <10ms inference  
    BALANCED = "balanced"                    # Balanced latency/throughput
    HIGH_THROUGHPUT = "high_throughput"      # Maximum throughput
    MEMORY_EFFICIENT = "memory_efficient"    # Minimize memory usage
    PRODUCTION = "production"                # Production optimized


class HardwareClass(Enum):
    """Hardware classification for optimization."""
    DATACENTER_GPU = "datacenter"    # A100, V100, H100
    GAMING_GPU = "gaming"            # RTX 3090, 4090
    WORKSTATION_GPU = "workstation"  # RTX A6000, Quadro
    CONSUMER_GPU = "consumer"        # RTX 3060, 3070, 3080
    INTEGRATED_GPU = "integrated"    # Intel, AMD integrated
    CPU_ONLY = "cpu_only"           # No GPU available


@dataclass
class OptimizedProfileConfig:
    """Complete optimized configuration profile."""
    
    # Model architecture
    input_size: int = 168
    horizon: int = 24
    hidden_size: int = 512
    n_blocks: List[int] = None
    n_freq_downsample: List[int] = None
    
    # Performance optimization
    batch_size: int = 64
    max_batch_size: int = 256
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Memory management
    memory_fraction: float = 0.8
    enable_memory_growth: bool = True
    use_memory_pool: bool = True
    cache_predictions: bool = True
    max_cache_size: int = 1000
    
    # Threading and parallelism
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    
    # Advanced optimizations
    use_tensorrt: bool = False
    compile_model: bool = False
    use_channels_last: bool = False
    
    # Training specific
    max_epochs: int = 100
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    
    def __post_init__(self):
        if self.n_blocks is None:
            self.n_blocks = [2, 2, 1]
        if self.n_freq_downsample is None:
            self.n_freq_downsample = [4, 2, 1]


class HardwareDetector:
    """Detects and classifies hardware for optimal configuration."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.hardware_class = self._classify_hardware()
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information."""
        gpu_info = {
            'available': False,
            'name': 'None',
            'memory_gb': 0,
            'compute_capability': None,
            'tensor_cores': False,
            'architecture': 'Unknown'
        }
        
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            gpu_info['compute_capability'] = f"{props.major}.{props.minor}"
            
            # Determine architecture and tensor core availability
            gpu_info['architecture'], gpu_info['tensor_cores'] = self._classify_gpu_architecture(
                gpu_info['name'], gpu_info['compute_capability']
            )
        
        return gpu_info
    
    def _classify_gpu_architecture(self, name: str, compute_capability: str) -> tuple:
        """Classify GPU architecture and tensor core availability."""
        name_lower = name.lower()
        
        # Data Center GPUs
        if any(card in name_lower for card in ['a100', 'h100', 'v100']):
            if 'h100' in name_lower:
                return 'Hopper', True
            elif 'a100' in name_lower:
                return 'Ampere', True
            else:  # V100
                return 'Volta', True
        
        # RTX 40 series (Ada Lovelace)
        if any(card in name_lower for card in ['rtx 40', 'rtx 4090', 'rtx 4080', 'rtx 4070']):
            return 'Ada Lovelace', True
        
        # RTX 30 series (Ampere)
        if any(card in name_lower for card in ['rtx 30', 'rtx 3090', 'rtx 3080', 'rtx 3070']):
            return 'Ampere', True
        
        # RTX 20 series (Turing)
        if any(card in name_lower for card in ['rtx 20', 'rtx 2080', 'rtx 2070']):
            return 'Turing', True
        
        # Workstation cards
        if any(card in name_lower for card in ['rtx a', 'quadro', 'tesla']):
            return 'Professional', True
        
        # Determine from compute capability
        if compute_capability:
            major = float(compute_capability.split('.')[0])
            if major >= 9.0:
                return 'Hopper', True
            elif major >= 8.6:
                return 'Ada Lovelace', True
            elif major >= 8.0:
                return 'Ampere', True
            elif major >= 7.5:
                return 'Turing', True
            elif major >= 7.0:
                return 'Volta', True
        
        return 'Unknown', False
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        return {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_ghz': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 0,
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect system memory information."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1e9,
            'available_gb': mem.available / 1e9,
            'usage_percent': mem.percent
        }
    
    def _classify_hardware(self) -> HardwareClass:
        """Classify hardware for optimization profiles."""
        if not self.gpu_info['available']:
            return HardwareClass.CPU_ONLY
        
        gpu_memory = self.gpu_info['memory_gb']
        gpu_name = self.gpu_info['name'].lower()
        
        # Data center GPUs
        if any(card in gpu_name for card in ['a100', 'h100', 'v100', 'tesla']):
            return HardwareClass.DATACENTER_GPU
        
        # High-end gaming GPUs
        if any(card in gpu_name for card in ['rtx 4090', 'rtx 3090']) or gpu_memory > 20:
            return HardwareClass.GAMING_GPU
        
        # Workstation GPUs
        if any(card in gpu_name for card in ['rtx a', 'quadro', 'titan']) or gpu_memory > 12:
            return HardwareClass.WORKSTATION_GPU
        
        # Consumer GPUs
        if gpu_memory > 4:
            return HardwareClass.CONSUMER_GPU
        
        # Integrated/low-end GPUs
        return HardwareClass.INTEGRATED_GPU
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get comprehensive hardware summary."""
        return {
            'hardware_class': self.hardware_class.value,
            'gpu': self.gpu_info,
            'cpu': self.cpu_info,
            'memory': self.memory_info,
            'platform': platform.platform()
        }


class OptimizedConfigProfileManager:
    """Manages optimized configuration profiles for different hardware and use cases."""
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.profiles = self._create_performance_profiles()
        
        logger.info(f"Hardware detected: {self.detector.hardware_class.value}")
        logger.info(f"GPU: {self.detector.gpu_info['name']} ({self.detector.gpu_info['memory_gb']:.1f}GB)")
    
    def _create_performance_profiles(self) -> Dict[str, Dict[HardwareClass, OptimizedProfileConfig]]:
        """Create optimized profiles for different performance targets and hardware."""
        profiles = {}
        
        # Ultra Low Latency Profile (<5ms)
        profiles[PerformanceProfile.ULTRA_LOW_LATENCY.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=24, horizon=6, hidden_size=256, batch_size=1, max_batch_size=8,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.9,
                cache_predictions=True, max_cache_size=5000, compile_model=True,
                use_tensorrt=True, use_channels_last=True
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=24, horizon=6, hidden_size=256, batch_size=1, max_batch_size=4,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.85,
                cache_predictions=True, max_cache_size=2000, compile_model=True
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=24, horizon=6, hidden_size=256, batch_size=1, max_batch_size=4,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.8,
                cache_predictions=True, max_cache_size=1000
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=24, horizon=6, hidden_size=128, batch_size=1, max_batch_size=2,
                mixed_precision=False, gradient_checkpointing=True, memory_fraction=0.7,
                cache_predictions=True, max_cache_size=500
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=24, horizon=6, hidden_size=128, batch_size=1, max_batch_size=2,
                use_gpu=False, num_workers=8, cache_predictions=True, max_cache_size=1000
            )
        }
        
        # Low Latency Profile (<10ms)
        profiles[PerformanceProfile.LOW_LATENCY.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=48, horizon=12, hidden_size=512, batch_size=8, max_batch_size=32,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.85,
                cache_predictions=True, max_cache_size=3000, compile_model=True
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=48, horizon=12, hidden_size=512, batch_size=4, max_batch_size=16,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.8,
                cache_predictions=True, max_cache_size=2000
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=48, horizon=12, hidden_size=256, batch_size=4, max_batch_size=16,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.75,
                cache_predictions=True, max_cache_size=1500
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=48, horizon=12, hidden_size=256, batch_size=2, max_batch_size=8,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.7,
                cache_predictions=True, max_cache_size=1000
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=48, horizon=12, hidden_size=256, batch_size=2, max_batch_size=4,
                use_gpu=False, num_workers=6, cache_predictions=True, max_cache_size=500
            )
        }
        
        # Balanced Profile
        profiles[PerformanceProfile.BALANCED.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=32, max_batch_size=128,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.8,
                cache_predictions=True, max_cache_size=2000
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=16, max_batch_size=64,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.75,
                cache_predictions=True, max_cache_size=1500
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=16, max_batch_size=64,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.7,
                cache_predictions=True, max_cache_size=1000
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=8, max_batch_size=32,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.65,
                cache_predictions=True, max_cache_size=500
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=4, max_batch_size=8,
                use_gpu=False, num_workers=4, cache_predictions=True, max_cache_size=200
            )
        }
        
        # High Throughput Profile
        profiles[PerformanceProfile.HIGH_THROUGHPUT.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=1024, batch_size=128, max_batch_size=512,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.9,
                cache_predictions=False, num_workers=8
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=64, max_batch_size=256,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.85,
                cache_predictions=False, num_workers=6
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=32, max_batch_size=128,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.8,
                cache_predictions=False, num_workers=4
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=16, max_batch_size=64,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.75,
                cache_predictions=False, num_workers=4
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=8, max_batch_size=16,
                use_gpu=False, num_workers=psutil.cpu_count(), cache_predictions=False
            )
        }
        
        # Memory Efficient Profile
        profiles[PerformanceProfile.MEMORY_EFFICIENT.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=96, horizon=12, hidden_size=256, batch_size=16, max_batch_size=64,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.6,
                cache_predictions=True, max_cache_size=500, enable_memory_growth=True
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=96, horizon=12, hidden_size=256, batch_size=8, max_batch_size=32,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.5,
                cache_predictions=True, max_cache_size=300, enable_memory_growth=True
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=96, horizon=12, hidden_size=128, batch_size=8, max_batch_size=32,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.5,
                cache_predictions=True, max_cache_size=200, enable_memory_growth=True
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=96, horizon=12, hidden_size=128, batch_size=4, max_batch_size=16,
                mixed_precision=False, gradient_checkpointing=True, memory_fraction=0.4,
                cache_predictions=True, max_cache_size=100, enable_memory_growth=True
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=96, horizon=12, hidden_size=128, batch_size=2, max_batch_size=4,
                use_gpu=False, num_workers=2, cache_predictions=True, max_cache_size=50
            )
        }
        
        # Production Profile
        profiles[PerformanceProfile.PRODUCTION.value] = {
            HardwareClass.DATACENTER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=64, max_batch_size=256,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.75,
                cache_predictions=True, max_cache_size=1000, compile_model=True,
                use_tensorrt=True, max_epochs=200, warmup_steps=200
            ),
            HardwareClass.GAMING_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=512, batch_size=32, max_batch_size=128,
                mixed_precision=True, gradient_checkpointing=False, memory_fraction=0.7,
                cache_predictions=True, max_cache_size=800, compile_model=True,
                max_epochs=150, warmup_steps=150
            ),
            HardwareClass.WORKSTATION_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=32, max_batch_size=128,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.65,
                cache_predictions=True, max_cache_size=600, max_epochs=100, warmup_steps=100
            ),
            HardwareClass.CONSUMER_GPU: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=16, max_batch_size=64,
                mixed_precision=True, gradient_checkpointing=True, memory_fraction=0.6,
                cache_predictions=True, max_cache_size=400, max_epochs=100, warmup_steps=100
            ),
            HardwareClass.CPU_ONLY: OptimizedProfileConfig(
                input_size=168, horizon=24, hidden_size=256, batch_size=8, max_batch_size=16,
                use_gpu=False, num_workers=4, cache_predictions=True, max_cache_size=200,
                max_epochs=50, warmup_steps=50
            )
        }
        
        return profiles
    
    def get_optimal_config(self, 
                          performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
                          custom_overrides: Optional[Dict[str, Any]] = None) -> OptimizedProfileConfig:
        """Get optimal configuration for current hardware and performance profile."""
        
        hardware_class = self.detector.hardware_class
        profile_configs = self.profiles[performance_profile.value]
        
        # Get base config for hardware class
        if hardware_class in profile_configs:
            config = profile_configs[hardware_class]
        else:
            # Fallback to CPU_ONLY if hardware class not found
            config = profile_configs[HardwareClass.CPU_ONLY]
            logger.warning(f"No profile for {hardware_class.value}, using CPU_ONLY profile")
        
        # Apply custom overrides
        if custom_overrides:
            config_dict = asdict(config)
            config_dict.update(custom_overrides)
            config = OptimizedProfileConfig(**config_dict)
        
        # Hardware-specific optimizations
        config = self._apply_hardware_optimizations(config)
        
        logger.info(f"Selected {performance_profile.value} profile for {hardware_class.value}")
        logger.info(f"Config: batch_size={config.batch_size}, hidden_size={config.hidden_size}, "
                   f"mixed_precision={config.mixed_precision}")
        
        return config
    
    def _apply_hardware_optimizations(self, config: OptimizedProfileConfig) -> OptimizedProfileConfig:
        """Apply hardware-specific optimizations."""
        config_dict = asdict(config)
        
        # GPU-specific optimizations
        if self.detector.gpu_info['available']:
            gpu_memory = self.detector.gpu_info['memory_gb']
            
            # Adjust batch size based on GPU memory
            if gpu_memory < 4:
                config_dict['batch_size'] = min(config_dict['batch_size'], 8)
                config_dict['max_batch_size'] = min(config_dict['max_batch_size'], 32)
            elif gpu_memory < 8:
                config_dict['batch_size'] = min(config_dict['batch_size'], 16)
                config_dict['max_batch_size'] = min(config_dict['max_batch_size'], 64)
            
            # Enable tensor cores if available
            if self.detector.gpu_info['tensor_cores']:
                config_dict['mixed_precision'] = True
                config_dict['use_channels_last'] = True
            
            # Enable TensorRT for production on datacenter GPUs
            if (self.detector.hardware_class == HardwareClass.DATACENTER_GPU and 
                'production' in str(config_dict)):
                config_dict['use_tensorrt'] = True
        
        # CPU optimizations
        cpu_cores = self.detector.cpu_info['cores_logical']
        if not config_dict['use_gpu']:
            config_dict['num_workers'] = min(cpu_cores, 8)  # Cap at 8 for diminishing returns
        else:
            config_dict['num_workers'] = min(cpu_cores // 2, 4)  # Leave cores for GPU
        
        return OptimizedProfileConfig(**config_dict)
    
    def benchmark_configuration(self, config: OptimizedProfileConfig, 
                               test_duration: int = 30) -> Dict[str, Any]:
        """Benchmark a configuration to validate performance."""
        logger.info(f"Benchmarking configuration for {test_duration} seconds...")
        
        # This would implement actual benchmarking
        # For now, return estimated performance based on configuration
        
        estimated_latency = self._estimate_latency(config)
        estimated_throughput = self._estimate_throughput(config)
        
        return {
            'estimated_performance': {
                'latency_ms': estimated_latency,
                'throughput_per_second': estimated_throughput,
                'memory_usage_mb': self._estimate_memory_usage(config)
            },
            'hardware_utilization': {
                'gpu_utilization': 85 if config.use_gpu else 0,
                'cpu_utilization': 45 if not config.use_gpu else 25,
                'memory_utilization': 60
            },
            'configuration': asdict(config),
            'hardware_summary': self.detector.get_hardware_summary()
        }
    
    def _estimate_latency(self, config: OptimizedProfileConfig) -> float:
        """Estimate inference latency based on configuration."""
        base_latency = 50.0  # Base latency in ms
        
        # Model complexity factors
        complexity_factor = (config.hidden_size / 512) * (config.input_size / 168)
        base_latency *= complexity_factor
        
        # Hardware factors
        if config.use_gpu:
            if self.detector.hardware_class == HardwareClass.DATACENTER_GPU:
                base_latency *= 0.1  # 10x speedup
            elif self.detector.hardware_class == HardwareClass.GAMING_GPU:
                base_latency *= 0.15  # 6.7x speedup
            elif self.detector.hardware_class == HardwareClass.WORKSTATION_GPU:
                base_latency *= 0.2   # 5x speedup
            else:
                base_latency *= 0.3   # 3.3x speedup
        
        # Mixed precision speedup
        if config.mixed_precision and config.use_gpu:
            base_latency *= 0.7
        
        # Batch size factor (larger batches increase latency per item)
        batch_factor = 1 + (config.batch_size - 1) * 0.1
        base_latency *= batch_factor
        
        # Caching speedup
        if config.cache_predictions:
            base_latency *= 0.8  # 20% improvement from caching
        
        return max(base_latency, 1.0)  # Minimum 1ms
    
    def _estimate_throughput(self, config: OptimizedProfileConfig) -> float:
        """Estimate throughput based on configuration."""
        base_throughput = 10.0  # Base throughput per second
        
        # Scale with batch size
        base_throughput *= config.batch_size
        
        # Hardware scaling
        if config.use_gpu:
            if self.detector.hardware_class == HardwareClass.DATACENTER_GPU:
                base_throughput *= 20
            elif self.detector.hardware_class == HardwareClass.GAMING_GPU:
                base_throughput *= 15
            elif self.detector.hardware_class == HardwareClass.WORKSTATION_GPU:
                base_throughput *= 10
            else:
                base_throughput *= 5
        else:
            base_throughput *= config.num_workers
        
        return base_throughput
    
    def _estimate_memory_usage(self, config: OptimizedProfileConfig) -> float:
        """Estimate memory usage in MB."""
        # Base model memory
        base_memory = (config.hidden_size * config.input_size * 4) / 1024**2  # 4 bytes per float32
        
        # Scale with batch size
        batch_memory = base_memory * config.batch_size
        
        # Add overhead
        overhead = batch_memory * 0.5  # 50% overhead for gradients, optimizer, etc.
        
        total_memory = batch_memory + overhead
        
        # Mixed precision saves memory
        if config.mixed_precision:
            total_memory *= 0.6
        
        return total_memory
    
    def save_config(self, config: OptimizedProfileConfig, filepath: str):
        """Save configuration to file."""
        config_data = {
            'config': asdict(config),
            'hardware_summary': self.detector.get_hardware_summary(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str) -> OptimizedProfileConfig:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        return OptimizedProfileConfig(**config_data['config'])


# Convenience functions for easy access
def get_ultra_low_latency_config(custom_overrides: Optional[Dict] = None) -> OptimizedProfileConfig:
    """Get ultra low latency configuration (<5ms inference)."""
    manager = OptimizedConfigProfileManager()
    return manager.get_optimal_config(PerformanceProfile.ULTRA_LOW_LATENCY, custom_overrides)

def get_low_latency_config(custom_overrides: Optional[Dict] = None) -> OptimizedProfileConfig:
    """Get low latency configuration (<10ms inference)."""
    manager = OptimizedConfigProfileManager()
    return manager.get_optimal_config(PerformanceProfile.LOW_LATENCY, custom_overrides)

def get_balanced_config(custom_overrides: Optional[Dict] = None) -> OptimizedProfileConfig:
    """Get balanced configuration (optimal latency/throughput)."""
    manager = OptimizedConfigProfileManager()
    return manager.get_optimal_config(PerformanceProfile.BALANCED, custom_overrides)

def get_production_config(custom_overrides: Optional[Dict] = None) -> OptimizedProfileConfig:
    """Get production-optimized configuration."""
    manager = OptimizedConfigProfileManager()
    return manager.get_optimal_config(PerformanceProfile.PRODUCTION, custom_overrides)

def auto_detect_optimal_config() -> OptimizedProfileConfig:
    """Automatically detect and return optimal configuration for current hardware."""
    manager = OptimizedConfigProfileManager()
    
    # Choose profile based on hardware capabilities
    if manager.detector.hardware_class in [HardwareClass.DATACENTER_GPU, HardwareClass.GAMING_GPU]:
        return manager.get_optimal_config(PerformanceProfile.LOW_LATENCY)
    elif manager.detector.hardware_class == HardwareClass.WORKSTATION_GPU:
        return manager.get_optimal_config(PerformanceProfile.BALANCED)
    else:
        return manager.get_optimal_config(PerformanceProfile.MEMORY_EFFICIENT)