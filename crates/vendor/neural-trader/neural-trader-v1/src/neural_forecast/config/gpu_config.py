"""
GPU Configuration Manager

Provides GPU hardware detection, configuration management, and optimization
for neural forecasting workloads with automatic performance tuning.
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import platform
import psutil

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendors."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class GPUArchitecture(Enum):
    """GPU architectures."""
    PASCAL = "Pascal"
    VOLTA = "Volta"
    TURING = "Turing"
    AMPERE = "Ampere"
    ADA_LOVELACE = "Ada Lovelace"
    HOPPER = "Hopper"
    RDNA2 = "RDNA2"
    RDNA3 = "RDNA3"
    UNKNOWN = "Unknown"


@dataclass
class GPUInfo:
    """GPU device information."""
    device_id: int
    name: str
    vendor: GPUVendor
    architecture: GPUArchitecture
    memory_total: int  # MB
    memory_free: int   # MB
    compute_capability: Optional[str] = None
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[bool] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    utilization: Optional[float] = None
    temperature: Optional[int] = None
    power_usage: Optional[int] = None


@dataclass
class GPUConfiguration:
    """GPU configuration settings."""
    enabled: bool
    device_ids: List[int]
    memory_fraction: float
    enable_growth: bool
    mixed_precision: bool
    benchmark_mode: bool
    optimization_level: str
    batch_size_scaling: bool
    gradient_accumulation: bool
    multi_gpu_strategy: str


class GPUConfigManager:
    """
    Manages GPU configuration with automatic hardware detection,
    performance optimization, and resource management.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize GPU configuration manager.
        
        Args:
            config_file: Path to GPU configuration file
        """
        self.config_file = Path(config_file) if config_file else Path.home() / ".neural_forecast" / "gpu_config.json"
        self.detected_gpus = []
        self.current_config = None
        self.performance_profiles = {}
        
        self._detect_hardware()
        self._create_performance_profiles()
        self._load_or_create_config()
        
        logger.info(f"GPU configuration manager initialized with {len(self.detected_gpus)} GPUs")
    
    def _detect_hardware(self):
        """Detect available GPU hardware."""
        self.detected_gpus = []
        
        # Try NVIDIA GPUs first
        nvidia_gpus = self._detect_nvidia_gpus()
        self.detected_gpus.extend(nvidia_gpus)
        
        # Try AMD GPUs
        amd_gpus = self._detect_amd_gpus()
        self.detected_gpus.extend(amd_gpus)
        
        # Try Intel GPUs
        intel_gpus = self._detect_intel_gpus()
        self.detected_gpus.extend(intel_gpus)
        
        logger.info(f"Detected {len(self.detected_gpus)} GPU(s)")
        for gpu in self.detected_gpus:
            logger.info(f"  GPU {gpu.device_id}: {gpu.name} ({gpu.vendor.value})")
    
    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-ml-py or nvidia-smi."""
        gpus = []
        
        # Try using nvidia-ml-py first
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = memory_info.total // (1024 * 1024)  # Convert to MB
                memory_free = memory_info.free // (1024 * 1024)
                
                # Driver and CUDA version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                except:
                    driver_version = None
                
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    cuda_version = None
                
                # Compute capability
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = None
                
                # Current utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                except:
                    gpu_util = None
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
                except:
                    power_usage = None
                
                # Determine architecture
                architecture = self._determine_nvidia_architecture(name, compute_capability)
                
                # Check for Tensor Cores
                tensor_cores = self._has_tensor_cores(architecture, compute_capability)
                
                gpu_info = GPUInfo(
                    device_id=i,
                    name=name,
                    vendor=GPUVendor.NVIDIA,
                    architecture=architecture,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    compute_capability=compute_capability,
                    tensor_cores=tensor_cores,
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    utilization=gpu_util,
                    temperature=temperature,
                    power_usage=power_usage
                )
                
                gpus.append(gpu_info)
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            # Fallback to nvidia-smi
            gpus = self._detect_nvidia_gpus_smi()
        except Exception as e:
            logger.debug(f"Error detecting NVIDIA GPUs with pynvml: {e}")
            gpus = self._detect_nvidia_gpus_smi()
        
        return gpus
    
    def _detect_nvidia_gpus_smi(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi command."""
        gpus = []
        
        try:
            # Run nvidia-smi to get GPU information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        device_id = int(parts[0])
                        name = parts[1]
                        memory_total = int(parts[2])
                        memory_free = int(parts[3])
                        driver_version = parts[4]
                        compute_capability = parts[5]
                        
                        utilization = None
                        temperature = None
                        power_usage = None
                        
                        if len(parts) > 6 and parts[6] != '[Not Supported]':
                            try:
                                utilization = float(parts[6])
                            except:
                                pass
                        
                        if len(parts) > 7 and parts[7] != '[Not Supported]':
                            try:
                                temperature = int(float(parts[7]))
                            except:
                                pass
                        
                        if len(parts) > 8 and parts[8] != '[Not Supported]':
                            try:
                                power_usage = int(float(parts[8]))
                            except:
                                pass
                        
                        architecture = self._determine_nvidia_architecture(name, compute_capability)
                        tensor_cores = self._has_tensor_cores(architecture, compute_capability)
                        
                        gpu_info = GPUInfo(
                            device_id=device_id,
                            name=name,
                            vendor=GPUVendor.NVIDIA,
                            architecture=architecture,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            compute_capability=compute_capability,
                            tensor_cores=tensor_cores,
                            driver_version=driver_version,
                            utilization=utilization,
                            temperature=temperature,
                            power_usage=power_usage
                        )
                        
                        gpus.append(gpu_info)
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("nvidia-smi not available or failed")
        except Exception as e:
            logger.debug(f"Error running nvidia-smi: {e}")
        
        return gpus
    
    def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi or other methods."""
        gpus = []
        
        try:
            # Try rocm-smi for AMD GPUs
            result = subprocess.run(['rocm-smi', '--showid', '--showproductname', '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse rocm-smi output (implementation would need to be added based on actual output format)
                pass
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("rocm-smi not available")
        except Exception as e:
            logger.debug(f"Error detecting AMD GPUs: {e}")
        
        return gpus
    
    def _detect_intel_gpus(self) -> List[GPUInfo]:
        """Detect Intel GPUs."""
        gpus = []
        
        # Intel GPU detection would be implemented here
        # This might involve checking for Intel Arc or integrated graphics
        
        return gpus
    
    def _determine_nvidia_architecture(self, name: str, compute_capability: Optional[str]) -> GPUArchitecture:
        """Determine NVIDIA GPU architecture from name and compute capability."""
        name_lower = name.lower()
        
        # RTX 40 series (Ada Lovelace)
        if any(card in name_lower for card in ['rtx 40', 'rtx 4090', 'rtx 4080', 'rtx 4070', 'rtx 4060']):
            return GPUArchitecture.ADA_LOVELACE
        
        # H100, A100 (Hopper/Ampere)
        if any(card in name_lower for card in ['h100', 'a100']):
            if 'h100' in name_lower:
                return GPUArchitecture.HOPPER
            else:
                return GPUArchitecture.AMPERE
        
        # RTX 30 series (Ampere)
        if any(card in name_lower for card in ['rtx 30', 'rtx 3090', 'rtx 3080', 'rtx 3070', 'rtx 3060', 'a40', 'a30', 'a10']):
            return GPUArchitecture.AMPERE
        
        # RTX 20 series, GTX 16 series (Turing)
        if any(card in name_lower for card in ['rtx 20', 'rtx 2080', 'rtx 2070', 'rtx 2060', 'gtx 16', 'gtx 1660', 'gtx 1650', 't4']):
            return GPUArchitecture.TURING
        
        # Titan V, V100 (Volta)
        if any(card in name_lower for card in ['titan v', 'v100']):
            return GPUArchitecture.VOLTA
        
        # GTX 10 series, Titan X (Pascal)
        if any(card in name_lower for card in ['gtx 10', 'gtx 1080', 'gtx 1070', 'gtx 1060', 'gtx 1050', 'titan x', 'p40', 'p100']):
            return GPUArchitecture.PASCAL
        
        # Use compute capability as fallback
        if compute_capability:
            major = float(compute_capability.split('.')[0])
            if major >= 9.0:
                return GPUArchitecture.HOPPER
            elif major >= 8.6:
                return GPUArchitecture.ADA_LOVELACE
            elif major >= 8.0:
                return GPUArchitecture.AMPERE
            elif major >= 7.5:
                return GPUArchitecture.TURING
            elif major >= 7.0:
                return GPUArchitecture.VOLTA
            elif major >= 6.0:
                return GPUArchitecture.PASCAL
        
        return GPUArchitecture.UNKNOWN
    
    def _has_tensor_cores(self, architecture: GPUArchitecture, compute_capability: Optional[str]) -> Optional[bool]:
        """Determine if GPU has Tensor Cores."""
        if architecture in [GPUArchitecture.VOLTA, GPUArchitecture.TURING, GPUArchitecture.AMPERE, 
                          GPUArchitecture.ADA_LOVELACE, GPUArchitecture.HOPPER]:
            return True
        
        if compute_capability:
            major = float(compute_capability.split('.')[0])
            return major >= 7.0
        
        return None
    
    def _create_performance_profiles(self):
        """Create GPU performance profiles for different use cases."""
        self.performance_profiles = {
            'development': {
                'memory_fraction': 0.5,
                'enable_growth': True,
                'mixed_precision': False,
                'benchmark_mode': False,
                'batch_size_scaling': False,
                'gradient_accumulation': False,
                'optimization_level': 'basic'
            },
            'training': {
                'memory_fraction': 0.8,
                'enable_growth': True,
                'mixed_precision': True,
                'benchmark_mode': True,
                'batch_size_scaling': True,
                'gradient_accumulation': True,
                'optimization_level': 'aggressive'
            },
            'inference': {
                'memory_fraction': 0.6,
                'enable_growth': False,
                'mixed_precision': True,
                'benchmark_mode': True,
                'batch_size_scaling': False,
                'gradient_accumulation': False,
                'optimization_level': 'inference'
            },
            'production': {
                'memory_fraction': 0.7,
                'enable_growth': False,
                'mixed_precision': True,
                'benchmark_mode': True,
                'batch_size_scaling': True,
                'gradient_accumulation': False,
                'optimization_level': 'production'
            }
        }
    
    def _load_or_create_config(self):
        """Load existing configuration or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                self.current_config = GPUConfiguration(**config_data)
                logger.info("Loaded existing GPU configuration")
            except Exception as e:
                logger.warning(f"Error loading GPU config: {e}, creating default")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default GPU configuration."""
        if self.detected_gpus:
            # Use first available GPU by default
            device_ids = [0]
            enabled = True
            
            # Determine best profile based on hardware
            best_gpu = self.detected_gpus[0]
            if best_gpu.memory_total > 8000:  # > 8GB
                profile = 'training'
            elif best_gpu.memory_total > 4000:  # > 4GB
                profile = 'inference'
            else:
                profile = 'development'
        else:
            device_ids = []
            enabled = False
            profile = 'development'
        
        profile_config = self.performance_profiles[profile]
        
        self.current_config = GPUConfiguration(
            enabled=enabled,
            device_ids=device_ids,
            memory_fraction=profile_config['memory_fraction'],
            enable_growth=profile_config['enable_growth'],
            mixed_precision=profile_config['mixed_precision'],
            benchmark_mode=profile_config['benchmark_mode'],
            optimization_level=profile_config['optimization_level'],
            batch_size_scaling=profile_config['batch_size_scaling'],
            gradient_accumulation=profile_config['gradient_accumulation'],
            multi_gpu_strategy='data_parallel' if len(device_ids) > 1 else 'single'
        )
        
        self.save_config()
        logger.info(f"Created default GPU configuration with profile: {profile}")
    
    def get_gpu_info(self, device_id: Optional[int] = None) -> Union[List[GPUInfo], GPUInfo, None]:
        """
        Get GPU information.
        
        Args:
            device_id: Specific device ID (None for all GPUs)
            
        Returns:
            GPU information
        """
        if device_id is None:
            return self.detected_gpus
        
        for gpu in self.detected_gpus:
            if gpu.device_id == device_id:
                return gpu
        
        return None
    
    def get_current_config(self) -> GPUConfiguration:
        """Get current GPU configuration."""
        return self.current_config
    
    def update_config(self, **kwargs) -> bool:
        """
        Update GPU configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            True if updated successfully
        """
        try:
            # Update configuration
            for key, value in kwargs.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
            
            # Validate configuration
            if self._validate_config():
                self.save_config()
                return True
            else:
                logger.error("Configuration validation failed")
                return False
        
        except Exception as e:
            logger.error(f"Error updating GPU configuration: {e}")
            return False
    
    def apply_profile(self, profile_name: str) -> bool:
        """
        Apply a performance profile.
        
        Args:
            profile_name: Name of profile to apply
            
        Returns:
            True if applied successfully
        """
        if profile_name not in self.performance_profiles:
            logger.error(f"Unknown profile: {profile_name}")
            return False
        
        profile = self.performance_profiles[profile_name]
        
        return self.update_config(**profile)
    
    def optimize_for_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize GPU configuration for specific model.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Optimized GPU configuration suggestions
        """
        if not self.detected_gpus or not self.current_config.enabled:
            return {'enabled': False, 'reason': 'No GPU available'}
        
        suggestions = {}
        
        # Get model parameters
        batch_size = model_config.get('batch_size', 32)
        input_size = model_config.get('input_size', 168)
        hidden_size = model_config.get('hidden_size', 256)
        max_steps = model_config.get('max_steps', 500)
        
        # Estimate memory requirements (rough calculation)
        estimated_memory_mb = (batch_size * input_size * hidden_size * 4) // (1024 * 1024)  # 4 bytes per float32
        estimated_memory_mb *= 3  # Account for gradients and optimizer state
        
        best_gpu = max(self.detected_gpus, key=lambda x: x.memory_total)
        
        # Memory fraction optimization
        if estimated_memory_mb > best_gpu.memory_total * 0.9:
            suggestions['memory_fraction'] = 0.95
            suggestions['enable_growth'] = True
            suggestions['batch_size_scaling'] = True
            suggestions['gradient_accumulation'] = True
        elif estimated_memory_mb > best_gpu.memory_total * 0.7:
            suggestions['memory_fraction'] = 0.8
            suggestions['enable_growth'] = True
        else:
            suggestions['memory_fraction'] = 0.7
            suggestions['enable_growth'] = False
        
        # Mixed precision optimization
        if best_gpu.tensor_cores:
            suggestions['mixed_precision'] = True
        
        # Multi-GPU optimization
        if len(self.detected_gpus) > 1 and estimated_memory_mb > best_gpu.memory_total * 0.5:
            suggestions['device_ids'] = [gpu.device_id for gpu in self.detected_gpus[:2]]
            suggestions['multi_gpu_strategy'] = 'data_parallel'
        
        # Benchmark mode for consistent input sizes
        suggestions['benchmark_mode'] = True
        
        return suggestions
    
    def _validate_config(self) -> bool:
        """Validate current configuration."""
        config = self.current_config
        
        # Check device IDs are valid
        available_ids = [gpu.device_id for gpu in self.detected_gpus]
        for device_id in config.device_ids:
            if device_id not in available_ids:
                logger.error(f"Invalid device ID: {device_id}")
                return False
        
        # Check memory fraction
        if not 0.1 <= config.memory_fraction <= 1.0:
            logger.error(f"Invalid memory fraction: {config.memory_fraction}")
            return False
        
        return True
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.current_config), f, indent=2)
            
            logger.debug("Saved GPU configuration")
        except Exception as e:
            logger.error(f"Error saving GPU configuration: {e}")
    
    def benchmark_gpu(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run GPU benchmark test.
        
        Args:
            duration_seconds: Benchmark duration
            
        Returns:
            Benchmark results
        """
        if not self.detected_gpus:
            return {'error': 'No GPU available for benchmarking'}
        
        try:
            import torch
            import time
            
            device = torch.device(f'cuda:{self.current_config.device_ids[0]}')
            
            # Tensor operations benchmark
            tensor_size = (1000, 1000)
            iterations = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                a = torch.randn(tensor_size, device=device)
                b = torch.randn(tensor_size, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                iterations += 1
            
            elapsed_time = time.time() - start_time
            ops_per_second = iterations / elapsed_time
            
            # Memory bandwidth test
            memory_size = 100 * 1024 * 1024  # 100MB
            data = torch.randn(memory_size // 4, device=device)  # 4 bytes per float
            
            start_time = time.time()
            for _ in range(10):
                data_copy = data.clone()
                torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            bandwidth_gb_s = (memory_size * 10) / (elapsed_time * 1024**3)
            
            return {
                'device': str(device),
                'compute_performance': {
                    'operations_per_second': ops_per_second,
                    'tensor_operations_tested': iterations
                },
                'memory_bandwidth': {
                    'bandwidth_gb_s': bandwidth_gb_s,
                    'test_duration_s': elapsed_time
                },
                'benchmark_duration': duration_seconds
            }
        
        except ImportError:
            return {'error': 'PyTorch not available for benchmarking'}
        except Exception as e:
            return {'error': f'Benchmark failed: {str(e)}'}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_info': {
                'processor': platform.processor(),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory_info': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'gpu_info': [asdict(gpu) for gpu in self.detected_gpus],
            'gpu_config': asdict(self.current_config)
        }
        
        # Add CUDA information if available
        try:
            import torch
            if torch.cuda.is_available():
                system_info['cuda_info'] = {
                    'version': torch.version.cuda,
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name()
                }
        except ImportError:
            pass
        
        return system_info