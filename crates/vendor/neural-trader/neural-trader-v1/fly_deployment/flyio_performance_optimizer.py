#!/usr/bin/env python3
"""
Fly.io Performance Optimizer for AI News Trading Platform
Advanced GPU and system optimization specifically for Fly.io infrastructure
"""

import os
import sys
import time
import json
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import psutil
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceProfile:
    """Performance optimization profile."""
    name: str
    gpu_memory_fraction: float
    batch_size: int
    precision: str
    tensorrt_enabled: bool
    cpu_threads: int
    memory_pool_size: str
    cache_size_mb: int

class FlyIOPerformanceOptimizer:
    """Advanced performance optimizer for Fly.io GPU instances."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_configuration()
        self.system_info = self._detect_system_configuration()
        self.profiles = self._create_optimization_profiles()
        self.current_profile = None
        
    def _detect_gpu_configuration(self) -> Dict[str, Any]:
        """Detect GPU configuration and capabilities."""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": 0,
            "devices": []
        }
        
        if not gpu_info["available"]:
            return gpu_info
        
        gpu_info["device_count"] = torch.cuda.device_count()
        
        for i in range(gpu_info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "tensor_cores": self._has_tensor_cores(props.major, props.minor)
            }
            gpu_info["devices"].append(device_info)
        
        return gpu_info
    
    def _detect_system_configuration(self) -> Dict[str, Any]:
        """Detect system configuration."""
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_space_gb": psutil.disk_usage('/').total / (1024**3)
        }
    
    def _has_tensor_cores(self, major: int, minor: int) -> bool:
        """Check if GPU has Tensor Cores."""
        # Volta (7.0), Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0)
        return major >= 7 and (major > 7 or minor >= 0)
    
    def _create_optimization_profiles(self) -> Dict[str, PerformanceProfile]:
        """Create performance profiles based on detected hardware."""
        profiles = {}
        
        if not self.gpu_info["available"]:
            profiles["cpu_only"] = PerformanceProfile(
                name="cpu_only",
                gpu_memory_fraction=0.0,
                batch_size=8,
                precision="fp32",
                tensorrt_enabled=False,
                cpu_threads=self.system_info["logical_cpu_count"],
                memory_pool_size="2g",
                cache_size_mb=512
            )
            return profiles
        
        gpu_device = self.gpu_info["devices"][0]
        gpu_memory = gpu_device["total_memory_gb"]
        has_tensor_cores = gpu_device["tensor_cores"]
        
        # Ultra-Low Latency Profile (for real-time trading)
        profiles["ultra_low_latency"] = PerformanceProfile(
            name="ultra_low_latency",
            gpu_memory_fraction=0.6,
            batch_size=1,
            precision="fp16" if has_tensor_cores else "fp32",
            tensorrt_enabled=True,
            cpu_threads=4,
            memory_pool_size="1g",
            cache_size_mb=128
        )
        
        # High Throughput Profile (for batch processing)
        profiles["high_throughput"] = PerformanceProfile(
            name="high_throughput",
            gpu_memory_fraction=0.9,
            batch_size=min(64, int(gpu_memory * 8)),  # Scale with GPU memory
            precision="fp16" if has_tensor_cores else "fp32",
            tensorrt_enabled=True,
            cpu_threads=self.system_info["logical_cpu_count"],
            memory_pool_size="4g",
            cache_size_mb=1024
        )
        
        # Balanced Profile (general purpose)
        profiles["balanced"] = PerformanceProfile(
            name="balanced",
            gpu_memory_fraction=0.8,
            batch_size=min(32, int(gpu_memory * 4)),
            precision="fp16" if has_tensor_cores else "fp32",
            tensorrt_enabled=True,
            cpu_threads=max(4, self.system_info["logical_cpu_count"] // 2),
            memory_pool_size="2g",
            cache_size_mb=512
        )
        
        # Memory Optimized Profile (for large models)
        profiles["memory_optimized"] = PerformanceProfile(
            name="memory_optimized",
            gpu_memory_fraction=0.95,
            batch_size=min(16, int(gpu_memory * 2)),
            precision="fp16",
            tensorrt_enabled=False,  # TensorRT may increase memory usage
            cpu_threads=2,
            memory_pool_size="6g",
            cache_size_mb=2048
        )
        
        return profiles
    
    def select_optimal_profile(self, workload_type: str = "balanced") -> PerformanceProfile:
        """Select optimal profile based on workload type."""
        workload_mapping = {
            "real_time": "ultra_low_latency",
            "batch": "high_throughput", 
            "balanced": "balanced",
            "memory_intensive": "memory_optimized"
        }
        
        profile_name = workload_mapping.get(workload_type, "balanced")
        
        if profile_name not in self.profiles:
            profile_name = list(self.profiles.keys())[0]
        
        self.current_profile = self.profiles[profile_name]
        logger.info(f"Selected optimization profile: {profile_name}")
        return self.current_profile
    
    def apply_gpu_optimizations(self, profile: PerformanceProfile):
        """Apply GPU-specific optimizations."""
        if not self.gpu_info["available"]:
            logger.info("GPU not available, skipping GPU optimizations")
            return
        
        logger.info(f"Applying GPU optimizations for profile: {profile.name}")
        
        # Set GPU memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(profile.gpu_memory_fraction)
            torch.cuda.empty_cache()
        
        # Configure PyTorch CUDA settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{int(profile.gpu_memory_fraction * 1000)},garbage_collection_threshold:0.8"
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set precision settings
        if profile.precision == "fp16":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        logger.info("GPU optimizations applied successfully")
    
    def apply_system_optimizations(self, profile: PerformanceProfile):
        """Apply system-level optimizations."""
        logger.info(f"Applying system optimizations for profile: {profile.name}")
        
        # Set CPU thread count
        os.environ["OMP_NUM_THREADS"] = str(profile.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(profile.cpu_threads)
        os.environ["NUMBA_NUM_THREADS"] = str(profile.cpu_threads)
        
        # Configure memory settings
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:512"
        
        # Set cache directories
        cache_dir = "/app/cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = f"{cache_dir}/numba"
        os.environ["MPLCONFIGDIR"] = f"{cache_dir}/matplotlib"
        os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
        
        logger.info("System optimizations applied successfully")
    
    def optimize_nvidia_settings(self):
        """Optimize NVIDIA GPU settings for maximum performance."""
        if not self.gpu_info["available"]:
            return
        
        logger.info("Optimizing NVIDIA GPU settings...")
        
        optimizations = [
            # Set persistence mode
            ["nvidia-smi", "-pm", "1"],
            # Set maximum power limit
            ["nvidia-smi", "-pl", "400"],  # 400W for A100
            # Set application clocks for optimal performance
            ["nvidia-smi", "-ac", "1215,1410"],  # Memory, Graphics clocks for A100
            # Set compute mode to exclusive process
            ["nvidia-smi", "-c", "3"]
        ]
        
        for cmd in optimizations:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"Applied: {' '.join(cmd)}")
                else:
                    logger.warning(f"Failed to apply {' '.join(cmd)}: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
                logger.warning(f"Could not apply {' '.join(cmd)}: {e}")
    
    def benchmark_configuration(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Benchmark the current configuration."""
        logger.info(f"Benchmarking configuration: {profile.name}")
        
        results = {
            "profile": profile.name,
            "timestamp": time.time()
        }
        
        if self.gpu_info["available"]:
            results.update(self._benchmark_gpu_performance(profile))
        
        results.update(self._benchmark_cpu_performance(profile))
        results.update(self._benchmark_memory_performance())
        
        return results
    
    def _benchmark_gpu_performance(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Benchmark GPU performance."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.device("cuda:0")
        
        # Matrix multiplication benchmark
        sizes = [512, 1024, 2048]
        matmul_times = []
        
        for size in sizes:
            # Warm up
            for _ in range(5):
                a = torch.randn(size, size, device=device, dtype=torch.float16 if profile.precision == "fp16" else torch.float32)
                b = torch.randn(size, size, device=device, dtype=torch.float16 if profile.precision == "fp16" else torch.float32)
                torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            matmul_times.append(np.mean(times))
        
        # Memory bandwidth test
        memory_size = int(self.gpu_info["devices"][0]["total_memory_gb"] * 0.8 * 1024**3)
        tensor_size = memory_size // 4  # Float32 = 4 bytes
        
        start = time.time()
        large_tensor = torch.randn(tensor_size // 4, device=device)
        large_tensor = large_tensor * 2.0  # Simple operation
        torch.cuda.synchronize()
        memory_bandwidth_time = time.time() - start
        
        return {
            "gpu_matmul_512_ms": matmul_times[0] * 1000,
            "gpu_matmul_1024_ms": matmul_times[1] * 1000,
            "gpu_matmul_2048_ms": matmul_times[2] * 1000,
            "gpu_memory_bandwidth_gbs": (memory_size / memory_bandwidth_time) / (1024**3),
            "gpu_utilization": self._get_gpu_utilization()
        }
    
    def _benchmark_cpu_performance(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Benchmark CPU performance."""
        # CPU matrix multiplication
        size = 1000
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        start = time.time()
        np.matmul(a, b)
        cpu_matmul_time = time.time() - start
        
        return {
            "cpu_matmul_1000_ms": cpu_matmul_time * 1000,
            "cpu_threads_used": profile.cpu_threads,
            "cpu_utilization": psutil.cpu_percent(interval=1)
        }
    
    def _benchmark_memory_performance(self) -> Dict[str, float]:
        """Benchmark memory performance."""
        memory_info = psutil.virtual_memory()
        
        return {
            "memory_total_gb": memory_info.total / (1024**3),
            "memory_available_gb": memory_info.available / (1024**3),
            "memory_usage_percent": memory_info.percent
        }
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return 0.0
    
    def create_tensorrt_optimization(self, model_path: str, output_path: str, profile: PerformanceProfile):
        """Create TensorRT optimized model."""
        if not profile.tensorrt_enabled:
            logger.info("TensorRT optimization disabled in profile")
            return
        
        logger.info("Creating TensorRT optimized model...")
        
        try:
            # This would require the actual model to optimize
            # For now, create a placeholder script
            tensorrt_script = f"""
import torch
import tensorrt as trt
import torch_tensorrt

# Load model
model = torch.load('{model_path}')
model.eval()

# Create example input
example_input = torch.randn(1, 168, 1).cuda()  # NHITS input shape

# Convert to TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 168, 1],
        opt_shape=[{profile.batch_size}, 168, 1],
        max_shape=[{profile.batch_size * 2}, 168, 1],
        dtype=torch.{'half' if profile.precision == 'fp16' else 'float'}
    )],
    enabled_precisions={{torch.{'half' if profile.precision == 'fp16' else 'float'}}},
    workspace_size=2 << 30,  # 2GB
    min_block_size=1,
    use_python_runtime=True
)

# Save optimized model
torch.jit.save(trt_model, '{output_path}')
print("TensorRT optimization completed")
"""
            
            with open("/app/tensorrt_optimize.py", "w") as f:
                f.write(tensorrt_script)
            
            logger.info(f"TensorRT optimization script created: /app/tensorrt_optimize.py")
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
    
    def save_optimization_config(self, profile: PerformanceProfile, benchmark_results: Dict):
        """Save optimization configuration for future use."""
        config = {
            "profile": profile.__dict__,
            "benchmark_results": benchmark_results,
            "gpu_info": self.gpu_info,
            "system_info": self.system_info,
            "timestamp": time.time()
        }
        
        config_path = "/app/optimization_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Optimization configuration saved: {config_path}")
    
    def apply_full_optimization(self, workload_type: str = "balanced") -> Dict:
        """Apply complete optimization suite."""
        logger.info("🚀 Starting Fly.io GPU optimization...")
        
        # Select optimal profile
        profile = self.select_optimal_profile(workload_type)
        
        # Apply optimizations
        self.optimize_nvidia_settings()
        self.apply_gpu_optimizations(profile)
        self.apply_system_optimizations(profile)
        
        # Benchmark performance
        benchmark_results = self.benchmark_configuration(profile)
        
        # Save configuration
        self.save_optimization_config(profile, benchmark_results)
        
        logger.info("✅ Fly.io GPU optimization completed")
        
        return {
            "profile": profile.__dict__,
            "benchmark_results": benchmark_results,
            "optimization_status": "completed"
        }

def main():
    """Main entry point for performance optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fly.io Performance Optimizer")
    parser.add_argument("--workload", choices=["real_time", "batch", "balanced", "memory_intensive"], 
                       default="balanced", help="Workload type for optimization")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks only")
    parser.add_argument("--profile", help="Use specific profile")
    
    args = parser.parse_args()
    
    optimizer = FlyIOPerformanceOptimizer()
    
    if args.benchmark:
        # Run benchmarks with current configuration
        if args.profile and args.profile in optimizer.profiles:
            profile = optimizer.profiles[args.profile]
        else:
            profile = optimizer.select_optimal_profile(args.workload)
        
        results = optimizer.benchmark_configuration(profile)
        print(json.dumps(results, indent=2))
    else:
        # Apply full optimization
        results = optimizer.apply_full_optimization(args.workload)
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()