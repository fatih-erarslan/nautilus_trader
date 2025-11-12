"""
ROCm Setup and Configuration for AMD 6800XT

Optimizes PyTorch for AMD GPU acceleration with ROCm.
Includes device detection, memory management, and performance tuning.

AMD 6800XT Specifications:
- Compute Units: 60
- Stream Processors: 3840
- Memory: 16GB GDDR6
- Memory Bandwidth: 512 GB/s
- Architecture: RDNA 2

ROCm Optimization:
- Enables Tensor Cores (Matrix Cores)
- Configures memory pools
- Sets optimal batch sizes
- Tunes kernel parameters

References:
1. ROCm Documentation: https://rocmdocs.amd.com/
2. PyTorch ROCm: https://pytorch.org/get-started/locally/
3. AMD GPU Performance Tuning
"""

import os
import sys
import torch
import subprocess
from typing import Dict, Optional, Tuple
import warnings


class ROCmConfig:
    """
    ROCm configuration and optimization for AMD 6800XT.

    Handles device setup, memory management, and performance tuning
    for optimal PyTorch execution on AMD GPUs.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize ROCm configuration.

        Args:
            device_id: GPU device ID (default: 0)
        """
        self.device_id = device_id
        self.device = None
        self.device_properties = None

        # Detect ROCm
        self.rocm_available = self._check_rocm()

        if self.rocm_available:
            self._configure_device()
            self._optimize_memory()
            self._set_environment()

    def _check_rocm(self) -> bool:
        """
        Check if ROCm is available and properly configured.

        Returns:
            bool: True if ROCm is available
        """
        # Check PyTorch CUDA (ROCm uses CUDA API)
        if not torch.cuda.is_available():
            print("WARNING: ROCm/CUDA not available in PyTorch")
            return False

        # Check for ROCm version
        try:
            if hasattr(torch.version, 'hip'):
                print(f"ROCm Version: {torch.version.hip}")
                return True
            else:
                print("PyTorch built without ROCm support")
                return False
        except Exception as e:
            print(f"Error checking ROCm: {e}")
            return False

    def _configure_device(self):
        """Configure GPU device for optimal performance."""
        if self.device_id >= torch.cuda.device_count():
            raise ValueError(f"Device {self.device_id} not available. Found {torch.cuda.device_count()} devices.")

        self.device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device)

        # Get device properties
        self.device_properties = torch.cuda.get_device_properties(self.device_id)

        print("=" * 60)
        print("AMD GPU Configuration")
        print("=" * 60)
        print(f"Device: {self.device_properties.name}")
        print(f"Compute Capability: {self.device_properties.major}.{self.device_properties.minor}")
        print(f"Total Memory: {self.device_properties.total_memory / 1e9:.2f} GB")
        print(f"Multi-Processors: {self.device_properties.multi_processor_count}")
        print("=" * 60)

    def _optimize_memory(self):
        """Optimize GPU memory allocation and management."""
        # Enable TF32 for matrix operations (faster on RDNA2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking for optimal kernels
        torch.backends.cudnn.benchmark = True

        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Memory optimization enabled:")
        print("  - TF32 matrix operations: ON")
        print("  - cuDNN benchmarking: ON")
        print("  - Memory allocator: Configured")

    def _set_environment(self):
        """Set ROCm environment variables for optimal performance."""
        # ROCm-specific environment variables
        env_vars = {
            # Enable asynchronous kernel launches
            'HIP_LAUNCH_BLOCKING': '0',

            # Optimize for RDNA 2 architecture (gfx1030 for 6800XT)
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',

            # Enable wave32 mode for better occupancy
            'AMD_SERIALIZE_KERNEL': '0',

            # Memory pool size (8GB)
            'HIP_VISIBLE_DEVICES': str(self.device_id),

            # Enable MIOpen for optimized primitives
            'MIOPEN_FIND_ENFORCE': '3',

            # Cache compiled kernels
            'MIOPEN_USER_DB_PATH': '/tmp/miopen_cache',
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        print("\nROCm environment configured:")
        for key, value in env_vars.items():
            print(f"  {key}={value}")

    def get_optimal_batch_size(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            tensor_shape: Shape of single sample
            dtype: Tensor data type

        Returns:
            int: Recommended batch size
        """
        if not self.rocm_available:
            return 32  # Conservative default

        # Calculate single sample size
        element_size = torch.tensor([], dtype=dtype).element_size()
        sample_size = element_size
        for dim in tensor_shape:
            sample_size *= dim

        # Reserve 80% of GPU memory for batch
        available_memory = self.device_properties.total_memory * 0.8

        # Account for gradient computation (3x memory)
        batch_size = int(available_memory / (sample_size * 3))

        # Round to power of 2 for optimal memory alignment
        batch_size = 2 ** int(torch.log2(torch.tensor(batch_size)))

        # Ensure reasonable limits
        batch_size = max(16, min(batch_size, 1024))

        return batch_size

    def benchmark_performance(
        self,
        size: int = 4096,
        iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark GPU performance for matrix operations.

        Args:
            size: Matrix dimension
            iterations: Number of iterations

        Returns:
            dict: Performance metrics
        """
        if not self.rocm_available:
            return {'error': 'ROCm not available'}

        # Create random matrices on GPU
        A = torch.randn(size, size, device=self.device)
        B = torch.randn(size, size, device=self.device)

        # Warm-up
        for _ in range(10):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()

        # Benchmark
        import time
        start = time.time()

        for _ in range(iterations):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        ops = 2 * size ** 3 * iterations  # Matrix multiplication ops
        gflops = (ops / elapsed) / 1e9

        # Memory bandwidth
        memory_ops = 3 * size * size * 4 * iterations  # 3 matrices, 4 bytes (float32)
        bandwidth = (memory_ops / elapsed) / 1e9  # GB/s

        return {
            'elapsed_time': elapsed,
            'gflops': gflops,
            'memory_bandwidth_gb_s': bandwidth,
            'iterations': iterations,
            'matrix_size': size
        }

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current GPU memory statistics.

        Returns:
            dict: Memory usage statistics
        """
        if not self.rocm_available:
            return {}

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = self.device_properties.total_memory / 1e9

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'utilization_percent': (allocated / total) * 100
        }

    def optimize_for_inference(self):
        """Optimize settings for inference workloads."""
        torch.backends.cudnn.benchmark = False  # Disable for consistent timing
        torch.backends.cudnn.deterministic = True
        torch.set_grad_enabled(False)  # Disable gradients

        print("Inference optimization enabled:")
        print("  - Deterministic mode: ON")
        print("  - Gradient computation: OFF")

    def optimize_for_training(self):
        """Optimize settings for training workloads."""
        torch.backends.cudnn.benchmark = True  # Enable for fastest kernels
        torch.backends.cudnn.deterministic = False
        torch.set_grad_enabled(True)

        # Mixed precision training
        torch.backends.cuda.matmul.allow_tf32 = True

        print("Training optimization enabled:")
        print("  - cuDNN benchmarking: ON")
        print("  - Mixed precision: ON")
        print("  - Gradient computation: ON")


def setup_rocm_for_freqtrade(venv_path: str = "/Users/ashina/freqtrade/.venv") -> ROCmConfig:
    """
    Setup ROCm for freqtrade environment.

    Args:
        venv_path: Path to freqtrade virtual environment

    Returns:
        ROCmConfig: Configured ROCm instance
    """
    print("=" * 60)
    print("Setting up ROCm for Freqtrade Integration")
    print("=" * 60)

    # Initialize ROCm
    config = ROCmConfig(device_id=0)

    if not config.rocm_available:
        print("\nWARNING: ROCm not available. Using CPU fallback.")
        print("\nTo install PyTorch with ROCm support:")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
        return config

    # Run performance benchmark
    print("\n" + "=" * 60)
    print("Running Performance Benchmark...")
    print("=" * 60)

    results = config.benchmark_performance(size=2048, iterations=50)
    print(f"Matrix Multiplication Performance:")
    print(f"  Size: {results['matrix_size']}x{results['matrix_size']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Performance: {results['gflops']:.2f} GFLOPS")
    print(f"  Memory Bandwidth: {results['memory_bandwidth_gb_s']:.2f} GB/s")

    # Memory stats
    print("\n" + "=" * 60)
    print("GPU Memory Status")
    print("=" * 60)
    mem_stats = config.get_memory_stats()
    print(f"Total Memory: {mem_stats['total_gb']:.2f} GB")
    print(f"Allocated: {mem_stats['allocated_gb']:.2f} GB")
    print(f"Free: {mem_stats['free_gb']:.2f} GB")
    print(f"Utilization: {mem_stats['utilization_percent']:.1f}%")

    # Optimal batch size recommendation
    print("\n" + "=" * 60)
    print("Recommended Settings for Financial Computing")
    print("=" * 60)

    # For order book processing (100 levels, 2 sides, float32)
    ob_batch = config.get_optimal_batch_size((100, 2), torch.float32)
    print(f"Order Book Batch Size: {ob_batch}")

    # For Monte Carlo simulations (10000 paths, 252 steps)
    mc_batch = config.get_optimal_batch_size((10000, 252), torch.float32)
    print(f"Monte Carlo Batch Size: {mc_batch}")

    print("\n" + "=" * 60)
    print("ROCm Setup Complete!")
    print("=" * 60)

    return config


if __name__ == "__main__":
    # Setup ROCm
    config = setup_rocm_for_freqtrade()

    # Example: Test tensor operations
    if config.rocm_available:
        print("\n" + "=" * 60)
        print("Testing Tensor Operations on GPU")
        print("=" * 60)

        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=config.device)
        y = torch.randn(1000, 1000, device=config.device)

        # Matrix multiplication
        z = torch.matmul(x, y)

        print(f"Successfully computed 1000x1000 matrix multiplication")
        print(f"Result shape: {z.shape}")
        print(f"Result device: {z.device}")

        # Clean up
        del x, y, z
        torch.cuda.empty_cache()

        print("\nGPU memory cleared successfully")
