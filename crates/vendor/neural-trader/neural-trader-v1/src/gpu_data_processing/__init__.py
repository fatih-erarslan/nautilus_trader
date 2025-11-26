"""
GPU Data Processing and Signal Generation Optimization Framework

This module provides GPU-accelerated data processing and signal generation
components for the AI News Trading Platform, targeting 5,000x+ speedup
vs CPU implementations.

Main Components:
- GPUDataProcessor: cuDF-based market data processing
- GPUSignalGenerator: CuPy-based signal generation
- GPUFeatureEngine: GPU-accelerated feature engineering
- CUDA Kernels: Custom optimized kernels for technical indicators

Performance Targets:
- 5,000x+ speedup vs CPU
- Process 100,000+ data points in <1 second
- Real-time processing with <10ms latency
- Memory efficiency >70% on GPU
"""

from .core.gpu_data_processor import GPUDataProcessor
from .core.gpu_signal_generator import GPUSignalGenerator
from .core.gpu_feature_engine import GPUFeatureEngine
from .indicators.gpu_technical_indicators import GPUTechnicalIndicators
from .patterns.gpu_pattern_recognition import GPUPatternRecognition
from .benchmarks.performance_benchmark import GPUPerformanceBenchmark

__version__ = "1.0.0"
__author__ = "AI News Trading Platform"

__all__ = [
    "GPUDataProcessor",
    "GPUSignalGenerator", 
    "GPUFeatureEngine",
    "GPUTechnicalIndicators",
    "GPUPatternRecognition",
    "GPUPerformanceBenchmark"
]

# GPU availability check
try:
    import cudf
    import cupy as cp
    import numba.cuda
    
    GPU_AVAILABLE = numba.cuda.is_available()
    GPU_COUNT = len(numba.cuda.gpus) if GPU_AVAILABLE else 0
    
except ImportError as e:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    import warnings
    warnings.warn(f"GPU libraries not available: {e}. Falling back to CPU processing.")

# Module configuration
CONFIG = {
    "gpu_available": GPU_AVAILABLE,
    "gpu_count": GPU_COUNT,
    "default_memory_pool": "managed" if GPU_AVAILABLE else None,
    "enable_fallback": True,
    "performance_targets": {
        "speedup_factor": 5000,
        "max_processing_time": 1.0,  # seconds for 100k data points
        "max_latency": 0.01,  # 10ms for real-time
        "min_memory_efficiency": 0.70
    }
}