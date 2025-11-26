"""
Neural Forecasting Test Suite

Comprehensive test suite for neural forecasting components including:
- NHITS model integration
- MCP tool neural extensions  
- CLI command testing
- Performance benchmarking
- GPU/CPU compatibility
- Data pipeline validation
- Model lifecycle management
- Stress testing
"""

__version__ = "1.0.0"
__author__ = "AI News Trading Platform Team"

# Test configuration
TEST_CONFIG = {
    "gpu_tests_enabled": True,
    "performance_thresholds": {
        "inference_latency_ms": 50,
        "throughput_predictions_per_sec": 1000,
        "memory_usage_mb": 512,
        "accuracy_mape_threshold": 5.0
    },
    "test_data_samples": 1000,
    "benchmark_iterations": 100
}

# Import test utilities
from .utils.fixtures import *
from .utils.data_generators import *
from .utils.gpu_utils import *
from .utils.mock_objects import *

__all__ = [
    "TEST_CONFIG",
    # Utilities will be imported via star imports above
]