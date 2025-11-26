"""
Neural Forecasting Test Configuration

Shared pytest configuration and fixtures for neural forecasting tests.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import asyncio
import warnings
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import fixtures from utils
from tests.neural.utils.fixtures import *
from tests.neural.utils.data_generators import *
from tests.neural.utils.gpu_utils import *
from tests.neural.utils.mock_objects import *
from tests.neural.utils.performance_utils import *


def pytest_configure(config):
    """Configure pytest for neural forecasting tests."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests between components")
    config.addinivalue_line("markers", "performance: Performance and benchmarking tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU hardware")
    config.addinivalue_line("markers", "slow: Tests that take longer than 30 seconds")
    config.addinivalue_line("markers", "stress: Stress tests with high resource usage")
    config.addinivalue_line("markers", "regression: Regression tests against baselines")
    config.addinivalue_line("markers", "mock: Tests using mock objects only")
    config.addinivalue_line("markers", "real: Tests requiring real neural components")
    
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*GPU.*not available.*")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Auto-mark performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Auto-mark slow tests
        if "stress" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark unit vs integration tests based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow"):
        if not item.config.getoption("--runslow", default=False):
            pytest.skip("Slow test skipped (use --runslow to run)")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


def pytest_runtest_teardown(item):
    """Cleanup after individual test runs."""
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset random seeds
    np.random.seed(42)
    torch.manual_seed(42)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="Run slow tests"
    )
    parser.addoption(
        "--rungpu", 
        action="store_true", 
        default=False, 
        help="Run GPU tests even if GPU not detected"
    )
    parser.addoption(
        "--runstress", 
        action="store_true", 
        default=False, 
        help="Run stress tests"
    )
    parser.addoption(
        "--performance-baseline", 
        action="store", 
        default=None, 
        help="Path to performance baseline file for regression testing"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_output_dir():
    """Create temporary directory for test outputs."""
    output_dir = Path(tempfile.mkdtemp(prefix="neural_test_"))
    yield output_dir
    
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def performance_baseline_data(request):
    """Load performance baseline data if provided."""
    baseline_path = request.config.getoption("--performance-baseline")
    
    if baseline_path and Path(baseline_path).exists():
        import json
        with open(baseline_path, 'r') as f:
            return json.load(f)
    else:
        # Return default baseline if no file provided
        return {
            'inference_latency': {
                'mean_ms': 25.0,
                'p95_ms': 40.0,
                'p99_ms': 50.0
            },
            'throughput': {
                'predictions_per_second': 1500,
                'batches_per_second': 100
            },
            'memory_usage': {
                'gpu_memory_mb': 256,
                'cpu_memory_mb': 128
            },
            'accuracy': {
                'mae': 0.05,
                'mape': 3.5,
                'rmse': 0.08
            }
        }


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'test_data_samples': 1000,
        'benchmark_iterations': 100,
        'performance_thresholds': {
            'inference_latency_ms': 50,
            'throughput_predictions_per_sec': 1000,
            'memory_usage_mb': 512,
            'accuracy_mape_threshold': 5.0
        },
        'gpu_tests_enabled': torch.cuda.is_available(),
        'timeout_seconds': 300
    }


@pytest.fixture
def neural_test_data():
    """Generate test data for neural forecasting."""
    generator = SyntheticTimeSeriesGenerator(seed=42)
    
    # Generate different types of test data
    params_basic = TimeSeriesParams(
        n_points=1000,
        trend=0.01,
        seasonality_periods=[24, 168],
        seasonality_amplitudes=[1.0, 0.5],
        noise_level=0.1
    )
    
    params_volatile = TimeSeriesParams(
        n_points=1000,
        trend=0.005,
        seasonality_periods=[24],
        seasonality_amplitudes=[2.0],
        noise_level=0.3,
        volatility_clustering=True
    )
    
    return {
        'basic': generator.generate_single_series(params_basic),
        'volatile': generator.generate_single_series(params_volatile),
        'multi_asset': generator.generate_multi_asset_series(
            ['AAPL', 'GOOGL', 'MSFT'], params_basic
        )
    }


@pytest.fixture
def mock_neural_components():
    """Create mock neural components for testing."""
    config = MockNHITSConfig()
    
    return {
        'config': config,
        'model': create_mock_nhits_model(config),
        'engine': create_mock_real_time_engine(config),
        'mcp_server': create_mock_mcp_server(),
        'multi_processor': create_mock_multi_asset_processor(['AAPL', 'GOOGL'], config)
    }


@pytest.fixture
def benchmark_suite():
    """Create benchmark suite for performance testing."""
    config = BenchmarkConfig(
        warmup_iterations=5,  # Reduced for testing
        benchmark_iterations=20,  # Reduced for testing
        timeout_seconds=60
    )
    
    return {
        'latency': LatencyBenchmark(config),
        'throughput': ThroughputBenchmark(config),
        'memory': MemoryBenchmark(),
        'config': config
    }


@pytest.fixture
def gpu_test_environment():
    """Setup GPU testing environment."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available for testing")
    
    # Get GPU info
    detector = GPUDetector()
    gpu_info = detector.get_gpu_info()
    
    # Check minimum requirements
    if not detector.check_gpu_requirements(min_memory_gb=2.0):
        pytest.skip("GPU does not meet minimum requirements")
    
    # Setup memory tracking
    tracker = GPUMemoryTracker()
    
    yield {
        'info': gpu_info,
        'tracker': tracker,
        'detector': detector
    }
    
    # Cleanup
    torch.cuda.empty_cache()


@pytest.fixture
def temp_model_storage(tmp_path):
    """Create temporary model storage for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Create some dummy model files
    dummy_model_data = {
        'model_state': 'dummy_state',
        'config': {'h': 24, 'input_size': 168},
        'metadata': {'version': '1.0', 'created': datetime.now().isoformat()}
    }
    
    import json
    
    # Save dummy models
    for i in range(3):
        model_file = model_dir / f"model_v{i+1}.pt"
        config_file = model_dir / f"model_v{i+1}_config.json"
        
        # Dummy model file (just text for testing)
        model_file.write_text(f"dummy_model_v{i+1}")
        
        # Config file
        with open(config_file, 'w') as f:
            json.dump(dummy_model_data, f)
    
    return model_dir


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure test isolation."""
    # Clear any global state
    if hasattr(torch, '_C') and hasattr(torch._C, '_clear_cublas_workspace'):
        try:
            torch._C._clear_cublas_workspace()
        except:
            pass
    
    yield
    
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_price_series(n_points: int = 1000, **kwargs) -> pd.DataFrame:
        """Create price time series."""
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(n_points=n_points, **kwargs)
        return generator.generate_single_series(params)
    
    @staticmethod
    def create_tensor_data(shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """Create tensor data for testing."""
        return torch.randn(shape, device=device)
    
    @staticmethod
    def create_nhits_data(series: pd.DataFrame, asset_id: str = 'TEST') -> pd.DataFrame:
        """Convert series to NHITS format."""
        return prepare_nhits_format(series, asset_id)


@pytest.fixture
def data_factory():
    """Provide data factory for tests."""
    return TestDataFactory()


# Custom assertions for neural testing
def assert_neural_output_valid(output: Dict[str, torch.Tensor], expected_shape: tuple):
    """Assert neural network output is valid."""
    assert isinstance(output, dict)
    assert 'point_forecast' in output
    assert output['point_forecast'].shape == expected_shape
    assert not torch.isnan(output['point_forecast']).any()
    assert torch.isfinite(output['point_forecast']).all()


def assert_performance_acceptable(metrics: Dict[str, float], thresholds: Dict[str, float]):
    """Assert performance metrics meet thresholds."""
    for metric, value in metrics.items():
        if metric in thresholds:
            threshold = thresholds[metric]
            assert value <= threshold, f"{metric} {value} exceeds threshold {threshold}"


def assert_memory_efficient(memory_stats: Dict[str, Any], max_increase_mb: float = 100):
    """Assert memory usage is efficient."""
    if 'increase' in memory_stats:
        total_increase = (memory_stats['increase'].get('cpu_memory_mb', 0) + 
                         memory_stats['increase'].get('gpu_memory_mb', 0))
        assert total_increase <= max_increase_mb, \
            f"Memory increase {total_increase:.1f}MB > {max_increase_mb}MB"


# Add custom assertions to pytest namespace
pytest.assert_neural_output_valid = assert_neural_output_valid
pytest.assert_performance_acceptable = assert_performance_acceptable
pytest.assert_memory_efficient = assert_memory_efficient