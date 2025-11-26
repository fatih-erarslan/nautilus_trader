"""
Neural Forecasting Test Fixtures

Common pytest fixtures for neural forecasting tests.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, MagicMock, AsyncMock
import asyncio
from pathlib import Path

# Import neural forecasting components if they exist
try:
    from plans.neuralforecast.NHITS_Implementation_Guide import (
        NHITSConfig, OptimizedNHITS, RealTimeNHITSEngine,
        MultiAssetNHITSProcessor, EventAwareNHITS
    )
    NEURAL_COMPONENTS_AVAILABLE = True
except ImportError:
    NEURAL_COMPONENTS_AVAILABLE = False
    # Create mock classes for testing
    class NHITSConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class OptimizedNHITS:
        def __init__(self, config):
            self.config = config


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session") 
def device(gpu_available):
    """Get appropriate device for testing."""
    return torch.device("cuda" if gpu_available else "cpu")


@pytest.fixture
def basic_nhits_config():
    """Basic NHITS configuration for testing."""
    return NHITSConfig(
        h=24,  # 24-hour forecast horizon
        input_size=168,  # 1 week of hourly data
        n_freq_downsample=[4, 2, 1],
        n_pool_kernel_size=[4, 2, 1],
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=10,  # Reduced for testing
        early_stop_patience=5,
        use_gpu=torch.cuda.is_available(),
        mixed_precision=False,  # Disabled for testing stability
        prediction_interval=5,
        confidence_levels=[0.1, 0.5, 0.9]
    )


@pytest.fixture
def high_performance_nhits_config():
    """High-performance NHITS configuration for stress testing."""
    return NHITSConfig(
        h=96,  # 4-day forecast horizon
        input_size=720,  # 30 days of hourly data
        n_freq_downsample=[8, 4, 1],
        n_pool_kernel_size=[8, 4, 1],
        batch_size=256,
        learning_rate=1e-3,
        max_epochs=50,
        early_stop_patience=10,
        use_gpu=torch.cuda.is_available(),
        mixed_precision=True,
        prediction_interval=1,  # 1-minute intervals
        confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
    )


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    # Create 30 days of hourly data
    n_points = 24 * 30
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        periods=n_points,
        freq='H'
    )
    
    # Generate synthetic price data with trend, seasonality, and noise
    t = np.arange(n_points)
    trend = 100 + 0.01 * t  # Upward trend
    daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    weekly_seasonality = 3 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
    noise = np.random.normal(0, 1, n_points)
    
    prices = trend + daily_seasonality + weekly_seasonality + noise
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': np.random.lognormal(10, 0.5, n_points),
        'returns': np.diff(np.log(prices), prepend=0)
    })


@pytest.fixture
def multi_asset_data():
    """Generate multi-asset time series data."""
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    data = {}
    
    for asset in assets:
        n_points = 24 * 7  # 1 week of hourly data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            periods=n_points,
            freq='H'
        )
        
        # Asset-specific characteristics
        base_price = np.random.uniform(50, 300)
        volatility = np.random.uniform(0.5, 2.0)
        
        t = np.arange(n_points)
        trend = base_price + np.random.uniform(-0.1, 0.1) * t
        seasonality = volatility * np.sin(2 * np.pi * t / 24)
        noise = np.random.normal(0, volatility, n_points)
        
        prices = trend + seasonality + noise
        
        data[asset] = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': np.random.lognormal(8, 0.5, n_points),
            'returns': np.diff(np.log(prices), prepend=0)
        })
    
    return data


@pytest.fixture
def news_events_data():
    """Generate sample news events data."""
    events = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(20):  # 20 news events over the week
        event_time = base_time + timedelta(
            hours=np.random.randint(0, 24*7)
        )
        
        events.append({
            'timestamp': event_time,
            'title': f'News Event {i+1}',
            'content': f'Sample news content for event {i+1}',
            'sentiment_score': np.random.uniform(-1, 1),
            'magnitude': np.random.uniform(0, 1),
            'asset': np.random.choice(['AAPL', 'GOOGL', 'MSFT']),
            'category': np.random.choice(['earnings', 'product', 'management', 'market'])
        })
    
    return pd.DataFrame(events)


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for testing MCP tool integration."""
    client = AsyncMock()
    
    # Mock successful responses
    client.call_tool.return_value = {
        'status': 'success',
        'result': {
            'forecast': np.random.randn(24).tolist(),
            'confidence_intervals': {
                '80%': {
                    'lower': np.random.randn(24).tolist(),
                    'upper': np.random.randn(24).tolist()
                },
                '95%': {
                    'lower': np.random.randn(24).tolist(),
                    'upper': np.random.randn(24).tolist()
                }
            },
            'metadata': {
                'inference_time_ms': 15.5,
                'model_version': 'nhits_v1.0',
                'timestamp': datetime.now().isoformat()
            }
        }
    }
    
    return client


@pytest.fixture
def mock_neural_forecaster():
    """Mock neural forecaster for testing integration."""
    forecaster = Mock()
    
    # Mock async methods
    async def mock_fit(df, val_size=None):
        return {
            'training_time': 100,
            'final_loss': 0.05,
            'validation_loss': 0.07
        }
    
    async def mock_predict(df, level=[80, 95]):
        n_forecasts = 24
        return pd.DataFrame({
            'timestamp': pd.date_range(
                start=datetime.now(),
                periods=n_forecasts,
                freq='H'
            ),
            'forecast': np.random.randn(n_forecasts),
            'forecast_lo_80': np.random.randn(n_forecasts),
            'forecast_hi_80': np.random.randn(n_forecasts),
            'forecast_lo_95': np.random.randn(n_forecasts),
            'forecast_hi_95': np.random.randn(n_forecasts)
        })
    
    async def mock_cross_validate(df, n_windows=5, step_size=24):
        return pd.DataFrame({
            'window': range(n_windows),
            'mae': np.random.uniform(0.01, 0.1, n_windows),
            'mape': np.random.uniform(1, 5, n_windows),
            'rmse': np.random.uniform(0.02, 0.12, n_windows)
        })
    
    forecaster.fit = mock_fit
    forecaster.predict = mock_predict
    forecaster.cross_validate = mock_cross_validate
    forecaster._is_fitted = True
    
    return forecaster


@pytest.fixture
def performance_baseline():
    """Performance baseline metrics for regression testing."""
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
def temp_model_dir(tmp_path):
    """Temporary directory for model storage during tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Create some dummy model files
    (model_dir / "nhits_v1.0.pt").write_text("dummy model")
    (model_dir / "nhits_v1.1.pt").write_text("dummy model v1.1")
    (model_dir / "config.json").write_text('{"version": "1.0"}')
    
    return model_dir


@pytest.fixture
def gpu_memory_monitor():
    """GPU memory monitor for testing memory leaks."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available for memory monitoring")
    
    class GPUMemoryMonitor:
        def __init__(self):
            self.initial_memory = torch.cuda.memory_allocated()
            self.peak_memory = self.initial_memory
        
        def checkpoint(self, name: str = ""):
            current = torch.cuda.memory_allocated()
            self.peak_memory = max(self.peak_memory, current)
            print(f"GPU Memory {name}: {current / 1024**2:.1f} MB")
        
        def assert_no_leak(self, tolerance_mb: float = 10.0):
            final_memory = torch.cuda.memory_allocated()
            leak = (final_memory - self.initial_memory) / 1024**2
            assert leak <= tolerance_mb, f"Memory leak detected: {leak:.1f} MB"
        
        def clear_cache(self):
            torch.cuda.empty_cache()
    
    monitor = GPUMemoryMonitor()
    yield monitor
    monitor.clear_cache()


@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0  # 30 seconds


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        'warmup_iterations': 10,
        'benchmark_iterations': 100,
        'batch_sizes': [1, 16, 64, 256],
        'acceptable_variance': 0.2,  # 20% variance allowed
        'timeout_seconds': 60
    }


@pytest.fixture
def stress_test_config():
    """Configuration for stress tests."""
    return {
        'duration_seconds': 60,
        'concurrent_requests': 100,
        'ramp_up_seconds': 10,
        'error_rate_threshold': 0.05,  # Max 5% errors
        'memory_growth_threshold_mb': 100
    }


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=[
    {'batch_size': 1, 'sequence_length': 60},
    {'batch_size': 16, 'sequence_length': 168},
    {'batch_size': 64, 'sequence_length': 720},
])
def varied_input_sizes(request):
    """Varied input sizes for testing scalability."""
    return request.param


@pytest.fixture(params=[True, False])
def gpu_cpu_modes(request):
    """Test both GPU and CPU modes."""
    if request.param and not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return request.param


# Test data generators
class TestDataGenerator:
    """Generate various types of test data."""
    
    @staticmethod
    def generate_tensor_data(batch_size: int, sequence_length: int, 
                           device: torch.device) -> torch.Tensor:
        """Generate random tensor data."""
        return torch.randn(batch_size, sequence_length, device=device)
    
    @staticmethod
    def generate_market_scenario(scenario_type: str) -> pd.DataFrame:
        """Generate specific market scenarios."""
        scenarios = {
            'bull_market': lambda: np.random.normal(0.01, 0.02, 1000),
            'bear_market': lambda: np.random.normal(-0.01, 0.03, 1000), 
            'sideways': lambda: np.random.normal(0, 0.01, 1000),
            'volatile': lambda: np.random.normal(0, 0.05, 1000),
            'crash': lambda: np.concatenate([
                np.random.normal(0, 0.01, 800),
                np.random.normal(-0.1, 0.05, 200)  # Crash period
            ])
        }
        
        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_type}")
        
        returns = scenarios[scenario_type]()
        prices = 100 * np.exp(np.cumsum(returns))
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=len(prices)),
            periods=len(prices),
            freq='H'
        )
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'returns': returns,
            'volatility': np.abs(returns)
        })


@pytest.fixture
def test_data_generator():
    """Test data generator instance."""
    return TestDataGenerator()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Automatically cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
    # Reset again after test
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)