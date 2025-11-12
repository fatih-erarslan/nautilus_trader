"""
Integration tests for HyperPhysics PyTorch GPU bridge.

Tests all components:
1. ROCm device configuration
2. Order book GPU processing
3. Risk calculations
4. Greeks computation
5. Freqtrade integration

Run with:
    pytest tests/python/test_torch_integration.py -v
    pytest tests/python/test_torch_integration.py --benchmark-only
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add python module to path
python_dir = Path(__file__).parent.parent.parent / "python"
sys.path.insert(0, str(python_dir))

from hyperphysics_torch import (
    HyperbolicOrderBook,
    GPURiskEngine,
    get_device_info
)
from rocm_setup import ROCmConfig
from integration_example import HyperPhysicsFinancialEngine


# Fixtures
@pytest.fixture(scope="module")
def device():
    """Get available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


@pytest.fixture(scope="module")
def sample_order_book_data():
    """Generate sample order book data."""
    bids = np.array([
        [100.0, 10.0],
        [99.5, 15.0],
        [99.0, 20.0],
        [98.5, 25.0],
        [98.0, 30.0]
    ], dtype=np.float32)

    asks = np.array([
        [100.5, 12.0],
        [101.0, 18.0],
        [101.5, 23.0],
        [102.0, 28.0],
        [102.5, 33.0]
    ], dtype=np.float32)

    return bids, asks


@pytest.fixture(scope="module")
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    return np.random.randn(1000) * 0.02  # 2% daily volatility


class TestDeviceInfo:
    """Test device detection and configuration."""

    def test_get_device_info(self):
        """Test device information retrieval."""
        info = get_device_info()

        assert 'cuda_available' in info
        assert 'device_count' in info
        assert 'torch_version' in info

        if info['cuda_available']:
            assert info['device_count'] > 0
            assert 'device_name' in info
            assert 'total_memory' in info

    def test_device_availability(self, device):
        """Test device is accessible."""
        if device == "cuda:0":
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() > 0
        else:
            assert device == "cpu"


class TestROCmConfig:
    """Test ROCm configuration and optimization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_rocm_initialization(self):
        """Test ROCm configuration initialization."""
        config = ROCmConfig(device_id=0)

        assert config.device is not None
        assert config.device_properties is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        config = ROCmConfig(device_id=0)

        batch_size = config.get_optimal_batch_size((100, 2), torch.float32)

        assert batch_size >= 16
        assert batch_size <= 1024
        # Should be power of 2
        assert batch_size & (batch_size - 1) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_memory_stats(self):
        """Test GPU memory statistics."""
        config = ROCmConfig(device_id=0)

        stats = config.get_memory_stats()

        assert 'allocated_gb' in stats
        assert 'reserved_gb' in stats
        assert 'total_gb' in stats
        assert 'free_gb' in stats
        assert 'utilization_percent' in stats

        assert stats['total_gb'] > 0
        assert stats['free_gb'] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    @pytest.mark.slow
    def test_performance_benchmark(self):
        """Test GPU performance benchmarking."""
        config = ROCmConfig(device_id=0)

        results = config.benchmark_performance(size=1024, iterations=10)

        assert 'elapsed_time' in results
        assert 'gflops' in results
        assert 'memory_bandwidth_gb_s' in results

        assert results['gflops'] > 0
        assert results['elapsed_time'] > 0


class TestHyperbolicOrderBook:
    """Test GPU-accelerated order book processing."""

    def test_initialization(self, device):
        """Test order book initialization."""
        ob = HyperbolicOrderBook(device=device, max_levels=100)

        assert ob.device.type in ['cuda', 'cpu']
        assert ob.max_levels == 100
        assert ob.decay_lambda == 1.0

    def test_update_basic(self, device, sample_order_book_data):
        """Test basic order book update."""
        ob = HyperbolicOrderBook(device=device)
        bids, asks = sample_order_book_data

        state = ob.update(bids, asks, apply_hyperbolic=False)

        assert state['best_bid'] is not None
        assert state['best_ask'] is not None
        assert state['total_bid_qty'] > 0
        assert state['total_ask_qty'] > 0

    def test_update_with_hyperbolic(self, device, sample_order_book_data):
        """Test order book update with hyperbolic distance modeling."""
        ob = HyperbolicOrderBook(device=device)
        bids, asks = sample_order_book_data

        state = ob.update(bids, asks, apply_hyperbolic=True)

        # With hyperbolic decay, quantities should be adjusted
        assert state['best_bid'] is not None
        assert state['best_ask'] is not None
        assert state['spread'] is not None

        # Spread should be positive
        assert state['spread'] > 0

    def test_hyperbolic_mapping(self, device, sample_order_book_data):
        """Test price mapping to hyperbolic coordinates."""
        ob = HyperbolicOrderBook(device=device)
        bids, asks = sample_order_book_data

        ob.update(bids, asks, apply_hyperbolic=True)

        # Check coordinates are in unit disk
        coords = ob.bid_coords[:len(bids)]
        norms = torch.sqrt(torch.sum(coords ** 2, dim=-1))

        # All coordinates should be in unit disk (|x| < 1)
        assert torch.all(norms < 1.0)

    def test_state_consistency(self, device, sample_order_book_data):
        """Test order book state consistency."""
        ob = HyperbolicOrderBook(device=device)
        bids, asks = sample_order_book_data

        state = ob.update(bids, asks)

        # Best bid should be less than best ask
        if state['best_bid'] is not None and state['best_ask'] is not None:
            best_bid = state['best_bid'].item()
            best_ask = state['best_ask'].item()
            assert best_bid < best_ask

            # Mid price should be between bid and ask
            if 'mid_price' in state and state['mid_price'] is not None:
                mid = state['mid_price'].item()
                assert best_bid < mid < best_ask

    @pytest.mark.benchmark
    def test_update_performance(self, device, sample_order_book_data, benchmark):
        """Benchmark order book update performance."""
        ob = HyperbolicOrderBook(device=device)
        bids, asks = sample_order_book_data

        def update_orderbook():
            return ob.update(bids, asks, apply_hyperbolic=True)

        result = benchmark(update_orderbook)
        assert result is not None


class TestGPURiskEngine:
    """Test GPU-accelerated risk calculations."""

    def test_initialization(self, device):
        """Test risk engine initialization."""
        engine = GPURiskEngine(device=device, mc_simulations=1000)

        assert engine.device.type in ['cuda', 'cpu']
        assert engine.mc_simulations == 1000

    def test_var_monte_carlo(self, device, sample_returns):
        """Test Monte Carlo VaR calculation."""
        engine = GPURiskEngine(device=device, mc_simulations=1000)

        var_95, es_95 = engine.var_monte_carlo(sample_returns, confidence=0.95)

        # VaR should be positive
        assert var_95 > 0

        # Expected Shortfall should be >= VaR
        assert es_95 >= var_95

        # VaR should be reasonable (< 10% for 95% confidence)
        assert var_95 < 0.1

    def test_var_confidence_levels(self, device, sample_returns):
        """Test VaR at different confidence levels."""
        engine = GPURiskEngine(device=device, mc_simulations=1000)

        var_95, _ = engine.var_monte_carlo(sample_returns, confidence=0.95)
        var_99, _ = engine.var_monte_carlo(sample_returns, confidence=0.99)

        # VaR at 99% should be higher than at 95%
        assert var_99 > var_95

    def test_greeks_calculation(self, device):
        """Test option Greeks calculation."""
        engine = GPURiskEngine(device=device)

        greeks = engine.calculate_greeks(
            spot=100.0,
            strike=100.0,  # ATM
            volatility=0.2,
            time_to_expiry=1.0,
            risk_free_rate=0.05
        )

        # Check all Greeks are present
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks
        assert 'price' in greeks

        # ATM call should have delta around 0.5
        assert 0.4 < greeks['delta'] < 0.6

        # Gamma should be positive
        assert greeks['gamma'] > 0

        # Vega should be positive
        assert greeks['vega'] > 0

    def test_greeks_put_call_parity(self, device):
        """Test put-call parity for Greeks."""
        engine = GPURiskEngine(device=device)

        params = {
            'spot': 100.0,
            'strike': 100.0,
            'volatility': 0.2,
            'time_to_expiry': 1.0,
            'risk_free_rate': 0.05
        }

        call_greeks = engine.calculate_greeks(**params, option_type='call')
        put_greeks = engine.calculate_greeks(**params, option_type='put')

        # Put delta = Call delta - 1
        assert abs((put_greeks['delta'] - call_greeks['delta']) + 1.0) < 0.01

        # Gamma should be equal
        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 0.001

    @pytest.mark.benchmark
    def test_var_performance(self, device, sample_returns, benchmark):
        """Benchmark VaR calculation performance."""
        engine = GPURiskEngine(device=device, mc_simulations=10000)

        def calculate_var():
            return engine.var_monte_carlo(sample_returns, confidence=0.95)

        result = benchmark(calculate_var)
        assert result is not None


class TestHyperPhysicsFinancialEngine:
    """Test complete financial engine integration."""

    def test_initialization(self, device):
        """Test engine initialization."""
        engine = HyperPhysicsFinancialEngine(
            device=device,
            use_gpu=(device != "cpu"),
            max_levels=50
        )

        assert engine.device == device
        assert engine.order_book is not None
        assert engine.risk_engine is not None

    def test_process_market_data(self, device):
        """Test market data processing."""
        engine = HyperPhysicsFinancialEngine(device=device)

        bids = [(100.0, 10.0), (99.5, 15.0)]
        asks = [(100.5, 12.0), (101.0, 18.0)]

        state = engine.process_market_data(bids, asks)

        assert 'best_bid' in state
        assert 'best_ask' in state
        assert 'spread' in state
        assert state['best_bid'] is not None
        assert state['best_ask'] is not None

    def test_calculate_risk_metrics(self, device, sample_returns):
        """Test risk metrics calculation."""
        engine = HyperPhysicsFinancialEngine(device=device)

        metrics = engine.calculate_risk_metrics(sample_returns)

        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'expected_shortfall_95' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics

        assert metrics['var_95'] > 0
        assert metrics['volatility'] > 0

    def test_generate_trading_signals(self, device):
        """Test trading signal generation."""
        engine = HyperPhysicsFinancialEngine(device=device)

        # Create sample dataframe
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices + np.random.randn(100) * 0.2,
            'high': prices + np.abs(np.random.randn(100)) * 0.5,
            'low': prices - np.abs(np.random.randn(100)) * 0.5,
            'volume': np.random.rand(100) * 1000
        })

        signals = engine.generate_trading_signals(df, lookback=20)

        assert 'signal_long' in signals.columns
        assert 'signal_short' in signals.columns
        assert 'volatility' in signals.columns
        assert 'sharpe' in signals.columns

    def test_performance_stats(self, device):
        """Test performance statistics tracking."""
        engine = HyperPhysicsFinancialEngine(device=device)

        # Process some data
        bids = [(100.0, 10.0)]
        asks = [(100.5, 12.0)]
        engine.process_market_data(bids, asks)

        stats = engine.get_performance_stats()

        assert 'total_updates' in stats
        assert 'avg_update_time_ms' in stats
        assert stats['total_updates'] > 0


class TestFreqtradeIntegration:
    """Test freqtrade integration scenarios."""

    def test_real_time_processing(self, device):
        """Test real-time order book processing."""
        engine = HyperPhysicsFinancialEngine(device=device, max_levels=20)

        # Simulate real-time updates
        for i in range(10):
            price_base = 50000 + i * 10
            bids = [(price_base - j * 5, 0.1 * (j + 1)) for j in range(5)]
            asks = [(price_base + j * 5, 0.1 * (j + 1)) for j in range(1, 6)]

            state = engine.process_market_data(bids, asks)

            assert state['best_bid'] is not None
            assert state['best_ask'] is not None
            assert state['spread'] > 0

        # Check performance
        stats = engine.get_performance_stats()
        assert stats['total_updates'] == 10

    def test_risk_management_workflow(self, device):
        """Test complete risk management workflow."""
        engine = HyperPhysicsFinancialEngine(device=device)

        # Generate realistic returns (crypto-like)
        returns = np.random.randn(500) * 0.03  # 3% daily vol

        # Calculate risk metrics
        risk = engine.calculate_risk_metrics(returns, confidence=0.95)

        # Should get realistic crypto values
        assert 0.01 < risk['volatility'] < 2.0  # 1-200% annualized
        assert risk['var_95'] > 0

        # Calculate position sizing based on VaR
        max_loss = risk['var_95']
        account_size = 10000
        position_size = account_size * 0.01 / max_loss  # 1% risk

        assert position_size > 0


# Performance benchmarks
@pytest.mark.benchmark(group="orderbook")
def test_benchmark_orderbook_cpu(benchmark, sample_order_book_data):
    """Benchmark order book on CPU."""
    ob = HyperbolicOrderBook(device="cpu")
    bids, asks = sample_order_book_data

    benchmark(lambda: ob.update(bids, asks, apply_hyperbolic=True))


@pytest.mark.benchmark(group="orderbook")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_benchmark_orderbook_gpu(benchmark, sample_order_book_data):
    """Benchmark order book on GPU."""
    ob = HyperbolicOrderBook(device="cuda:0")
    bids, asks = sample_order_book_data

    benchmark(lambda: ob.update(bids, asks, apply_hyperbolic=True))


@pytest.mark.benchmark(group="risk")
def test_benchmark_var_cpu(benchmark, sample_returns):
    """Benchmark VaR on CPU."""
    engine = GPURiskEngine(device="cpu", mc_simulations=1000)

    benchmark(lambda: engine.var_monte_carlo(sample_returns, confidence=0.95))


@pytest.mark.benchmark(group="risk")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_benchmark_var_gpu(benchmark, sample_returns):
    """Benchmark VaR on GPU."""
    engine = GPURiskEngine(device="cuda:0", mc_simulations=10000)

    benchmark(lambda: engine.var_monte_carlo(sample_returns, confidence=0.95))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
