"""
Performance Benchmarks and Load Tests

This module contains comprehensive performance tests including API response
time benchmarks, concurrent connection stress tests, memory profiling,
and high-frequency trading scenario simulations.
"""

import asyncio
import gc
import json
import os
import psutil
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import aiohttp
import websockets

from polymarket.api import (
    CLOBClient, GammaClient, PolymarketClient, 
    RateLimiter, WebSocketClient
)
from polymarket.models import Market, Order, OrderSide, OrderType
from polymarket.strategies import (
    MomentumStrategy, ArbitrageStrategy, EnsembleStrategy
)
from polymarket.utils.monitoring import PerformanceMonitor


class TestPerformance:
    """Performance benchmarks and load tests."""

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitoring instance."""
        return PerformanceMonitor()

    @pytest.fixture
    async def load_test_client(self):
        """Create client configured for load testing."""
        config = {
            "api_key": "test_key",
            "base_url": "https://api.polymarket.com",
            "rate_limit": 100,  # Higher for load testing
            "timeout": 5
        }
        client = AsyncMock(spec=PolymarketClient)
        
        # Configure realistic response times
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(np.random.uniform(0.01, 0.05))  # 10-50ms
            return self._generate_mock_response(args[0] if args else None)
        
        client.get_markets.side_effect = delayed_response
        client.get_orderbook.side_effect = delayed_response
        client.place_order.side_effect = delayed_response
        
        return client

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_api_response_time_benchmarks(self, load_test_client, performance_monitor):
        """Benchmark API response times under various loads."""
        results = {
            "single_request": {},
            "burst_requests": {},
            "sustained_load": {}
        }
        
        # Test 1: Single request baseline
        start = time.time()
        await load_test_client.get_markets(limit=10)
        results["single_request"]["latency"] = (time.time() - start) * 1000  # ms
        
        # Test 2: Burst of requests
        burst_size = 50
        start = time.time()
        tasks = [load_test_client.get_markets(limit=10) for _ in range(burst_size)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        results["burst_requests"]["total_time"] = elapsed
        results["burst_requests"]["avg_latency"] = (elapsed / burst_size) * 1000
        results["burst_requests"]["requests_per_second"] = burst_size / elapsed
        
        # Test 3: Sustained load
        duration = 10  # seconds
        request_count = 0
        latencies = []
        
        start = time.time()
        while time.time() - start < duration:
            req_start = time.time()
            await load_test_client.get_markets(limit=10)
            latencies.append((time.time() - req_start) * 1000)
            request_count += 1
            await asyncio.sleep(0.01)  # 100 requests/second target
        
        results["sustained_load"]["total_requests"] = request_count
        results["sustained_load"]["avg_latency"] = np.mean(latencies)
        results["sustained_load"]["p95_latency"] = np.percentile(latencies, 95)
        results["sustained_load"]["p99_latency"] = np.percentile(latencies, 99)
        results["sustained_load"]["requests_per_second"] = request_count / duration
        
        # Verify performance meets requirements
        assert results["single_request"]["latency"] < 100  # < 100ms
        assert results["burst_requests"]["avg_latency"] < 200  # < 200ms avg
        assert results["sustained_load"]["p95_latency"] < 500  # < 500ms p95
        assert results["sustained_load"]["requests_per_second"] > 50  # > 50 RPS
        
        performance_monitor.record_benchmark("api_response_times", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_websocket_connections(self, performance_monitor):
        """Test WebSocket performance with many concurrent connections."""
        connection_counts = [10, 50, 100, 200]
        results = {}
        
        async def mock_websocket_connection(connection_id: int):
            """Simulate WebSocket connection and message handling."""
            messages_received = 0
            start_time = time.time()
            
            # Simulate connection and message stream
            for _ in range(100):  # 100 messages per connection
                await asyncio.sleep(0.01)  # Simulate message arrival
                messages_received += 1
            
            return {
                "connection_id": connection_id,
                "messages_received": messages_received,
                "duration": time.time() - start_time
            }
        
        for count in connection_counts:
            start = time.time()
            
            # Create concurrent connections
            tasks = [mock_websocket_connection(i) for i in range(count)]
            connection_results = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start
            
            # Calculate metrics
            total_messages = sum(r["messages_received"] for r in connection_results)
            avg_msg_rate = total_messages / elapsed
            
            results[f"{count}_connections"] = {
                "total_time": elapsed,
                "total_messages": total_messages,
                "messages_per_second": avg_msg_rate,
                "avg_connection_time": elapsed / count,
                "successful_connections": len(connection_results)
            }
            
            # Verify scalability
            assert len(connection_results) == count  # All connections successful
            assert avg_msg_rate > count * 5  # At least 5 msg/sec per connection
        
        performance_monitor.record_benchmark("websocket_concurrency", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_profiling(self, load_test_client, performance_monitor):
        """Profile memory usage for large datasets."""
        tracemalloc.start()
        process = psutil.Process()
        
        results = {
            "baseline": {},
            "large_dataset": {},
            "memory_leak_test": {}
        }
        
        # Baseline memory
        gc.collect()
        results["baseline"]["memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # Test 1: Large dataset handling
        large_markets = []
        for i in range(10000):  # 10k markets
            market = Market(
                id=f"market_{i}",
                question=f"Test market {i}?",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.5"), Decimal("0.5")],
                volume_24h=Decimal(str(np.random.uniform(100, 100000)))
            )
            large_markets.append(market)
        
        current, peak = tracemalloc.get_traced_memory()
        results["large_dataset"]["current_mb"] = current / 1024 / 1024
        results["large_dataset"]["peak_mb"] = peak / 1024 / 1024
        results["large_dataset"]["items_count"] = len(large_markets)
        
        # Test 2: Memory leak detection
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        for iteration in range(100):
            # Simulate repeated operations
            temp_data = []
            for _ in range(1000):
                temp_data.append({
                    "id": f"item_{iteration}_{_}",
                    "data": np.random.rand(100).tolist()
                })
            
            # Process and discard
            processed = [json.dumps(item) for item in temp_data]
            del temp_data
            del processed
            
            if iteration % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        results["memory_leak_test"]["initial_mb"] = initial_memory
        results["memory_leak_test"]["final_mb"] = final_memory
        results["memory_leak_test"]["growth_mb"] = memory_growth
        results["memory_leak_test"]["growth_percentage"] = (memory_growth / initial_memory) * 100
        
        # Verify memory efficiency
        assert results["large_dataset"]["peak_mb"] < 500  # < 500MB for 10k markets
        assert results["memory_leak_test"]["growth_percentage"] < 10  # < 10% growth
        
        tracemalloc.stop()
        performance_monitor.record_benchmark("memory_usage", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_high_frequency_trading_scenario(self, load_test_client, performance_monitor):
        """Simulate high-frequency trading scenario."""
        # HFT parameters
        trading_duration = 30  # seconds
        target_orders_per_second = 10
        markets_to_monitor = 5
        
        results = {
            "orders_placed": 0,
            "orders_cancelled": 0,
            "market_updates_processed": 0,
            "latencies": [],
            "errors": 0
        }
        
        # Mock market data updates
        async def market_data_generator():
            """Generate continuous market data updates."""
            while True:
                for i in range(markets_to_monitor):
                    yield {
                        "market_id": f"market_{i}",
                        "bid": Decimal(str(np.random.uniform(0.4, 0.6))),
                        "ask": Decimal(str(np.random.uniform(0.4, 0.6))),
                        "timestamp": time.time()
                    }
                await asyncio.sleep(0.1)  # 10 updates/second per market
        
        # HFT strategy logic
        async def hft_strategy():
            """Simple HFT strategy for testing."""
            data_gen = market_data_generator()
            start_time = time.time()
            
            while time.time() - start_time < trading_duration:
                # Get market update
                market_update = await anext(data_gen)
                results["market_updates_processed"] += 1
                
                # Make trading decision
                spread = market_update["ask"] - market_update["bid"]
                
                if spread > Decimal("0.02"):  # Wide spread, place orders
                    # Place buy order
                    order_start = time.time()
                    try:
                        buy_order = await load_test_client.place_order(
                            market_id=market_update["market_id"],
                            side=OrderSide.BUY,
                            price=market_update["bid"] + Decimal("0.001"),
                            size=Decimal("10")
                        )
                        results["orders_placed"] += 1
                        
                        # Immediately cancel if spread narrows
                        if np.random.random() > 0.5:
                            await load_test_client.cancel_order(buy_order.id)
                            results["orders_cancelled"] += 1
                        
                    except Exception as e:
                        results["errors"] += 1
                    
                    latency = (time.time() - order_start) * 1000
                    results["latencies"].append(latency)
                
                # Rate limiting
                await asyncio.sleep(1 / target_orders_per_second)
        
        # Run HFT simulation
        await hft_strategy()
        
        # Calculate final metrics
        results["avg_latency_ms"] = np.mean(results["latencies"]) if results["latencies"] else 0
        results["p99_latency_ms"] = np.percentile(results["latencies"], 99) if results["latencies"] else 0
        results["orders_per_second"] = results["orders_placed"] / trading_duration
        results["updates_per_second"] = results["market_updates_processed"] / trading_duration
        results["error_rate"] = results["errors"] / results["orders_placed"] if results["orders_placed"] > 0 else 0
        
        # Verify HFT performance
        assert results["avg_latency_ms"] < 50  # < 50ms average latency
        assert results["p99_latency_ms"] < 200  # < 200ms p99 latency
        assert results["orders_per_second"] > 5  # > 5 orders/second
        assert results["error_rate"] < 0.01  # < 1% error rate
        
        performance_monitor.record_benchmark("hft_simulation", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_strategy_computation_performance(self, performance_monitor):
        """Benchmark strategy computation and analysis speed."""
        # Create test data
        market_count = 100
        historical_data_points = 1000
        
        markets = []
        for i in range(market_count):
            markets.append({
                "id": f"market_{i}",
                "prices": np.random.rand(historical_data_points).tolist(),
                "volumes": np.random.randint(100, 10000, historical_data_points).tolist(),
                "timestamps": [(datetime.now() - timedelta(hours=x)).isoformat() 
                              for x in range(historical_data_points)]
            })
        
        results = {}
        
        # Test 1: Momentum calculation
        start = time.time()
        for market in markets:
            prices = np.array(market["prices"])
            # Calculate various momentum indicators
            sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
            sma_50 = np.convolve(prices, np.ones(50)/50, mode='valid')
            rsi = self._calculate_rsi(prices)
            macd = self._calculate_macd(prices)
        
        results["momentum_calculation"] = {
            "total_time": time.time() - start,
            "markets_per_second": market_count / (time.time() - start),
            "indicators_calculated": ["SMA20", "SMA50", "RSI", "MACD"]
        }
        
        # Test 2: Correlation analysis
        start = time.time()
        price_matrix = np.array([m["prices"] for m in markets])
        correlation_matrix = np.corrcoef(price_matrix)
        
        results["correlation_analysis"] = {
            "matrix_size": f"{market_count}x{market_count}",
            "calculation_time": time.time() - start,
            "correlations_per_second": (market_count * market_count) / (time.time() - start)
        }
        
        # Test 3: Signal generation
        start = time.time()
        signals_generated = 0
        
        for market in markets:
            prices = np.array(market["prices"])
            volumes = np.array(market["volumes"])
            
            # Generate signals based on multiple factors
            momentum_signal = self._generate_momentum_signal(prices)
            volume_signal = self._generate_volume_signal(volumes)
            combined_signal = (momentum_signal + volume_signal) / 2
            
            if abs(combined_signal) > 0.5:
                signals_generated += 1
        
        results["signal_generation"] = {
            "total_time": time.time() - start,
            "signals_generated": signals_generated,
            "markets_per_second": market_count / (time.time() - start)
        }
        
        # Verify performance targets
        assert results["momentum_calculation"]["markets_per_second"] > 50
        assert results["correlation_analysis"]["calculation_time"] < 1.0
        assert results["signal_generation"]["markets_per_second"] > 100
        
        performance_monitor.record_benchmark("strategy_computation", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_operation_performance(self, performance_monitor):
        """Test database read/write performance for trade history."""
        # Simulate database operations
        results = {
            "write_performance": {},
            "read_performance": {},
            "query_performance": {}
        }
        
        # Test data
        trades = []
        for i in range(10000):
            trades.append({
                "id": f"trade_{i}",
                "market_id": f"market_{i % 100}",
                "timestamp": datetime.now() - timedelta(seconds=i),
                "price": float(np.random.uniform(0.1, 0.9)),
                "size": float(np.random.uniform(10, 1000)),
                "side": "buy" if i % 2 == 0 else "sell"
            })
        
        # Test 1: Bulk write performance
        start = time.time()
        # Simulate bulk insert
        batch_size = 1000
        for i in range(0, len(trades), batch_size):
            batch = trades[i:i+batch_size]
            # Simulate write delay
            await asyncio.sleep(0.01)
        
        write_time = time.time() - start
        results["write_performance"]["total_records"] = len(trades)
        results["write_performance"]["total_time"] = write_time
        results["write_performance"]["records_per_second"] = len(trades) / write_time
        
        # Test 2: Read performance
        start = time.time()
        # Simulate reads
        for _ in range(100):
            # Random range query
            start_idx = np.random.randint(0, len(trades) - 100)
            selected_trades = trades[start_idx:start_idx + 100]
            await asyncio.sleep(0.001)  # Simulate read delay
        
        read_time = time.time() - start
        results["read_performance"]["queries_executed"] = 100
        results["read_performance"]["total_time"] = read_time
        results["read_performance"]["queries_per_second"] = 100 / read_time
        
        # Test 3: Complex query performance
        start = time.time()
        # Simulate aggregation queries
        df = pd.DataFrame(trades)
        
        # Group by market and calculate stats
        market_stats = df.groupby('market_id').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'size': ['sum', 'mean'],
            'id': 'count'
        })
        
        # Time-based aggregations
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        hourly_volume = df.groupby('market_id').resample('1H')['size'].sum()
        
        query_time = time.time() - start
        results["query_performance"]["aggregation_time"] = query_time
        results["query_performance"]["markets_analyzed"] = len(market_stats)
        results["query_performance"]["time_periods"] = len(hourly_volume)
        
        # Verify performance
        assert results["write_performance"]["records_per_second"] > 5000
        assert results["read_performance"]["queries_per_second"] > 500
        assert results["query_performance"]["aggregation_time"] < 0.5
        
        performance_monitor.record_benchmark("database_operations", results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.gpu
    async def test_gpu_vs_cpu_performance(self, performance_monitor):
        """Compare GPU vs CPU performance for computational tasks."""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
            
        if not gpu_available:
            pytest.skip("GPU not available for testing")
        
        results = {
            "monte_carlo": {},
            "neural_network": {},
            "matrix_operations": {}
        }
        
        # Test 1: Monte Carlo simulation
        n_simulations = 1000000
        n_steps = 252  # Trading days
        
        # CPU version
        start = time.time()
        cpu_paths = np.random.randn(n_simulations, n_steps)
        cpu_prices = 100 * np.exp(np.cumsum(0.01 + 0.2 * cpu_paths / np.sqrt(252), axis=1))
        cpu_payoffs = np.maximum(cpu_prices[:, -1] - 100, 0)
        cpu_price = np.mean(cpu_payoffs)
        cpu_time = time.time() - start
        
        # GPU version
        start = time.time()
        gpu_paths = torch.randn(n_simulations, n_steps, device='cuda')
        gpu_prices = 100 * torch.exp(torch.cumsum(0.01 + 0.2 * gpu_paths / np.sqrt(252), dim=1))
        gpu_payoffs = torch.maximum(gpu_prices[:, -1] - 100, torch.tensor(0.0, device='cuda'))
        gpu_price = torch.mean(gpu_payoffs).cpu().numpy()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        results["monte_carlo"]["cpu_time"] = cpu_time
        results["monte_carlo"]["gpu_time"] = gpu_time
        results["monte_carlo"]["speedup"] = cpu_time / gpu_time
        results["monte_carlo"]["simulations"] = n_simulations
        
        # Test 2: Neural network inference
        batch_size = 1000
        input_size = 100
        hidden_size = 256
        output_size = 2
        
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=1)
        )
        
        # CPU inference
        model_cpu = model.cpu()
        input_cpu = torch.randn(batch_size, input_size)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                output_cpu = model_cpu(input_cpu)
        cpu_inference_time = time.time() - start
        
        # GPU inference
        model_gpu = model.cuda()
        input_gpu = torch.randn(batch_size, input_size, device='cuda')
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                output_gpu = model_gpu(input_gpu)
        torch.cuda.synchronize()
        gpu_inference_time = time.time() - start
        
        results["neural_network"]["cpu_time"] = cpu_inference_time
        results["neural_network"]["gpu_time"] = gpu_inference_time
        results["neural_network"]["speedup"] = cpu_inference_time / gpu_inference_time
        results["neural_network"]["batch_size"] = batch_size
        
        # Test 3: Large matrix operations
        matrix_size = 5000
        
        # CPU matrix multiplication
        A_cpu = np.random.rand(matrix_size, matrix_size)
        B_cpu = np.random.rand(matrix_size, matrix_size)
        
        start = time.time()
        C_cpu = np.matmul(A_cpu, B_cpu)
        cpu_matmul_time = time.time() - start
        
        # GPU matrix multiplication
        A_gpu = torch.rand(matrix_size, matrix_size, device='cuda')
        B_gpu = torch.rand(matrix_size, matrix_size, device='cuda')
        
        start = time.time()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        gpu_matmul_time = time.time() - start
        
        results["matrix_operations"]["cpu_time"] = cpu_matmul_time
        results["matrix_operations"]["gpu_time"] = gpu_matmul_time
        results["matrix_operations"]["speedup"] = cpu_matmul_time / gpu_matmul_time
        results["matrix_operations"]["matrix_size"] = matrix_size
        
        # Verify GPU provides significant speedup
        assert results["monte_carlo"]["speedup"] > 10  # 10x+ speedup
        assert results["neural_network"]["speedup"] > 5  # 5x+ speedup
        assert results["matrix_operations"]["speedup"] > 20  # 20x+ speedup
        
        performance_monitor.record_benchmark("gpu_vs_cpu", results)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MACD indicator."""
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            "macd": macd.values,
            "signal": signal.values,
            "histogram": histogram.values
        }

    def _generate_momentum_signal(self, prices: np.ndarray) -> float:
        """Generate momentum-based trading signal."""
        if len(prices) < 50:
            return 0.0
        
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        current_price = prices[-1]
        
        momentum = (current_price - sma_50) / sma_50
        trend = (sma_20 - sma_50) / sma_50
        
        signal = momentum * 0.6 + trend * 0.4
        return np.clip(signal * 10, -1, 1)  # Normalize to [-1, 1]

    def _generate_volume_signal(self, volumes: np.ndarray) -> float:
        """Generate volume-based trading signal."""
        if len(volumes) < 20:
            return 0.0
        
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[-20:])
        
        if avg_volume == 0:
            return 0.0
        
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 2:
            return 1.0  # High volume spike
        elif volume_ratio < 0.5:
            return -1.0  # Low volume
        else:
            return (volume_ratio - 1) * 2  # Linear scaling

    def _generate_mock_response(self, endpoint: Optional[str]) -> Any:
        """Generate mock API response based on endpoint."""
        if endpoint and "markets" in endpoint:
            return [{"id": f"market_{i}", "data": "test"} for i in range(10)]
        elif endpoint and "orderbook" in endpoint:
            return {"bids": [], "asks": []}
        elif endpoint and "order" in endpoint:
            return {"id": "order_123", "status": "filled"}
        return {}


class PerformanceMonitor:
    """Monitor and record performance metrics."""
    
    def __init__(self):
        self.benchmarks = {}
        self.start_time = time.time()
    
    def record_benchmark(self, name: str, results: Dict[str, Any]):
        """Record benchmark results."""
        self.benchmarks[name] = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "total_runtime": time.time() - self.start_time,
            "benchmarks": self.benchmarks,
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            "total_benchmarks": len(self.benchmarks),
            "key_metrics": {}
        }
        
        # Extract key metrics from each benchmark
        if "api_response_times" in self.benchmarks:
            api_data = self.benchmarks["api_response_times"]["results"]
            summary["key_metrics"]["avg_api_latency_ms"] = api_data["sustained_load"]["avg_latency"]
            summary["key_metrics"]["api_requests_per_second"] = api_data["sustained_load"]["requests_per_second"]
        
        if "gpu_vs_cpu" in self.benchmarks:
            gpu_data = self.benchmarks["gpu_vs_cpu"]["results"]
            summary["key_metrics"]["gpu_speedup_monte_carlo"] = gpu_data["monte_carlo"]["speedup"]
            summary["key_metrics"]["gpu_speedup_neural_net"] = gpu_data["neural_network"]["speedup"]
        
        return summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])