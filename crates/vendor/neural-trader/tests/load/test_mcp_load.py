"""
Load tests for integrated MCP server
Tests system performance under high-volume scenarios
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import random
import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mcp.mcp_server_integrated import (
    mcp, NEWS_AGGREGATION_AVAILABLE, STRATEGY_MANAGER_AVAILABLE,
    GPU_AVAILABLE, POLYMARKET_TOOLS_AVAILABLE
)

# Load test configuration
LOAD_TEST_CONFIG = {
    "concurrent_users": [10, 50, 100, 200],
    "test_duration_seconds": 60,
    "request_delay_ms": [0, 10, 50, 100],
    "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM", "BAC", "WMT"],
    "strategies": ["momentum_trading", "swing_trading", "mean_reversion", "mirror_trading"]
}

class LoadTestMetrics:
    """Track load test metrics"""
    
    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def record_request(self, duration: float, success: bool):
        self.request_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def finish(self):
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.request_times:
            return {"error": "No requests recorded"}
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        total_requests = len(self.request_times)
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0,
            "total_time_seconds": round(total_time, 2),
            "requests_per_second": round(total_requests / total_time, 2) if total_time > 0 else 0,
            "response_times": {
                "min_ms": round(min(self.request_times) * 1000, 2),
                "max_ms": round(max(self.request_times) * 1000, 2),
                "mean_ms": round(statistics.mean(self.request_times) * 1000, 2),
                "median_ms": round(statistics.median(self.request_times) * 1000, 2),
                "p95_ms": round(np.percentile(self.request_times, 95) * 1000, 2),
                "p99_ms": round(np.percentile(self.request_times, 99) * 1000, 2)
            }
        }

# Load test scenarios
class TestHighVolumeTrading:
    """Test high-volume trading scenarios"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrent_users", [10, 50, 100])
    async def test_concurrent_trading(self, concurrent_users):
        """Test concurrent trading operations"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def trading_user(user_id: int):
            """Simulate a trading user"""
            for _ in range(10):  # Each user makes 10 trades
                try:
                    start = time.time()
                    
                    # Random symbol and strategy
                    symbol = random.choice(LOAD_TEST_CONFIG["symbols"])
                    strategy = random.choice(LOAD_TEST_CONFIG["strategies"])
                    
                    # Quick analysis
                    analysis = await mcp.call_tool("quick_analysis", {
                        "symbol": symbol,
                        "use_gpu": GPU_AVAILABLE
                    })
                    
                    # Simulate trade
                    if analysis.get("status") == "success":
                        action = random.choice(["buy", "sell"])
                        trade = await mcp.call_tool("simulate_trade", {
                            "strategy": strategy,
                            "symbol": symbol,
                            "action": action,
                            "use_gpu": GPU_AVAILABLE
                        })
                        
                        success = trade.get("status") == "executed"
                    else:
                        success = False
                    
                    duration = time.time() - start
                    metrics.record_request(duration, success)
                    
                    # Small random delay
                    await asyncio.sleep(random.uniform(0.01, 0.1))
                    
                except Exception as e:
                    metrics.record_request(time.time() - start, False)
        
        # Run concurrent users
        tasks = [trading_user(i) for i in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary["error_rate"] < 0.1, f"High error rate: {summary['error_rate']}"
        assert summary["response_times"]["p95_ms"] < 1000, f"P95 latency too high: {summary['response_times']['p95_ms']}ms"
        
        print(f"\nTrading Load Test Results ({concurrent_users} users):")
        print(f"  Total requests: {summary['total_requests']}")
        print(f"  Requests/second: {summary['requests_per_second']}")
        print(f"  P95 latency: {summary['response_times']['p95_ms']}ms")
        print(f"  Error rate: {summary['error_rate']:.2%}")
    
    @pytest.mark.asyncio
    async def test_burst_trading(self):
        """Test burst trading scenarios"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        # Generate burst of 1000 trades in 10 seconds
        async def burst_trade():
            symbol = random.choice(LOAD_TEST_CONFIG["symbols"])
            strategy = random.choice(LOAD_TEST_CONFIG["strategies"])
            
            start = time.time()
            try:
                result = await mcp.call_tool("simulate_trade", {
                    "strategy": strategy,
                    "symbol": symbol,
                    "action": random.choice(["buy", "sell"]),
                    "use_gpu": GPU_AVAILABLE
                })
                success = result.get("status") == "executed"
            except:
                success = False
            
            duration = time.time() - start
            metrics.record_request(duration, success)
        
        # Create burst
        tasks = [burst_trade() for _ in range(1000)]
        start_burst = time.time()
        await asyncio.gather(*tasks)
        burst_duration = time.time() - start_burst
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nBurst Trading Test Results:")
        print(f"  1000 trades in {burst_duration:.2f} seconds")
        print(f"  Throughput: {1000/burst_duration:.2f} trades/second")
        print(f"  P99 latency: {summary['response_times']['p99_ms']}ms")
        
        assert burst_duration < 30, f"Burst took too long: {burst_duration}s"
        assert summary["error_rate"] < 0.05, f"High error rate: {summary['error_rate']}"

class TestNewsAggregationLoad:
    """Test news aggregation under load"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_concurrent_news_fetching(self):
        """Test concurrent news fetching"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def fetch_news_for_symbol(symbol: str):
            for _ in range(5):
                start = time.time()
                try:
                    result = await mcp.call_tool("fetch_filtered_news", {
                        "symbols": [symbol],
                        "sentiment_filter": random.choice([None, "positive", "negative"]),
                        "relevance_threshold": 0.5,
                        "limit": 20
                    })
                    success = result.get("status") == "success"
                except:
                    success = False
                
                duration = time.time() - start
                metrics.record_request(duration, success)
                await asyncio.sleep(0.1)
        
        # Fetch news for all symbols concurrently
        tasks = [fetch_news_for_symbol(symbol) for symbol in LOAD_TEST_CONFIG["symbols"]]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nNews Fetching Load Test Results:")
        print(f"  Total requests: {summary['total_requests']}")
        print(f"  Mean response time: {summary['response_times']['mean_ms']}ms")
        print(f"  Error rate: {summary['error_rate']:.2%}")
        
        assert summary["error_rate"] < 0.2, f"High error rate: {summary['error_rate']}"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_news_trend_analysis_load(self):
        """Test news trend analysis under load"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def analyze_trends():
            symbols = random.sample(LOAD_TEST_CONFIG["symbols"], 3)
            start = time.time()
            try:
                result = await mcp.call_tool("get_news_trends", {
                    "symbols": symbols,
                    "time_intervals": [1, 6, 24]
                })
                success = result.get("status") == "success"
            except:
                success = False
            
            duration = time.time() - start
            metrics.record_request(duration, success)
        
        # Run 100 trend analyses concurrently
        tasks = [analyze_trends() for _ in range(100)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        assert summary["response_times"]["p95_ms"] < 2000, "Trend analysis too slow"

class TestNeuralForecastingLoad:
    """Test neural forecasting under load"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrent_forecasts", [10, 25, 50])
    async def test_concurrent_neural_forecasts(self, concurrent_forecasts):
        """Test concurrent neural forecasting"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def run_forecast():
            symbol = random.choice(LOAD_TEST_CONFIG["symbols"])
            horizon = random.choice([1, 7, 30])
            
            start = time.time()
            try:
                result = await mcp.call_tool("neural_forecast", {
                    "symbol": symbol,
                    "horizon": horizon,
                    "confidence_level": 0.95,
                    "use_gpu": GPU_AVAILABLE
                })
                success = result.get("status") == "success"
            except:
                success = False
            
            duration = time.time() - start
            metrics.record_request(duration, success)
        
        # Run concurrent forecasts
        tasks = [run_forecast() for _ in range(concurrent_forecasts)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nNeural Forecasting Load Test ({concurrent_forecasts} concurrent):")
        print(f"  Mean response time: {summary['response_times']['mean_ms']}ms")
        print(f"  P95 response time: {summary['response_times']['p95_ms']}ms")
        print(f"  GPU enabled: {GPU_AVAILABLE}")
        
        # GPU should provide significant speedup
        if GPU_AVAILABLE:
            assert summary["response_times"]["p95_ms"] < 2000, "GPU forecasting too slow"
        else:
            assert summary["response_times"]["p95_ms"] < 10000, "CPU forecasting too slow"

class TestMultiAssetTradingLoad:
    """Test multi-asset trading under load"""
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_load(self):
        """Test portfolio rebalancing under load"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def rebalance_portfolio():
            # Generate random target allocations
            symbols = random.sample(LOAD_TEST_CONFIG["symbols"], 5)
            allocations = np.random.dirichlet(np.ones(5))
            target_allocations = {symbol: float(alloc) for symbol, alloc in zip(symbols, allocations)}
            
            start = time.time()
            try:
                result = await mcp.call_tool("portfolio_rebalance", {
                    "target_allocations": target_allocations,
                    "rebalance_threshold": 0.05
                })
                success = result.get("status") == "success"
            except:
                success = False
            
            duration = time.time() - start
            metrics.record_request(duration, success)
        
        # Run 50 rebalancing operations
        tasks = [rebalance_portfolio() for _ in range(50)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        assert summary["error_rate"] < 0.1, "High error rate in rebalancing"
        assert summary["response_times"]["mean_ms"] < 500, "Rebalancing too slow"
    
    @pytest.mark.asyncio
    async def test_multi_asset_execution_load(self):
        """Test multi-asset trade execution under load"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def execute_multi_trade():
            # Generate 5-10 trades
            num_trades = random.randint(5, 10)
            trades = []
            
            for _ in range(num_trades):
                trades.append({
                    "symbol": random.choice(LOAD_TEST_CONFIG["symbols"]),
                    "action": random.choice(["buy", "sell"]),
                    "quantity": random.randint(10, 100),
                    "price": round(random.uniform(50, 500), 2)
                })
            
            start = time.time()
            try:
                result = await mcp.call_tool("execute_multi_asset_trade", {
                    "trades": trades,
                    "strategy": random.choice(LOAD_TEST_CONFIG["strategies"]),
                    "risk_limit": 1000000,
                    "execute_parallel": True
                })
                success = result.get("status") == "success"
            except:
                success = False
            
            duration = time.time() - start
            metrics.record_request(duration, success)
        
        # Run 100 multi-asset trades
        tasks = [execute_multi_trade() for _ in range(100)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nMulti-Asset Execution Load Test:")
        print(f"  Total batches: {summary['total_requests']}")
        print(f"  Success rate: {(1 - summary['error_rate']) * 100:.1f}%")
        print(f"  P95 latency: {summary['response_times']['p95_ms']}ms")

class TestSystemMonitoringLoad:
    """Test system monitoring under load"""
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self):
        """Test continuous system monitoring"""
        metrics = LoadTestMetrics()
        monitoring_data = []
        
        async def monitor_system():
            start = time.time()
            try:
                result = await mcp.call_tool("get_system_metrics", {
                    "metrics": ["cpu", "memory", "latency", "throughput"],
                    "include_history": False
                })
                if result.get("status") == "success":
                    monitoring_data.append({
                        "timestamp": datetime.now(),
                        "metrics": result["current_metrics"]
                    })
                success = result.get("status") == "success"
            except:
                success = False
            
            duration = time.time() - start
            return duration, success
        
        # Monitor for 30 seconds with high frequency
        metrics.start()
        start_time = time.time()
        
        while time.time() - start_time < 30:
            duration, success = await monitor_system()
            metrics.record_request(duration, success)
            await asyncio.sleep(0.1)  # 10 Hz monitoring
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nContinuous Monitoring Results:")
        print(f"  Monitoring frequency: {summary['requests_per_second']:.1f} Hz")
        print(f"  Mean latency: {summary['response_times']['mean_ms']}ms")
        
        assert summary["requests_per_second"] > 5, "Monitoring frequency too low"
        assert summary["response_times"]["mean_ms"] < 100, "Monitoring latency too high"

class TestMixedWorkloadScenarios:
    """Test realistic mixed workload scenarios"""
    
    @pytest.mark.asyncio
    async def test_realistic_trading_day(self):
        """Simulate a realistic trading day workload"""
        metrics = LoadTestMetrics()
        metrics.start()
        
        # Different user types
        async def day_trader(trader_id: int):
            """High-frequency trader"""
            for _ in range(50):
                symbol = random.choice(LOAD_TEST_CONFIG["symbols"][:5])  # Focus on few symbols
                
                start = time.time()
                try:
                    # Quick analysis
                    analysis = await mcp.call_tool("quick_analysis", {
                        "symbol": symbol,
                        "use_gpu": GPU_AVAILABLE
                    })
                    
                    # Execute trade
                    if analysis.get("analysis", {}).get("recommendation") in ["buy", "sell"]:
                        await mcp.call_tool("simulate_trade", {
                            "strategy": "momentum_trading",
                            "symbol": symbol,
                            "action": analysis["analysis"]["recommendation"],
                            "use_gpu": GPU_AVAILABLE
                        })
                    
                    success = True
                except:
                    success = False
                
                metrics.record_request(time.time() - start, success)
                await asyncio.sleep(random.uniform(0.5, 2))  # Trade every 0.5-2 seconds
        
        async def swing_trader(trader_id: int):
            """Less frequent trader with more analysis"""
            for _ in range(10):
                symbols = random.sample(LOAD_TEST_CONFIG["symbols"], 3)
                
                start = time.time()
                try:
                    # Analyze multiple symbols
                    for symbol in symbols:
                        await mcp.call_tool("analyze_news", {
                            "symbol": symbol,
                            "lookback_hours": 24,
                            "use_gpu": False
                        })
                        
                        await mcp.call_tool("neural_forecast", {
                            "symbol": symbol,
                            "horizon": 7,
                            "use_gpu": GPU_AVAILABLE
                        })
                    
                    # Make trading decision
                    best_symbol = random.choice(symbols)
                    await mcp.call_tool("simulate_trade", {
                        "strategy": "swing_trading",
                        "symbol": best_symbol,
                        "action": random.choice(["buy", "sell"]),
                        "use_gpu": False
                    })
                    
                    success = True
                except:
                    success = False
                
                metrics.record_request(time.time() - start, success)
                await asyncio.sleep(random.uniform(10, 30))  # Trade every 10-30 seconds
        
        async def portfolio_manager(manager_id: int):
            """Portfolio rebalancing and monitoring"""
            for _ in range(5):
                start = time.time()
                try:
                    # Check correlations
                    await mcp.call_tool("cross_asset_correlation_matrix", {
                        "assets": random.sample(LOAD_TEST_CONFIG["symbols"], 5),
                        "lookback_days": 30,
                        "include_prediction_confidence": True
                    })
                    
                    # Monitor portfolio
                    await mcp.call_tool("get_portfolio_status", {
                        "include_analytics": True
                    })
                    
                    # Rebalance if needed
                    await mcp.call_tool("portfolio_rebalance", {
                        "target_allocations": {
                            "AAPL": 0.2,
                            "GOOGL": 0.2,
                            "MSFT": 0.2,
                            "CASH": 0.4
                        }
                    })
                    
                    success = True
                except:
                    success = False
                
                metrics.record_request(time.time() - start, success)
                await asyncio.sleep(random.uniform(30, 60))  # Check every 30-60 seconds
        
        # Simulate different user types
        tasks = []
        tasks.extend([day_trader(i) for i in range(5)])  # 5 day traders
        tasks.extend([swing_trader(i) for i in range(10)])  # 10 swing traders
        tasks.extend([portfolio_manager(i) for i in range(3)])  # 3 portfolio managers
        
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nRealistic Trading Day Simulation:")
        print(f"  Total operations: {summary['total_requests']}")
        print(f"  Success rate: {(1 - summary['error_rate']) * 100:.1f}%")
        print(f"  Mean response time: {summary['response_times']['mean_ms']}ms")
        print(f"  P95 response time: {summary['response_times']['p95_ms']}ms")
        
        assert summary["error_rate"] < 0.05, "Too many errors in realistic scenario"
        assert summary["response_times"]["p95_ms"] < 5000, "Response times too high"

class TestStressLimits:
    """Test system stress limits"""
    
    @pytest.mark.asyncio
    async def test_maximum_concurrent_connections(self):
        """Test maximum concurrent connections"""
        max_concurrent = 500
        metrics = LoadTestMetrics()
        
        async def simple_request():
            start = time.time()
            try:
                result = await mcp.call_tool("ping", {})
                success = result == "pong"
            except:
                success = False
            
            duration = time.time() - start
            return duration, success
        
        # Create many concurrent requests
        metrics.start()
        tasks = [simple_request() for _ in range(max_concurrent)]
        results = await asyncio.gather(*tasks)
        
        for duration, success in results:
            metrics.record_request(duration, success)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nMaximum Concurrent Connections Test ({max_concurrent}):")
        print(f"  Success rate: {(1 - summary['error_rate']) * 100:.1f}%")
        print(f"  Mean response time: {summary['response_times']['mean_ms']}ms")
        
        assert summary["error_rate"] < 0.5, "System cannot handle high concurrency"
    
    @pytest.mark.asyncio
    async def test_sustained_high_load(self):
        """Test sustained high load for extended period"""
        test_duration = 120  # 2 minutes
        metrics = LoadTestMetrics()
        metrics.start()
        
        async def continuous_load():
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                tool = random.choice([
                    "quick_analysis",
                    "list_strategies",
                    "get_system_metrics",
                    "neural_model_status"
                ])
                
                params = {}
                if tool == "quick_analysis":
                    params = {"symbol": random.choice(LOAD_TEST_CONFIG["symbols"]), "use_gpu": False}
                elif tool == "get_system_metrics":
                    params = {"metrics": ["cpu", "memory"]}
                
                start = time.time()
                try:
                    await mcp.call_tool(tool, params)
                    success = True
                except:
                    success = False
                
                metrics.record_request(time.time() - start, success)
                await asyncio.sleep(0.01)  # High frequency
        
        # Run multiple continuous load generators
        tasks = [continuous_load() for _ in range(20)]
        await asyncio.gather(*tasks)
        
        metrics.finish()
        summary = metrics.get_summary()
        
        print(f"\nSustained High Load Test (2 minutes):")
        print(f"  Total requests: {summary['total_requests']}")
        print(f"  Requests/second: {summary['requests_per_second']}")
        print(f"  Error rate: {summary['error_rate']:.2%}")
        print(f"  P99 latency: {summary['response_times']['p99_ms']}ms")
        
        assert summary["error_rate"] < 0.1, "System degrades under sustained load"

# Performance comparison tests
class TestPerformanceComparison:
    """Compare performance with and without optimizations"""
    
    @pytest.mark.asyncio
    async def test_gpu_vs_cpu_performance(self):
        """Compare GPU vs CPU performance"""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")
        
        # Test neural forecasting
        async def test_forecast_performance(use_gpu: bool) -> Dict[str, Any]:
            metrics = LoadTestMetrics()
            metrics.start()
            
            for _ in range(20):
                start = time.time()
                try:
                    await mcp.call_tool("neural_forecast", {
                        "symbol": "AAPL",
                        "horizon": 30,
                        "use_gpu": use_gpu
                    })
                    success = True
                except:
                    success = False
                
                metrics.record_request(time.time() - start, success)
            
            metrics.finish()
            return metrics.get_summary()
        
        # Test with GPU
        gpu_results = await test_forecast_performance(True)
        
        # Test without GPU
        cpu_results = await test_forecast_performance(False)
        
        # Calculate speedup
        gpu_mean = gpu_results["response_times"]["mean_ms"]
        cpu_mean = cpu_results["response_times"]["mean_ms"]
        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0
        
        print(f"\nGPU vs CPU Performance Comparison:")
        print(f"  GPU mean time: {gpu_mean:.1f}ms")
        print(f"  CPU mean time: {cpu_mean:.1f}ms")
        print(f"  GPU speedup: {speedup:.1f}x")
        
        assert speedup > 2, "GPU should provide significant speedup"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])