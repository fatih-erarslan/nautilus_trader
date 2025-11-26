#!/usr/bin/env python3
"""
Performance tests for low-latency trading APIs
Tests latency, throughput, and system optimization effectiveness
"""

import time
import asyncio
import statistics
import pytest
from unittest.mock import Mock
import numpy as np

class LatencyBenchmark:
    """Benchmark class for measuring trading system latency"""
    
    def __init__(self):
        self.measurements = []
        
    def measure_latency(self, func):
        """Measure function execution latency in microseconds"""
        start = time.time_ns()
        result = func()
        end = time.time_ns()
        latency_us = (end - start) / 1000
        self.measurements.append(latency_us)
        return result, latency_us
    
    def get_stats(self):
        """Get latency statistics"""
        if not self.measurements:
            return {}
        
        return {
            'count': len(self.measurements),
            'mean': statistics.mean(self.measurements),
            'median': statistics.median(self.measurements),
            'min': min(self.measurements),
            'max': max(self.measurements),
            'std': statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0,
            'p95': np.percentile(self.measurements, 95),
            'p99': np.percentile(self.measurements, 99),
        }

class MockMarketData:
    """Mock market data for testing"""
    
    def __init__(self):
        self.price = 100.0
        self.volume = 1000
        self.timestamp = time.time_ns()
    
    def get_quote(self):
        """Simulate getting market quote"""
        return {
            'symbol': 'AAPL',
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp
        }
    
    def process_quote(self, quote):
        """Simulate processing market quote"""
        # Simulate some processing time
        processed = {
            'symbol': quote['symbol'],
            'price': quote['price'],
            'volume': quote['volume'],
            'timestamp': quote['timestamp'],
            'processed_at': time.time_ns()
        }
        return processed

class TestLatencyBenchmarks:
    """Test suite for latency benchmarks"""
    
    def setup_method(self):
        """Setup test environment"""
        self.benchmark = LatencyBenchmark()
        self.market_data = MockMarketData()
    
    def test_quote_processing_latency(self):
        """Test market data quote processing latency"""
        # Target: < 100 microseconds
        target_latency = 100.0
        
        # Run multiple iterations
        for _ in range(1000):
            quote = self.market_data.get_quote()
            _, latency = self.benchmark.measure_latency(
                lambda: self.market_data.process_quote(quote)
            )
        
        stats = self.benchmark.get_stats()
        
        # Assert performance requirements
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.1f}µs exceeds target {target_latency}µs"
        assert stats['mean'] < target_latency / 2, f"Mean latency {stats['mean']:.1f}µs exceeds target {target_latency/2}µs"
        
        print(f"Quote processing latency stats: {stats}")
    
    def test_order_submission_latency(self):
        """Test order submission latency"""
        # Target: < 10 milliseconds
        target_latency = 10000.0  # 10ms in microseconds
        
        def mock_order_submission():
            # Simulate order validation and submission
            order = {
                'symbol': 'AAPL',
                'quantity': 100,
                'side': 'buy',
                'order_type': 'limit',
                'price': 150.0
            }
            # Simulate network call
            time.sleep(0.001)  # 1ms simulated network delay
            return {'order_id': '12345', 'status': 'submitted'}
        
        # Run multiple iterations
        for _ in range(100):
            _, latency = self.benchmark.measure_latency(mock_order_submission)
        
        stats = self.benchmark.get_stats()
        
        # Assert performance requirements
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.1f}µs exceeds target {target_latency}µs"
        assert stats['mean'] < target_latency / 2, f"Mean latency {stats['mean']:.1f}µs exceeds target {target_latency/2}µs"
        
        print(f"Order submission latency stats: {stats}")
    
    def test_memory_allocation_performance(self):
        """Test memory allocation performance"""
        # Test pre-allocated vs dynamic allocation
        
        # Pre-allocated arrays
        pre_allocated = np.zeros(10000, dtype=np.float64)
        
        def test_pre_allocated():
            for i in range(1000):
                pre_allocated[i % len(pre_allocated)] = i * 1.5
        
        def test_dynamic_allocation():
            data = []
            for i in range(1000):
                data.append(i * 1.5)
        
        # Measure pre-allocated performance
        benchmark_pre = LatencyBenchmark()
        for _ in range(100):
            benchmark_pre.measure_latency(test_pre_allocated)
        
        # Measure dynamic allocation performance
        benchmark_dynamic = LatencyBenchmark()
        for _ in range(100):
            benchmark_dynamic.measure_latency(test_dynamic_allocation)
        
        stats_pre = benchmark_pre.get_stats()
        stats_dynamic = benchmark_dynamic.get_stats()
        
        # Pre-allocated should be significantly faster
        assert stats_pre['mean'] < stats_dynamic['mean'], "Pre-allocated memory should be faster"
        
        print(f"Pre-allocated mean: {stats_pre['mean']:.1f}µs")
        print(f"Dynamic allocation mean: {stats_dynamic['mean']:.1f}µs")
        print(f"Speedup: {stats_dynamic['mean'] / stats_pre['mean']:.2f}x")
    
    def test_json_vs_msgpack_performance(self):
        """Test JSON vs MessagePack serialization performance"""
        import json
        try:
            import msgpack
            msgpack_available = True
        except ImportError:
            msgpack_available = False
            pytest.skip("msgpack not available")
        
        # Test data
        data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 10000,
            'timestamp': time.time_ns(),
            'metadata': {
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'tags': ['tech', 'large_cap']
            }
        }
        
        # JSON serialization
        benchmark_json = LatencyBenchmark()
        for _ in range(1000):
            benchmark_json.measure_latency(lambda: json.dumps(data))
        
        # MessagePack serialization
        benchmark_msgpack = LatencyBenchmark()
        for _ in range(1000):
            benchmark_msgpack.measure_latency(lambda: msgpack.packb(data))
        
        stats_json = benchmark_json.get_stats()
        stats_msgpack = benchmark_msgpack.get_stats()
        
        # MessagePack should be faster
        assert stats_msgpack['mean'] < stats_json['mean'], "MessagePack should be faster than JSON"
        
        print(f"JSON serialization mean: {stats_json['mean']:.1f}µs")
        print(f"MessagePack serialization mean: {stats_msgpack['mean']:.1f}µs")
        print(f"Speedup: {stats_json['mean'] / stats_msgpack['mean']:.2f}x")
    
    def test_system_optimization_effectiveness(self):
        """Test if system optimizations are effective"""
        import os
        import psutil
        
        # Check CPU governor
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
            assert governor == 'performance', f"CPU governor should be 'performance', got '{governor}'"
        except FileNotFoundError:
            pytest.skip("CPU governor not available")
        
        # Check process priority
        process = psutil.Process()
        nice_value = process.nice()
        assert nice_value <= 0, f"Process should have high priority (nice <= 0), got {nice_value}"
        
        # Check memory settings
        try:
            with open('/proc/sys/vm/swappiness', 'r') as f:
                swappiness = int(f.read().strip())
            assert swappiness == 0, f"Swappiness should be 0, got {swappiness}"
        except FileNotFoundError:
            pytest.skip("Swappiness setting not available")
        
        print("System optimization checks passed")
    
    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async operation performance"""
        
        async def mock_async_operation():
            # Simulate async network call
            await asyncio.sleep(0.001)  # 1ms
            return {'status': 'success'}
        
        # Measure async performance
        benchmark = LatencyBenchmark()
        
        for _ in range(100):
            start = time.time_ns()
            result = await mock_async_operation()
            end = time.time_ns()
            latency_us = (end - start) / 1000
            benchmark.measurements.append(latency_us)
        
        stats = benchmark.get_stats()
        
        # Should be close to 1ms (1000µs)
        assert stats['mean'] < 2000, f"Async operation too slow: {stats['mean']:.1f}µs"
        
        print(f"Async operation latency stats: {stats}")

def run_performance_suite():
    """Run the complete performance test suite"""
    print("Running Low-Latency Trading API Performance Tests")
    print("=" * 50)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])

if __name__ == "__main__":
    run_performance_suite()