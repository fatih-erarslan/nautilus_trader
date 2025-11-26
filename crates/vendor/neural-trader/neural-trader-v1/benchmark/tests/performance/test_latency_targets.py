"""
Latency performance validation test suite for AI News Trading benchmark system.

This module validates that signal generation latency meets the target of <100ms (P99).
Tests include:
- Signal generation latency
- Data processing latency  
- End-to-end pipeline latency
- Latency under load
- Latency optimization validation
"""

import asyncio
import time
import statistics
import pytest
from unittest.mock import Mock, patch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.profiling.profiler import LatencyProfiler


class TestSignalGenerationLatency:
    """Test signal generation latency performance."""
    
    @pytest.fixture
    async def latency_test_system(self):
        """Create system optimized for latency testing."""
        config = {
            'optimization_mode': 'latency',
            'cache_enabled': True,
            'parallel_processing': True,
            'buffer_size': 1000
        }
        
        system = {
            'data_manager': RealtimeManager(config),
            'simulator': MarketSimulator(config),
            'profiler': LatencyProfiler()
        }
        
        await system['data_manager'].initialize()
        await system['simulator'].initialize()
        await system['profiler'].initialize()
        
        yield system
        
        await system['data_manager'].shutdown()
        await system['simulator'].shutdown()
        await system['profiler'].shutdown()
    
    @pytest.mark.asyncio
    async def test_single_signal_generation_latency(self, latency_test_system):
        """Test latency of single signal generation."""
        simulator = latency_test_system['simulator']
        profiler = latency_test_system['profiler']
        
        # Warm up the system
        for _ in range(10):
            await simulator.generate_signal('AAPL')
        
        # Measure signal generation latency
        latencies = []
        
        for i in range(1000):
            market_data = {
                'symbol': 'AAPL',
                'price': 150 + np.random.normal(0, 1),
                'volume': 1000,
                'timestamp': time.time()
            }
            
            start_time = time.perf_counter()
            
            with profiler.measure('signal_generation'):
                signal = await simulator.generate_signal('AAPL', market_data)
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.001)
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        
        # Validate latency requirements
        assert avg_latency < 25, f"Average latency {avg_latency:.2f}ms exceeds 25ms"
        assert p50_latency < 20, f"P50 latency {p50_latency:.2f}ms exceeds 20ms"
        assert p95_latency < 50, f"P95 latency {p95_latency:.2f}ms exceeds 50ms"
        assert p99_latency < 100, f"P99 latency {p99_latency:.2f}ms exceeds 100ms target"
        assert max_latency < 200, f"Max latency {max_latency:.2f}ms exceeds 200ms"
        
        # Log detailed statistics
        profiler_stats = profiler.get_statistics('signal_generation')
        print(f"\nSignal Generation Latency Statistics:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P50: {p50_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        print(f"  Profiler avg: {profiler_stats['avg_duration_ms']:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_batch_signal_generation_latency(self, latency_test_system):
        """Test latency of batch signal generation."""
        simulator = latency_test_system['simulator']
        
        # Test different batch sizes
        batch_sizes = [5, 10, 25, 50, 100]
        symbols = [f'SYM{i:03d}' for i in range(100)]
        
        for batch_size in batch_sizes:
            batch_symbols = symbols[:batch_size]
            batch_latencies = []
            
            for _ in range(50):  # 50 batches per size
                start_time = time.perf_counter()
                
                signals = await simulator.generate_signals_batch(batch_symbols)
                
                end_time = time.perf_counter()
                
                batch_latency = (end_time - start_time) * 1000
                batch_latencies.append(batch_latency)
                
                assert len(signals) == batch_size, f"Expected {batch_size} signals, got {len(signals)}"
            
            # Calculate per-signal latency
            avg_batch_latency = statistics.mean(batch_latencies)
            per_signal_latency = avg_batch_latency / batch_size
            p99_batch_latency = np.percentile(batch_latencies, 99)
            p99_per_signal = p99_batch_latency / batch_size
            
            # Batch processing should be more efficient
            assert per_signal_latency < 50, f"Batch size {batch_size}: per-signal latency {per_signal_latency:.2f}ms > 50ms"
            assert p99_per_signal < 100, f"Batch size {batch_size}: P99 per-signal latency {p99_per_signal:.2f}ms > 100ms"
            
            print(f"Batch size {batch_size}: avg {per_signal_latency:.2f}ms/signal, P99 {p99_per_signal:.2f}ms/signal")
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation_latency(self, latency_test_system):
        """Test latency under concurrent signal generation load."""
        simulator = latency_test_system['simulator']
        
        # Test concurrent signal generation
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        concurrent_levels = [1, 2, 4, 8, 16]
        
        for concurrency in concurrent_levels:
            latency_results = []
            
            async def generate_concurrent_signals():
                tasks = []
                start_time = time.perf_counter()
                
                for _ in range(concurrency):
                    for symbol in symbols:
                        task = asyncio.create_task(simulator.generate_signal(symbol))
                        tasks.append(task)
                
                signals = await asyncio.gather(*tasks)
                end_time = time.perf_counter()
                
                total_latency = (end_time - start_time) * 1000
                per_signal_latency = total_latency / len(signals)
                
                return per_signal_latency, len(signals)
            
            # Run multiple concurrent batches
            for _ in range(20):
                per_signal_latency, signal_count = await generate_concurrent_signals()
                latency_results.append(per_signal_latency)
            
            # Analyze concurrent performance
            avg_concurrent_latency = statistics.mean(latency_results)
            p99_concurrent_latency = np.percentile(latency_results, 99)
            
            # Latency should remain reasonable under concurrency
            max_acceptable_latency = 100 + (concurrency * 10)  # Allow some degradation
            assert avg_concurrent_latency < max_acceptable_latency, \
                f"Concurrency {concurrency}: avg latency {avg_concurrent_latency:.2f}ms > {max_acceptable_latency}ms"
            assert p99_concurrent_latency < max_acceptable_latency * 1.5, \
                f"Concurrency {concurrency}: P99 latency {p99_concurrent_latency:.2f}ms > {max_acceptable_latency * 1.5}ms"
            
            print(f"Concurrency {concurrency}: avg {avg_concurrent_latency:.2f}ms, P99 {p99_concurrent_latency:.2f}ms")


class TestDataProcessingLatency:
    """Test data processing pipeline latency."""
    
    @pytest.fixture
    async def data_processing_system(self):
        """Create system for data processing latency tests."""
        config = {
            'processing_mode': 'realtime',
            'buffer_size': 5000,
            'batch_processing': True,
            'parallel_workers': 4
        }
        
        system = {
            'data_manager': RealtimeManager(config),
            'profiler': LatencyProfiler()
        }
        
        await system['data_manager'].initialize()
        await system['profiler'].initialize()
        
        yield system
        
        await system['data_manager'].shutdown()
        await system['profiler'].shutdown()
    
    @pytest.mark.asyncio
    async def test_market_data_processing_latency(self, data_processing_system):
        """Test market data processing latency."""
        data_manager = data_processing_system['data_manager']
        profiler = data_processing_system['profiler']
        
        # Generate market data updates
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
        processing_latencies = []
        
        for i in range(1000):
            symbol = symbols[i % len(symbols)]
            market_update = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 5),
                'volume': np.random.randint(100, 5000),
                'timestamp': time.time(),
                'bid': 100 + np.random.normal(0, 5),
                'ask': 100 + np.random.normal(0, 5) + 0.01
            }
            
            start_time = time.perf_counter()
            
            with profiler.measure('market_data_processing'):
                await data_manager.process_market_update(market_update)
            
            end_time = time.perf_counter()
            
            processing_latency = (end_time - start_time) * 1000
            processing_latencies.append(processing_latency)
        
        # Analyze processing latencies
        avg_processing = statistics.mean(processing_latencies)
        p95_processing = np.percentile(processing_latencies, 95)
        p99_processing = np.percentile(processing_latencies, 99)
        
        # Validate processing latency targets
        assert avg_processing < 5, f"Average processing latency {avg_processing:.2f}ms > 5ms"
        assert p95_processing < 10, f"P95 processing latency {p95_processing:.2f}ms > 10ms" 
        assert p99_processing < 20, f"P99 processing latency {p99_processing:.2f}ms > 20ms"
        
        print(f"\nMarket Data Processing Latency:")
        print(f"  Average: {avg_processing:.2f}ms")
        print(f"  P95: {p95_processing:.2f}ms")
        print(f"  P99: {p99_processing:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_news_data_processing_latency(self, data_processing_system):
        """Test news data processing latency."""
        data_manager = data_processing_system['data_manager']
        profiler = data_processing_system['profiler']
        
        # Generate news updates
        news_headlines = [
            "Company reports strong earnings",
            "Market volatility increases amid uncertainty",
            "Technology sector shows resilience",
            "Central bank announces policy changes",
            "Major acquisition deal announced"
        ]
        
        news_processing_latencies = []
        
        for i in range(200):
            news_update = {
                'headline': news_headlines[i % len(news_headlines)],
                'content': f"Full news content for headline {i}",
                'sentiment': np.random.uniform(-1, 1),
                'relevance': np.random.uniform(0.5, 1.0),
                'symbols': ['AAPL', 'GOOGL'],
                'timestamp': time.time(),
                'source': 'test_feed'
            }
            
            start_time = time.perf_counter()
            
            with profiler.measure('news_processing'):
                await data_manager.process_news_update(news_update)
            
            end_time = time.perf_counter()
            
            processing_latency = (end_time - start_time) * 1000
            news_processing_latencies.append(processing_latency)
        
        # Analyze news processing latencies
        avg_news_processing = statistics.mean(news_processing_latencies)
        p95_news_processing = np.percentile(news_processing_latencies, 95)
        p99_news_processing = np.percentile(news_processing_latencies, 99)
        
        # News processing can be slightly slower due to NLP
        assert avg_news_processing < 30, f"Average news processing {avg_news_processing:.2f}ms > 30ms"
        assert p95_news_processing < 50, f"P95 news processing {p95_news_processing:.2f}ms > 50ms"
        assert p99_news_processing < 100, f"P99 news processing {p99_news_processing:.2f}ms > 100ms"
        
        print(f"\nNews Data Processing Latency:")
        print(f"  Average: {avg_news_processing:.2f}ms")
        print(f"  P95: {p95_news_processing:.2f}ms")
        print(f"  P99: {p99_news_processing:.2f}ms")


class TestEndToEndLatency:
    """Test end-to-end pipeline latency."""
    
    @pytest.fixture
    async def e2e_system(self):
        """Create end-to-end test system."""
        config = {
            'mode': 'performance',
            'optimizations': ['caching', 'prefetch', 'parallel'],
            'latency_target_ms': 100
        }
        
        system = {
            'data_manager': RealtimeManager(config),
            'simulator': MarketSimulator(config),
            'benchmark_runner': BenchmarkRunner(config),
            'profiler': LatencyProfiler()
        }
        
        # Initialize all components
        for component in system.values():
            await component.initialize()
        
        yield system
        
        # Cleanup
        for component in system.values():
            await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_latency(self, e2e_system):
        """Test complete data-to-signal-to-execution pipeline latency."""
        data_manager = e2e_system['data_manager']
        simulator = e2e_system['simulator']
        profiler = e2e_system['profiler']
        
        # Test complete pipeline
        pipeline_latencies = []
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        for i in range(500):
            symbol = symbols[i % len(symbols)]
            
            # Create market update
            market_update = {
                'symbol': symbol,
                'price': 150 + np.random.normal(0, 2),
                'volume': np.random.randint(500, 3000),
                'timestamp': time.time()
            }
            
            start_time = time.perf_counter()
            
            with profiler.measure('e2e_pipeline'):
                # Step 1: Process market data
                await data_manager.process_market_update(market_update)
                
                # Step 2: Generate signal
                signal = await simulator.generate_signal(symbol)
                
                # Step 3: Execute trade if signal generated
                if signal and signal.get('action') != 'hold':
                    trade_result = await simulator.execute_trade(signal)
            
            end_time = time.perf_counter()
            
            e2e_latency = (end_time - start_time) * 1000
            pipeline_latencies.append(e2e_latency)
        
        # Analyze end-to-end latencies
        avg_e2e = statistics.mean(pipeline_latencies)
        p95_e2e = np.percentile(pipeline_latencies, 95)
        p99_e2e = np.percentile(pipeline_latencies, 99)
        max_e2e = max(pipeline_latencies)
        
        # Validate end-to-end latency targets
        assert avg_e2e < 50, f"Average E2E latency {avg_e2e:.2f}ms > 50ms"
        assert p95_e2e < 80, f"P95 E2E latency {p95_e2e:.2f}ms > 80ms"
        assert p99_e2e < 100, f"P99 E2E latency {p99_e2e:.2f}ms > 100ms target"
        assert max_e2e < 200, f"Max E2E latency {max_e2e:.2f}ms > 200ms"
        
        print(f"\nEnd-to-End Pipeline Latency:")
        print(f"  Average: {avg_e2e:.2f}ms")
        print(f"  P95: {p95_e2e:.2f}ms")
        print(f"  P99: {p99_e2e:.2f}ms")
        print(f"  Max: {max_e2e:.2f}ms")
        
        # Get profiler breakdown
        profiler_stats = profiler.get_detailed_statistics('e2e_pipeline')
        print(f"  Profiler breakdown: {profiler_stats}")
    
    @pytest.mark.asyncio
    async def test_latency_under_stress(self, e2e_system):
        """Test latency performance under high load conditions."""
        data_manager = e2e_system['data_manager']
        simulator = e2e_system['simulator']
        
        # Stress test parameters
        symbols = [f'STRESS{i:02d}' for i in range(20)]
        updates_per_second = 1000
        test_duration_seconds = 10
        
        stress_latencies = []
        update_count = 0
        
        async def stress_generator():
            nonlocal update_count
            end_time = time.time() + test_duration_seconds
            
            while time.time() < end_time:
                symbol = symbols[update_count % len(symbols)]
                
                market_update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 3),
                    'volume': np.random.randint(100, 2000),
                    'timestamp': time.time()
                }
                
                start_time = time.perf_counter()
                
                # Process update and generate signal
                await data_manager.process_market_update(market_update)
                signal = await simulator.generate_signal(symbol)
                
                end_processing_time = time.perf_counter()
                
                latency = (end_processing_time - start_time) * 1000
                stress_latencies.append(latency)
                
                update_count += 1
                
                # Control rate
                await asyncio.sleep(1.0 / updates_per_second)
        
        # Run stress test
        await stress_generator()
        
        # Analyze stress test results
        stress_avg = statistics.mean(stress_latencies)
        stress_p95 = np.percentile(stress_latencies, 95)
        stress_p99 = np.percentile(stress_latencies, 99)
        
        actual_rate = update_count / test_duration_seconds
        
        # Latency should degrade gracefully under stress
        assert stress_avg < 100, f"Stress average latency {stress_avg:.2f}ms > 100ms"
        assert stress_p95 < 150, f"Stress P95 latency {stress_p95:.2f}ms > 150ms"
        assert stress_p99 < 200, f"Stress P99 latency {stress_p99:.2f}ms > 200ms"
        assert actual_rate > updates_per_second * 0.8, f"Actual rate {actual_rate:.0f} < 80% of target {updates_per_second}"
        
        print(f"\nStress Test Results:")
        print(f"  Target rate: {updates_per_second} updates/sec")
        print(f"  Actual rate: {actual_rate:.0f} updates/sec")
        print(f"  Stress avg latency: {stress_avg:.2f}ms")
        print(f"  Stress P95 latency: {stress_p95:.2f}ms")
        print(f"  Stress P99 latency: {stress_p99:.2f}ms")


class TestLatencyOptimization:
    """Test latency optimization features and effectiveness."""
    
    @pytest.mark.asyncio
    async def test_caching_latency_improvement(self):
        """Test latency improvement from caching."""
        # Test without caching
        no_cache_config = {'cache_enabled': False}
        no_cache_simulator = MarketSimulator(no_cache_config)
        await no_cache_simulator.initialize()
        
        no_cache_latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            await no_cache_simulator.generate_signal('AAPL')
            end_time = time.perf_counter()
            no_cache_latencies.append((end_time - start_time) * 1000)
        
        await no_cache_simulator.shutdown()
        
        # Test with caching
        cache_config = {'cache_enabled': True, 'cache_size': 1000}
        cache_simulator = MarketSimulator(cache_config)
        await cache_simulator.initialize()
        
        # Warm up cache
        for _ in range(10):
            await cache_simulator.generate_signal('AAPL')
        
        cache_latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            await cache_simulator.generate_signal('AAPL')
            end_time = time.perf_counter()
            cache_latencies.append((end_time - start_time) * 1000)
        
        await cache_simulator.shutdown()
        
        # Compare performance
        no_cache_avg = statistics.mean(no_cache_latencies)
        cache_avg = statistics.mean(cache_latencies)
        improvement = (no_cache_avg - cache_avg) / no_cache_avg
        
        assert improvement > 0.2, f"Caching improvement {improvement:.1%} < 20%"
        assert cache_avg < 50, f"Cached latency {cache_avg:.2f}ms still > 50ms"
        
        print(f"\nCaching Performance:")
        print(f"  No cache avg: {no_cache_avg:.2f}ms")
        print(f"  With cache avg: {cache_avg:.2f}ms")
        print(f"  Improvement: {improvement:.1%}")
    
    @pytest.mark.asyncio
    async def test_parallel_processing_latency(self):
        """Test latency improvement from parallel processing."""
        # Sequential processing
        sequential_config = {'parallel_processing': False}
        sequential_simulator = MarketSimulator(sequential_config)
        await sequential_simulator.initialize()
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        start_time = time.perf_counter()
        for symbol in symbols:
            await sequential_simulator.generate_signal(symbol)
        sequential_time = (time.perf_counter() - start_time) * 1000
        
        await sequential_simulator.shutdown()
        
        # Parallel processing
        parallel_config = {'parallel_processing': True, 'max_workers': 4}
        parallel_simulator = MarketSimulator(parallel_config)
        await parallel_simulator.initialize()
        
        start_time = time.perf_counter()
        signals = await parallel_simulator.generate_signals_batch(symbols)
        parallel_time = (time.perf_counter() - start_time) * 1000
        
        await parallel_simulator.shutdown()
        
        # Validate improvement
        speedup = sequential_time / parallel_time
        assert speedup > 2.0, f"Parallel speedup {speedup:.1f}x < 2.0x"
        assert parallel_time < 100, f"Parallel time {parallel_time:.2f}ms > 100ms"
        
        print(f"\nParallel Processing Performance:")
        print(f"  Sequential time: {sequential_time:.2f}ms")
        print(f"  Parallel time: {parallel_time:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")


if __name__ == '__main__':
    pytest.main([__file__])