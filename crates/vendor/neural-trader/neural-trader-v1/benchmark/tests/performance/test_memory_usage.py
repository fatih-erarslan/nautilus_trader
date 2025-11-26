"""
Memory usage performance validation test suite for AI News Trading benchmark system.

This module validates that memory usage stays within the 2GB limit during 8-hour simulations.
Tests include:
- Memory usage during extended simulations
- Memory leak detection
- Memory efficiency optimization
- Garbage collection effectiveness
- Memory scaling with load
"""

import asyncio
import gc
import time
import pytest
import psutil
import os
from unittest.mock import Mock, patch
import numpy as np
import threading
from memory_profiler import profile
import tracemalloc

from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.profiling.profiler import MemoryProfiler


class TestMemoryUsageLimits:
    """Test memory usage within specified limits."""
    
    @pytest.fixture
    async def memory_monitored_system(self):
        """Create system with memory monitoring enabled."""
        config = {
            'memory_monitoring': True,
            'memory_limit_gb': 2,
            'gc_threshold_mb': 100,
            'memory_cleanup_interval': 60,
            'buffer_management': 'adaptive'
        }
        
        # Start memory tracing
        tracemalloc.start()
        
        system = {
            'simulator': MarketSimulator(config),
            'data_manager': RealtimeManager(config),
            'memory_profiler': MemoryProfiler(),
            'process': psutil.Process(os.getpid())
        }
        
        for component_name, component in system.items():
            if hasattr(component, 'initialize'):
                await component.initialize()
        
        yield system
        
        # Cleanup and stop tracing
        for component_name, component in system.items():
            if hasattr(component, 'shutdown'):
                await component.shutdown()
        
        tracemalloc.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_simulation(self, memory_monitored_system):
        """Test memory usage during extended simulation."""
        simulator = memory_monitored_system['simulator']
        data_manager = memory_monitored_system['data_manager']
        process = memory_monitored_system['process']
        memory_profiler = memory_monitored_system['memory_profiler']
        
        # Record initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_measurements = [initial_memory]
        
        # Simulate 8 hours compressed into 8 minutes (60x speed)
        simulation_minutes = 8
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'INTC', 'NFLX']
        
        print(f"Starting memory test - Initial memory: {initial_memory:.1f} MB")
        
        for minute in range(simulation_minutes):
            minute_start_memory = process.memory_info().rss / 1024 / 1024
            
            # Simulate 1 hour of trading data in 1 minute
            for second in range(60):
                # Generate market updates for all symbols
                for symbol in symbols:
                    market_update = {
                        'symbol': symbol,
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 5000),
                        'bid': 100 + np.random.normal(0, 5),
                        'ask': 100 + np.random.normal(0, 5) + 0.01,
                        'timestamp': time.time()
                    }
                    await data_manager.process_market_update(market_update)
                
                # Generate signals and execute trades
                for symbol in symbols:
                    signal = await simulator.generate_signal(symbol)
                    if signal and signal.get('action') != 'hold':
                        await simulator.execute_trade(signal)
                
                # Generate occasional news
                if second % 10 == 0:
                    news_update = {
                        'headline': f'Market update minute {minute}, second {second}',
                        'sentiment': np.random.uniform(-1, 1),
                        'relevance': np.random.uniform(0.5, 1.0),
                        'symbols': [symbols[second % len(symbols)]],
                        'timestamp': time.time()
                    }
                    await data_manager.process_news_update(news_update)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            # Record memory after each "hour"
            minute_end_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(minute_end_memory)
            
            # Force garbage collection
            gc.collect()
            gc_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"Hour {minute + 1}: {minute_start_memory:.1f} -> {minute_end_memory:.1f} -> {gc_memory:.1f} MB (after GC)")
            
            # Check memory growth
            memory_growth = gc_memory - initial_memory
            assert gc_memory < 2048, f"Memory usage {gc_memory:.1f}MB exceeds 2GB limit at hour {minute + 1}"
            
            # Memory growth should be reasonable
            max_acceptable_growth = 200 * (minute + 1)  # 200MB per hour max
            assert memory_growth < max_acceptable_growth, \
                f"Memory growth {memory_growth:.1f}MB exceeds {max_acceptable_growth:.1f}MB at hour {minute + 1}"
        
        # Final memory analysis
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        growth_rate = total_growth / simulation_minutes  # MB per hour
        
        # Validate final memory state
        assert final_memory < 2048, f"Final memory {final_memory:.1f}MB exceeds 2GB limit"
        assert total_growth < 1024, f"Total memory growth {total_growth:.1f}MB exceeds 1GB"
        assert growth_rate < 100, f"Memory growth rate {growth_rate:.1f}MB/hour too high"
        
        print(f"\nMemory Test Summary:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {total_growth:.1f} MB")
        print(f"  Growth rate: {growth_rate:.1f} MB/hour")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, memory_monitored_system):
        """Test for memory leaks during repeated operations."""
        simulator = memory_monitored_system['simulator']
        data_manager = memory_monitored_system['data_manager']
        process = memory_monitored_system['process']
        
        # Take initial memory snapshot
        tracemalloc_snapshot1 = tracemalloc.take_snapshot()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform repeated operations that should not leak memory
        symbols = ['LEAK_TEST_1', 'LEAK_TEST_2', 'LEAK_TEST_3']
        iterations = 1000
        
        for i in range(iterations):
            # Create and process market data
            for symbol in symbols:
                market_data = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 2),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time(),
                    'iteration': i
                }
                await data_manager.process_market_update(market_data)
                
                # Generate signal
                signal = await simulator.generate_signal(symbol)
                
                # Execute trade if signal generated
                if signal and signal.get('action') != 'hold':
                    trade_result = await simulator.execute_trade(signal)
            
            # Periodic garbage collection
            if i % 100 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                assert memory_growth < 100, f"Iteration {i}: Memory growth {memory_growth:.1f}MB suggests leak"
        
        # Force final garbage collection
        gc.collect()
        
        # Take final memory snapshot
        tracemalloc_snapshot2 = tracemalloc.take_snapshot()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Compare snapshots to detect leaks
        top_stats = tracemalloc_snapshot2.compare_to(tracemalloc_snapshot1, 'lineno')
        
        # Check for significant memory increases
        significant_increases = [stat for stat in top_stats[:10] if stat.size_diff > 1024 * 1024]  # > 1MB increase
        
        # Memory increase should be minimal after GC
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB after {iterations} iterations"
        
        # Log memory leak analysis
        if significant_increases:
            print(f"\nPotential memory increases detected:")
            for stat in significant_increases[:5]:
                print(f"  {stat}")
        
        print(f"\nMemory Leak Test:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Iterations: {iterations}")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_optimization(self, memory_monitored_system):
        """Test memory efficiency optimizations."""
        simulator = memory_monitored_system['simulator']
        data_manager = memory_monitored_system['data_manager']
        process = memory_monitored_system['process']
        
        # Test without optimizations
        unoptimized_config = {
            'buffer_management': 'static',
            'cache_enabled': False,
            'gc_optimization': False,
            'memory_pooling': False
        }
        
        unoptimized_simulator = MarketSimulator(unoptimized_config)
        await unoptimized_simulator.initialize()
        
        # Run workload without optimizations
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        symbols = ['OPT_TEST_1', 'OPT_TEST_2', 'OPT_TEST_3', 'OPT_TEST_4']
        workload_size = 5000
        
        for i in range(workload_size):
            symbol = symbols[i % len(symbols)]
            
            market_data = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 3),
                'volume': np.random.randint(100, 2000),
                'timestamp': time.time()
            }
            
            # Process with unoptimized system
            await data_manager.process_market_update(market_data)
            signal = await unoptimized_simulator.generate_signal(symbol)
        
        unoptimized_memory = process.memory_info().rss / 1024 / 1024
        unoptimized_usage = unoptimized_memory - initial_memory
        
        await unoptimized_simulator.shutdown()
        gc.collect()
        
        # Test with optimizations enabled
        optimized_config = {
            'buffer_management': 'adaptive',
            'cache_enabled': True,
            'gc_optimization': True,
            'memory_pooling': True,
            'lazy_loading': True
        }
        
        optimized_simulator = MarketSimulator(optimized_config)
        await optimized_simulator.initialize()
        
        # Reset memory baseline
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Run same workload with optimizations
        for i in range(workload_size):
            symbol = symbols[i % len(symbols)]
            
            market_data = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 3),
                'volume': np.random.randint(100, 2000),
                'timestamp': time.time()
            }
            
            # Process with optimized system
            await data_manager.process_market_update(market_data)
            signal = await optimized_simulator.generate_signal(symbol)
        
        optimized_memory = process.memory_info().rss / 1024 / 1024
        optimized_usage = optimized_memory - baseline_memory
        
        await optimized_simulator.shutdown()
        
        # Compare memory efficiency
        efficiency_improvement = (unoptimized_usage - optimized_usage) / unoptimized_usage
        
        assert optimized_usage < unoptimized_usage, \
            f"Optimized memory {optimized_usage:.1f}MB not better than unoptimized {unoptimized_usage:.1f}MB"
        assert efficiency_improvement > 0.2, \
            f"Memory optimization improvement {efficiency_improvement:.1%} < 20%"
        
        print(f"\nMemory Efficiency Test:")
        print(f"  Unoptimized usage: {unoptimized_usage:.1f} MB")
        print(f"  Optimized usage: {optimized_usage:.1f} MB")
        print(f"  Improvement: {efficiency_improvement:.1%}")


class TestMemoryScaling:
    """Test memory usage scaling with system load."""
    
    @pytest.mark.asyncio
    async def test_memory_scaling_with_symbols(self):
        """Test memory usage scaling with number of symbols."""
        symbol_counts = [10, 25, 50, 100, 200]
        memory_measurements = []
        
        for symbol_count in symbol_counts:
            # Create system for this symbol count
            config = {
                'symbol_count': symbol_count,
                'buffer_per_symbol': 1000,
                'memory_optimization': True
            }
            
            simulator = MarketSimulator(config)
            data_manager = RealtimeManager(config)
            
            await simulator.initialize()
            await data_manager.initialize()
            
            # Generate symbols
            symbols = [f'SCALE{i:03d}' for i in range(symbol_count)]
            
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Load data for all symbols
            for symbol in symbols:
                # Create market data buffer
                for i in range(100):  # 100 data points per symbol
                    market_data = {
                        'symbol': symbol,
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 1000),
                        'timestamp': time.time() + i
                    }
                    await data_manager.process_market_update(market_data)
                
                # Generate initial signal
                await simulator.generate_signal(symbol)
            
            # Measure memory after loading
            loaded_memory = process.memory_info().rss / 1024 / 1024
            memory_per_symbol = (loaded_memory - initial_memory) / symbol_count
            memory_measurements.append((symbol_count, loaded_memory - initial_memory, memory_per_symbol))
            
            await simulator.shutdown()
            await data_manager.shutdown()
            
            print(f"Symbols {symbol_count}: {loaded_memory - initial_memory:.1f} MB total, {memory_per_symbol:.2f} MB/symbol")
        
        # Analyze scaling characteristics
        # Memory per symbol should remain relatively constant (good scaling)
        memory_per_symbol_list = [measurement[2] for measurement in memory_measurements]
        
        # Memory per symbol should not increase dramatically
        min_per_symbol = min(memory_per_symbol_list)
        max_per_symbol = max(memory_per_symbol_list)
        scaling_ratio = max_per_symbol / min_per_symbol
        
        assert scaling_ratio < 2.0, f"Memory per symbol scaling ratio {scaling_ratio:.1f} > 2.0 (poor scaling)"
        
        # Total memory should be reasonable even at high symbol counts
        max_total_memory = memory_measurements[-1][1]  # Highest symbol count total memory
        assert max_total_memory < 1024, f"Memory for {symbol_counts[-1]} symbols: {max_total_memory:.1f}MB > 1GB"
        
        print(f"\nMemory Scaling Analysis:")
        print(f"  Memory per symbol range: {min_per_symbol:.2f} - {max_per_symbol:.2f} MB")
        print(f"  Scaling ratio: {scaling_ratio:.1f}")
    
    @pytest.mark.asyncio
    async def test_memory_scaling_with_load(self):
        """Test memory usage scaling with processing load."""
        load_levels = [100, 500, 1000, 2000, 5000]  # Updates per minute
        
        config = {
            'memory_monitoring': True,
            'adaptive_buffering': True
        }
        
        simulator = MarketSimulator(config)
        data_manager = RealtimeManager(config)
        
        await simulator.initialize()
        await data_manager.initialize()
        
        process = psutil.Process(os.getpid())
        symbols = ['LOAD_A', 'LOAD_B', 'LOAD_C', 'LOAD_D', 'LOAD_E']
        
        for load_level in load_levels:
            # Reset system state
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate load for 1 minute
            updates_per_second = load_level / 60
            test_duration = 60  # seconds
            
            update_count = 0
            start_time = time.time()
            
            while time.time() - start_time < test_duration:
                symbol = symbols[update_count % len(symbols)]
                
                market_update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 3),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time()
                }
                
                await data_manager.process_market_update(market_update)
                signal = await simulator.generate_signal(symbol)
                
                update_count += 1
                
                # Control update rate
                if updates_per_second > 0:
                    await asyncio.sleep(1.0 / updates_per_second)
            
            # Measure memory after load test
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            memory_per_update = memory_increase / update_count if update_count > 0 else 0
            
            # Memory increase should be bounded
            max_acceptable_increase = min(200, load_level * 0.1)  # Max 200MB or 0.1MB per update/min
            assert memory_increase < max_acceptable_increase, \
                f"Load {load_level}: memory increase {memory_increase:.1f}MB > {max_acceptable_increase:.1f}MB"
            
            print(f"Load {load_level} updates/min: {memory_increase:.1f} MB increase, {memory_per_update*1000:.3f} KB/update")
        
        await simulator.shutdown()
        await data_manager.shutdown()


class TestGarbageCollectionEffectiveness:
    """Test garbage collection effectiveness and tuning."""
    
    @pytest.mark.asyncio
    async def test_gc_effectiveness(self):
        """Test garbage collection effectiveness."""
        config = {
            'gc_monitoring': True,
            'gc_tuning': True
        }
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        process = psutil.Process(os.getpid())
        
        # Generate objects that should be garbage collected
        symbols = ['GC_TEST_1', 'GC_TEST_2', 'GC_TEST_3']
        objects_created = 0
        
        # Disable automatic GC to test manual collection
        gc.disable()
        
        try:
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create many temporary objects
            for cycle in range(10):
                temp_data = []
                
                for i in range(1000):
                    # Create objects that should be collectible
                    market_data = {
                        'symbol': symbols[i % len(symbols)],
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 1000),
                        'timestamp': time.time(),
                        'large_data': [np.random.random() for _ in range(100)]  # Make objects larger
                    }
                    
                    temp_data.append(market_data)
                    objects_created += 1
                
                # Process data
                for data in temp_data:
                    await simulator.generate_signal(data['symbol'])
                
                # Clear references (should make objects collectible)
                temp_data.clear()
                del temp_data
                
                # Measure memory before GC
                pre_gc_memory = process.memory_info().rss / 1024 / 1024
                
                # Force garbage collection
                collected = gc.collect()
                
                # Measure memory after GC
                post_gc_memory = process.memory_info().rss / 1024 / 1024
                memory_freed = pre_gc_memory - post_gc_memory
                
                print(f"Cycle {cycle}: {collected} objects collected, {memory_freed:.1f} MB freed")
                
                # GC should free some memory
                if cycle > 2:  # Allow initial cycles for warmup
                    assert memory_freed > 0 or collected > 0, f"Cycle {cycle}: GC freed {memory_freed:.1f}MB, collected {collected} objects"
            
            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_change = final_memory - initial_memory
            
            # Memory should not have grown excessively
            assert total_memory_change < 100, f"Memory grew by {total_memory_change:.1f}MB despite GC"
            
            print(f"\nGC Effectiveness Test:")
            print(f"  Objects created: {objects_created}")
            print(f"  Initial memory: {initial_memory:.1f} MB")
            print(f"  Final memory: {final_memory:.1f} MB")
            print(f"  Net change: {total_memory_change:.1f} MB")
            
        finally:
            gc.enable()
            await simulator.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_pooling_effectiveness(self):
        """Test memory pooling effectiveness."""
        # Test without memory pooling
        no_pool_config = {'memory_pooling': False}
        no_pool_simulator = MarketSimulator(no_pool_config)
        await no_pool_simulator.initialize()
        
        process = psutil.Process(os.getpid())
        
        # Measure memory usage without pooling
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create and destroy many objects
        for i in range(5000):
            market_data = {
                'symbol': f'POOL_TEST_{i % 10}',
                'price': 100 + np.random.normal(0, 3),
                'volume': np.random.randint(100, 1000),
                'data_array': np.random.random(50).tolist()
            }
            
            await no_pool_simulator.generate_signal(market_data['symbol'])
        
        no_pool_memory = process.memory_info().rss / 1024 / 1024
        no_pool_usage = no_pool_memory - initial_memory
        
        await no_pool_simulator.shutdown()
        
        # Test with memory pooling
        pool_config = {'memory_pooling': True, 'pool_size': 1000}
        pool_simulator = MarketSimulator(pool_config)
        await pool_simulator.initialize()
        
        gc.collect()
        pool_initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Same workload with pooling
        for i in range(5000):
            market_data = {
                'symbol': f'POOL_TEST_{i % 10}',
                'price': 100 + np.random.normal(0, 3),
                'volume': np.random.randint(100, 1000),
                'data_array': np.random.random(50).tolist()
            }
            
            await pool_simulator.generate_signal(market_data['symbol'])
        
        pool_memory = process.memory_info().rss / 1024 / 1024
        pool_usage = pool_memory - pool_initial_memory
        
        await pool_simulator.shutdown()
        
        # Compare memory efficiency
        pooling_efficiency = (no_pool_usage - pool_usage) / no_pool_usage if no_pool_usage > 0 else 0
        
        assert pool_usage <= no_pool_usage, \
            f"Pooled memory usage {pool_usage:.1f}MB > no-pool usage {no_pool_usage:.1f}MB"
        
        print(f"\nMemory Pooling Test:")
        print(f"  No pooling: {no_pool_usage:.1f} MB")
        print(f"  With pooling: {pool_usage:.1f} MB")
        print(f"  Efficiency gain: {pooling_efficiency:.1%}")


if __name__ == '__main__':
    pytest.main([__file__])