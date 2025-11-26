"""
Scalability performance validation test suite for AI News Trading benchmark system.

This module validates that the system can handle 100+ concurrent symbols and scales effectively.
Tests include:
- Concurrent symbol processing
- User concurrency scaling
- System resource scaling
- Load balancing effectiveness
- Performance degradation under load
"""

import asyncio
import time
import statistics
import pytest
from unittest.mock import Mock, patch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing
import psutil

from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.profiling.profiler import ScalabilityProfiler


class TestConcurrentSymbolScaling:
    """Test scaling with concurrent symbol processing."""
    
    @pytest.fixture
    async def scalable_system(self):
        """Create system optimized for scalability testing."""
        config = {
            'scalability_mode': True,
            'max_concurrent_symbols': 200,
            'symbol_processing_pool_size': 16,
            'load_balancing': True,
            'adaptive_resource_allocation': True,
            'performance_monitoring': True
        }
        
        system = {
            'simulator': MarketSimulator(config),
            'data_manager': RealtimeManager(config),
            'benchmark_runner': BenchmarkRunner(config),
            'profiler': ScalabilityProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing_scaling(self, scalable_system):
        """Test processing scaling with increasing number of concurrent symbols."""
        simulator = scalable_system['simulator']
        data_manager = scalable_system['data_manager']
        profiler = scalable_system['profiler']
        
        # Test different numbers of concurrent symbols
        symbol_counts = [10, 25, 50, 100, 150, 200]
        scaling_results = []
        
        for symbol_count in symbol_counts:
            symbols = [f'SCALE{i:03d}' for i in range(symbol_count)]
            
            # Configure system for this symbol count
            await data_manager.configure_symbols(symbols)
            
            # Generate market data for all symbols
            market_updates = []
            for symbol in symbols:
                update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': np.random.randint(100, 2000),
                    'timestamp': time.time()
                }
                market_updates.append(update)
            
            # Measure concurrent processing performance
            start_time = time.perf_counter()
            
            with profiler.measure(f'concurrent_symbols_{symbol_count}'):
                # Process all market updates concurrently
                update_tasks = []
                for update in market_updates:
                    task = asyncio.create_task(data_manager.process_market_update(update))
                    update_tasks.append(task)
                
                await asyncio.gather(*update_tasks)
                
                # Generate signals for all symbols concurrently
                signal_tasks = []
                for symbol in symbols:
                    task = asyncio.create_task(simulator.generate_signal(symbol))
                    signal_tasks.append(task)
                
                signals = await asyncio.gather(*signal_tasks)
            
            end_time = time.perf_counter()
            
            # Calculate performance metrics
            duration = end_time - start_time
            symbols_per_second = symbol_count / duration
            
            # Validate scaling performance
            min_expected_rate = symbol_count * 5  # At least 5 symbols/sec per symbol
            assert symbols_per_second > min_expected_rate, \
                f"Symbols {symbol_count}: rate {symbols_per_second:.0f}/sec < {min_expected_rate}/sec"
            
            scaling_results.append({
                'symbol_count': symbol_count,
                'duration': duration,
                'symbols_per_second': symbols_per_second,
                'signals_generated': len([s for s in signals if s])
            })
            
            print(f"Symbols {symbol_count}: {symbols_per_second:.0f} symbols/sec, {duration:.2f}s")
        
        # Analyze scaling characteristics
        self._analyze_scaling_efficiency(scaling_results)
    
    def _analyze_scaling_efficiency(self, results):
        """Analyze scaling efficiency from results."""
        # Calculate scaling efficiency
        base_result = results[0]
        base_rate_per_symbol = base_result['symbols_per_second'] / base_result['symbol_count']
        
        for result in results[1:]:
            current_rate_per_symbol = result['symbols_per_second'] / result['symbol_count']
            efficiency = current_rate_per_symbol / base_rate_per_symbol
            
            # Efficiency should not degrade too much
            min_efficiency = 0.5  # Should maintain at least 50% efficiency
            assert efficiency > min_efficiency, \
                f"Scaling efficiency {efficiency:.2f} < {min_efficiency} at {result['symbol_count']} symbols"
            
            print(f"  {result['symbol_count']} symbols: {efficiency:.2f} efficiency")
    
    @pytest.mark.asyncio
    async def test_symbol_load_balancing(self, scalable_system):
        """Test load balancing across symbol processing workers."""
        simulator = scalable_system['simulator']
        data_manager = scalable_system['data_manager']
        
        # Configure load balancing
        worker_count = 8
        symbols_per_worker = 25
        total_symbols = worker_count * symbols_per_worker
        
        symbols = [f'LB{i:03d}' for i in range(total_symbols)]
        
        # Track per-worker performance
        worker_performance = {i: [] for i in range(worker_count)}
        
        async def worker_load_test(worker_id, worker_symbols):
            worker_start = time.perf_counter()
            
            for symbol in worker_symbols:
                # Process market data
                market_update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 3),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time(),
                    'worker_id': worker_id
                }
                
                await data_manager.process_market_update(market_update)
                
                # Generate signal
                signal = await simulator.generate_signal(symbol)
                
                if signal:
                    worker_performance[worker_id].append(signal)
            
            worker_end = time.perf_counter()
            return worker_id, worker_end - worker_start, len(worker_performance[worker_id])
        
        # Create worker tasks
        tasks = []
        for worker_id in range(worker_count):
            start_idx = worker_id * symbols_per_worker
            end_idx = start_idx + symbols_per_worker
            worker_symbols = symbols[start_idx:end_idx]
            
            task = asyncio.create_task(worker_load_test(worker_id, worker_symbols))
            tasks.append(task)
        
        # Execute load balancing test
        start_time = time.perf_counter()
        worker_results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        # Analyze load balancing effectiveness
        total_duration = end_time - start_time
        worker_durations = [result[1] for result in worker_results]
        worker_signal_counts = [result[2] for result in worker_results]
        
        # Load should be balanced across workers
        avg_duration = statistics.mean(worker_durations)
        max_duration = max(worker_durations)
        min_duration = min(worker_durations)
        
        load_balance_ratio = max_duration / min_duration if min_duration > 0 else float('inf')
        
        # Load balancing should keep workers relatively equal
        assert load_balance_ratio < 2.0, f"Load balance ratio {load_balance_ratio:.1f} > 2.0 (poor balancing)"
        
        # All workers should complete work
        assert min(worker_signal_counts) > 0, "Some workers generated no signals"
        
        print(f"\nLoad Balancing Test:")
        print(f"  Workers: {worker_count}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Worker duration range: {min_duration:.2f}s - {max_duration:.2f}s")
        print(f"  Load balance ratio: {load_balance_ratio:.1f}")
        print(f"  Signal counts: {worker_signal_counts}")
    
    @pytest.mark.asyncio
    async def test_dynamic_symbol_addition(self, scalable_system):
        """Test dynamic addition of symbols during runtime."""
        simulator = scalable_system['simulator']
        data_manager = scalable_system['data_manager']
        
        # Start with initial symbols
        initial_symbols = [f'DYN{i:02d}' for i in range(20)]
        await data_manager.configure_symbols(initial_symbols)
        
        # Track system performance
        performance_metrics = []
        
        # Process initial symbols
        for symbol in initial_symbols:
            start_time = time.perf_counter()
            
            market_update = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 2),
                'volume': np.random.randint(100, 1000),
                'timestamp': time.time()
            }
            
            await data_manager.process_market_update(market_update)
            signal = await simulator.generate_signal(symbol)
            
            end_time = time.perf_counter()
            performance_metrics.append(end_time - start_time)
        
        initial_avg_latency = statistics.mean(performance_metrics) * 1000  # ms
        
        # Dynamically add more symbols in batches
        for batch in range(5):
            new_symbols = [f'NEW{batch:02d}_{i:02d}' for i in range(20)]
            
            # Add new symbols to system
            await data_manager.add_symbols(new_symbols)
            
            # Test performance with expanded symbol set
            all_symbols = initial_symbols + [s for b in range(batch + 1) for s in [f'NEW{b:02d}_{i:02d}' for i in range(20)]]
            
            batch_metrics = []
            for symbol in all_symbols[-20:]:  # Test last 20 symbols
                start_time = time.perf_counter()
                
                market_update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 2),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time()
                }
                
                await data_manager.process_market_update(market_update)
                signal = await simulator.generate_signal(symbol)
                
                end_time = time.perf_counter()
                batch_metrics.append(end_time - start_time)
            
            batch_avg_latency = statistics.mean(batch_metrics) * 1000  # ms
            
            # Performance should not degrade significantly
            degradation = (batch_avg_latency - initial_avg_latency) / initial_avg_latency
            assert degradation < 0.5, f"Batch {batch}: latency degradation {degradation:.1%} > 50%"
            
            print(f"Batch {batch}: {len(all_symbols)} total symbols, {batch_avg_latency:.1f}ms avg latency")


class TestUserConcurrencyScaling:
    """Test scaling with concurrent users/sessions."""
    
    @pytest.fixture
    async def multi_user_system(self):
        """Create system for multi-user testing."""
        config = {
            'multi_user_mode': True,
            'max_concurrent_users': 50,
            'session_isolation': True,
            'resource_quotas': True,
            'user_load_balancing': True
        }
        
        system = {
            'benchmark_runner': BenchmarkRunner(config),
            'profiler': ScalabilityProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, multi_user_system):
        """Test concurrent user session handling."""
        benchmark_runner = multi_user_system['benchmark_runner']
        profiler = multi_user_system['profiler']
        
        # Test increasing numbers of concurrent users
        user_counts = [5, 10, 20, 30, 40, 50]
        
        for user_count in user_counts:
            user_performance = []
            
            async def user_session(user_id):
                session_start = time.perf_counter()
                
                # Each user runs a mini benchmark
                user_config = {
                    'user_id': user_id,
                    'symbols': [f'USER{user_id}_{i}' for i in range(5)],
                    'strategies': ['momentum'],
                    'duration': 10  # 10 second benchmark
                }
                
                try:
                    results = await benchmark_runner.run_user_benchmark(user_config)
                    session_end = time.perf_counter()
                    
                    return {
                        'user_id': user_id,
                        'duration': session_end - session_start,
                        'success': True,
                        'results': results
                    }
                except Exception as e:
                    session_end = time.perf_counter()
                    return {
                        'user_id': user_id,
                        'duration': session_end - session_start,
                        'success': False,
                        'error': str(e)
                    }
            
            # Run concurrent user sessions
            start_time = time.perf_counter()
            
            with profiler.measure(f'concurrent_users_{user_count}'):
                user_tasks = [user_session(user_id) for user_id in range(user_count)]
                user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            
            # Analyze concurrent user performance
            total_duration = end_time - start_time
            successful_users = sum(1 for result in user_results if isinstance(result, dict) and result.get('success'))
            success_rate = successful_users / user_count
            
            # Validate concurrent user handling
            assert success_rate > 0.9, f"Users {user_count}: success rate {success_rate:.2%} < 90%"
            assert total_duration < 30, f"Users {user_count}: total time {total_duration:.1f}s > 30s"
            
            # Calculate user latency metrics
            user_durations = [result['duration'] for result in user_results if isinstance(result, dict)]
            avg_user_duration = statistics.mean(user_durations) if user_durations else 0
            max_user_duration = max(user_durations) if user_durations else 0
            
            print(f"Users {user_count}: {success_rate:.1%} success, avg {avg_user_duration:.1f}s, max {max_user_duration:.1f}s")
    
    @pytest.mark.asyncio
    async def test_user_isolation_effectiveness(self, multi_user_system):
        """Test that user sessions are properly isolated."""
        benchmark_runner = multi_user_system['benchmark_runner']
        
        # Create users with different workloads
        async def heavy_user(user_id):
            """User with heavy computational workload."""
            config = {
                'user_id': user_id,
                'symbols': [f'HEAVY{user_id}_{i}' for i in range(20)],
                'strategies': ['momentum', 'arbitrage', 'news_sentiment'],
                'computation_intensity': 'high',
                'duration': 15
            }
            
            start_time = time.perf_counter()
            results = await benchmark_runner.run_user_benchmark(config)
            end_time = time.perf_counter()
            
            return {
                'user_type': 'heavy',
                'user_id': user_id,
                'duration': end_time - start_time,
                'results': results
            }
        
        async def light_user(user_id):
            """User with light computational workload."""
            config = {
                'user_id': user_id,
                'symbols': [f'LIGHT{user_id}_{i}' for i in range(3)],
                'strategies': ['momentum'],
                'computation_intensity': 'low',
                'duration': 5
            }
            
            start_time = time.perf_counter()
            results = await benchmark_runner.run_user_benchmark(config)
            end_time = time.perf_counter()
            
            return {
                'user_type': 'light',
                'user_id': user_id,
                'duration': end_time - start_time,
                'results': results
            }
        
        # Run mixed workload
        heavy_users = 3
        light_users = 12
        
        tasks = []
        tasks.extend([heavy_user(i) for i in range(heavy_users)])
        tasks.extend([light_user(i + heavy_users) for i in range(light_users)])
        
        results = await asyncio.gather(*tasks)
        
        # Analyze isolation effectiveness
        heavy_results = [r for r in results if r['user_type'] == 'heavy']
        light_results = [r for r in results if r['user_type'] == 'light']
        
        heavy_durations = [r['duration'] for r in heavy_results]
        light_durations = [r['duration'] for r in light_results]
        
        avg_heavy_duration = statistics.mean(heavy_durations)
        avg_light_duration = statistics.mean(light_durations)
        
        # Light users should not be significantly impacted by heavy users
        expected_light_duration = 5  # Base duration for light users
        light_impact = (avg_light_duration - expected_light_duration) / expected_light_duration
        
        assert light_impact < 0.5, f"Light user impact {light_impact:.1%} > 50% (poor isolation)"
        
        print(f"\nUser Isolation Test:")
        print(f"  Heavy users: {len(heavy_results)}, avg duration: {avg_heavy_duration:.1f}s")
        print(f"  Light users: {len(light_results)}, avg duration: {avg_light_duration:.1f}s")
        print(f"  Light user impact: {light_impact:.1%}")


class TestSystemResourceScaling:
    """Test system resource scaling."""
    
    @pytest.mark.asyncio
    async def test_cpu_scaling_effectiveness(self):
        """Test CPU resource scaling."""
        # Test with different CPU worker configurations
        cpu_configs = [
            {'cpu_workers': 1, 'thread_pool_size': 2},
            {'cpu_workers': 2, 'thread_pool_size': 4},
            {'cpu_workers': 4, 'thread_pool_size': 8},
            {'cpu_workers': 8, 'thread_pool_size': 16}
        ]
        
        workload_size = 2000  # Number of computations
        scaling_results = []
        
        for config in cpu_configs:
            simulator = MarketSimulator(config)
            await simulator.initialize()
            
            # CPU-intensive workload
            symbols = [f'CPU{i:03d}' for i in range(100)]
            
            start_time = time.perf_counter()
            
            # Distribute work across CPU workers
            tasks = []
            for i in range(workload_size):
                symbol = symbols[i % len(symbols)]
                
                # CPU-intensive signal generation
                task = asyncio.create_task(simulator.generate_complex_signal(symbol))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            throughput = workload_size / duration
            
            scaling_results.append({
                'cpu_workers': config['cpu_workers'],
                'duration': duration,
                'throughput': throughput
            })
            
            await simulator.shutdown()
            
            print(f"CPU workers {config['cpu_workers']}: {throughput:.0f} ops/sec")
        
        # Analyze CPU scaling
        base_throughput = scaling_results[0]['throughput']
        
        for result in scaling_results[1:]:
            expected_speedup = result['cpu_workers']
            actual_speedup = result['throughput'] / base_throughput
            scaling_efficiency = actual_speedup / expected_speedup
            
            # Scaling efficiency should be reasonable
            assert scaling_efficiency > 0.5, \
                f"CPU scaling efficiency {scaling_efficiency:.2f} < 0.5 with {result['cpu_workers']} workers"
            
            print(f"  {result['cpu_workers']} workers: {actual_speedup:.1f}x speedup, {scaling_efficiency:.2f} efficiency")
    
    @pytest.mark.asyncio
    async def test_memory_scaling_under_load(self):
        """Test memory scaling under different load conditions."""
        load_configs = [
            {'concurrent_operations': 100, 'data_size': 'small'},
            {'concurrent_operations': 500, 'data_size': 'medium'},
            {'concurrent_operations': 1000, 'data_size': 'large'},
            {'concurrent_operations': 2000, 'data_size': 'xlarge'}
        ]
        
        process = psutil.Process()
        
        for config in load_configs:
            # Configure system for this load level
            system_config = {
                'memory_management': 'adaptive',
                'max_concurrent_ops': config['concurrent_operations'],
                'data_optimization': True
            }
            
            simulator = MarketSimulator(system_config)
            data_manager = RealtimeManager(system_config)
            
            await simulator.initialize()
            await data_manager.initialize()
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate load
            symbols = [f'MEM{i:03d}' for i in range(config['concurrent_operations'])]
            
            # Create data of specified size
            data_sizes = {
                'small': 10,
                'medium': 100,
                'large': 1000,
                'xlarge': 5000
            }
            data_points = data_sizes[config['data_size']]
            
            # Process concurrent operations
            tasks = []
            for symbol in symbols:
                market_data = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': np.random.randint(100, 1000),
                    'large_data': [np.random.random() for _ in range(data_points)],
                    'timestamp': time.time()
                }
                
                task = asyncio.create_task(data_manager.process_market_update(market_data))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Generate signals for all symbols
            signal_tasks = [simulator.generate_signal(symbol) for symbol in symbols]
            await asyncio.gather(*signal_tasks)
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            memory_per_operation = memory_increase / config['concurrent_operations']
            
            # Cleanup
            await simulator.shutdown()
            await data_manager.shutdown()
            
            # Memory increase should be reasonable
            max_acceptable_increase = config['concurrent_operations'] * 0.5  # 0.5MB per operation max
            assert memory_increase < max_acceptable_increase, \
                f"Memory increase {memory_increase:.1f}MB > {max_acceptable_increase:.1f}MB"
            
            print(f"Load {config['concurrent_operations']} ops, {config['data_size']}: {memory_increase:.1f}MB increase, {memory_per_operation:.2f}MB/op")


class TestPerformanceDegradationUnderLoad:
    """Test performance degradation characteristics under load."""
    
    @pytest.mark.asyncio
    async def test_latency_degradation_curve(self):
        """Test how latency degrades with increasing load."""
        load_levels = [100, 250, 500, 750, 1000, 1500, 2000]  # Operations per minute
        
        config = {
            'performance_monitoring': True,
            'adaptive_throttling': False  # Disable to see raw degradation
        }
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        degradation_data = []
        
        for load_level in load_levels:
            latencies = []
            
            # Generate load for 1 minute
            ops_per_second = load_level / 60
            test_duration = 60  # seconds
            
            start_test = time.time()
            op_count = 0
            
            while time.time() - start_test < test_duration:
                op_start = time.perf_counter()
                
                # Simulate operation
                market_data = {
                    'symbol': f'LOAD_{op_count % 20}',
                    'price': 100 + np.random.normal(0, 3),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time()
                }
                
                signal = await simulator.generate_signal(market_data['symbol'])
                
                op_end = time.perf_counter()
                latency = (op_end - op_start) * 1000  # ms
                latencies.append(latency)
                
                op_count += 1
                
                # Control rate
                if ops_per_second > 0:
                    await asyncio.sleep(1.0 / ops_per_second)
            
            # Analyze latency at this load level
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            degradation_data.append({
                'load_level': load_level,
                'avg_latency': avg_latency,
                'p95_latency': p95_latency,
                'p99_latency': p99_latency,
                'operations_completed': len(latencies)
            })
            
            # P99 latency should stay within bounds
            assert p99_latency < 500, f"Load {load_level}: P99 latency {p99_latency:.1f}ms > 500ms"
            
            print(f"Load {load_level} ops/min: avg {avg_latency:.1f}ms, P95 {p95_latency:.1f}ms, P99 {p99_latency:.1f}ms")
        
        await simulator.shutdown()
        
        # Analyze degradation curve
        base_latency = degradation_data[0]['avg_latency']
        
        for data in degradation_data[1:]:
            degradation_factor = data['avg_latency'] / base_latency
            
            # Degradation should be gradual, not cliff-like
            max_acceptable_degradation = 1 + (data['load_level'] / 1000)  # Linear degradation model
            assert degradation_factor < max_acceptable_degradation, \
                f"Load {data['load_level']}: degradation factor {degradation_factor:.1f} > {max_acceptable_degradation:.1f}"
    
    @pytest.mark.asyncio
    async def test_throughput_saturation_point(self):
        """Test system throughput saturation characteristics."""
        # Gradually increase load until saturation
        load_levels = range(100, 3000, 200)  # 100 to 3000 ops/min in steps of 200
        
        config = {
            'max_throughput_mode': True,
            'queue_size': 10000,
            'workers': 8
        }
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        throughput_data = []
        saturation_detected = False
        
        for target_load in load_levels:
            if saturation_detected:
                break
            
            # Test throughput at this load level
            test_duration = 30  # seconds
            target_ops_per_second = target_load / 60
            
            operations_completed = 0
            start_time = time.perf_counter()
            
            # Generate target load
            async def load_generator():
                nonlocal operations_completed
                end_time = time.time() + test_duration
                
                while time.time() < end_time:
                    # Generate operation
                    market_data = {
                        'symbol': f'THRU_{operations_completed % 50}',
                        'price': 100 + np.random.normal(0, 2),
                        'volume': np.random.randint(100, 1000),
                        'timestamp': time.time()
                    }
                    
                    await simulator.generate_signal(market_data['symbol'])
                    operations_completed += 1
                    
                    # Control rate
                    await asyncio.sleep(1.0 / target_ops_per_second)
            
            await load_generator()
            
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            actual_throughput = operations_completed / actual_duration * 60  # ops/min
            
            throughput_efficiency = actual_throughput / target_load
            
            throughput_data.append({
                'target_load': target_load,
                'actual_throughput': actual_throughput,
                'efficiency': throughput_efficiency
            })
            
            # Detect saturation (efficiency drops below 80%)
            if throughput_efficiency < 0.8:
                saturation_detected = True
                print(f"Saturation detected at {target_load} ops/min (efficiency: {throughput_efficiency:.2f})")
            
            print(f"Target {target_load} ops/min: actual {actual_throughput:.0f} ops/min, efficiency {throughput_efficiency:.2f}")
        
        await simulator.shutdown()
        
        # Validate saturation behavior
        assert len(throughput_data) > 3, "Should test multiple load levels before saturation"
        
        # Find maximum sustainable throughput
        max_sustainable = max(data['actual_throughput'] for data in throughput_data if data['efficiency'] > 0.9)
        assert max_sustainable > 1000, f"Maximum sustainable throughput {max_sustainable:.0f} ops/min < 1000"
        
        print(f"\nThroughput Analysis:")
        print(f"  Maximum sustainable: {max_sustainable:.0f} ops/min")
        print(f"  Saturation detected: {saturation_detected}")


if __name__ == '__main__':
    pytest.main([__file__])