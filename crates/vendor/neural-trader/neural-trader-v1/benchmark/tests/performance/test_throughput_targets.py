"""
Throughput performance validation test suite for AI News Trading benchmark system.

This module validates that the system can handle >1000 trades/second throughput.
Tests include:
- Trade execution throughput
- Signal generation throughput
- Data processing throughput
- Concurrent user throughput
- System scalability validation
"""

import asyncio
import time
import statistics
import pytest
from unittest.mock import Mock, patch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.profiling.profiler import ThroughputProfiler


class TestTradeExecutionThroughput:
    """Test trade execution throughput performance."""
    
    @pytest.fixture
    async def throughput_system(self):
        """Create system optimized for throughput testing."""
        config = {
            'mode': 'high_throughput',
            'batch_processing': True,
            'parallel_execution': True,
            'max_workers': 8,
            'queue_size': 10000,
            'optimization_level': 'aggressive'
        }
        
        system = {
            'simulator': MarketSimulator(config),
            'data_manager': RealtimeManager(config),
            'profiler': ThroughputProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_single_thread_trade_throughput(self, throughput_system):
        """Test trade execution throughput on single thread."""
        simulator = throughput_system['simulator']
        profiler = throughput_system['profiler']
        
        # Prepare trades
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        num_trades = 5000
        trades = []
        
        for i in range(num_trades):
            trade = {
                'symbol': symbols[i % len(symbols)],
                'action': 'buy' if i % 2 == 0 else 'sell',
                'quantity': np.random.randint(1, 100),
                'price': 100 + np.random.normal(0, 5),
                'timestamp': time.time()
            }
            trades.append(trade)
        
        # Execute trades and measure throughput
        start_time = time.perf_counter()
        
        with profiler.measure('single_thread_trades'):
            executed_trades = []
            for trade in trades:
                result = await simulator.execute_trade(trade)
                executed_trades.append(result)
        
        end_time = time.perf_counter()
        
        # Calculate throughput metrics
        duration = end_time - start_time
        throughput = num_trades / duration
        
        # Validate single-thread throughput
        assert throughput > 500, f"Single-thread throughput {throughput:.0f} trades/sec < 500"
        assert len(executed_trades) == num_trades, f"Expected {num_trades} trades, executed {len(executed_trades)}"
        
        # Validate trade execution success rate
        successful_trades = sum(1 for trade in executed_trades if trade.get('status') == 'executed')
        success_rate = successful_trades / num_trades
        assert success_rate > 0.95, f"Trade success rate {success_rate:.2%} < 95%"
        
        print(f"\nSingle-Thread Trade Throughput:")
        print(f"  Throughput: {throughput:.0f} trades/sec")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Success rate: {success_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_concurrent_trade_throughput(self, throughput_system):
        """Test concurrent trade execution throughput."""
        simulator = throughput_system['simulator']
        
        # Test different concurrency levels
        concurrency_levels = [2, 4, 8, 16]
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'INTC', 'NFLX']
        
        for concurrency in concurrency_levels:
            trades_per_worker = 1000
            total_trades = trades_per_worker * concurrency
            
            async def worker_trades(worker_id):
                worker_trades = []
                for i in range(trades_per_worker):
                    trade = {
                        'symbol': symbols[(worker_id * trades_per_worker + i) % len(symbols)],
                        'action': 'buy' if i % 2 == 0 else 'sell',
                        'quantity': np.random.randint(1, 50),
                        'price': 100 + np.random.normal(0, 3),
                        'worker_id': worker_id,
                        'timestamp': time.time()
                    }
                    
                    result = await simulator.execute_trade(trade)
                    worker_trades.append(result)
                
                return worker_trades
            
            # Execute concurrent trades
            start_time = time.perf_counter()
            
            tasks = [worker_trades(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            
            # Analyze results
            duration = end_time - start_time
            throughput = total_trades / duration
            
            # Flatten results
            all_trades = []
            for worker_results in results:
                all_trades.extend(worker_results)
            
            successful_trades = sum(1 for trade in all_trades if trade.get('status') == 'executed')
            success_rate = successful_trades / total_trades
            
            # Validate concurrent throughput
            min_expected_throughput = 200 * concurrency  # Scale with concurrency
            assert throughput > min_expected_throughput, \
                f"Concurrency {concurrency}: throughput {throughput:.0f} < {min_expected_throughput}"
            assert success_rate > 0.95, f"Concurrency {concurrency}: success rate {success_rate:.2%} < 95%"
            
            print(f"Concurrency {concurrency}: {throughput:.0f} trades/sec, {success_rate:.2%} success")
    
    @pytest.mark.asyncio
    async def test_batch_trade_throughput(self, throughput_system):
        """Test batch trade execution throughput."""
        simulator = throughput_system['simulator']
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 250, 500]
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for batch_size in batch_sizes:
            num_batches = 20
            total_trades = batch_size * num_batches
            
            batch_throughputs = []
            
            for batch_num in range(num_batches):
                # Create batch of trades
                batch_trades = []
                for i in range(batch_size):
                    trade = {
                        'symbol': symbols[i % len(symbols)],
                        'action': 'buy' if i % 2 == 0 else 'sell',
                        'quantity': np.random.randint(1, 100),
                        'price': 100 + np.random.normal(0, 5),
                        'batch_id': batch_num,
                        'timestamp': time.time()
                    }
                    batch_trades.append(trade)
                
                # Execute batch
                start_time = time.perf_counter()
                batch_results = await simulator.execute_trades_batch(batch_trades)
                end_time = time.perf_counter()
                
                batch_duration = end_time - start_time
                batch_throughput = batch_size / batch_duration
                batch_throughputs.append(batch_throughput)
                
                # Validate batch execution
                assert len(batch_results) == batch_size, f"Batch {batch_num}: expected {batch_size} results"
            
            # Analyze batch performance
            avg_batch_throughput = statistics.mean(batch_throughputs)
            max_batch_throughput = max(batch_throughputs)
            
            # Batch processing should be more efficient
            min_expected = batch_size * 10  # At least 10 trades/sec per trade in batch
            assert avg_batch_throughput > min_expected, \
                f"Batch size {batch_size}: avg throughput {avg_batch_throughput:.0f} < {min_expected}"
            
            print(f"Batch size {batch_size}: avg {avg_batch_throughput:.0f} trades/sec, max {max_batch_throughput:.0f}")
    
    @pytest.mark.asyncio
    async def test_sustained_throughput(self, throughput_system):
        """Test sustained throughput over extended period."""
        simulator = throughput_system['simulator']
        
        # Test sustained throughput for 30 seconds
        test_duration = 30
        target_throughput = 1000
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        throughput_measurements = []
        total_trades_executed = 0
        
        async def sustained_trade_generator():
            nonlocal total_trades_executed
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                # Generate batch of trades
                batch_size = 50
                batch_trades = []
                
                for i in range(batch_size):
                    trade = {
                        'symbol': symbols[i % len(symbols)],
                        'action': 'buy' if total_trades_executed % 2 == 0 else 'sell',
                        'quantity': np.random.randint(1, 50),
                        'price': 100 + np.random.normal(0, 3),
                        'timestamp': time.time()
                    }
                    batch_trades.append(trade)
                
                # Execute batch and measure throughput
                batch_start = time.perf_counter()
                results = await simulator.execute_trades_batch(batch_trades)
                batch_end = time.perf_counter()
                
                batch_duration = batch_end - batch_start
                batch_throughput = len(results) / batch_duration
                throughput_measurements.append(batch_throughput)
                
                total_trades_executed += len(results)
                
                # Control rate to avoid overwhelming system
                await asyncio.sleep(0.01)
        
        # Run sustained test
        test_start = time.perf_counter()
        await sustained_trade_generator()
        test_end = time.perf_counter()
        
        actual_duration = test_end - test_start
        overall_throughput = total_trades_executed / actual_duration
        
        # Analyze sustained performance
        avg_throughput = statistics.mean(throughput_measurements)
        min_throughput = min(throughput_measurements) 
        throughput_std = statistics.stdev(throughput_measurements)
        
        # Validate sustained throughput requirements
        assert overall_throughput > target_throughput, \
            f"Overall sustained throughput {overall_throughput:.0f} < {target_throughput}"
        assert avg_throughput > target_throughput * 0.8, \
            f"Average sustained throughput {avg_throughput:.0f} < {target_throughput * 0.8}"
        assert min_throughput > target_throughput * 0.5, \
            f"Minimum sustained throughput {min_throughput:.0f} < {target_throughput * 0.5}"
        
        # Throughput should be relatively stable
        cv = throughput_std / avg_throughput  # Coefficient of variation
        assert cv < 0.5, f"Throughput coefficient of variation {cv:.2f} > 0.5 (too unstable)"
        
        print(f"\nSustained Throughput Test ({test_duration}s):")
        print(f"  Overall throughput: {overall_throughput:.0f} trades/sec")
        print(f"  Average throughput: {avg_throughput:.0f} trades/sec")
        print(f"  Minimum throughput: {min_throughput:.0f} trades/sec")
        print(f"  Throughput stability (CV): {cv:.2f}")
        print(f"  Total trades: {total_trades_executed}")


class TestSignalGenerationThroughput:
    """Test signal generation throughput performance."""
    
    @pytest.fixture
    async def signal_system(self):
        """Create system for signal generation testing."""
        config = {
            'signal_processing': 'optimized',
            'parallel_signals': True,
            'max_signal_workers': 6,
            'signal_cache_size': 5000
        }
        
        system = {
            'simulator': MarketSimulator(config),
            'profiler': ThroughputProfiler()
        }
        
        await system['simulator'].initialize()
        await system['profiler'].initialize()
        
        yield system
        
        await system['simulator'].shutdown()
        await system['profiler'].shutdown()
    
    @pytest.mark.asyncio
    async def test_signal_generation_throughput(self, signal_system):
        """Test signal generation throughput."""
        simulator = signal_system['simulator']
        profiler = signal_system['profiler']
        
        # Test signal generation for multiple symbols
        symbols = [f'SYM{i:03d}' for i in range(100)]
        num_iterations = 50
        
        with profiler.measure('signal_generation_throughput'):
            total_signals = 0
            start_time = time.perf_counter()
            
            for iteration in range(num_iterations):
                # Generate market data for all symbols
                market_updates = []
                for symbol in symbols:
                    update = {
                        'symbol': symbol,
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 1000),
                        'timestamp': time.time()
                    }
                    market_updates.append(update)
                
                # Generate signals for all symbols
                signals = await simulator.generate_signals_batch(symbols, market_updates)
                total_signals += len(signals)
            
            end_time = time.perf_counter()
        
        # Calculate signal generation throughput
        duration = end_time - start_time
        signal_throughput = total_signals / duration
        
        # Validate signal generation throughput
        assert signal_throughput > 1000, f"Signal throughput {signal_throughput:.0f} signals/sec < 1000"
        assert total_signals == len(symbols) * num_iterations, \
            f"Expected {len(symbols) * num_iterations} signals, got {total_signals}"
        
        print(f"\nSignal Generation Throughput:")
        print(f"  Throughput: {signal_throughput:.0f} signals/sec")
        print(f"  Total signals: {total_signals}")
        print(f"  Duration: {duration:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, signal_system):
        """Test concurrent signal generation across multiple strategies."""
        simulator = signal_system['simulator']
        
        # Test multiple strategies concurrently
        strategies = ['momentum', 'arbitrage', 'news_sentiment', 'mean_reversion']
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        signals_per_strategy = 500
        
        async def strategy_signal_generator(strategy_name):
            strategy_signals = []
            
            for i in range(signals_per_strategy):
                symbol = symbols[i % len(symbols)]
                
                market_data = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 3),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time()
                }
                
                signal = await simulator.generate_signal(symbol, market_data, strategy=strategy_name)
                if signal:
                    strategy_signals.append(signal)
            
            return strategy_name, strategy_signals
        
        # Run all strategies concurrently
        start_time = time.perf_counter()
        
        tasks = [strategy_signal_generator(strategy) for strategy in strategies]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        
        # Analyze concurrent signal generation
        duration = end_time - start_time
        total_signals = sum(len(signals) for _, signals in results)
        concurrent_throughput = total_signals / duration
        
        # Validate concurrent performance
        min_expected_throughput = len(strategies) * 200  # 200 signals/sec per strategy
        assert concurrent_throughput > min_expected_throughput, \
            f"Concurrent signal throughput {concurrent_throughput:.0f} < {min_expected_throughput}"
        
        # Validate each strategy generated signals
        for strategy_name, signals in results:
            assert len(signals) > 0, f"Strategy {strategy_name} generated no signals"
        
        print(f"Concurrent Signal Generation:")
        print(f"  Total throughput: {concurrent_throughput:.0f} signals/sec")
        print(f"  Strategies: {len(strategies)}")
        print(f"  Total signals: {total_signals}")


class TestDataProcessingThroughput:
    """Test data processing throughput performance."""
    
    @pytest.fixture
    async def data_system(self):
        """Create system for data processing testing."""
        config = {
            'data_processing': 'high_throughput',
            'batch_size': 1000,
            'parallel_processing': True,
            'buffer_size': 50000,
            'processing_workers': 8
        }
        
        system = {
            'data_manager': RealtimeManager(config),
            'profiler': ThroughputProfiler()
        }
        
        await system['data_manager'].initialize()
        await system['profiler'].initialize()
        
        yield system
        
        await system['data_manager'].shutdown()
        await system['profiler'].shutdown()
    
    @pytest.mark.asyncio
    async def test_market_data_processing_throughput(self, data_system):
        """Test market data processing throughput."""
        data_manager = data_system['data_manager']
        profiler = data_system['profiler']
        
        # Generate high volume of market data
        symbols = [f'DATA{i:03d}' for i in range(50)]
        updates_per_symbol = 1000
        total_updates = len(symbols) * updates_per_symbol
        
        market_updates = []
        for symbol in symbols:
            for i in range(updates_per_symbol):
                update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': np.random.randint(100, 5000),
                    'bid': 100 + np.random.normal(0, 5),
                    'ask': 100 + np.random.normal(0, 5) + 0.01,
                    'timestamp': time.time() + i * 0.001
                }
                market_updates.append(update)
        
        # Process all market updates
        start_time = time.perf_counter()
        
        with profiler.measure('market_data_processing'):
            processed_count = 0
            
            # Process in batches for efficiency
            batch_size = 1000
            for i in range(0, len(market_updates), batch_size):
                batch = market_updates[i:i + batch_size]
                await data_manager.process_market_updates_batch(batch)
                processed_count += len(batch)
        
        end_time = time.perf_counter()
        
        # Calculate processing throughput
        duration = end_time - start_time
        processing_throughput = processed_count / duration
        
        # Validate market data processing throughput
        assert processing_throughput > 5000, f"Market data throughput {processing_throughput:.0f} updates/sec < 5000"
        assert processed_count == total_updates, f"Processed {processed_count} != expected {total_updates}"
        
        print(f"\nMarket Data Processing Throughput:")
        print(f"  Throughput: {processing_throughput:.0f} updates/sec")
        print(f"  Total updates: {processed_count}")
        print(f"  Duration: {duration:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_mixed_data_processing_throughput(self, data_system):
        """Test throughput with mixed market and news data."""
        data_manager = data_system['data_manager']
        
        # Generate mixed data stream
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        market_updates = []
        news_updates = []
        
        # Generate market data (80% of stream)
        for i in range(4000):
            update = {
                'symbol': symbols[i % len(symbols)],
                'price': 100 + np.random.normal(0, 3),
                'volume': np.random.randint(100, 2000),
                'timestamp': time.time() + i * 0.01
            }
            market_updates.append(update)
        
        # Generate news data (20% of stream)
        news_headlines = [
            "Company reports earnings",
            "Market analysis update",
            "Sector performance review",
            "Economic indicators released",
            "Industry news update"
        ]
        
        for i in range(1000):
            update = {
                'headline': news_headlines[i % len(news_headlines)],
                'sentiment': np.random.uniform(-1, 1),
                'relevance': np.random.uniform(0.5, 1.0),
                'symbols': [symbols[i % len(symbols)]],
                'timestamp': time.time() + i * 0.04
            }
            news_updates.append(update)
        
        # Process mixed data stream
        start_time = time.perf_counter()
        
        # Interleave market and news processing
        market_tasks = []
        news_tasks = []
        
        for update in market_updates:
            task = asyncio.create_task(data_manager.process_market_update(update))
            market_tasks.append(task)
        
        for update in news_updates:
            task = asyncio.create_task(data_manager.process_news_update(update))
            news_tasks.append(task)
        
        # Wait for all updates to complete
        await asyncio.gather(*market_tasks, *news_tasks)
        
        end_time = time.perf_counter()
        
        # Calculate mixed processing throughput
        duration = end_time - start_time
        total_updates = len(market_updates) + len(news_updates)
        mixed_throughput = total_updates / duration
        
        # Validate mixed data throughput
        assert mixed_throughput > 2000, f"Mixed data throughput {mixed_throughput:.0f} updates/sec < 2000"
        
        print(f"Mixed Data Processing Throughput:")
        print(f"  Throughput: {mixed_throughput:.0f} updates/sec")
        print(f"  Market updates: {len(market_updates)}")
        print(f"  News updates: {len(news_updates)}")
        print(f"  Total updates: {total_updates}")


class TestSystemScalabilityThroughput:
    """Test system throughput scalability."""
    
    @pytest.mark.asyncio
    async def test_horizontal_scaling_throughput(self):
        """Test throughput scaling with multiple system instances."""
        # Test with different numbers of "instances"
        instance_counts = [1, 2, 4, 8]
        base_throughput_per_instance = 500  # trades/sec per instance
        
        for instance_count in instance_counts:
            # Simulate multiple instances
            instance_configs = []
            for i in range(instance_count):
                config = {
                    'instance_id': i,
                    'optimization_mode': 'throughput',
                    'max_workers': 4,
                    'batch_size': 100
                }
                instance_configs.append(config)
            
            # Create simulators for each "instance"
            simulators = []
            for config in instance_configs:
                simulator = MarketSimulator(config)
                await simulator.initialize()
                simulators.append(simulator)
            
            # Test throughput across all instances
            trades_per_instance = 1000
            
            async def instance_workload(simulator, instance_id):
                trades = []
                for i in range(trades_per_instance):
                    trade = {
                        'symbol': f'SYM{i % 10}',
                        'action': 'buy' if i % 2 == 0 else 'sell',
                        'quantity': 10,
                        'price': 100,
                        'instance_id': instance_id
                    }
                    result = await simulator.execute_trade(trade)
                    trades.append(result)
                return trades
            
            # Execute workload across all instances
            start_time = time.perf_counter()
            
            tasks = []
            for i, simulator in enumerate(simulators):
                task = asyncio.create_task(instance_workload(simulator, i))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            
            # Calculate scaling metrics
            duration = end_time - start_time
            total_trades = sum(len(instance_results) for instance_results in results)
            scaled_throughput = total_trades / duration
            
            # Expected throughput should scale linearly
            expected_min_throughput = base_throughput_per_instance * instance_count * 0.8
            scaling_efficiency = scaled_throughput / (base_throughput_per_instance * instance_count)
            
            assert scaled_throughput > expected_min_throughput, \
                f"Instances {instance_count}: throughput {scaled_throughput:.0f} < {expected_min_throughput:.0f}"
            assert scaling_efficiency > 0.7, \
                f"Instances {instance_count}: scaling efficiency {scaling_efficiency:.2f} < 0.7"
            
            print(f"Instances {instance_count}: {scaled_throughput:.0f} trades/sec, efficiency {scaling_efficiency:.2f}")
            
            # Cleanup
            for simulator in simulators:
                await simulator.shutdown()
    
    @pytest.mark.asyncio
    async def test_load_balancing_throughput(self):
        """Test throughput with load balancing across workers."""
        # Test different worker configurations
        worker_configs = [
            {'workers': 1, 'queue_size': 1000},
            {'workers': 2, 'queue_size': 2000},
            {'workers': 4, 'queue_size': 4000},
            {'workers': 8, 'queue_size': 8000}
        ]
        
        for config in worker_configs:
            system_config = {
                'load_balancing': True,
                'max_workers': config['workers'],
                'queue_size': config['queue_size'],
                'worker_strategy': 'round_robin'
            }
            
            simulator = MarketSimulator(system_config)
            await simulator.initialize()
            
            # Generate high-volume workload
            num_trades = 2000
            trades = []
            
            for i in range(num_trades):
                trade = {
                    'symbol': f'SYM{i % 20}',
                    'action': 'buy' if i % 2 == 0 else 'sell',
                    'quantity': np.random.randint(1, 50),
                    'price': 100 + np.random.normal(0, 2),
                    'trade_id': i
                }
                trades.append(trade)
            
            # Execute with load balancing
            start_time = time.perf_counter()
            results = await simulator.execute_trades_with_load_balancing(trades)
            end_time = time.perf_counter()
            
            # Analyze load-balanced performance
            duration = end_time - start_time
            lb_throughput = len(results) / duration
            
            # Throughput should improve with more workers
            min_expected = config['workers'] * 150  # 150 trades/sec per worker minimum
            assert lb_throughput > min_expected, \
                f"Workers {config['workers']}: throughput {lb_throughput:.0f} < {min_expected}"
            
            print(f"Workers {config['workers']}: {lb_throughput:.0f} trades/sec")
            
            await simulator.shutdown()


if __name__ == '__main__':
    pytest.main([__file__])