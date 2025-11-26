"""
Test suite for measuring and validating performance gains.

Tests that optimization achieves target performance improvements
in latency, throughput, memory usage, and trading metrics.
"""

import pytest
import numpy as np
import time
import psutil
import asyncio
from typing import Dict, List, Callable, Any
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from benchmark.src.optimization.performance_optimizer import (
    PerformanceOptimizer,
    LatencyOptimizer,
    ThroughputOptimizer,
    MemoryOptimizer,
    TradingMetricsOptimizer
)


class TestLatencyOptimization:
    """Test latency reduction optimizations."""
    
    @pytest.fixture
    def mock_trading_system(self):
        """Mock trading system for testing."""
        system = Mock()
        system.process_signal = Mock(return_value={'status': 'processed'})
        system.execute_trade = Mock(return_value={'status': 'executed'})
        system.update_positions = Mock(return_value={'status': 'updated'})
        return system
    
    def test_baseline_latency_measurement(self, mock_trading_system):
        """Test measuring baseline latency."""
        optimizer = LatencyOptimizer()
        
        # Simulate baseline latency
        def slow_process(signal):
            time.sleep(0.1)  # 100ms baseline
            return mock_trading_system.process_signal(signal)
        
        baseline = optimizer.measure_baseline(
            function=slow_process,
            test_data=[{'signal': 'buy', 'symbol': 'AAPL'} for _ in range(10)],
            iterations=10
        )
        
        assert baseline['mean_latency'] >= 100  # At least 100ms
        assert baseline['p99_latency'] >= 100
        assert 'latency_distribution' in baseline
    
    def test_caching_optimization(self, mock_trading_system):
        """Test latency reduction through caching."""
        optimizer = LatencyOptimizer()
        
        # Function with expensive computation
        call_count = 0
        def expensive_calculation(params):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)  # Simulate expensive operation
            return params['value'] ** 2
        
        # Apply caching optimization
        optimized = optimizer.apply_caching(
            expensive_calculation,
            cache_size=100,
            ttl_seconds=60
        )
        
        # Test performance improvement
        test_params = [{'value': i % 10} for i in range(100)]
        
        # First run - populate cache
        start = time.time()
        for params in test_params:
            optimized(params)
        first_run_time = time.time() - start
        
        # Second run - use cache
        start = time.time()
        for params in test_params:
            optimized(params)
        second_run_time = time.time() - start
        
        assert second_run_time < first_run_time * 0.1  # 90%+ improvement
        assert call_count == 10  # Only unique values computed
    
    def test_vectorization_optimization(self):
        """Test latency reduction through vectorization."""
        optimizer = LatencyOptimizer()
        
        # Scalar implementation
        def calculate_indicators_scalar(prices):
            sma = []
            for i in range(len(prices)):
                if i >= 20:
                    sma.append(sum(prices[i-20:i]) / 20)
                else:
                    sma.append(None)
            return sma
        
        # Apply vectorization
        optimized = optimizer.apply_vectorization(calculate_indicators_scalar)
        
        # Test with large dataset
        prices = np.random.randn(10000).cumsum() + 100
        
        # Measure improvement
        start = time.time()
        result_scalar = calculate_indicators_scalar(prices.tolist())
        scalar_time = time.time() - start
        
        start = time.time()
        result_vector = optimized(prices)
        vector_time = time.time() - start
        
        assert vector_time < scalar_time * 0.2  # 80%+ improvement
        
        # Verify correctness
        assert len(result_scalar) == len(result_vector)
        for i in range(20, 100):  # Check sample of results
            if result_scalar[i] is not None:
                assert abs(result_scalar[i] - result_vector[i]) < 1e-6
    
    def test_async_optimization(self):
        """Test latency reduction through async operations."""
        optimizer = LatencyOptimizer()
        
        # Simulate I/O bound operations
        async def fetch_market_data(symbol):
            await asyncio.sleep(0.05)  # Simulate API call
            return {'symbol': symbol, 'price': 100 + np.random.randn()}
        
        async def fetch_news_data(symbol):
            await asyncio.sleep(0.03)  # Simulate API call
            return {'symbol': symbol, 'sentiment': np.random.rand()}
        
        # Synchronous version
        def sync_fetch_all(symbols):
            results = {}
            for symbol in symbols:
                market = asyncio.run(fetch_market_data(symbol))
                news = asyncio.run(fetch_news_data(symbol))
                results[symbol] = {'market': market, 'news': news}
            return results
        
        # Optimized async version
        async def async_fetch_all(symbols):
            tasks = []
            for symbol in symbols:
                tasks.append(fetch_market_data(symbol))
                tasks.append(fetch_news_data(symbol))
            
            results = await asyncio.gather(*tasks)
            
            # Organize results
            organized = {}
            for i in range(0, len(results), 2):
                symbol = results[i]['symbol']
                organized[symbol] = {
                    'market': results[i],
                    'news': results[i + 1]
                }
            return organized
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Measure sync version
        start = time.time()
        sync_results = sync_fetch_all(symbols)
        sync_time = time.time() - start
        
        # Measure async version
        start = time.time()
        async_results = asyncio.run(async_fetch_all(symbols))
        async_time = time.time() - start
        
        assert async_time < sync_time * 0.3  # 70%+ improvement
        assert len(sync_results) == len(async_results)
    
    def test_achieve_target_latency(self, mock_trading_system):
        """Test achieving target <50ms latency."""
        optimizer = LatencyOptimizer()
        
        # Original slow implementation
        def process_trading_signal(signal):
            # Simulate various processing steps
            time.sleep(0.02)  # Data validation
            time.sleep(0.03)  # Feature calculation
            time.sleep(0.04)  # Model inference
            time.sleep(0.02)  # Risk checks
            time.sleep(0.01)  # Order preparation
            return {'action': 'buy', 'confidence': 0.8}
        
        # Apply multiple optimizations
        optimized = optimizer.optimize_for_target(
            function=process_trading_signal,
            target_latency_ms=50,
            optimization_techniques=[
                'caching',
                'vectorization', 
                'parallel_processing',
                'jit_compilation'
            ]
        )
        
        # Test optimized version
        test_signals = [{'id': i, 'data': np.random.randn(100)} for i in range(100)]
        
        latencies = []
        for signal in test_signals[:10]:  # Test subset
            start = time.perf_counter()
            result = optimized(signal)
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        assert avg_latency < 50, f"Average latency {avg_latency}ms exceeds target"
        assert p99_latency < 75, f"P99 latency {p99_latency}ms too high"


class TestThroughputOptimization:
    """Test throughput improvement optimizations."""
    
    def test_baseline_throughput_measurement(self):
        """Test measuring baseline throughput."""
        optimizer = ThroughputOptimizer()
        
        # Simulate trade processing
        def process_trade(trade):
            time.sleep(0.001)  # 1ms per trade
            return {'processed': True, 'id': trade['id']}
        
        baseline = optimizer.measure_baseline(
            function=process_trade,
            test_duration_seconds=1.0,
            test_data_generator=lambda: {'id': np.random.randint(1000000)}
        )
        
        assert baseline['throughput'] < 1200  # Less than theoretical max
        assert baseline['throughput'] > 800   # But reasonable
        assert 'latency_stats' in baseline
    
    def test_batch_processing_optimization(self):
        """Test throughput improvement through batching."""
        optimizer = ThroughputOptimizer()
        
        # Single item processing
        def process_single(item):
            time.sleep(0.001)  # Fixed overhead
            return item * 2
        
        # Optimized batch processing
        optimized = optimizer.apply_batching(
            process_single,
            batch_size=100,
            max_wait_ms=10
        )
        
        # Generate test load
        items = list(range(10000))
        
        # Process without batching
        start = time.time()
        results_single = [process_single(item) for item in items[:1000]]
        single_time = time.time() - start
        single_throughput = 1000 / single_time
        
        # Process with batching
        start = time.time()
        results_batch = optimized.process_many(items[:1000])
        batch_time = time.time() - start
        batch_throughput = 1000 / batch_time
        
        assert batch_throughput > single_throughput * 5  # 5x+ improvement
        assert results_single == results_batch  # Correct results
    
    def test_parallel_processing_optimization(self):
        """Test throughput improvement through parallelization."""
        optimizer = ThroughputOptimizer()
        
        # CPU-bound task
        def compute_indicators(price_data):
            # Simulate expensive calculation
            result = 0
            for i in range(1000):
                result += sum(price_data) / len(price_data)
                result = result ** 0.99
            return result
        
        # Apply parallel processing
        optimized = optimizer.apply_parallel_processing(
            compute_indicators,
            num_workers=4,
            chunk_size=10
        )
        
        # Generate test data
        test_data = [np.random.randn(100) for _ in range(100)]
        
        # Sequential processing
        start = time.time()
        results_seq = [compute_indicators(data) for data in test_data]
        seq_time = time.time() - start
        
        # Parallel processing
        start = time.time()
        results_par = optimized.process_many(test_data)
        par_time = time.time() - start
        
        assert par_time < seq_time * 0.4  # 60%+ improvement with 4 workers
        
        # Verify correctness
        for r1, r2 in zip(results_seq, results_par):
            assert abs(r1 - r2) < 1e-10
    
    def test_pipeline_optimization(self):
        """Test throughput improvement through pipelining."""
        optimizer = ThroughputOptimizer()
        
        # Multi-stage processing
        def stage1_parse(data):
            time.sleep(0.001)
            return {'parsed': data['raw']}
        
        def stage2_analyze(data):
            time.sleep(0.002)
            return {'analyzed': data['parsed'] * 2}
        
        def stage3_decide(data):
            time.sleep(0.001)
            return {'decision': 'buy' if data['analyzed'] > 0 else 'sell'}
        
        # Create optimized pipeline
        pipeline = optimizer.create_pipeline([
            ('parse', stage1_parse, 2),     # 2 workers
            ('analyze', stage2_analyze, 3),  # 3 workers
            ('decide', stage3_decide, 2)     # 2 workers
        ])
        
        # Generate test stream
        test_items = [{'raw': np.random.randn()} for _ in range(1000)]
        
        # Sequential processing
        start = time.time()
        results_seq = []
        for item in test_items:
            r1 = stage1_parse(item)
            r2 = stage2_analyze(r1)
            r3 = stage3_decide(r2)
            results_seq.append(r3)
        seq_time = time.time() - start
        
        # Pipeline processing
        start = time.time()
        results_pipe = list(pipeline.process_stream(test_items))
        pipe_time = time.time() - start
        
        assert pipe_time < seq_time * 0.5  # 50%+ improvement
        assert len(results_pipe) == len(results_seq)
    
    def test_achieve_target_throughput(self):
        """Test achieving target 1000+ trades/sec throughput."""
        optimizer = ThroughputOptimizer()
        
        # Simulate complete trade processing
        def process_trade_order(order):
            # Validation
            if order['quantity'] <= 0:
                return {'error': 'Invalid quantity'}
            
            # Risk check
            risk_score = abs(order['quantity'] * order['price']) / 10000
            if risk_score > 1:
                return {'error': 'Risk limit exceeded'}
            
            # Calculate fees
            fee = order['quantity'] * order['price'] * 0.001
            
            # Simulate order matching
            time.sleep(0.0001)  # Small delay
            
            return {
                'status': 'executed',
                'fill_price': order['price'] * (1 + np.random.randn() * 0.0001),
                'fee': fee,
                'timestamp': time.time()
            }
        
        # Apply comprehensive optimizations
        optimized_processor = optimizer.optimize_for_target(
            function=process_trade_order,
            target_throughput=1000,
            techniques=[
                'batching',
                'parallel_processing',
                'memory_pool',
                'lock_free_queues',
                'numa_optimization'
            ]
        )
        
        # Test with realistic load
        def generate_orders():
            return {
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT']),
                'quantity': np.random.randint(1, 1000),
                'price': 100 + np.random.randn(),
                'side': np.random.choice(['buy', 'sell'])
            }
        
        # Measure throughput
        duration = 5.0  # 5 second test
        processed = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            orders = [generate_orders() for _ in range(100)]
            results = optimized_processor.process_batch(orders)
            processed += len([r for r in results if r.get('status') == 'executed'])
        
        actual_duration = time.time() - start_time
        throughput = processed / actual_duration
        
        assert throughput > 1000, f"Throughput {throughput:.1f} below target"
        print(f"Achieved throughput: {throughput:.1f} trades/sec")


class TestMemoryOptimization:
    """Test memory usage optimizations."""
    
    def test_baseline_memory_measurement(self):
        """Test measuring baseline memory usage."""
        optimizer = MemoryOptimizer()
        
        # Function that uses memory
        def create_large_dataset(size):
            data = {
                'prices': np.random.randn(size, 100),
                'volumes': np.random.randint(1000, 10000, size=(size, 100)),
                'timestamps': list(range(size * 100)),
                'metadata': [{'id': i, 'info': 'x' * 1000} for i in range(size)]
            }
            return data
        
        baseline = optimizer.measure_baseline(
            function=create_large_dataset,
            args=(1000,),
            include_children=True
        )
        
        assert baseline['peak_memory_mb'] > 10  # Uses significant memory
        assert 'memory_profile' in baseline
        assert 'object_counts' in baseline
    
    def test_memory_pool_optimization(self):
        """Test memory reduction through object pooling."""
        optimizer = MemoryOptimizer()
        
        # Class that allocates frequently
        class Order:
            def __init__(self, symbol, quantity, price):
                self.symbol = symbol
                self.quantity = quantity
                self.price = price
                self.metadata = {'created': time.time()}
        
        # Apply object pooling
        OrderPool = optimizer.create_object_pool(
            Order,
            pool_size=1000,
            reset_function=lambda obj: obj.__dict__.clear()
        )
        
        # Test memory usage without pooling
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        orders_no_pool = []
        for i in range(10000):
            orders_no_pool.append(Order('AAPL', 100, 150.0))
        
        mem_after_no_pool = process.memory_info().rss / 1024 / 1024
        mem_used_no_pool = mem_after_no_pool - mem_before
        
        # Clear
        orders_no_pool.clear()
        import gc
        gc.collect()
        
        # Test with pooling
        mem_before_pool = process.memory_info().rss / 1024 / 1024
        
        orders_pool = []
        for i in range(10000):
            order = OrderPool.acquire()
            order.symbol = 'AAPL'
            order.quantity = 100
            order.price = 150.0
            orders_pool.append(order)
        
        mem_after_pool = process.memory_info().rss / 1024 / 1024
        mem_used_pool = mem_after_pool - mem_before_pool
        
        # Return objects to pool
        for order in orders_pool:
            OrderPool.release(order)
        
        assert mem_used_pool < mem_used_no_pool * 0.5  # 50%+ reduction
    
    def test_data_structure_optimization(self):
        """Test memory reduction through efficient data structures."""
        optimizer = MemoryOptimizer()
        
        # Original implementation with dictionaries
        def create_price_history_dict(symbols, days):
            history = {}
            for symbol in symbols:
                history[symbol] = {
                    'dates': [f'2024-01-{d:02d}' for d in range(1, days+1)],
                    'open': list(np.random.randn(days) * 5 + 100),
                    'high': list(np.random.randn(days) * 5 + 105),
                    'low': list(np.random.randn(days) * 5 + 95),
                    'close': list(np.random.randn(days) * 5 + 100),
                    'volume': list(np.random.randint(1000000, 10000000, days))
                }
            return history
        
        # Optimized with numpy arrays
        optimized_creator = optimizer.optimize_data_structures(
            create_price_history_dict,
            strategies=['numpy_arrays', 'struct_arrays', 'compression']
        )
        
        # Test memory usage
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] * 25  # 100 symbols
        days = 252  # One year
        
        # Measure original
        mem_stats_original = optimizer.profile_memory(
            create_price_history_dict,
            args=(symbols, days)
        )
        
        # Measure optimized
        mem_stats_optimized = optimizer.profile_memory(
            optimized_creator,
            args=(symbols, days)
        )
        
        reduction = 1 - (mem_stats_optimized['peak_memory'] / 
                        mem_stats_original['peak_memory'])
        
        assert reduction > 0.3  # 30%+ memory reduction
    
    def test_achieve_target_memory_reduction(self):
        """Test achieving 30%+ memory reduction target."""
        optimizer = MemoryOptimizer()
        
        # Simulate complete trading system memory usage
        class TradingSystem:
            def __init__(self):
                # Large data structures
                self.price_history = {}
                self.order_book = []
                self.positions = {}
                self.indicators = {}
                self.cache = {}
                
            def load_historical_data(self, symbols, days):
                for symbol in symbols:
                    self.price_history[symbol] = {
                        'data': np.random.randn(days, 6) * 10 + 100,
                        'timestamps': list(range(days)),
                        'metadata': {'symbol': symbol} 
                    }
                    
            def calculate_indicators(self):
                for symbol, data in self.price_history.items():
                    prices = data['data'][:, 3]  # Close prices
                    self.indicators[symbol] = {
                        'sma_20': self._sma(prices, 20),
                        'sma_50': self._sma(prices, 50),
                        'rsi': self._rsi(prices),
                        'macd': self._macd(prices)
                    }
                    
            def _sma(self, prices, period):
                return np.convolve(prices, np.ones(period)/period, 'valid')
                
            def _rsi(self, prices, period=14):
                # Simplified RSI
                deltas = np.diff(prices)
                gains = deltas.copy()
                gains[gains < 0] = 0
                return gains
                
            def _macd(self, prices):
                # Simplified MACD
                return prices[1:] - prices[:-1]
        
        # Create optimized version
        OptimizedTradingSystem = optimizer.optimize_class(
            TradingSystem,
            techniques=[
                'lazy_loading',
                'memory_mapping',
                'object_pooling',
                'gc_optimization',
                'dtype_optimization'
            ]
        )
        
        # Test memory usage
        symbols = ['AAPL', 'GOOGL', 'MSFT'] * 10  # 30 symbols
        days = 1000
        
        # Original system
        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        system1 = TradingSystem()
        system1.load_historical_data(symbols, days)
        system1.calculate_indicators()
        
        mem_original = process.memory_info().rss / 1024 / 1024 - mem_before
        
        # Cleanup
        del system1
        gc.collect()
        
        # Optimized system
        mem_before = process.memory_info().rss / 1024 / 1024
        
        system2 = OptimizedTradingSystem()
        system2.load_historical_data(symbols, days)
        system2.calculate_indicators()
        
        mem_optimized = process.memory_info().rss / 1024 / 1024 - mem_before
        
        reduction = 1 - (mem_optimized / mem_original)
        assert reduction > 0.3, f"Memory reduction {reduction:.1%} below 30% target"
        print(f"Achieved memory reduction: {reduction:.1%}")


class TestTradingMetricsOptimization:
    """Test trading performance metrics optimization."""
    
    def test_sharpe_ratio_improvement(self):
        """Test improving Sharpe ratio by 20%+."""
        optimizer = TradingMetricsOptimizer()
        
        # Original strategy
        def basic_momentum_strategy(prices, params):
            ma_short = params.get('ma_short', 20)
            ma_long = params.get('ma_long', 50)
            
            # Calculate moving averages
            sma_short = np.convolve(prices, np.ones(ma_short)/ma_short, 'valid')
            sma_long = np.convolve(prices, np.ones(ma_long)/ma_long, 'valid')
            
            # Align arrays
            min_len = min(len(sma_short), len(sma_long))
            sma_short = sma_short[-min_len:]
            sma_long = sma_long[-min_len:]
            prices_aligned = prices[-min_len:]
            
            # Generate signals
            signals = np.where(sma_short > sma_long, 1, -1)
            
            # Calculate returns
            price_returns = np.diff(prices_aligned) / prices_aligned[:-1]
            strategy_returns = signals[:-1] * price_returns
            
            return strategy_returns
        
        # Generate test data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(1000) * 0.01))
        
        # Baseline Sharpe
        baseline_returns = basic_momentum_strategy(prices, {})
        baseline_sharpe = np.mean(baseline_returns) / np.std(baseline_returns) * np.sqrt(252)
        
        # Optimize strategy
        optimized_params = optimizer.optimize_sharpe_ratio(
            strategy_function=basic_momentum_strategy,
            price_data=prices,
            initial_params={'ma_short': 20, 'ma_long': 50},
            optimization_method='bayesian',
            target_improvement=0.2
        )
        
        # Calculate optimized Sharpe
        optimized_returns = basic_momentum_strategy(prices, optimized_params['params'])
        optimized_sharpe = np.mean(optimized_returns) / np.std(optimized_returns) * np.sqrt(252)
        
        improvement = (optimized_sharpe - baseline_sharpe) / abs(baseline_sharpe)
        assert improvement > 0.2, f"Sharpe improvement {improvement:.1%} below 20% target"
        
        # Additional risk metrics should also improve
        assert optimized_params['max_drawdown'] < 0.2  # Less than 20% drawdown
        assert optimized_params['win_rate'] > 0.5  # Positive win rate
    
    def test_risk_adjusted_optimization(self):
        """Test optimization with risk constraints."""
        optimizer = TradingMetricsOptimizer()
        
        # Strategy with position sizing
        def risk_managed_strategy(prices, params):
            # Extract parameters
            entry_threshold = params['entry_threshold']
            exit_threshold = params['exit_threshold']
            max_position = params['max_position']
            stop_loss = params['stop_loss']
            
            positions = []
            returns = []
            current_position = 0
            entry_price = 0
            
            for i in range(1, len(prices)):
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                
                # Entry logic
                if current_position == 0 and price_change > entry_threshold:
                    current_position = min(max_position, 1.0)
                    entry_price = prices[i]
                    
                # Exit logic
                elif current_position > 0:
                    # Stop loss
                    if prices[i] < entry_price * (1 - stop_loss):
                        returns.append(current_position * (prices[i] / entry_price - 1))
                        current_position = 0
                    # Take profit
                    elif price_change < -exit_threshold:
                        returns.append(current_position * (prices[i] / entry_price - 1))
                        current_position = 0
                
                positions.append(current_position)
            
            return np.array(returns)
        
        # Optimize with risk constraints
        constraints = {
            'max_drawdown': 0.15,      # Max 15% drawdown
            'var_95': 0.02,            # 95% VaR of 2%
            'max_position_size': 0.5,   # Max 50% position
            'min_sharpe': 1.0          # Minimum Sharpe of 1.0
        }
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(2000) * 0.005))
        
        result = optimizer.optimize_with_constraints(
            strategy_function=risk_managed_strategy,
            price_data=prices,
            constraints=constraints,
            initial_params={
                'entry_threshold': 0.01,
                'exit_threshold': 0.005,
                'max_position': 1.0,
                'stop_loss': 0.02
            }
        )
        
        assert result['success']
        assert result['metrics']['max_drawdown'] <= 0.15
        assert result['metrics']['sharpe_ratio'] >= 1.0
        assert result['params']['max_position'] <= 0.5
    
    def test_multi_asset_optimization(self):
        """Test optimization across multiple assets."""
        optimizer = TradingMetricsOptimizer()
        
        # Generate correlated asset prices
        n_assets = 5
        n_periods = 1000
        
        # Correlation matrix
        correlation = np.array([
            [1.0, 0.5, 0.3, 0.2, 0.1],
            [0.5, 1.0, 0.4, 0.3, 0.2],
            [0.3, 0.4, 1.0, 0.5, 0.3],
            [0.2, 0.3, 0.5, 1.0, 0.4],
            [0.1, 0.2, 0.3, 0.4, 1.0]
        ])
        
        # Generate returns
        cov = correlation * 0.01  # Daily volatility ~1%
        returns = np.random.multivariate_normal(
            mean=[0.0002] * n_assets,  # Small positive drift
            cov=cov,
            size=n_periods
        )
        
        prices = 100 * np.exp(np.cumsum(returns, axis=0))
        
        # Multi-asset strategy
        def portfolio_strategy(all_prices, params):
            weights = params['weights']
            rebalance_freq = params['rebalance_frequency']
            
            # Calculate portfolio returns
            portfolio_returns = []
            
            for i in range(rebalance_freq, len(all_prices)):
                # Calculate period returns for each asset
                period_returns = (all_prices[i] - all_prices[i-1]) / all_prices[i-1]
                
                # Portfolio return is weighted average
                portfolio_return = np.dot(weights, period_returns)
                portfolio_returns.append(portfolio_return)
            
            return np.array(portfolio_returns)
        
        # Optimize portfolio
        result = optimizer.optimize_portfolio(
            asset_prices=prices,
            strategy_function=portfolio_strategy,
            constraints={
                'min_weight': 0.0,
                'max_weight': 0.4,
                'sum_weights': 1.0
            },
            target_metrics={
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.1,
                'correlation_with_market': 0.7
            }
        )
        
        assert result['success']
        assert abs(sum(result['params']['weights']) - 1.0) < 0.01
        assert max(result['params']['weights']) <= 0.4
        assert result['metrics']['sharpe_ratio'] > 1.2  # Good Sharpe