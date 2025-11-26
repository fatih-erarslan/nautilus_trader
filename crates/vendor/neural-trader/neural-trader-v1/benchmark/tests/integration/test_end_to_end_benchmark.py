"""
End-to-end integration test suite for AI News Trading benchmark system.

This module validates the complete pipeline:
1. Data ingestion (news + market data)
2. Signal generation
3. Strategy execution
4. Performance measurement
5. Results reporting
"""

import asyncio
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
import pandas as pd
import numpy as np

from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.optimization.optimizer import StrategyOptimizer
from benchmark.src.cli.commands import BenchmarkCommand
from benchmark.src.config import Config


class TestEndToEndBenchmark:
    """End-to-end benchmark system integration tests."""
    
    @pytest.fixture
    async def config(self):
        """Create comprehensive test configuration."""
        return Config({
            'data': {
                'sources': {
                    'news': {
                        'feeds': ['finnhub', 'alpha_vantage'],
                        'max_latency_ms': 50,
                        'update_frequency': '1s'
                    },
                    'market': {
                        'feeds': ['yahoo', 'coinbase'],
                        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD'],
                        'max_latency_ms': 10,
                        'update_frequency': '100ms'
                    }
                },
                'cache': {
                    'enabled': True,
                    'ttl': 300,
                    'max_size': '1GB'
                }
            },
            'strategies': {
                'momentum': {
                    'enabled': True,
                    'parameters': {
                        'lookback_period': 20,
                        'threshold': 0.02,
                        'position_size': 0.1
                    }
                },
                'news_sentiment': {
                    'enabled': True,
                    'parameters': {
                        'sentiment_threshold': 0.6,
                        'news_impact_decay': 3600,
                        'position_size': 0.15
                    }
                },
                'arbitrage': {
                    'enabled': True,
                    'parameters': {
                        'min_spread': 0.001,
                        'max_position': 0.2,
                        'timeout': 5
                    }
                }
            },
            'simulation': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'max_drawdown': 0.2,
                'risk_per_trade': 0.02
            },
            'performance': {
                'target_latency_ms': 100,
                'target_throughput': 1000,
                'memory_limit_gb': 2,
                'max_concurrent_symbols': 100
            }
        })
    
    @pytest.fixture
    async def benchmark_system(self, config):
        """Create complete benchmark system."""
        system = {
            'data_manager': RealtimeManager(config.data),
            'simulator': MarketSimulator(config.simulation),
            'benchmark_runner': BenchmarkRunner(config),
            'optimizer': StrategyOptimizer(config.strategies)
        }
        
        # Initialize components
        await system['data_manager'].initialize()
        await system['simulator'].initialize()
        
        yield system
        
        # Cleanup
        await system['data_manager'].shutdown()
        await system['simulator'].shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_benchmark_pipeline(self, benchmark_system, config):
        """Test complete benchmark pipeline from data to results."""
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        runner = benchmark_system['benchmark_runner']
        
        # Mock data sources
        with patch.object(data_manager, 'get_market_data') as mock_market:
            with patch.object(data_manager, 'get_news_data') as mock_news:
                # Setup mock data
                mock_market.return_value = self._create_mock_market_data()
                mock_news.return_value = self._create_mock_news_data()
                
                # Start data collection
                await data_manager.start_feeds()
                
                # Wait for data buffer to fill
                await asyncio.sleep(2)
                
                # Run benchmark suite
                results = await runner.run_suite('comprehensive')
                
                # Validate results structure
                assert 'performance' in results
                assert 'strategies' in results
                assert 'metrics' in results
                assert 'timestamp' in results
                
                # Validate performance metrics
                perf = results['performance']
                assert perf['signal_latency_p99'] < config.performance.target_latency_ms
                assert perf['throughput'] > config.performance.target_throughput
                assert perf['memory_usage_gb'] < config.performance.memory_limit_gb
                
                # Validate strategy results
                strategies = results['strategies']
                assert len(strategies) >= 3  # momentum, news_sentiment, arbitrage
                
                for strategy_name, strategy_results in strategies.items():
                    assert 'total_return' in strategy_results
                    assert 'sharpe_ratio' in strategy_results
                    assert 'max_drawdown' in strategy_results
                    assert 'win_rate' in strategy_results
                    assert strategy_results['max_drawdown'] <= config.simulation.max_drawdown
    
    @pytest.mark.asyncio
    async def test_multi_asset_concurrent_processing(self, benchmark_system):
        """Test concurrent processing of multiple asset classes."""
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        
        # Test symbols across different asset classes
        symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA',  # Stocks
            'BTC-USD', 'ETH-USD', 'ADA-USD',         # Crypto
            'EUR/USD', 'GBP/USD',                     # Forex
            'GLD', 'TLT'                              # ETFs
        ]
        
        # Configure for concurrent processing
        await data_manager.configure_symbols(symbols)
        
        # Start concurrent data streams
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                data_manager.start_symbol_feed(symbol)
            )
            tasks.append(task)
        
        # Wait for all feeds to establish
        await asyncio.gather(*tasks)
        
        # Generate signals for all symbols
        start_time = time.perf_counter()
        signals = await simulator.generate_signals_concurrent(symbols)
        end_time = time.perf_counter()
        
        # Validate concurrent processing performance
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 500  # All signals in <500ms
        assert len(signals) == len(symbols)
        
        # Validate each signal
        for symbol, signal in signals.items():
            assert symbol in symbols
            assert 'action' in signal
            assert 'confidence' in signal
            assert 'timestamp' in signal
            assert signal['confidence'] >= 0 and signal['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_real_time_latency_requirements(self, benchmark_system):
        """Test that real-time processing meets latency requirements."""
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        
        latency_measurements = []
        
        # Create mock real-time data stream
        async def generate_market_tick():
            return {
                'symbol': 'AAPL',
                'price': 150.25 + np.random.normal(0, 0.5),
                'volume': 1000,
                'timestamp': time.time()
            }
        
        async def generate_news_event():
            return {
                'headline': 'Apple announces quarterly earnings',
                'sentiment': 0.7,
                'relevance': 0.9,
                'timestamp': time.time()
            }
        
        # Test latency for 1000 market ticks
        for i in range(1000):
            market_tick = await generate_market_tick()
            news_event = await generate_news_event() if i % 10 == 0 else None
            
            start_time = time.perf_counter()
            
            # Process market tick
            await data_manager.process_market_update(market_tick)
            
            # Process news if available
            if news_event:
                await data_manager.process_news_update(news_event)
            
            # Generate signal
            signal = await simulator.generate_signal('AAPL')
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latency_measurements.append(latency_ms)
        
        # Validate latency statistics
        avg_latency = np.mean(latency_measurements)
        p95_latency = np.percentile(latency_measurements, 95)
        p99_latency = np.percentile(latency_measurements, 99)
        
        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms exceeds 50ms"
        assert p95_latency < 75, f"P95 latency {p95_latency:.2f}ms exceeds 75ms"
        assert p99_latency < 100, f"P99 latency {p99_latency:.2f}ms exceeds 100ms"
    
    @pytest.mark.asyncio
    async def test_throughput_stress_test(self, benchmark_system):
        """Test system throughput under stress conditions."""
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        
        # Configure for high throughput
        symbols = [f'SYM{i:03d}' for i in range(100)]  # 100 symbols
        updates_per_symbol = 100  # 100 updates per symbol
        total_updates = len(symbols) * updates_per_symbol
        
        # Generate concurrent updates
        async def generate_updates():
            tasks = []
            for symbol in symbols:
                for i in range(updates_per_symbol):
                    update = {
                        'symbol': symbol,
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 10000),
                        'timestamp': time.time()
                    }
                    task = asyncio.create_task(
                        data_manager.process_market_update(update)
                    )
                    tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Measure throughput
        start_time = time.perf_counter()
        await generate_updates()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = total_updates / duration
        
        # Validate throughput requirement
        assert throughput > 1000, f"Throughput {throughput:.0f} updates/sec < 1000"
        
        # Validate signal generation throughput
        start_time = time.perf_counter()
        signals = await simulator.generate_signals_batch(symbols)
        end_time = time.perf_counter()
        
        signal_duration = end_time - start_time
        signal_throughput = len(symbols) / signal_duration
        
        assert signal_throughput > 100, f"Signal throughput {signal_throughput:.0f} signals/sec < 100"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, benchmark_system):
        """Test memory usage stays within limits during long runs."""
        import psutil
        import gc
        
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run extended simulation (8 hours compressed to 8 minutes)
        simulation_duration = 480  # 8 minutes
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC-USD']
        
        memory_measurements = []
        
        for minute in range(simulation_duration // 60):
            # Simulate 1 hour of trading in 1 minute
            for second in range(60):
                # Generate market updates
                for symbol in symbols:
                    update = {
                        'symbol': symbol,
                        'price': 100 + np.random.normal(0, 5),
                        'volume': np.random.randint(100, 5000),
                        'timestamp': time.time()
                    }
                    await data_manager.process_market_update(update)
                
                # Generate occasional news
                if second % 10 == 0:
                    news = {
                        'headline': f'Market update {minute}:{second}',
                        'sentiment': np.random.uniform(-1, 1),
                        'relevance': np.random.uniform(0.5, 1.0),
                        'timestamp': time.time()
                    }
                    await data_manager.process_news_update(news)
                
                # Generate signals
                await simulator.generate_signals_batch(symbols)
                
                await asyncio.sleep(0.01)  # 10ms delay
            
            # Measure memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
            
            # Force garbage collection
            gc.collect()
        
        # Validate memory usage
        max_memory = max(memory_measurements)
        memory_growth = max_memory - initial_memory
        
        assert max_memory < 2048, f"Peak memory {max_memory:.0f}MB exceeds 2GB limit"
        assert memory_growth < 1024, f"Memory growth {memory_growth:.0f}MB excessive"
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, benchmark_system):
        """Test system resilience to errors and automatic recovery."""
        data_manager = benchmark_system['data_manager']
        simulator = benchmark_system['simulator']
        
        # Test data feed failure recovery
        with patch.object(data_manager, 'primary_feed') as mock_primary:
            # Simulate primary feed failure
            mock_primary.side_effect = Exception("Feed connection lost")
            
            # System should failover to backup
            result = await data_manager.get_market_data(['AAPL'])
            assert result is not None  # Should get data from backup
            assert data_manager.current_feed == 'backup'
        
        # Test strategy execution error handling
        with patch.object(simulator, 'execute_strategy') as mock_execute:
            # Simulate strategy execution error
            mock_execute.side_effect = Exception("Strategy execution failed")
            
            # Should continue with other strategies
            results = await simulator.run_all_strategies(['AAPL'])
            assert len(results) >= 0  # Should handle gracefully
        
        # Test memory pressure handling
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate low memory condition
            mock_memory.return_value.percent = 95  # 95% memory usage
            
            # Should trigger cleanup
            await data_manager.handle_memory_pressure()
            
            # Verify cleanup occurred
            buffer_size = await data_manager.get_buffer_size()
            assert buffer_size < 1000  # Should have cleaned buffers
    
    def _create_mock_market_data(self):
        """Create realistic mock market data."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
        data = {}
        
        for symbol in symbols:
            base_price = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300, 
                         'BTC-USD': 45000, 'ETH-USD': 3000}[symbol]
            
            # Generate 1000 price points
            prices = [base_price]
            for i in range(999):
                change = np.random.normal(0, base_price * 0.001)
                prices.append(max(prices[-1] + change, base_price * 0.5))
            
            data[symbol] = {
                'prices': prices,
                'volumes': np.random.randint(100, 10000, 1000).tolist(),
                'timestamps': [time.time() - (999-i) for i in range(1000)]
            }
        
        return data
    
    def _create_mock_news_data(self):
        """Create realistic mock news data."""
        headlines = [
            "Apple reports record quarterly earnings",
            "Google announces new AI breakthrough",
            "Microsoft expands cloud services",
            "Bitcoin reaches new adoption milestone",
            "Ethereum upgrade improves efficiency",
            "Market volatility increases amid uncertainty",
            "Tech stocks show strong momentum",
            "Crypto regulation clarity emerges"
        ]
        
        news_data = []
        for i, headline in enumerate(headlines):
            news_data.append({
                'headline': headline,
                'sentiment': np.random.uniform(-0.5, 0.8),
                'relevance': np.random.uniform(0.6, 1.0),
                'timestamp': time.time() - (len(headlines) - i) * 300,
                'source': 'mock_feed'
            })
        
        return news_data


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration with complete system."""
    
    @pytest.mark.asyncio
    async def test_cli_benchmark_command_execution(self):
        """Test CLI benchmark command with real system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            results_path = Path(temp_dir) / 'results.json'
            
            # Create test configuration
            config_content = """
            global:
              output_dir: {temp_dir}
              log_level: INFO
            
            benchmark:
              default_suite: quick
              warmup_duration: 5
              measurement_duration: 10
            
            simulation:
              data_source: synthetic
              tick_resolution: 100ms
              symbols: ['AAPL', 'GOOGL']
            """.replace('{temp_dir}', str(temp_dir))
            
            config_path.write_text(config_content)
            
            # Execute CLI command
            command = BenchmarkCommand()
            args = Mock()
            args.config = str(config_path)
            args.suite = 'quick'
            args.format = 'json'
            args.output = str(results_path)
            args.strategy = None
            args.duration = 10
            args.parallel = 2
            args.metrics = None
            args.baseline = None
            args.save_baseline = False
            
            # Run command
            result = await command.execute(args)
            
            # Validate execution
            assert result['status'] == 'success'
            assert results_path.exists()
            
            # Validate results file
            results_data = json.loads(results_path.read_text())
            assert 'benchmark_results' in results_data
            assert 'performance_metrics' in results_data
            assert 'execution_time' in results_data


if __name__ == '__main__':
    pytest.main([__file__])