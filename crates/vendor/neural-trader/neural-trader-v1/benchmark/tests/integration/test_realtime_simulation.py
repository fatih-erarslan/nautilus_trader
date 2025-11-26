"""
Real-time simulation integration test suite for AI News Trading benchmark system.

This module tests the integration between real-time data feeds and simulation engine:
- Real-time data processing
- Live strategy execution
- Latency optimization
- Concurrent symbol handling
- Market condition simulation
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from datetime import datetime, timedelta

from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.simulation.event_simulator import EventSimulator
from benchmark.src.data.realtime_feed import RealtimeFeed
from benchmark.src.simulation.scenarios import *


class TestRealtimeSimulationIntegration:
    """Test real-time simulation system integration."""
    
    @pytest.fixture
    async def realtime_config(self):
        """Create real-time configuration."""
        return {
            'data_sources': {
                'primary': {
                    'type': 'websocket',
                    'url': 'wss://api.test.com/stream',
                    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD'],
                    'max_latency_ms': 10
                },
                'secondary': {
                    'type': 'rest',
                    'url': 'https://api.test.com/quotes',
                    'polling_interval_ms': 100
                }
            },
            'simulation': {
                'mode': 'realtime',
                'speed_multiplier': 1.0,
                'buffer_size': 10000,
                'max_latency_ms': 50
            },
            'strategies': {
                'momentum': {
                    'enabled': True,
                    'params': {'period': 20, 'threshold': 0.02}
                },
                'arbitrage': {
                    'enabled': True,
                    'params': {'min_spread': 0.001, 'timeout': 5}
                },
                'news_sentiment': {
                    'enabled': True,
                    'params': {'sentiment_threshold': 0.6}
                }
            }
        }
    
    @pytest.fixture
    async def realtime_system(self, realtime_config):
        """Create integrated real-time system."""
        system = {
            'data_manager': RealtimeManager(realtime_config['data_sources']),
            'simulator': MarketSimulator(realtime_config['simulation']),
            'event_simulator': EventSimulator()
        }
        
        # Initialize components
        await system['data_manager'].initialize()
        await system['simulator'].initialize()
        await system['event_simulator'].initialize()
        
        yield system
        
        # Cleanup
        await system['data_manager'].shutdown()
        await system['simulator'].shutdown()
        await system['event_simulator'].shutdown()
    
    @pytest.mark.asyncio
    async def test_realtime_data_to_simulation_pipeline(self, realtime_system, realtime_config):
        """Test complete pipeline from real-time data to simulation results."""
        data_manager = realtime_system['data_manager']
        simulator = realtime_system['simulator']
        
        # Mock real-time data stream
        market_updates = []
        news_updates = []
        
        # Generate realistic market data
        base_prices = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300}
        
        for i in range(100):
            timestamp = time.time() + i * 0.1
            for symbol, base_price in base_prices.items():
                price_change = np.random.normal(0, base_price * 0.001)
                market_updates.append({
                    'symbol': symbol,
                    'price': base_price + price_change,
                    'volume': np.random.randint(100, 5000),
                    'timestamp': timestamp
                })
        
        # Generate news events
        news_events = [
            {'headline': 'Apple announces new product', 'sentiment': 0.8, 'timestamp': time.time() + 2},
            {'headline': 'Google reports strong earnings', 'sentiment': 0.7, 'timestamp': time.time() + 5},
            {'headline': 'Microsoft expands cloud services', 'sentiment': 0.6, 'timestamp': time.time() + 8}
        ]
        
        # Start data processing
        await data_manager.start_feeds()
        
        # Process market updates
        latency_measurements = []
        for update in market_updates:
            start_time = time.perf_counter()
            
            # Process market data
            await data_manager.process_market_update(update)
            
            # Generate signal
            signal = await simulator.generate_signal(update['symbol'])
            
            # Execute strategy if signal generated
            if signal and signal.get('action') != 'hold':
                trade_result = await simulator.execute_trade(signal)
                
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latency_measurements.append(latency_ms)
            
            await asyncio.sleep(0.01)  # Small delay to simulate real-time
        
        # Process news events
        for news in news_events:
            await data_manager.process_news_update(news)
            
            # Check for news-driven signals
            for symbol in base_prices.keys():
                signal = await simulator.generate_signal(symbol, news_context=news)
                if signal and signal.get('action') != 'hold':
                    await simulator.execute_trade(signal)
        
        # Validate latency requirements
        avg_latency = np.mean(latency_measurements)
        p95_latency = np.percentile(latency_measurements, 95)
        p99_latency = np.percentile(latency_measurements, 99)
        
        assert avg_latency < 25, f"Average latency {avg_latency:.2f}ms exceeds 25ms"
        assert p95_latency < 40, f"P95 latency {p95_latency:.2f}ms exceeds 40ms"
        assert p99_latency < 50, f"P99 latency {p99_latency:.2f}ms exceeds 50ms"
        
        # Validate simulation results
        portfolio = await simulator.get_portfolio_state()
        trades = await simulator.get_executed_trades()
        
        assert len(trades) > 0, "No trades were executed"
        assert portfolio['total_value'] != 100000, "Portfolio value unchanged"
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self, realtime_system):
        """Test concurrent processing of multiple symbols in real-time."""
        data_manager = realtime_system['data_manager']
        simulator = realtime_system['simulator']
        
        # Test with 50 symbols
        symbols = [f'SYM{i:03d}' for i in range(50)]
        
        # Configure data manager for all symbols
        await data_manager.configure_symbols(symbols)
        
        # Generate concurrent updates
        async def generate_symbol_updates(symbol):
            updates = []
            for i in range(20):  # 20 updates per symbol
                update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time() + i * 0.05
                }
                updates.append(update)
            return updates
        
        # Generate all updates concurrently
        all_updates = []
        for symbol in symbols:
            symbol_updates = await generate_symbol_updates(symbol)
            all_updates.extend(symbol_updates)
        
        # Process all updates concurrently
        start_time = time.perf_counter()
        
        tasks = []
        for update in all_updates:
            task = asyncio.create_task(
                data_manager.process_market_update(update)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Generate signals for all symbols
        signal_tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                simulator.generate_signal(symbol)
            )
            signal_tasks.append(task)
        
        signals = await asyncio.gather(*signal_tasks)
        
        end_time = time.perf_counter()
        
        # Validate concurrent processing performance
        total_time = (end_time - start_time) * 1000
        updates_per_ms = len(all_updates) / total_time
        
        assert total_time < 2000, f"Concurrent processing took {total_time:.0f}ms > 2000ms"
        assert updates_per_ms > 0.5, f"Processing rate {updates_per_ms:.2f} updates/ms too slow"
        assert len(signals) == len(symbols), "Not all symbols generated signals"
    
    @pytest.mark.asyncio
    async def test_market_scenario_simulation(self, realtime_system):
        """Test different market scenarios in real-time simulation."""
        simulator = realtime_system['simulator']
        event_simulator = realtime_system['event_simulator']
        
        # Test bull market scenario
        bull_scenario = BullMarketScenario(
            duration_minutes=5,
            trend_strength=0.8,
            volatility=0.15
        )
        
        await event_simulator.load_scenario(bull_scenario)
        
        # Run bull market simulation
        bull_results = await simulator.run_scenario(bull_scenario)
        
        assert bull_results['scenario_type'] == 'bull_market'
        assert bull_results['total_return'] > 0, "Bull market should generate positive returns"
        assert bull_results['max_drawdown'] < 0.1, "Bull market drawdown should be limited"
        
        # Test bear market scenario
        bear_scenario = BearMarketScenario(
            duration_minutes=5,
            decline_rate=0.6,
            volatility=0.25
        )
        
        await event_simulator.load_scenario(bear_scenario)
        bear_results = await simulator.run_scenario(bear_scenario)
        
        assert bear_results['scenario_type'] == 'bear_market'
        # In bear market, good strategies should limit losses
        assert bear_results['max_drawdown'] < 0.2, "Bear market drawdown should be controlled"
        
        # Test high volatility scenario
        volatility_scenario = HighVolatilityScenario(
            duration_minutes=3,
            volatility_multiplier=3.0,
            event_frequency=10  # events per minute
        )
        
        await event_simulator.load_scenario(volatility_scenario)
        volatility_results = await simulator.run_scenario(volatility_scenario)
        
        assert volatility_results['scenario_type'] == 'high_volatility'
        assert volatility_results['volatility'] > 0.3, "High volatility scenario should show high volatility"
        
        # Test flash crash scenario
        flash_crash_scenario = FlashCrashScenario(
            trigger_time=60,  # 1 minute in
            crash_magnitude=0.1,  # 10% drop
            recovery_time=30  # 30 seconds to recover
        )
        
        await event_simulator.load_scenario(flash_crash_scenario)
        crash_results = await simulator.run_scenario(flash_crash_scenario)
        
        assert crash_results['scenario_type'] == 'flash_crash'
        assert 'crash_detected' in crash_results
        assert crash_results['recovery_time'] <= 30, "Flash crash recovery should be quick"
    
    @pytest.mark.asyncio
    async def test_news_driven_simulation(self, realtime_system):
        """Test news-driven real-time simulation."""
        data_manager = realtime_system['data_manager']
        simulator = realtime_system['simulator']
        
        # Create news event sequence
        news_sequence = [
            {
                'timestamp': time.time() + 1,
                'headline': 'Apple reports record earnings',
                'sentiment': 0.9,
                'relevance': 0.95,
                'symbols': ['AAPL'],
                'expected_impact': 'positive'
            },
            {
                'timestamp': time.time() + 3,
                'headline': 'Fed raises interest rates',
                'sentiment': -0.6,
                'relevance': 0.8,
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'expected_impact': 'negative'
            },
            {
                'timestamp': time.time() + 5,
                'headline': 'Tech sector shows resilience',
                'sentiment': 0.7,
                'relevance': 0.85,
                'symbols': ['GOOGL', 'MSFT'],
                'expected_impact': 'positive'
            }
        ]
        
        # Track news impact on signals
        news_signals = []
        
        for news_event in news_sequence:
            # Wait for news timestamp
            await asyncio.sleep(0.1)
            
            # Process news event
            await data_manager.process_news_update(news_event)
            
            # Generate signals for affected symbols
            for symbol in news_event['symbols']:
                signal = await simulator.generate_signal(symbol, news_context=news_event)
                if signal:
                    signal['news_triggered'] = True
                    signal['news_sentiment'] = news_event['sentiment']
                    news_signals.append(signal)
        
        # Validate news-driven signals
        assert len(news_signals) > 0, "No news-driven signals generated"
        
        # Check positive news generates buy signals
        positive_signals = [s for s in news_signals if s['news_sentiment'] > 0.5]
        buy_signals = [s for s in positive_signals if s['action'] == 'buy']
        assert len(buy_signals) > 0, "Positive news should generate buy signals"
        
        # Check negative news generates sell signals
        negative_signals = [s for s in news_signals if s['news_sentiment'] < -0.5]
        sell_signals = [s for s in negative_signals if s['action'] == 'sell']
        assert len(sell_signals) > 0, "Negative news should generate sell signals"
    
    @pytest.mark.asyncio
    async def test_arbitrage_opportunity_detection(self, realtime_system):
        """Test real-time arbitrage opportunity detection."""
        data_manager = realtime_system['data_manager']
        simulator = realtime_system['simulator']
        
        # Create price differences between exchanges
        exchanges = ['exchange_a', 'exchange_b', 'exchange_c']
        symbol = 'BTC-USD'
        
        # Base price with differences
        base_price = 45000
        price_differences = [0, 50, -25]  # Different prices on different exchanges
        
        arbitrage_opportunities = []
        
        for i in range(10):
            # Generate price updates with arbitrage opportunities
            for j, exchange in enumerate(exchanges):
                price = base_price + price_differences[j] + np.random.normal(0, 10)
                
                update = {
                    'symbol': symbol,
                    'price': price,
                    'volume': 1000,
                    'exchange': exchange,
                    'timestamp': time.time() + i * 0.1
                }
                
                await data_manager.process_market_update(update)
            
            # Check for arbitrage opportunities
            opportunity = await simulator.detect_arbitrage_opportunity(symbol)
            if opportunity:
                arbitrage_opportunities.append(opportunity)
                
                # Execute arbitrage if profitable
                if opportunity['profit_potential'] > 0.001:  # Min 0.1% profit
                    trade_result = await simulator.execute_arbitrage_trade(opportunity)
                    assert trade_result['status'] == 'executed'
                    assert trade_result['profit'] > 0
        
        # Validate arbitrage detection
        assert len(arbitrage_opportunities) > 0, "No arbitrage opportunities detected"
        
        # Check profit potential calculations
        for opp in arbitrage_opportunities:
            assert 'buy_exchange' in opp
            assert 'sell_exchange' in opp
            assert 'buy_price' in opp
            assert 'sell_price' in opp
            assert 'profit_potential' in opp
            assert opp['sell_price'] > opp['buy_price'], "Arbitrage opportunity should have sell > buy price"
    
    @pytest.mark.asyncio
    async def test_latency_optimization_features(self, realtime_system):
        """Test latency optimization features in real-time simulation."""
        data_manager = realtime_system['data_manager']
        simulator = realtime_system['simulator']
        
        # Enable latency optimization features
        await data_manager.enable_latency_optimization({
            'prefetch_enabled': True,
            'cache_predictions': True,
            'parallel_processing': True,
            'batch_processing': True
        })
        
        await simulator.enable_latency_optimization({
            'precompute_indicators': True,
            'signal_caching': True,
            'fast_execution_mode': True
        })
        
        # Test latency with optimization
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        latency_measurements = []
        
        for i in range(100):
            updates = []
            for symbol in symbols:
                update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': np.random.randint(100, 1000),
                    'timestamp': time.time()
                }
                updates.append(update)
            
            start_time = time.perf_counter()
            
            # Process all updates
            for update in updates:
                await data_manager.process_market_update(update)
            
            # Generate signals for all symbols
            signals = await simulator.generate_signals_batch(symbols)
            
            end_time = time.perf_counter()
            
            batch_latency = (end_time - start_time) * 1000
            latency_measurements.append(batch_latency)
            
            await asyncio.sleep(0.01)
        
        # Validate optimized latency
        avg_latency = np.mean(latency_measurements)
        p99_latency = np.percentile(latency_measurements, 99)
        
        assert avg_latency < 15, f"Optimized average latency {avg_latency:.2f}ms exceeds 15ms"
        assert p99_latency < 30, f"Optimized P99 latency {p99_latency:.2f}ms exceeds 30ms"
        
        # Compare with non-optimized baseline
        await data_manager.disable_latency_optimization()
        await simulator.disable_latency_optimization()
        
        baseline_measurements = []
        for i in range(10):  # Fewer iterations for baseline
            start_time = time.perf_counter()
            
            for symbol in symbols:
                update = {
                    'symbol': symbol,
                    'price': 100 + np.random.normal(0, 5),
                    'volume': 1000,
                    'timestamp': time.time()
                }
                await data_manager.process_market_update(update)
                await simulator.generate_signal(symbol)
            
            end_time = time.perf_counter()
            baseline_measurements.append((end_time - start_time) * 1000)
        
        baseline_avg = np.mean(baseline_measurements)
        improvement = (baseline_avg - avg_latency) / baseline_avg
        
        assert improvement > 0.3, f"Latency optimization improvement {improvement:.1%} < 30%"


@pytest.mark.integration
class TestRealtimeDataFeedIntegration:
    """Test real-time data feed integration components."""
    
    @pytest.mark.asyncio
    async def test_websocket_rest_failover(self):
        """Test WebSocket to REST API failover."""
        feed_config = {
            'websocket_url': 'wss://api.test.com/stream',
            'rest_url': 'https://api.test.com/quotes',
            'failover_enabled': True,
            'max_latency_ms': 10
        }
        
        feed = RealtimeFeed(feed_config)
        
        # Start with WebSocket
        with patch('websockets.connect') as mock_ws:
            mock_ws.return_value.__aenter__.return_value = AsyncMock()
            await feed.connect()
            assert feed.is_websocket_active
        
        # Simulate WebSocket failure
        with patch.object(feed, '_websocket_healthy', return_value=False):
            await feed._check_connection_health()
            
            # Should failover to REST
            assert not feed.is_websocket_active
            assert feed.is_rest_active
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self):
        """Test real-time data quality validation."""
        feed = RealtimeFeed({'quality_checks': True})
        
        # Valid data should pass
        valid_update = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'timestamp': time.time()
        }
        
        is_valid, errors = await feed.validate_data_quality(valid_update)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid data should fail
        invalid_update = {
            'symbol': 'AAPL',
            'price': -10,  # Negative price
            'volume': 0,   # Zero volume
            'timestamp': time.time() - 3600  # Old timestamp
        }
        
        is_valid, errors = await feed.validate_data_quality(invalid_update)
        assert not is_valid
        assert len(errors) > 0


if __name__ == '__main__':
    pytest.main([__file__])