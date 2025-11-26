"""
Tests for the complete market simulation engine.
"""
import asyncio
import pytest
import numpy as np
import time
from typing import List
from benchmark.src.simulation.market_simulator import (
    MarketSimulator, SimulationConfig, MarketParticipant,
    MarketMaker, RandomTrader, MomentumTrader,
    SimulationResult, MarketStats
)


class TestMarketSimulator:
    """Test suite for market simulator."""
    
    @pytest.fixture
    def config(self):
        """Create simulation configuration."""
        return SimulationConfig(
            symbols=["AAPL", "MSFT", "GOOGL"],
            duration=10.0,  # 10 seconds
            tick_rate=1000,  # 1000 ticks/second
            initial_prices={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2000.0},
            participant_counts={
                "market_maker": 2,
                "random_trader": 10,
                "momentum_trader": 5
            }
        )
    
    @pytest.fixture
    def simulator(self, config):
        """Create market simulator instance."""
        return MarketSimulator(config)
    
    def test_simulator_initialization(self, simulator):
        """Test simulator is properly initialized."""
        assert len(simulator.symbols) == 3
        assert len(simulator.order_books) == 3
        assert len(simulator.price_generators) == 3
        assert len(simulator.participants) > 0
    
    def test_create_participants(self, simulator):
        """Test participant creation."""
        participants = simulator.participants
        
        # Count participant types
        market_makers = sum(1 for p in participants if isinstance(p, MarketMaker))
        random_traders = sum(1 for p in participants if isinstance(p, RandomTrader))
        momentum_traders = sum(1 for p in participants if isinstance(p, MomentumTrader))
        
        assert market_makers == 2 * 3  # 2 per symbol
        assert random_traders == 10
        assert momentum_traders == 5
    
    @pytest.mark.asyncio
    async def test_run_simulation(self, simulator):
        """Test running a short simulation."""
        # Run for 1 second
        simulator.config.duration = 1.0
        
        result = await simulator.run()
        
        assert isinstance(result, SimulationResult)
        assert result.duration > 0.9
        assert result.total_ticks > 500  # Should have many ticks
        assert result.total_trades > 0
        assert len(result.market_stats) == 3  # One per symbol
    
    def test_market_maker_behavior(self):
        """Test market maker provides liquidity."""
        config = SimulationConfig(
            symbols=["TEST"],
            initial_prices={"TEST": 100.0}
        )
        simulator = MarketSimulator(config)
        
        # Get market maker
        market_maker = next(p for p in simulator.participants if isinstance(p, MarketMaker))
        
        # Generate orders
        orders = market_maker.generate_orders(
            symbol="TEST",
            current_price=100.0,
            order_book_snapshot=simulator.order_books["TEST"].get_snapshot()
        )
        
        # Should generate both buy and sell orders
        buy_orders = [o for o in orders if o.side.value == "BUY"]
        sell_orders = [o for o in orders if o.side.value == "SELL"]
        
        assert len(buy_orders) > 0
        assert len(sell_orders) > 0
        assert all(o.price < 100.0 for o in buy_orders)  # Below market
        assert all(o.price > 100.0 for o in sell_orders)  # Above market
    
    def test_momentum_trader_behavior(self):
        """Test momentum trader follows trends."""
        config = SimulationConfig(
            symbols=["TEST"],
            initial_prices={"TEST": 100.0}
        )
        simulator = MarketSimulator(config)
        
        # Get momentum trader
        momentum_trader = MomentumTrader("momentum_1", capital=100000)
        
        # Simulate upward price movement
        price_history = [100, 101, 102, 103, 104]
        
        # Trader should want to buy
        signal = momentum_trader.calculate_signal("TEST", price_history)
        assert signal > 0  # Bullish signal
        
        # Simulate downward movement
        price_history = [104, 103, 102, 101, 100]
        signal = momentum_trader.calculate_signal("TEST", price_history)
        assert signal < 0  # Bearish signal
    
    def test_random_trader_behavior(self):
        """Test random trader generates diverse orders."""
        trader = RandomTrader("random_1", capital=50000)
        
        orders = []
        for _ in range(100):
            order_list = trader.generate_orders(
                symbol="TEST",
                current_price=100.0,
                order_book_snapshot=None
            )
            orders.extend(order_list)
        
        # Should have mix of buy and sell
        buy_count = sum(1 for o in orders if o.side.value == "BUY")
        sell_count = sum(1 for o in orders if o.side.value == "SELL")
        
        assert buy_count > 20
        assert sell_count > 20
        assert buy_count != sell_count  # Very unlikely to be equal
    
    @pytest.mark.asyncio
    async def test_high_frequency_simulation(self, simulator):
        """Test simulation can handle high frequency trading."""
        # Configure for very high frequency
        simulator.config.duration = 0.1  # 100ms
        simulator.config.tick_rate = 10000  # 10k ticks/second
        
        start_time = time.perf_counter()
        result = await simulator.run()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # Should complete quickly
        assert elapsed < 0.5  # Should not take more than 500ms
        assert result.total_ticks >= 500  # At least 500 ticks
        
        # Calculate actual tick rate
        actual_rate = result.total_ticks / result.duration
        assert actual_rate > 5000  # Should achieve high tick rate
    
    def test_market_stats_calculation(self, simulator):
        """Test market statistics calculation."""
        # Run simulation
        simulator.config.duration = 2.0
        
        # Manually inject some trades for testing
        order_book = simulator.order_books["AAPL"]
        
        # Add orders and create trades
        from benchmark.src.simulation.order_book import Order, OrderSide, OrderType
        
        # Create spread
        sell_order = Order(
            order_id="S1",
            side=OrderSide.SELL,
            price=150.10,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        buy_order = Order(
            order_id="B1",
            side=OrderSide.BUY,
            price=149.90,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        
        order_book.add_order(sell_order)
        order_book.add_order(buy_order)
        
        # Get stats
        stats = simulator.calculate_market_stats("AAPL")
        
        assert stats.symbol == "AAPL"
        assert stats.average_spread == 0.20
        assert stats.total_volume == 0  # No trades yet
        assert stats.price_volatility >= 0
    
    @pytest.mark.asyncio
    async def test_multi_symbol_correlation(self):
        """Test correlation between multiple symbols."""
        # Create correlated market
        config = SimulationConfig(
            symbols=["AAPL", "MSFT"],
            initial_prices={"AAPL": 150.0, "MSFT": 300.0},
            duration=5.0,
            correlation_matrix=np.array([[1.0, 0.8], [0.8, 1.0]])
        )
        
        simulator = MarketSimulator(config)
        result = await simulator.run()
        
        # Extract price series
        aapl_prices = result.price_series["AAPL"]
        msft_prices = result.price_series["MSFT"]
        
        # Calculate returns
        aapl_returns = np.diff(aapl_prices) / aapl_prices[:-1]
        msft_returns = np.diff(msft_prices) / msft_prices[:-1]
        
        # Check correlation (should be positive)
        if len(aapl_returns) > 10:
            correlation = np.corrcoef(aapl_returns, msft_returns)[0, 1]
            assert correlation > 0.3  # Should show positive correlation
    
    def test_circuit_breaker_activation(self, simulator):
        """Test circuit breaker halts trading."""
        # Enable circuit breakers
        simulator.enable_circuit_breakers(threshold=0.05)  # 5% move
        
        # Trigger large price move
        price_gen = simulator.price_generators["AAPL"]
        price_gen.trigger_flash_crash(crash_magnitude=0.10, recovery_time=1.0)
        
        # Check if trading halts
        halted = False
        for _ in range(100):
            update = price_gen.generate_update()
            if update.halted:
                halted = True
                break
        
        assert halted  # Should have triggered halt
    
    @pytest.mark.asyncio
    async def test_event_injection(self, simulator):
        """Test injecting market events during simulation."""
        # Schedule events
        simulator.schedule_event(
            time=1.0,
            event_type="news",
            data={"symbol": "AAPL", "sentiment": "negative", "impact": 0.05}
        )
        
        simulator.schedule_event(
            time=2.0,
            event_type="halt",
            data={"symbol": "MSFT", "duration": 60}
        )
        
        # Run simulation
        result = await simulator.run()
        
        # Check events were processed
        assert len(result.events_processed) == 2
        assert result.events_processed[0]["event_type"] == "news"
        assert result.events_processed[1]["event_type"] == "halt"
    
    def test_simulation_result_export(self, simulator):
        """Test exporting simulation results."""
        # Create mock result
        result = SimulationResult(
            duration=10.0,
            total_ticks=10000,
            total_trades=5000,
            market_stats={
                "AAPL": MarketStats(
                    symbol="AAPL",
                    total_volume=1000000,
                    total_trades=2000,
                    average_spread=0.05,
                    price_volatility=0.02,
                    vwap=150.50
                )
            },
            price_series={"AAPL": [150, 150.1, 150.2]},
            events_processed=[]
        )
        
        # Export to dict
        exported = result.to_dict()
        
        assert exported["duration"] == 10.0
        assert exported["total_ticks"] == 10000
        assert "AAPL" in exported["market_stats"]
        assert len(exported["price_series"]["AAPL"]) == 3
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test simulator memory usage stays reasonable."""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run large simulation
        config = SimulationConfig(
            symbols=["TEST1", "TEST2", "TEST3", "TEST4", "TEST5"],
            duration=10.0,
            tick_rate=1000,
            initial_prices={f"TEST{i}": 100.0 for i in range(1, 6)}
        )
        
        simulator = MarketSimulator(config)
        await simulator.run()
        
        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 500  # Less than 500MB increase