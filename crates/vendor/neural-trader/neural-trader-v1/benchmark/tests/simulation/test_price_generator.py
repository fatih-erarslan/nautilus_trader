"""
Tests for price generation with realistic market dynamics.
"""
import pytest
import numpy as np
import asyncio
from typing import List
from benchmark.src.simulation.price_generator import (
    PriceGenerator, PriceModel, MarketRegime,
    PriceUpdate, PriceGeneratorConfig
)


class TestPriceGenerator:
    """Test suite for price generation."""
    
    @pytest.fixture
    def config(self):
        """Create default price generator configuration."""
        return PriceGeneratorConfig(
            initial_price=100.0,
            volatility=0.20,  # 20% annual volatility
            drift=0.05,       # 5% annual drift
            tick_size=0.01,
            update_frequency=1000  # 1000 updates per second
        )
    
    @pytest.fixture
    def price_generator(self, config):
        """Create price generator instance."""
        return PriceGenerator(symbol="AAPL", config=config)
    
    def test_price_generator_initialization(self, price_generator):
        """Test price generator is properly initialized."""
        assert price_generator.symbol == "AAPL"
        assert price_generator.current_price == 100.0
        assert price_generator.config.volatility == 0.20
        assert price_generator.config.drift == 0.05
    
    def test_generate_single_price_update(self, price_generator):
        """Test generating a single price update."""
        update = price_generator.generate_update()
        
        assert isinstance(update, PriceUpdate)
        assert update.symbol == "AAPL"
        assert update.price > 0
        assert update.volume > 0
        assert update.timestamp > 0
        assert abs(update.price - 100.0) < 5.0  # Reasonable price movement
    
    def test_price_model_brownian_motion(self, config):
        """Test Brownian motion price model."""
        model = PriceModel.create("brownian", config)
        
        # Generate multiple price paths
        prices = []
        current_price = 100.0
        
        for _ in range(1000):
            current_price = model.next_price(current_price, dt=0.001)
            prices.append(current_price)
        
        prices = np.array(prices)
        
        # Check statistical properties
        returns = np.diff(prices) / prices[:-1]
        assert np.mean(returns) < 0.001  # Near zero mean for short time
        assert np.std(returns) < 0.01    # Reasonable volatility
        assert prices.min() > 50         # No extreme moves
        assert prices.max() < 150
    
    def test_price_model_jump_diffusion(self, config):
        """Test jump diffusion price model with occasional jumps."""
        config.jump_frequency = 0.1  # 10% chance of jump
        config.jump_size_mean = 0.02  # 2% average jump
        config.jump_size_std = 0.01
        
        model = PriceModel.create("jump_diffusion", config)
        
        # Generate prices and detect jumps
        prices = []
        jumps = 0
        current_price = 100.0
        
        for _ in range(10000):
            new_price = model.next_price(current_price, dt=0.001)
            if abs(new_price - current_price) / current_price > 0.01:
                jumps += 1
            current_price = new_price
            prices.append(current_price)
        
        # Should have some jumps
        assert jumps > 50
        assert jumps < 2000  # Not too many
    
    def test_market_regime_changes(self, price_generator):
        """Test market regime changes affect volatility."""
        # Normal market
        price_generator.set_market_regime(MarketRegime.NORMAL)
        normal_updates = [price_generator.generate_update() for _ in range(100)]
        normal_returns = np.diff([u.price for u in normal_updates])
        
        # High volatility
        price_generator.set_market_regime(MarketRegime.HIGH_VOLATILITY)
        volatile_updates = [price_generator.generate_update() for _ in range(100)]
        volatile_returns = np.diff([u.price for u in volatile_updates])
        
        # Volatile regime should have higher price swings
        assert np.std(volatile_returns) > np.std(normal_returns)
    
    def test_intraday_patterns(self, price_generator):
        """Test intraday volume and volatility patterns."""
        # Enable intraday patterns
        price_generator.enable_intraday_patterns(
            market_open=9.5,   # 9:30 AM
            market_close=16.0  # 4:00 PM
        )
        
        # Generate updates throughout the day
        updates_by_hour = {hour: [] for hour in range(9, 17)}
        
        for hour in range(9, 17):
            for minute in range(60):
                time_of_day = hour + minute / 60.0
                price_generator.set_time_of_day(time_of_day)
                update = price_generator.generate_update()
                updates_by_hour[hour].append(update)
        
        # Check U-shaped volume pattern (high at open/close)
        open_volume = np.mean([u.volume for u in updates_by_hour[9]])
        midday_volume = np.mean([u.volume for u in updates_by_hour[12]])
        close_volume = np.mean([u.volume for u in updates_by_hour[15]])
        
        assert open_volume > midday_volume * 1.2
        assert close_volume > midday_volume * 1.2
    
    def test_correlated_price_generation(self):
        """Test generating correlated prices for multiple assets."""
        # Create correlation matrix
        symbols = ["AAPL", "MSFT", "GOOGL"]
        correlation_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
        
        # Create correlated generators
        generators = PriceGenerator.create_correlated(
            symbols=symbols,
            correlation_matrix=correlation_matrix,
            base_config=PriceGeneratorConfig(
                initial_price=100.0,
                volatility=0.20,
                drift=0.05
            )
        )
        
        # Generate prices
        prices = {symbol: [] for symbol in symbols}
        for _ in range(1000):
            updates = PriceGenerator.generate_correlated_updates(generators)
            for update in updates:
                prices[update.symbol].append(update.price)
        
        # Calculate actual correlations
        price_arrays = [np.array(prices[s]) for s in symbols]
        returns = [np.diff(p) / p[:-1] for p in price_arrays]
        
        # Check correlations are maintained
        actual_corr_01 = np.corrcoef(returns[0], returns[1])[0, 1]
        actual_corr_02 = np.corrcoef(returns[0], returns[2])[0, 1]
        actual_corr_12 = np.corrcoef(returns[1], returns[2])[0, 1]
        
        assert abs(actual_corr_01 - 0.8) < 0.1
        assert abs(actual_corr_02 - 0.6) < 0.1
        assert abs(actual_corr_12 - 0.7) < 0.1
    
    @pytest.mark.asyncio
    async def test_async_price_streaming(self, price_generator):
        """Test asynchronous price streaming."""
        updates = []
        
        async def collect_updates():
            async for update in price_generator.stream_prices(duration=0.1):
                updates.append(update)
        
        await collect_updates()
        
        # Should generate ~100 updates in 0.1 seconds at 1000 Hz
        assert len(updates) > 50
        assert len(updates) < 200
        
        # Check timestamps are increasing
        timestamps = [u.timestamp for u in updates]
        assert all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_historical_replay(self):
        """Test replaying historical price patterns."""
        # Create historical data
        historical_prices = [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(1000)]
        historical_volumes = [1000000 + np.random.randint(-100000, 100000) for _ in range(1000)]
        
        # Create replay generator
        generator = PriceGenerator.from_historical(
            symbol="AAPL",
            prices=historical_prices,
            volumes=historical_volumes,
            timestamps=list(range(1000))
        )
        
        # Replay should match historical data
        for i in range(10):
            update = generator.replay_next()
            assert abs(update.price - historical_prices[i]) < 0.01
            assert update.volume == historical_volumes[i]
    
    def test_extreme_market_conditions(self, price_generator):
        """Test behavior under extreme market conditions."""
        # Flash crash scenario
        price_generator.trigger_flash_crash(
            crash_magnitude=0.10,  # 10% drop
            recovery_time=60       # 60 seconds to recover
        )
        
        prices = []
        for _ in range(100):
            update = price_generator.generate_update()
            prices.append(update.price)
        
        # Should see significant drop
        min_price = min(prices)
        assert min_price < 95.0  # At least 5% drop observed
        
        # Circuit breaker test
        price_generator.set_circuit_breaker(threshold=0.07)  # 7% move triggers halt
        price_generator.reset(initial_price=100.0)
        
        # Force large move
        price_generator.config.volatility = 10.0  # Extreme volatility
        
        halted = False
        for _ in range(100):
            update = price_generator.generate_update()
            if update.halted:
                halted = True
                break
        
        assert halted  # Circuit breaker should trigger
    
    def test_order_flow_impact(self, price_generator):
        """Test price impact from order flow."""
        # Configure order flow impact
        price_generator.set_order_flow_impact(
            impact_coefficient=0.0001,  # 1 bp per $1M volume
            decay_rate=0.9
        )
        
        # Simulate buy pressure
        buy_volume = 10000000  # $10M buy volume
        price_before = price_generator.current_price
        
        update = price_generator.generate_update_with_order_flow(
            net_order_flow=buy_volume
        )
        
        # Price should increase
        assert update.price > price_before
        assert (update.price - price_before) / price_before > 0.0005  # At least 5 bps
    
    def test_volatility_clustering(self, price_generator):
        """Test GARCH-like volatility clustering."""
        price_generator.enable_volatility_clustering(
            persistence=0.9,
            mean_reversion=0.1
        )
        
        # Generate long series of prices
        prices = []
        for _ in range(5000):
            update = price_generator.generate_update()
            prices.append(update.price)
        
        # Calculate returns
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        
        # Calculate rolling volatility
        window = 20
        rolling_vol = []
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            rolling_vol.append(vol)
        
        # Check for volatility clustering (autocorrelation)
        if len(rolling_vol) > 1:
            vol_array = np.array(rolling_vol)
            autocorr = np.corrcoef(vol_array[:-1], vol_array[1:])[0, 1]
            
            # With such high persistence, we expect significant autocorrelation
            assert autocorr > 0.3  # Positive autocorrelation