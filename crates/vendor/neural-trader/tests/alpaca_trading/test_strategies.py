"""
Comprehensive tests for trading strategies.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np

from src.alpaca_trading.strategies import (
    TradingStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    VWAPStrategy,
    StrategyManager,
    Signal,
    SignalType
)


@pytest.fixture
def market_data():
    """Generate sample market data."""
    return {
        'trades': [
            {'symbol': 'AAPL', 'price': 150.0, 'size': 100, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 150.5, 'size': 200, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 151.0, 'size': 150, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 150.8, 'size': 100, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'price': 151.2, 'size': 300, 'timestamp': datetime.now()}
        ],
        'quotes': [
            {'symbol': 'AAPL', 'bid': 150.0, 'ask': 150.1, 'bid_size': 100, 'ask_size': 100},
            {'symbol': 'AAPL', 'bid': 150.5, 'ask': 150.6, 'bid_size': 200, 'ask_size': 150},
            {'symbol': 'AAPL', 'bid': 151.0, 'ask': 151.1, 'bid_size': 150, 'ask_size': 200}
        ]
    }


class TestSignal:
    """Test signal generation."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        signal = Signal(
            symbol='AAPL',
            type=SignalType.BUY,
            strength=0.8,
            price=150.0,
            quantity=100,
            reason='Momentum breakout',
            metadata={'indicator': 'RSI', 'value': 75}
        )
        
        assert signal.symbol == 'AAPL'
        assert signal.type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.price == 150.0
        assert signal.quantity == 100
        assert signal.reason == 'Momentum breakout'
        assert signal.metadata['indicator'] == 'RSI'
    
    def test_signal_validation(self):
        """Test signal validation."""
        # Valid signal
        signal = Signal('AAPL', SignalType.BUY, 0.8, 150.0, 100)
        assert signal.is_valid()
        
        # Invalid signals
        invalid_signals = [
            Signal('AAPL', SignalType.BUY, 1.5, 150.0, 100),  # strength > 1
            Signal('AAPL', SignalType.BUY, -0.1, 150.0, 100),  # negative strength
            Signal('AAPL', SignalType.BUY, 0.8, -150.0, 100),  # negative price
            Signal('AAPL', SignalType.BUY, 0.8, 150.0, -100),  # negative quantity
        ]
        
        for signal in invalid_signals:
            assert not signal.is_valid()


class TestMomentumStrategy:
    """Test momentum strategy."""
    
    @pytest.mark.asyncio
    async def test_momentum_calculation(self):
        """Test momentum calculation."""
        strategy = MomentumStrategy(
            symbols=['AAPL'],
            lookback_period=5,
            threshold=0.02
        )
        
        # Add price data
        prices = [100, 101, 102, 103, 104, 105, 106]
        for i, price in enumerate(prices):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': price,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        # Should generate buy signal (positive momentum)
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.BUY
        assert signal.strength > 0.5
    
    @pytest.mark.asyncio
    async def test_momentum_reversal(self):
        """Test momentum reversal detection."""
        strategy = MomentumStrategy(
            symbols=['AAPL'],
            lookback_period=5,
            threshold=0.02
        )
        
        # Add declining prices
        prices = [106, 105, 104, 103, 102, 101, 100]
        for i, price in enumerate(prices):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': price,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        # Should generate sell signal (negative momentum)
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.SELL
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = MomentumStrategy(
            symbols=['AAPL'],
            lookback_period=10
        )
        
        # Add only 3 data points
        for i in range(3):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': 100 + i,
                'size': 100,
                'timestamp': datetime.now()
            })
        
        # Should not generate signal
        signal = await strategy.generate_signal('AAPL')
        assert signal is None


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    @pytest.mark.asyncio
    async def test_mean_reversion_buy(self):
        """Test buy signal generation."""
        strategy = MeanReversionStrategy(
            symbols=['AAPL'],
            window=5,
            num_std=2.0
        )
        
        # Create price series with dip
        prices = [100, 101, 100, 99, 100, 101, 100, 95]  # Last price is 2 std below mean
        
        for i, price in enumerate(prices):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': price,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.BUY
        assert signal.reason is not None
    
    @pytest.mark.asyncio
    async def test_mean_reversion_sell(self):
        """Test sell signal generation."""
        strategy = MeanReversionStrategy(
            symbols=['AAPL'],
            window=5,
            num_std=2.0
        )
        
        # Create price series with spike
        prices = [100, 101, 100, 99, 100, 101, 100, 108]  # Last price is 2 std above mean
        
        for i, price in enumerate(prices):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': price,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.SELL
    
    @pytest.mark.asyncio
    async def test_no_deviation(self):
        """Test no signal when price is within bands."""
        strategy = MeanReversionStrategy(
            symbols=['AAPL'],
            window=5,
            num_std=2.0
        )
        
        # Stable prices
        prices = [100, 100.1, 99.9, 100, 100.1, 99.9, 100]
        
        for i, price in enumerate(prices):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': price,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        signal = await strategy.generate_signal('AAPL')
        assert signal is None


class TestVWAPStrategy:
    """Test VWAP strategy."""
    
    @pytest.mark.asyncio
    async def test_vwap_calculation(self):
        """Test VWAP calculation."""
        strategy = VWAPStrategy(
            symbols=['AAPL'],
            threshold=0.01
        )
        
        # Add trades with different volumes
        trades = [
            {'price': 100, 'size': 100},  # VWAP = 100
            {'price': 101, 'size': 200},  # VWAP = 100.67
            {'price': 102, 'size': 100},  # VWAP = 101
        ]
        
        for i, trade in enumerate(trades):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': trade['price'],
                'size': trade['size'],
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        # Check VWAP calculation
        vwap = strategy.get_vwap('AAPL')
        expected_vwap = (100*100 + 101*200 + 102*100) / 400
        assert abs(vwap - expected_vwap) < 0.01
    
    @pytest.mark.asyncio
    async def test_vwap_buy_signal(self):
        """Test buy signal when price crosses below VWAP."""
        strategy = VWAPStrategy(
            symbols=['AAPL'],
            threshold=0.01
        )
        
        # Build up VWAP around 100
        for i in range(10):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': 100,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        # Price drops below VWAP
        await strategy.on_trade({
            'symbol': 'AAPL',
            'price': 98,  # 2% below VWAP
            'size': 100,
            'timestamp': datetime.now() + timedelta(seconds=11)
        })
        
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.BUY
    
    @pytest.mark.asyncio
    async def test_vwap_sell_signal(self):
        """Test sell signal when price crosses above VWAP."""
        strategy = VWAPStrategy(
            symbols=['AAPL'],
            threshold=0.01
        )
        
        # Build up VWAP around 100
        for i in range(10):
            await strategy.on_trade({
                'symbol': 'AAPL',
                'price': 100,
                'size': 100,
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        # Price rises above VWAP
        await strategy.on_trade({
            'symbol': 'AAPL',
            'price': 102,  # 2% above VWAP
            'size': 100,
            'timestamp': datetime.now() + timedelta(seconds=11)
        })
        
        signal = await strategy.generate_signal('AAPL')
        assert signal is not None
        assert signal.type == SignalType.SELL


class TestStrategyManager:
    """Test strategy manager."""
    
    @pytest.mark.asyncio
    async def test_strategy_registration(self):
        """Test strategy registration and management."""
        manager = StrategyManager()
        
        # Create strategies
        momentum = MomentumStrategy(['AAPL'])
        mean_rev = MeanReversionStrategy(['AAPL'])
        
        # Register strategies
        manager.add_strategy('momentum', momentum)
        manager.add_strategy('mean_reversion', mean_rev)
        
        assert len(manager.strategies) == 2
        assert 'momentum' in manager.strategies
        assert 'mean_reversion' in manager.strategies
    
    @pytest.mark.asyncio
    async def test_signal_aggregation(self):
        """Test signal aggregation from multiple strategies."""
        manager = StrategyManager()
        
        # Mock strategies
        mock_strat1 = Mock()
        mock_strat1.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.BUY, 0.8, 150.0, 100
        ))
        
        mock_strat2 = Mock()
        mock_strat2.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.BUY, 0.6, 150.0, 100
        ))
        
        manager.add_strategy('strat1', mock_strat1, weight=0.6)
        manager.add_strategy('strat2', mock_strat2, weight=0.4)
        
        # Get aggregated signal
        signal = await manager.get_signal('AAPL')
        
        assert signal is not None
        assert signal.type == SignalType.BUY
        # Weighted average: 0.8 * 0.6 + 0.6 * 0.4 = 0.72
        assert abs(signal.strength - 0.72) < 0.01
    
    @pytest.mark.asyncio
    async def test_conflicting_signals(self):
        """Test handling of conflicting signals."""
        manager = StrategyManager()
        
        # Mock conflicting strategies
        mock_strat1 = Mock()
        mock_strat1.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.BUY, 0.8, 150.0, 100
        ))
        
        mock_strat2 = Mock()
        mock_strat2.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.SELL, 0.6, 150.0, 100
        ))
        
        manager.add_strategy('strat1', mock_strat1, weight=0.6)
        manager.add_strategy('strat2', mock_strat2, weight=0.4)
        
        # Should return stronger signal
        signal = await manager.get_signal('AAPL')
        
        assert signal is not None
        assert signal.type == SignalType.BUY  # Stronger signal wins
    
    @pytest.mark.asyncio
    async def test_data_distribution(self):
        """Test data distribution to strategies."""
        manager = StrategyManager()
        
        # Create strategies
        strat1 = MomentumStrategy(['AAPL'])
        strat2 = VWAPStrategy(['AAPL'])
        
        manager.add_strategy('momentum', strat1)
        manager.add_strategy('vwap', strat2)
        
        # Distribute trade data
        trade_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'size': 100,
            'timestamp': datetime.now()
        }
        
        await manager.on_trade(trade_data)
        
        # Both strategies should have received the data
        assert len(strat1._price_history['AAPL']) == 1
        assert strat2._vwap_data['AAPL']['volume'] == 100
    
    @pytest.mark.asyncio
    async def test_strategy_enabling_disabling(self):
        """Test enabling/disabling strategies."""
        manager = StrategyManager()
        
        mock_strat = Mock()
        mock_strat.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.BUY, 0.8, 150.0, 100
        ))
        
        manager.add_strategy('test', mock_strat)
        
        # Should work when enabled
        signal = await manager.get_signal('AAPL')
        assert signal is not None
        
        # Disable strategy
        manager.disable_strategy('test')
        
        # Should return None when disabled
        signal = await manager.get_signal('AAPL')
        assert signal is None
        
        # Re-enable
        manager.enable_strategy('test')
        signal = await manager.get_signal('AAPL')
        assert signal is not None
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test strategy performance tracking."""
        manager = StrategyManager()
        
        mock_strat = Mock()
        mock_strat.generate_signal = AsyncMock(return_value=Signal(
            'AAPL', SignalType.BUY, 0.8, 150.0, 100
        ))
        
        manager.add_strategy('test', mock_strat)
        
        # Generate signal
        signal = await manager.get_signal('AAPL')
        
        # Record outcome
        manager.record_signal_outcome('test', signal, profit=50.0)
        
        # Check performance
        perf = manager.get_strategy_performance('test')
        assert perf['total_signals'] == 1
        assert perf['profitable_signals'] == 1
        assert perf['total_profit'] == 50.0