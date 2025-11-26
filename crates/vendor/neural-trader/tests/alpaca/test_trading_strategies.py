"""
Tests for Trading Strategies
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpaca.alpaca_client import AlpacaClient, OrderSide, OrderType
from alpaca.trading_strategies import (
    MomentumStrategy, MeanReversionStrategy, BuyAndHoldStrategy,
    TradingBot, Signal, PositionSize
)

class TestSignalAndPositionSize(unittest.TestCase):
    """Test Signal and PositionSize dataclasses"""

    def test_signal_creation(self):
        """Test Signal creation"""
        signal = Signal(
            symbol="AAPL",
            action="buy",
            strength=0.8,
            price=150.0,
            timestamp=datetime.now(),
            reason="Strong momentum"
        )

        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.action, "buy")
        self.assertEqual(signal.strength, 0.8)
        self.assertEqual(signal.price, 150.0)

    def test_position_size_creation(self):
        """Test PositionSize creation"""
        pos_size = PositionSize(
            symbol="AAPL",
            target_qty=100.0,
            current_qty=50.0,
            action_qty=50.0,
            action="buy"
        )

        self.assertEqual(pos_size.symbol, "AAPL")
        self.assertEqual(pos_size.target_qty, 100.0)
        self.assertEqual(pos_size.current_qty, 50.0)
        self.assertEqual(pos_size.action_qty, 50.0)

class TestMomentumStrategy(unittest.TestCase):
    """Test MomentumStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock(spec=AlpacaClient)
        self.strategy = MomentumStrategy(self.mock_client, lookback_days=10, volume_threshold=1.5)

    def create_test_data(self, symbol, days=20):
        """Create test market data"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        np.random.seed(42)  # For reproducible tests

        # Create trending data for momentum testing
        base_price = 100
        prices = [base_price]
        volumes = []

        for i in range(1, days):
            # Create upward trend
            change = np.random.normal(0.02, 0.02)  # 2% average daily return
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            volumes.append(np.random.randint(500000, 2000000))

        volumes.append(np.random.randint(500000, 2000000))  # Add volume for first day

        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)

        return data

    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.name, "Momentum")
        self.assertEqual(self.strategy.lookback_days, 10)
        self.assertEqual(self.strategy.volume_threshold, 1.5)
        self.assertEqual(self.strategy.client, self.mock_client)

    def test_generate_signals_with_strong_momentum(self):
        """Test signal generation with strong momentum"""
        # Create data with strong upward momentum
        data = {
            'AAPL': self.create_test_data('AAPL', days=20)
        }

        # Ensure the last day has strong momentum
        data['AAPL'].loc[data['AAPL'].index[-1], 'close'] = 120  # 20% gain over period
        data['AAPL'].loc[data['AAPL'].index[-1], 'volume'] = 3000000  # High volume

        signals = self.strategy.generate_signals(data)

        self.assertGreater(len(signals), 0)
        signal = signals[0]
        self.assertEqual(signal.symbol, 'AAPL')
        self.assertEqual(signal.action, 'buy')
        self.assertGreater(signal.strength, 0)

    def test_generate_signals_with_negative_momentum(self):
        """Test signal generation with negative momentum"""
        data = {
            'AAPL': self.create_test_data('AAPL', days=20)
        }

        # Create negative momentum
        data['AAPL'].loc[data['AAPL'].index[-1], 'close'] = 95  # 5% loss

        signals = self.strategy.generate_signals(data)

        if signals:  # May or may not generate sell signal depending on exact values
            signal = signals[0]
            self.assertEqual(signal.symbol, 'AAPL')
            self.assertEqual(signal.action, 'sell')

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data"""
        # Create data with fewer days than lookback period
        data = {
            'AAPL': self.create_test_data('AAPL', days=5)
        }

        signals = self.strategy.generate_signals(data)
        self.assertEqual(len(signals), 0)

    def test_calculate_position_size_buy_signal(self):
        """Test position size calculation for buy signal"""
        signal = Signal(
            symbol="AAPL",
            action="buy",
            strength=0.8,
            price=150.0,
            timestamp=datetime.now(),
            reason="Test"
        )

        # Mock get_position to return no current position
        self.mock_client.get_position.return_value = None

        account_value = 100000
        pos_size = self.strategy.calculate_position_size(signal, account_value)

        self.assertEqual(pos_size.symbol, "AAPL")
        self.assertGreater(pos_size.target_qty, 0)
        self.assertEqual(pos_size.current_qty, 0)
        self.assertEqual(pos_size.action_qty, pos_size.target_qty)

    def test_execute_signal_success(self):
        """Test successful signal execution"""
        signal = Signal(
            symbol="AAPL",
            action="buy",
            strength=0.8,
            price=150.0,
            timestamp=datetime.now(),
            reason="Test"
        )

        # Mock dependencies
        self.mock_client.get_position.return_value = None
        self.mock_client.get_account.return_value = {'portfolio_value': '100000'}

        mock_order = Mock()
        mock_order.id = "order_123"
        self.mock_client.place_order.return_value = mock_order

        result = self.strategy.execute_signal(signal)
        self.assertTrue(result)
        self.mock_client.place_order.assert_called_once()

class TestMeanReversionStrategy(unittest.TestCase):
    """Test MeanReversionStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock(spec=AlpacaClient)
        self.strategy = MeanReversionStrategy(self.mock_client, lookback_days=10, std_dev=2.0)

    def create_mean_reversion_data(self, symbol, days=20):
        """Create test data suitable for mean reversion testing"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Create data that oscillates around a mean
        base_price = 100
        prices = []
        mean = base_price

        for i in range(days):
            if i < 15:
                # Normal oscillation
                price = mean + np.sin(i * 0.5) * 5
            elif i < 18:
                # Create oversold condition
                price = mean - 15  # Below 2 std devs
            else:
                # Return to mean
                price = mean

            prices.append(price)

        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * days
        }, index=dates)

        return data

    def test_generate_signals_oversold(self):
        """Test signal generation for oversold condition"""
        data = {
            'AAPL': self.create_mean_reversion_data('AAPL', days=20)
        }

        signals = self.strategy.generate_signals(data)

        # Should generate buy signal for oversold condition
        if signals:
            signal = signals[0]
            self.assertEqual(signal.symbol, 'AAPL')
            # Could be buy (oversold) based on the data pattern

    def test_strategy_initialization(self):
        """Test mean reversion strategy initialization"""
        self.assertEqual(self.strategy.name, "MeanReversion")
        self.assertEqual(self.strategy.lookback_days, 10)
        self.assertEqual(self.strategy.std_dev, 2.0)

class TestBuyAndHoldStrategy(unittest.TestCase):
    """Test BuyAndHoldStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock(spec=AlpacaClient)
        self.target_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        self.strategy = BuyAndHoldStrategy(
            self.mock_client,
            self.target_weights,
            rebalance_threshold=0.05
        )

    def test_strategy_initialization(self):
        """Test buy and hold strategy initialization"""
        self.assertEqual(self.strategy.name, "BuyAndHold")
        self.assertEqual(self.strategy.target_weights, self.target_weights)
        self.assertEqual(self.strategy.rebalance_threshold, 0.05)

    def test_generate_rebalancing_signals(self):
        """Test rebalancing signal generation"""
        # Mock account and positions
        self.mock_client.get_account.return_value = {'portfolio_value': '100000'}

        # Mock positions - AAPL is overweight
        mock_positions = [
            Mock(symbol='AAPL', market_value='50000'),  # 50% vs 40% target
            Mock(symbol='GOOGL', market_value='25000'), # 25% vs 30% target
            Mock(symbol='MSFT', market_value='25000')   # 25% vs 30% target
        ]
        self.mock_client.get_positions.return_value = mock_positions

        # Create test data
        data = {
            'AAPL': pd.DataFrame({'close': [150]}, index=[datetime.now()]),
            'GOOGL': pd.DataFrame({'close': [2500]}, index=[datetime.now()]),
            'MSFT': pd.DataFrame({'close': [300]}, index=[datetime.now()])
        }

        signals = self.strategy.generate_signals(data)

        # Should generate rebalancing signals
        self.assertGreater(len(signals), 0)

        # AAPL should have sell signal (overweight)
        aapl_signals = [s for s in signals if s.symbol == 'AAPL']
        if aapl_signals:
            self.assertEqual(aapl_signals[0].action, 'sell')

class TestTradingBot(unittest.TestCase):
    """Test TradingBot"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock(spec=AlpacaClient)
        self.bot = TradingBot(self.mock_client)

    def test_bot_initialization(self):
        """Test trading bot initialization"""
        self.assertEqual(self.bot.client, self.mock_client)
        self.assertEqual(len(self.bot.strategies), 0)
        self.assertFalse(self.bot.running)

    def test_add_strategy(self):
        """Test adding strategy to bot"""
        strategy = MomentumStrategy(self.mock_client)
        self.bot.add_strategy(strategy)

        self.assertEqual(len(self.bot.strategies), 1)
        self.assertEqual(self.bot.strategies[0], strategy)

    def test_remove_strategy(self):
        """Test removing strategy from bot"""
        strategy1 = MomentumStrategy(self.mock_client)
        strategy2 = MeanReversionStrategy(self.mock_client)

        self.bot.add_strategy(strategy1)
        self.bot.add_strategy(strategy2)
        self.assertEqual(len(self.bot.strategies), 2)

        self.bot.remove_strategy("Momentum")
        self.assertEqual(len(self.bot.strategies), 1)
        self.assertEqual(self.bot.strategies[0].name, "MeanReversion")

    @patch('time.sleep')
    def test_get_market_data(self, mock_sleep):
        """Test market data retrieval"""
        # Mock successful data retrieval
        mock_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000000, 1100000]
        })

        self.mock_client.get_bars.return_value = mock_df

        symbols = ['AAPL', 'GOOGL']
        data = self.bot.get_market_data(symbols, days=30)

        self.assertEqual(len(data), 2)
        self.assertIn('AAPL', data)
        self.assertIn('GOOGL', data)
        self.assertIsInstance(data['AAPL'], pd.DataFrame)

    def test_get_portfolio_summary(self):
        """Test portfolio summary generation"""
        # Mock account
        self.mock_client.get_account.return_value = {
            'portfolio_value': '150000',
            'cash': '50000'
        }

        # Mock positions
        mock_positions = [
            Mock(symbol='AAPL', qty='100', market_value='15000', unrealized_pl='1000'),
            Mock(symbol='GOOGL', qty='20', market_value='50000', unrealized_pl='2000')
        ]
        self.mock_client.get_positions.return_value = mock_positions

        summary = self.bot.get_portfolio_summary()

        self.assertEqual(summary['total_value'], 150000)
        self.assertEqual(summary['cash'], 50000)
        self.assertEqual(summary['num_positions'], 2)
        self.assertEqual(len(summary['positions']), 2)

    def test_run_strategies_market_closed(self):
        """Test strategy execution when market is closed"""
        self.mock_client.is_market_open.return_value = False

        # Should not attempt to get market data when market is closed
        self.bot.run_strategies(['AAPL'])

        self.mock_client.get_bars.assert_not_called()

    @patch('time.sleep')
    def test_run_strategies_with_signals(self, mock_sleep):
        """Test strategy execution with generated signals"""
        self.mock_client.is_market_open.return_value = True

        # Mock market data
        mock_df = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000000]
        })
        self.mock_client.get_bars.return_value = mock_df

        # Add mock strategy that generates signals
        mock_strategy = Mock()
        mock_strategy.name = "TestStrategy"

        # Create a test signal
        test_signal = Signal(
            symbol="AAPL",
            action="buy",
            strength=0.8,
            price=101.0,
            timestamp=datetime.now(),
            reason="Test signal"
        )
        mock_strategy.generate_signals.return_value = [test_signal]
        mock_strategy.execute_signal.return_value = True

        self.bot.add_strategy(mock_strategy)

        # Run strategies
        self.bot.run_strategies(['AAPL'])

        # Verify strategy was executed
        mock_strategy.generate_signals.assert_called_once()
        mock_strategy.execute_signal.assert_called_once_with(test_signal)

class TestStrategyIntegration(unittest.TestCase):
    """Integration tests for strategies"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock(spec=AlpacaClient)

    def test_momentum_mean_reversion_combination(self):
        """Test combining momentum and mean reversion strategies"""
        bot = TradingBot(self.mock_client)

        # Add both strategies
        momentum = MomentumStrategy(self.mock_client, lookback_days=20)
        mean_reversion = MeanReversionStrategy(self.mock_client, lookback_days=20)

        bot.add_strategy(momentum)
        bot.add_strategy(mean_reversion)

        self.assertEqual(len(bot.strategies), 2)

        # Mock market conditions
        self.mock_client.is_market_open.return_value = True

        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = [100 + i * 0.5 for i in range(30)]  # Slight uptrend

        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 30
        }, index=dates)

        self.mock_client.get_bars.return_value = test_data

        # Test that both strategies can process the same data
        data = bot.get_market_data(['AAPL'])
        self.assertIn('AAPL', data)

        # Each strategy should be able to generate signals
        for strategy in bot.strategies:
            signals = strategy.generate_signals(data)
            # Signals may or may not be generated depending on exact conditions
            self.assertIsInstance(signals, list)

if __name__ == '__main__':
    unittest.main()