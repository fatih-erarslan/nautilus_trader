"""
Comprehensive test suite for Crypto Momentum Strategy
Tests fee optimization, signal generation, and position management
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.crypto_momentum_strategy import (
    CryptoMomentumStrategy,
    FeeStructure,
    CryptoSignal,
    MarketRegime,
    SignalStrength
)


class TestFeeStructure(unittest.TestCase):
    """Test fee calculation logic"""
    
    def test_default_fees(self):
        """Test default fee structure"""
        fee_struct = FeeStructure()
        self.assertEqual(fee_struct.maker_fee, 0.001)
        self.assertEqual(fee_struct.taker_fee, 0.001)
        self.assertEqual(fee_struct.round_trip_fee, 0.002)
    
    def test_fee_token_discount(self):
        """Test fee token discount calculation"""
        fee_struct = FeeStructure(
            maker_fee=0.001,
            taker_fee=0.001,
            has_fee_token=True,
            fee_token_discount=0.25
        )
        self.assertEqual(fee_struct.effective_maker_fee, 0.00075)
        self.assertEqual(fee_struct.effective_taker_fee, 0.00075)
        self.assertEqual(fee_struct.round_trip_fee, 0.0015)
    
    def test_vip_tier_fees(self):
        """Test VIP tier fee structures"""
        fee_struct = FeeStructure(
            maker_fee=0.0002,  # VIP tier
            taker_fee=0.0004,
            vip_tier=3
        )
        self.assertEqual(fee_struct.round_trip_fee, 0.0008)


class TestCryptoMomentumStrategy(unittest.TestCase):
    """Test main strategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = CryptoMomentumStrategy(
            min_move_threshold=0.015,
            confidence_threshold=0.75,
            fee_structure=FeeStructure(),
            max_position_pct=0.1
        )
        
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        self.price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 46000, 100),
            'high': np.random.uniform(46000, 47000, 100),
            'low': np.random.uniform(44000, 45000, 100),
            'close': np.random.uniform(45000, 46000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Add trend to close prices
        trend = np.linspace(45000, 46500, 100) + np.random.normal(0, 100, 100)
        self.price_data['close'] = trend
        
        self.volume_data = pd.DataFrame({
            'timestamp': dates,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
    
    def test_fee_efficiency_calculation(self):
        """Test fee efficiency ratio calculation"""
        # Test profitable trade
        efficiency = self.strategy.calculate_fee_efficiency(
            expected_move=0.02,  # 2% move
            position_size=1.0,   # 1 BTC
            entry_price=50000
        )
        # Expected: (1 * 50000 * 0.02) / (1 * 50000 * 0.002) = 10
        self.assertAlmostEqual(efficiency, 10.0, places=2)
        
        # Test marginal trade
        efficiency = self.strategy.calculate_fee_efficiency(
            expected_move=0.005,  # 0.5% move
            position_size=1.0,
            entry_price=50000
        )
        # Expected: (1 * 50000 * 0.005) / (1 * 50000 * 0.002) = 2.5
        self.assertAlmostEqual(efficiency, 2.5, places=2)
    
    def test_market_regime_detection(self):
        """Test market volatility regime detection"""
        # Low volatility data
        low_vol_data = self.price_data.copy()
        low_vol_data['high'] = low_vol_data['close'] * 1.002
        low_vol_data['low'] = low_vol_data['close'] * 0.998
        
        regime = self.strategy.detect_market_regime(low_vol_data)
        self.assertEqual(regime, MarketRegime.LOW_VOL)
        
        # High volatility data
        high_vol_data = self.price_data.copy()
        high_vol_data['high'] = high_vol_data['close'] * 1.05
        high_vol_data['low'] = high_vol_data['close'] * 0.95
        
        regime = self.strategy.detect_market_regime(high_vol_data)
        self.assertIn(regime, [MarketRegime.HIGH_VOL, MarketRegime.EXTREME_VOL])
    
    def test_momentum_score_calculation(self):
        """Test momentum indicator calculations"""
        momentum_score, components = self.strategy.calculate_momentum_score(
            self.price_data,
            self.volume_data
        )
        
        # Check components exist
        self.assertIn('roc_5', components)
        self.assertIn('roc_10', components)
        self.assertIn('roc_20', components)
        self.assertIn('rsi', components)
        self.assertIn('macd_histogram', components)
        self.assertIn('volume_score', components)
        
        # Check momentum score is reasonable
        self.assertIsInstance(momentum_score, float)
        self.assertLess(abs(momentum_score), 1.0)  # Should be normalized
    
    def test_neural_integration(self):
        """Test integration with neural predictions"""
        base_signal = 0.02  # 2% momentum signal
        neural_forecast = {
            'predicted_return': 0.025,  # 2.5% neural prediction
            'confidence': 0.85
        }
        
        integrated = self.strategy.integrate_neural_prediction(
            base_signal,
            neural_forecast
        )
        
        # Check weighted combination
        expected = (0.025 * 0.85 * 0.4) + (0.02 * 0.6)
        self.assertAlmostEqual(integrated, expected, places=4)
    
    def test_position_sizing_kelly(self):
        """Test Kelly Criterion position sizing"""
        position_size = self.strategy.calculate_position_size(
            signal_strength=0.8,
            expected_move=0.03,  # 3% move
            portfolio_value=100000,
            current_positions=0
        )
        
        # Should be positive and within limits
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 100000 * 0.1)  # Max 10% of portfolio
        
        # Test with multiple positions (should reduce size)
        position_size_multi = self.strategy.calculate_position_size(
            signal_strength=0.8,
            expected_move=0.03,
            portfolio_value=100000,
            current_positions=3
        )
        
        self.assertLess(position_size_multi, position_size)
    
    def test_signal_generation(self):
        """Test complete signal generation"""
        # Add strong trend to data
        self.price_data['close'] = np.linspace(45000, 47000, 100)
        
        neural_forecast = {
            'predicted_return': 0.025,
            'confidence': 0.85,
            'horizon': 4
        }
        
        signal = self.strategy.generate_signal(
            symbol='BTC/USDT',
            price_data=self.price_data,
            volume_data=self.volume_data,
            neural_forecast=neural_forecast,
            portfolio_value=100000
        )
        
        if signal:  # Signal may be None if conditions not met
            self.assertIsInstance(signal, CryptoSignal)
            self.assertEqual(signal.symbol, 'BTC/USDT')
            self.assertIn(signal.direction, ['long', 'short'])
            self.assertGreater(signal.fee_efficiency_ratio, 7)
            self.assertGreater(signal.confidence, 0.75)
    
    def test_signal_rejection_small_move(self):
        """Test that small moves are rejected"""
        # Create flat price data (no momentum)
        self.price_data['close'] = 45000  # Flat prices
        
        neural_forecast = {
            'predicted_return': 0.005,  # Only 0.5% move
            'confidence': 0.9
        }
        
        signal = self.strategy.generate_signal(
            symbol='BTC/USDT',
            price_data=self.price_data,
            volume_data=self.volume_data,
            neural_forecast=neural_forecast,
            portfolio_value=100000
        )
        
        self.assertIsNone(signal)  # Should reject due to small move
    
    def test_pyramiding_logic(self):
        """Test position pyramiding logic"""
        # Create initial position
        signal = CryptoSignal(
            symbol='BTC/USDT',
            direction='long',
            entry_price=45000,
            predicted_move=0.03,
            confidence=0.85,
            signal_strength=SignalStrength.STRONG,
            stop_loss=44100,
            take_profit=46350,
            position_size=10000,
            fee_efficiency_ratio=10,
            expected_holding_hours=4
        )
        
        self.strategy.positions['BTC/USDT'] = signal
        
        # Test pyramiding on profit
        should_pyramid = self.strategy.should_pyramid(
            'BTC/USDT',
            45500,  # 1.1% profit
            signal
        )
        
        self.assertTrue(should_pyramid)
        
        # Test no pyramiding on loss
        should_pyramid = self.strategy.should_pyramid(
            'BTC/USDT',
            44500,  # Loss
            signal
        )
        
        self.assertFalse(should_pyramid)
    
    def test_position_tracking(self):
        """Test position tracking and updates"""
        signal = CryptoSignal(
            symbol='BTC/USDT',
            direction='long',
            entry_price=45000,
            predicted_move=0.02,
            confidence=0.8,
            signal_strength=SignalStrength.MODERATE,
            stop_loss=44100,
            take_profit=45900,
            position_size=10000,
            fee_efficiency_ratio=8,
            expected_holding_hours=6
        )
        
        self.strategy.update_position_tracking(signal, 45000)
        
        self.assertIn('BTC/USDT', self.strategy.positions)
        self.assertEqual(self.strategy.trades_executed, 1)
        self.assertGreater(self.strategy.total_fees_paid, 0)
    
    def test_position_closing(self):
        """Test position closing and P&L calculation"""
        # Set up position
        signal = CryptoSignal(
            symbol='BTC/USDT',
            direction='long',
            entry_price=45000,
            predicted_move=0.02,
            confidence=0.8,
            signal_strength=SignalStrength.MODERATE,
            stop_loss=44100,
            take_profit=45900,
            position_size=10000,
            fee_efficiency_ratio=8,
            expected_holding_hours=6
        )
        
        self.strategy.positions['BTC/USDT'] = signal
        
        # Close with profit
        result = self.strategy.close_position(
            'BTC/USDT',
            45900,  # 2% profit
            'target_reached'
        )
        
        self.assertIn('net_pnl', result)
        self.assertIn('fees', result)
        self.assertGreater(result['gross_pnl'], 0)
        self.assertGreater(result['fees'], 0)
        self.assertNotIn('BTC/USDT', self.strategy.positions)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute some trades
        self.strategy.trades_executed = 10
        self.strategy.winning_trades = 6
        self.strategy.total_gross_profit = 5000
        self.strategy.total_fees_paid = 400
        
        metrics = self.strategy.get_performance_metrics()
        
        self.assertEqual(metrics['total_trades'], 10)
        self.assertEqual(metrics['win_rate'], 0.6)
        self.assertEqual(metrics['net_profit'], 4600)
        self.assertEqual(metrics['fee_efficiency_ratio'], 12.5)
        self.assertAlmostEqual(metrics['fee_drag_pct'], 0.08, places=2)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete trading scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.strategy = CryptoMomentumStrategy(
            min_move_threshold=0.015,
            confidence_threshold=0.7,
            fee_structure=FeeStructure(has_fee_token=True),
            use_pyramiding=True
        )
    
    def test_bull_market_scenario(self):
        """Test strategy in bull market conditions"""
        # Create trending price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        prices = np.linspace(40000, 45000, 100) + np.random.normal(0, 200, 100)
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        volume_data = pd.DataFrame({
            'timestamp': dates,
            'volume': np.random.uniform(2000000, 5000000, 100)
        })
        
        # Strong bullish neural forecast
        neural_forecast = {
            'predicted_return': 0.035,  # 3.5% up
            'confidence': 0.88,
            'horizon': 6
        }
        
        signal = self.strategy.generate_signal(
            'BTC/USDT',
            price_data,
            volume_data,
            neural_forecast,
            100000
        )
        
        if signal:
            self.assertEqual(signal.direction, 'long')
            self.assertGreater(signal.predicted_move, 0.015)
            self.assertGreater(signal.fee_efficiency_ratio, 7)
    
    def test_high_volatility_scenario(self):
        """Test strategy in high volatility conditions"""
        # Create volatile price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        base_price = 45000
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(base_price * 0.95, base_price * 1.05, 100),
            'high': np.random.uniform(base_price * 1.02, base_price * 1.08, 100),
            'low': np.random.uniform(base_price * 0.92, base_price * 0.98, 100),
            'close': np.random.uniform(base_price * 0.95, base_price * 1.05, 100),
            'volume': np.random.uniform(100, 2000, 100)
        })
        
        regime = self.strategy.detect_market_regime(price_data)
        self.assertIn(regime, [MarketRegime.HIGH_VOL, MarketRegime.EXTREME_VOL])
        
        # In high volatility, strategy should be more selective
        signal = self.strategy.generate_signal(
            'BTC/USDT',
            price_data,
            None,
            {'predicted_return': 0.02, 'confidence': 0.7},
            100000
        )
        
        # May or may not generate signal depending on exact conditions
        if signal:
            self.assertGreater(signal.confidence, 0.7)


class TestMCPIntegration(unittest.TestCase):
    """Test integration with MCP tools"""
    
    @patch('src.strategies.crypto_momentum_strategy.logger')
    def test_mcp_neural_forecast_integration(self, mock_logger):
        """Test integration with MCP neural forecast"""
        strategy = CryptoMomentumStrategy()
        
        # Simulate MCP neural forecast response
        mcp_forecast = {
            'predicted_return': 0.0285,
            'confidence': 0.92,
            'horizon': 4,
            'model_id': 'neural_momentum_v2',
            'features_used': 50
        }
        
        # Create price data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1H')
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 46000, 50),
            'high': np.random.uniform(46000, 47000, 50),
            'low': np.random.uniform(44000, 45000, 50),
            'close': np.linspace(45000, 46000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        })
        
        signal = strategy.generate_signal(
            'BTC/USDT',
            price_data,
            None,
            mcp_forecast,
            100000
        )
        
        if signal:
            # Verify neural forecast was used
            self.assertIn('neural_confidence', signal.metadata)
            self.assertEqual(signal.metadata['neural_confidence'], 0.92)


if __name__ == '__main__':
    unittest.main()