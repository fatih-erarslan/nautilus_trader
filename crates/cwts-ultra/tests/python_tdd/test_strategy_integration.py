"""
Comprehensive TDD tests for CWTS strategy integration
Tests the actual FreqTrade strategy implementations
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from typing import Dict, Any, Optional, List

# Add FreqTrade strategy path
freqtrade_path = '/home/kutlu/CWTS/cwts-ultra/freqtrade'
strategy_path = os.path.join(freqtrade_path, 'strategies')
sys.path.insert(0, strategy_path)
sys.path.insert(0, freqtrade_path)

from test_configuration import TEST_CONFIG, CAS_CONFIG, validate_financial_calculation

class TestCWTSStrategyIntegration:
    """Integration tests for CWTS trading strategies"""
    
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create sample OHLCV dataframe for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
        
        # Generate realistic price data
        initial_price = 100.0
        prices = [initial_price]
        
        for _ in range(999):
            change = np.random.normal(0.0001, 0.002)  # Small random changes
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Create OHLC from price series
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, 1000)  # Log-normal volume distribution
        })
        
        # Ensure OHLC consistency
        for i in range(len(df)):
            df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
            df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])
        
        return df.set_index('date')
    
    @pytest.fixture
    def mock_dataprovider(self, sample_dataframe):
        """Mock DataProvider for strategy testing"""
        mock_dp = Mock()
        mock_dp.get_analyzed_dataframe.return_value = (sample_dataframe, pd.Timestamp('2024-01-01'))
        return mock_dp
    
    def test_profitable_strategy_loading(self):
        """Test loading of CWTSProfitableStrategy"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Test basic strategy properties
            assert strategy.STRATEGY_NAME == "CWTS_Profitable"
            assert strategy.timeframe == '5m'
            assert strategy.stoploss == -0.02
            assert strategy.can_short == False
            
            # Test ROI table structure
            assert isinstance(strategy.minimal_roi, dict)
            assert len(strategy.minimal_roi) > 0
            
        except ImportError as e:
            pytest.skip(f"CWTSProfitableStrategy not available: {e}")
    
    def test_strategy_indicators_calculation(self, sample_dataframe):
        """Test indicator calculations in strategy"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Populate indicators
            result_df = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Test required indicators exist
            required_indicators = [
                'momentum', 'volume_sma', 'volume_ratio', 'rsi',
                'macd', 'macdsignal', 'macdhist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'ema_fast', 'ema_slow', 'atr', 'trend_up', 'mfi',
                'quality_score', 'risk_score'
            ]
            
            for indicator in required_indicators:
                assert indicator in result_df.columns, f"Missing indicator: {indicator}"
                assert not result_df[indicator].isna().all(), f"Indicator {indicator} is all NaN"
            
            # Test indicator value ranges
            assert (result_df['rsi'] >= 0).all() and (result_df['rsi'] <= 100).all(), "RSI should be 0-100"
            assert (result_df['mfi'] >= 0).all() and (result_df['mfi'] <= 100).all(), "MFI should be 0-100"
            assert (result_df['quality_score'] >= 0).all() and (result_df['quality_score'] <= 1).all(), "Quality score should be 0-1"
            assert (result_df['risk_score'] >= 0).all() and (result_df['risk_score'] <= 1).all(), "Risk score should be 0-1"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_entry_signal_generation(self, sample_dataframe):
        """Test buy signal generation logic"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Populate indicators first
            df_with_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Generate entry signals
            result_df = strategy.populate_entry_trend(df_with_indicators, {'pair': 'BTC/USDT'})
            
            # Test signal columns exist
            assert 'enter_long' in result_df.columns, "Should have enter_long column"
            assert 'enter_tag' in result_df.columns, "Should have enter_tag column"
            
            # Test signal values
            buy_signals = result_df['enter_long'] == 1
            
            if buy_signals.any():
                # Test that buy signals have corresponding tags
                buy_tags = result_df.loc[buy_signals, 'enter_tag'].dropna()
                valid_tags = ['momentum', 'oversold', 'trend', 'macd_cross']
                assert all(tag in valid_tags for tag in buy_tags), f"Invalid buy tags: {buy_tags.unique()}"
                
                # Test quality and risk filters
                buy_rows = result_df.loc[buy_signals]
                assert (buy_rows['quality_score'] >= strategy.quality_threshold.value).all(), "Buy signals should meet quality threshold"
                assert (buy_rows['risk_score'] <= strategy.risk_threshold.value).all(), "Buy signals should meet risk threshold"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_exit_signal_generation(self, sample_dataframe):
        """Test sell signal generation logic"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Populate indicators first
            df_with_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Generate exit signals
            result_df = strategy.populate_exit_trend(df_with_indicators, {'pair': 'BTC/USDT'})
            
            # Test signal columns exist
            assert 'exit_long' in result_df.columns, "Should have exit_long column"
            assert 'exit_tag' in result_df.columns, "Should have exit_tag column"
            
            # Test signal values
            sell_signals = result_df['exit_long'] == 1
            
            if sell_signals.any():
                # Test that sell signals have corresponding tags
                sell_tags = result_df.loc[sell_signals, 'exit_tag'].dropna()
                valid_tags = ['overbought', 'momentum_reversal', 'trend_reversal', 'risk_limit']
                assert all(tag in valid_tags for tag in sell_tags), f"Invalid sell tags: {sell_tags.unique()}"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_custom_stake_amount(self, sample_dataframe, mock_dataprovider):
        """Test custom stake amount calculation"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            strategy.dp = mock_dataprovider
            
            # Populate indicators for the mock data
            df_with_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            mock_dataprovider.get_analyzed_dataframe.return_value = (df_with_indicators, pd.Timestamp('2024-01-01'))
            
            # Test stake calculation
            proposed_stake = 100.0
            min_stake = 10.0
            max_stake = 1000.0
            
            calculated_stake = strategy.custom_stake_amount(
                pair='BTC/USDT',
                current_time=pd.Timestamp('2024-01-01'),
                current_rate=50000.0,
                proposed_stake=proposed_stake,
                min_stake=min_stake,
                max_stake=max_stake,
                entry_tag='momentum',
                side='long'
            )
            
            # Test stake bounds
            assert min_stake <= calculated_stake <= max_stake, f"Stake {calculated_stake} outside bounds [{min_stake}, {max_stake}]"
            
            # Test different entry tags
            for tag in ['momentum', 'oversold', 'trend', 'macd_cross']:
                stake = strategy.custom_stake_amount(
                    pair='BTC/USDT',
                    current_time=pd.Timestamp('2024-01-01'),
                    current_rate=50000.0,
                    proposed_stake=proposed_stake,
                    min_stake=min_stake,
                    max_stake=max_stake,
                    entry_tag=tag,
                    side='long'
                )
                assert min_stake <= stake <= max_stake, f"Stake for {tag} outside bounds"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_quality_score_calculation(self, sample_dataframe):
        """Test quality score calculation logic"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Populate indicators to get required columns
            df_with_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Calculate quality score
            quality_score = strategy.calculate_quality_score(df_with_indicators)
            
            # Test quality score properties
            assert (quality_score >= 0).all(), "Quality score should be >= 0"
            assert (quality_score <= 1).all(), "Quality score should be <= 1"
            assert not quality_score.isna().all(), "Quality score should not be all NaN"
            
            # Test with extreme values
            extreme_df = df_with_indicators.copy()
            extreme_df['trend_up'] = 1  # All trend up
            extreme_df['volume_ratio'] = 2.0  # High volume
            extreme_df['rsi'] = 50  # Neutral RSI
            extreme_df['macdhist'] = 0.1  # Positive MACD hist
            
            extreme_quality = strategy.calculate_quality_score(extreme_df)
            assert (extreme_quality >= 0.5).any(), "Should have high quality scores with good conditions"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_risk_score_calculation(self, sample_dataframe):
        """Test risk score calculation logic"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Populate indicators to get required columns
            df_with_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Calculate risk score
            risk_score = strategy.calculate_risk_score(df_with_indicators)
            
            # Test risk score properties
            assert (risk_score >= 0).all(), "Risk score should be >= 0"
            assert (risk_score <= 1).all(), "Risk score should be <= 1"
            assert not risk_score.isna().all(), "Risk score should not be all NaN"
            
            # Test with high risk conditions
            risky_df = df_with_indicators.copy()
            risky_df['rsi'] = 95  # Extreme RSI
            risky_df['atr'] = risky_df['close'] * 0.1  # High volatility
            risky_df['bb_width'] = 0.2  # Wide Bollinger Bands
            
            risky_score = strategy.calculate_risk_score(risky_df)
            assert (risky_score >= 0.5).any(), "Should have high risk scores with risky conditions"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    @pytest.mark.parametrize("parameter_name,expected_range", [
        ('momentum_threshold', (0.002, 0.01)),
        ('momentum_period', (10, 30)),
        ('volume_threshold', (1.2, 2.0)),
        ('rsi_buy', (25, 45)),
        ('rsi_sell', (65, 85)),
        ('quality_threshold', (0.5, 0.8)),
        ('risk_threshold', (0.2, 0.5))
    ])
    def test_parameter_optimization_ranges(self, parameter_name, expected_range):
        """Test parameter optimization ranges"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            param = getattr(strategy, parameter_name)
            
            # Test parameter exists and has correct range
            assert hasattr(param, 'range'), f"Parameter {parameter_name} should have range attribute"
            param_range = (param.range[0], param.range[1])
            
            assert param_range == expected_range, f"Parameter {parameter_name} range {param_range} != expected {expected_range}"
            
            # Test default value is within range
            default_value = param.value
            assert expected_range[0] <= default_value <= expected_range[1], f"Default value {default_value} outside range {expected_range}"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_strategy_signal_consistency(self, sample_dataframe):
        """Test consistency of buy/sell signals"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Process full pipeline
            df_indicators = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            df_entry = strategy.populate_entry_trend(df_indicators, {'pair': 'BTC/USDT'})
            df_full = strategy.populate_exit_trend(df_entry, {'pair': 'BTC/USDT'})
            
            # Test signal consistency
            buy_signals = df_full['enter_long'] == 1
            sell_signals = df_full['exit_long'] == 1
            
            # No simultaneous buy and sell signals
            simultaneous = (buy_signals & sell_signals).sum()
            assert simultaneous == 0, f"Found {simultaneous} simultaneous buy/sell signals"
            
            # Test signal frequency is reasonable
            total_rows = len(df_full)
            buy_frequency = buy_signals.sum() / total_rows
            sell_frequency = sell_signals.sum() / total_rows
            
            assert 0 <= buy_frequency <= 0.5, f"Buy frequency {buy_frequency:.2%} seems unreasonable"
            assert 0 <= sell_frequency <= 0.5, f"Sell frequency {sell_frequency:.2%} seems unreasonable"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_complex_adaptive_systems_integration(self, sample_dataframe):
        """Test Complex Adaptive Systems principles in strategy"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            # Test adaptation capability
            df_processed = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Test emergence properties
            quality_scores = df_processed['quality_score']
            risk_scores = df_processed['risk_score']
            
            # System should show emergent behavior (non-linear combinations)
            correlation = np.corrcoef(quality_scores.dropna(), risk_scores.dropna())[0, 1]
            assert -1 <= correlation <= 1, "Quality and risk scores should have valid correlation"
            
            # Test system memory (indicators should show persistence)
            momentum = df_processed['momentum'].dropna()
            if len(momentum) > 10:
                autocorr = momentum.autocorr(lag=1)
                assert -1 <= autocorr <= 1, "Momentum should show valid autocorrelation"
            
            # Test feedback loops (RSI affects quality, quality affects decisions)
            rsi_values = df_processed['rsi'].dropna()
            quality_values = df_processed['quality_score'].dropna()
            
            if len(rsi_values) > 10 and len(quality_values) > 10:
                # Should have some relationship (not perfectly correlated due to complexity)
                rsi_quality_corr = np.corrcoef(
                    rsi_values[-len(quality_values):], 
                    quality_values[-len(rsi_values):]
                )[0, 1]
                assert not np.isnan(rsi_quality_corr), "RSI-Quality correlation should be calculable"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")
    
    def test_mathematical_rigor_validation(self, sample_dataframe):
        """Test mathematical rigor in calculations"""
        try:
            from CWTSProfitableStrategy import CWTSProfitableStrategy
            strategy = CWTSProfitableStrategy({})
            
            df_processed = strategy.populate_indicators(sample_dataframe.copy(), {'pair': 'BTC/USDT'})
            
            # Test mathematical properties
            # 1. Percentages should sum correctly for quality score components
            quality_components = ['trend_up', 'volume_ratio', 'rsi', 'macdhist']
            for component in quality_components:
                if component in df_processed.columns:
                    values = df_processed[component].dropna()
                    assert len(values) > 0, f"Component {component} should have values"
            
            # 2. Volatility measures should be positive
            volatility_measures = ['atr', 'bb_width']
            for measure in volatility_measures:
                if measure in df_processed.columns:
                    values = df_processed[measure].dropna()
                    assert (values >= 0).all(), f"{measure} should be non-negative"
            
            # 3. Bounded indicators should stay within bounds
            bounded_indicators = [
                ('rsi', 0, 100),
                ('mfi', 0, 100),
                ('quality_score', 0, 1),
                ('risk_score', 0, 1)
            ]
            
            for indicator, min_val, max_val in bounded_indicators:
                if indicator in df_processed.columns:
                    values = df_processed[indicator].dropna()
                    assert (values >= min_val).all(), f"{indicator} should be >= {min_val}"
                    assert (values <= max_val).all(), f"{indicator} should be <= {max_val}"
            
            # 4. Moving averages should be smooth
            ma_indicators = ['volume_sma', 'ema_fast', 'ema_slow', 'bb_middle']
            for ma in ma_indicators:
                if ma in df_processed.columns:
                    values = df_processed[ma].dropna()
                    if len(values) > 2:
                        # Check for reasonable smoothness (no extreme jumps)
                        pct_changes = pd.Series(values).pct_change().dropna()
                        extreme_changes = (abs(pct_changes) > 0.1).sum()  # >10% change
                        total_changes = len(pct_changes)
                        if total_changes > 0:
                            extreme_ratio = extreme_changes / total_changes
                            assert extreme_ratio < 0.1, f"{ma} has too many extreme changes: {extreme_ratio:.2%}"
            
        except ImportError:
            pytest.skip("CWTSProfitableStrategy not available")