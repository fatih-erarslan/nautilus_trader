"""Tests for technical indicators calculation"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestTechnicalIndicators:
    """Test suite for technical indicators calculations"""
    
    def test_simple_moving_average(self):
        """Test calculation of simple moving averages"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Test data
        prices = [100, 102, 101, 103, 105, 104, 106, 107, 105, 108]
        
        # Test SMA calculation
        sma_5 = indicators.calculate_sma(prices, period=5)
        
        # Check length (10 prices - 4 for SMA calculation = 6 values)
        assert len(sma_5) == 6
        
        # Check first SMA value (average of first 5 prices)
        expected_first = (100 + 102 + 101 + 103 + 105) / 5
        assert sma_5[0] == pytest.approx(expected_first, 0.01)
        
        # Check last SMA value
        expected_last = (104 + 106 + 107 + 105 + 108) / 5
        assert sma_5[-1] == pytest.approx(expected_last, 0.01)
        
    def test_exponential_moving_average(self):
        """Test calculation of exponential moving averages"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        prices = list(range(100, 120))  # Uptrending prices
        
        ema_10 = indicators.calculate_ema(prices, period=10)
        
        # EMA should have same length as prices
        assert len(ema_10) == len(prices)
        
        # EMA should be closer to recent prices than SMA
        sma_10 = indicators.calculate_sma(prices, period=10)
        assert ema_10[-1] > sma_10[-1]  # In uptrend, EMA > SMA
        
    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Test with known pattern
        # Overbought scenario
        prices_up = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
        rsi_up = indicators.calculate_rsi(prices_up, period=14)
        assert rsi_up > 70  # Should be overbought
        
        # Oversold scenario
        prices_down = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72]
        rsi_down = indicators.calculate_rsi(prices_down, period=14)
        assert rsi_down < 30  # Should be oversold
        
        # RSI should be between 0 and 100
        assert 0 <= rsi_up <= 100
        assert 0 <= rsi_down <= 100
        
    def test_macd_calculation(self):
        """Test MACD (Moving Average Convergence Divergence) calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Generate trending data
        prices = [100 + i + np.sin(i/5) * 2 for i in range(50)]
        
        macd_line, signal_line, histogram = indicators.calculate_macd(
            prices, 
            fast_period=12, 
            slow_period=26, 
            signal_period=9
        )
        
        # Check outputs
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None
        
        # MACD should be shorter than original prices
        assert len(macd_line) < len(prices)
        assert len(signal_line) < len(macd_line)
        assert len(histogram) == len(signal_line)
        
        # Histogram should be difference between MACD and Signal
        for i in range(len(histogram)):
            expected = macd_line[-(len(histogram)-i)] - signal_line[i]
            assert histogram[i] == pytest.approx(expected, 0.001)
            
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107]
        
        upper_band, middle_band, lower_band = indicators.calculate_bollinger_bands(
            prices, period=10, std_dev=2
        )
        
        # Check lengths
        assert len(upper_band) == len(middle_band) == len(lower_band)
        assert len(upper_band) == 6  # 15 prices - 9 for calculation
        
        # Check relationships
        for i in range(len(upper_band)):
            assert upper_band[i] > middle_band[i]
            assert middle_band[i] > lower_band[i]
            
        # Middle band should be SMA
        sma = indicators.calculate_sma(prices, period=10)
        np.testing.assert_array_almost_equal(middle_band, sma)
        
    def test_volume_indicators(self):
        """Test volume-based indicators"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Mock price and volume data
        data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105, 104, 106, 107, 105, 108],
            'Volume': [1000000, 1200000, 800000, 1500000, 2000000, 
                      900000, 1100000, 1300000, 950000, 1400000]
        })
        
        # Test Volume Weighted Average Price (VWAP)
        vwap = indicators.calculate_vwap(data)
        assert len(vwap) == len(data)
        assert all(vwap > 0)
        
        # Test On-Balance Volume (OBV)
        obv = indicators.calculate_obv(data)
        assert len(obv) == len(data)
        
        # OBV should increase when price goes up with volume
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                assert obv[i] > obv[i-1]
                
    def test_atr_calculation(self):
        """Test Average True Range calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Mock OHLC data
        data = pd.DataFrame({
            'High': [102, 104, 103, 105, 107, 106, 108, 109, 107, 110],
            'Low': [98, 100, 99, 101, 103, 102, 104, 105, 103, 106],
            'Close': [100, 102, 101, 103, 105, 104, 106, 107, 105, 108]
        })
        
        atr = indicators.calculate_atr(data, period=5)
        
        # ATR should be positive
        assert all(atr > 0)
        
        # ATR length should be data length - period + 1
        assert len(atr) == len(data) - 5 + 1
        
    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Mock OHLC data
        data = pd.DataFrame({
            'High': [102, 104, 103, 105, 107, 106, 108, 109, 107, 110, 112, 111, 113, 115, 114],
            'Low': [98, 100, 99, 101, 103, 102, 104, 105, 103, 106, 108, 107, 109, 111, 110],
            'Close': [100, 102, 101, 103, 105, 104, 106, 107, 105, 108, 110, 109, 111, 113, 112]
        })
        
        k_percent, d_percent = indicators.calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3)
        
        # Values should be between 0 and 100
        assert all(0 <= k <= 100 for k in k_percent)
        assert all(0 <= d <= 100 for d in d_percent)
        
        # D% should be smoother than K%
        assert np.std(d_percent) < np.std(k_percent)
        
    def test_fibonacci_levels(self):
        """Test Fibonacci retracement level calculation"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Test with known high and low
        high = 150
        low = 100
        
        levels = indicators.calculate_fibonacci_levels(high, low)
        
        # Check standard Fibonacci levels
        assert levels['0.0%'] == low
        assert levels['23.6%'] == pytest.approx(111.8, 0.1)
        assert levels['38.2%'] == pytest.approx(119.1, 0.1)
        assert levels['50.0%'] == pytest.approx(125.0, 0.1)
        assert levels['61.8%'] == pytest.approx(130.9, 0.1)
        assert levels['100.0%'] == high