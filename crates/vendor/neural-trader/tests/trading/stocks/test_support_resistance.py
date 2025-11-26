"""Tests for support and resistance level detection"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestSupportResistanceDetection:
    """Test suite for support and resistance level detection"""
    
    def test_basic_support_resistance_detection(self):
        """Test detection of clear support and resistance levels"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Price data with clear levels
        # Support around 98, resistance around 102-103
        prices = [100, 98, 99, 98, 102, 103, 102, 98, 99, 98, 104, 105, 
                 103, 102, 98, 99, 98, 103, 102, 103]
        
        levels = detector.detect_levels(prices)
        
        # Check structure
        assert "support" in levels
        assert "resistance" in levels
        assert "strength" in levels
        
        # Should detect support around 98
        assert any(abs(level - 98) < 0.5 for level in levels["support"])
        
        # Should detect resistance around 102-103
        assert any(102 <= level <= 103 for level in levels["resistance"])
        
        # Check strength ratings
        for price_level in levels["strength"]:
            assert 0 <= levels["strength"][price_level] <= 1
            
    def test_dynamic_level_detection(self):
        """Test detection of levels with price/volume data"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Create DataFrame with price and volume
        data = pd.DataFrame({
            'High': [101, 99, 100, 99, 103, 104, 103, 99, 100, 99, 105, 106,
                    104, 103, 99, 100, 99, 104, 103, 104],
            'Low': [99, 97, 98, 97, 101, 102, 101, 97, 98, 97, 103, 104,
                   102, 101, 97, 98, 97, 102, 101, 102],
            'Close': [100, 98, 99, 98, 102, 103, 102, 98, 99, 98, 104, 105,
                     103, 102, 98, 99, 98, 103, 102, 103],
            'Volume': [1000000, 1500000, 900000, 2000000, 1200000, 1100000,
                      1300000, 1800000, 1000000, 1900000, 1400000, 1600000,
                      1200000, 1300000, 2100000, 1000000, 2200000, 1500000,
                      1400000, 1300000]
        })
        
        levels = detector.detect_levels_with_volume(data)
        
        # Volume-weighted levels should be detected
        assert len(levels["support"]) > 0
        assert len(levels["resistance"]) > 0
        
        # High volume areas should have stronger levels
        # 98 had high volume (2000000+), should be strong support
        support_98 = next((s for s in levels["support"] if abs(s - 98) < 0.5), None)
        assert support_98 is not None
        assert levels["strength"][support_98] > 0.7
        
    def test_trend_line_detection(self):
        """Test detection of trend lines as dynamic support/resistance"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Create uptrending data with clear trend line
        base = 100
        prices = []
        for i in range(20):
            # Uptrend with noise
            price = base + i * 0.5 + np.sin(i) * 0.3
            prices.append(price)
            
        trend_lines = detector.detect_trend_lines(prices)
        
        assert "support_line" in trend_lines
        assert "resistance_line" in trend_lines
        
        # Support line should have positive slope in uptrend
        assert trend_lines["support_line"]["slope"] > 0
        
        # Check that trend line equation works
        support_line = trend_lines["support_line"]
        y_start = support_line["intercept"]
        y_end = support_line["intercept"] + support_line["slope"] * (len(prices) - 1)
        assert y_end > y_start  # Uptrending
        
    def test_breakout_detection(self):
        """Test detection of support/resistance breakouts"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Price data with breakout scenario
        # Resistance at 105, then breakout
        prices = [100, 102, 104, 105, 104, 103, 104, 105, 104, 105,  # Testing resistance
                 106, 107, 108, 109, 110]  # Breakout
        
        analysis = detector.analyze_breakout(prices, timeframe=10)
        
        assert analysis["breakout_detected"] == True
        assert analysis["breakout_type"] == "resistance"
        assert analysis["breakout_level"] == pytest.approx(105, 0.5)
        assert analysis["breakout_strength"] > 0.7  # Strong breakout
        
    def test_pivot_point_calculation(self):
        """Test calculation of pivot points"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Yesterday's OHLC data
        high = 110
        low = 100
        close = 105
        
        pivots = detector.calculate_pivot_points(high, low, close)
        
        # Check standard pivot calculation
        expected_pivot = (high + low + close) / 3
        assert pivots["pivot"] == pytest.approx(expected_pivot, 0.01)
        
        # Check support levels
        assert pivots["s1"] < pivots["pivot"]
        assert pivots["s2"] < pivots["s1"]
        assert pivots["s3"] < pivots["s2"]
        
        # Check resistance levels
        assert pivots["r1"] > pivots["pivot"]
        assert pivots["r2"] > pivots["r1"]
        assert pivots["r3"] > pivots["r2"]
        
    def test_volume_profile_levels(self):
        """Test volume profile based support/resistance"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Create data with high volume nodes
        data = []
        for i in range(100):
            price = 100 + np.sin(i/10) * 5
            # High volume at certain price levels
            if 98 <= price <= 99 or 104 <= price <= 105:
                volume = np.random.randint(2000000, 3000000)
            else:
                volume = np.random.randint(500000, 1000000)
            
            data.append({'price': price, 'volume': volume})
            
        df = pd.DataFrame(data)
        
        volume_levels = detector.calculate_volume_profile(df)
        
        # Should identify high volume nodes as support/resistance
        assert len(volume_levels["high_volume_nodes"]) > 0
        
        # Check that high volume areas are identified
        high_volume_prices = [node["price"] for node in volume_levels["high_volume_nodes"]]
        assert any(98 <= p <= 99 for p in high_volume_prices)
        assert any(104 <= p <= 105 for p in high_volume_prices)
        
    def test_multi_timeframe_levels(self):
        """Test support/resistance across multiple timeframes"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Mock data for different timeframes
        hourly_data = pd.DataFrame({
            'High': np.random.uniform(98, 102, 24),
            'Low': np.random.uniform(96, 100, 24),
            'Close': np.random.uniform(97, 101, 24)
        })
        
        daily_data = pd.DataFrame({
            'High': [102, 105, 103, 106, 104],
            'Low': [98, 101, 99, 102, 100],
            'Close': [100, 103, 101, 104, 102]
        })
        
        weekly_data = pd.DataFrame({
            'High': [106, 108],
            'Low': [96, 98],
            'Close': [102, 105]
        })
        
        combined_levels = detector.combine_timeframe_levels({
            '1H': hourly_data,
            '1D': daily_data,
            '1W': weekly_data
        })
        
        # Should have levels from all timeframes
        assert len(combined_levels["levels"]) > 0
        
        # Longer timeframe levels should have higher strength
        for level in combined_levels["levels"]:
            if level["timeframe"] == "1W":
                assert level["strength"] >= 0.8
            elif level["timeframe"] == "1D":
                assert level["strength"] >= 0.5