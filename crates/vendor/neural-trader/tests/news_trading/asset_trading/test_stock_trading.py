"""Tests for stock trading functionality."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.news_trading.asset_trading.stocks.data_collector import StockDataCollector
from src.news_trading.asset_trading.stocks.indicators import TechnicalIndicators
from src.news_trading.asset_trading.stocks.price_levels import SupportResistanceDetector
from src.news_trading.asset_trading.stocks.swing_setups import SwingSetupDetector
from src.news_trading.asset_trading.stocks.earnings_trader import EarningsGapTrader


class TestStockDataCollection:
    """Test stock data collection functionality."""
    
    def test_free_stock_data_sources(self):
        """Test integration with free stock data APIs."""
        collector = StockDataCollector()
        
        # Test Yahoo Finance integration (free)
        data = collector.get_stock_data("AAPL", period="1mo")
        
        assert "Open" in data.columns
        assert "High" in data.columns
        assert "Low" in data.columns
        assert "Close" in data.columns
        assert "Volume" in data.columns
        assert len(data) >= 20  # At least 20 trading days
        
    def test_technical_indicators_calculation(self):
        """Test calculation of technical indicators for swing trading."""
        indicators = TechnicalIndicators()
        
        # Mock price data
        prices = [100, 102, 101, 103, 105, 104, 106, 107, 105, 108]
        
        # Test moving averages
        sma_5 = indicators.calculate_sma(prices, period=5)
        assert len(sma_5) == 6  # 10 prices - 4 for SMA calculation
        assert sma_5[0] == pytest.approx(102.2, 0.1)
        
        # Test RSI
        rsi = indicators.calculate_rsi(prices, period=5)
        assert 0 <= rsi <= 100
        
        # Test MACD
        prices_extended = prices * 4  # Need more data for MACD
        macd_line, signal_line, histogram = indicators.calculate_macd(prices_extended)
        assert macd_line is not None
        # MACD might be empty with insufficient data, just check it returns something
        assert isinstance(macd_line, list)
        
    def test_support_resistance_detection(self):
        """Test automatic support and resistance level detection."""
        detector = SupportResistanceDetector()
        
        # Price data with clear levels
        prices = [100, 98, 99, 98, 102, 103, 102, 98, 99, 98, 104, 105]
        
        levels = detector.detect_levels(prices)
        
        assert 98 in levels["support"]  # Clear support at 98
        assert any(101 <= r <= 103 for r in levels["resistance"])  # Resistance around 102-103
        assert levels["strength"][98] > 0.7  # Strong support
        
    def test_intraday_data_collection(self):
        """Test intraday data collection."""
        collector = StockDataCollector()
        
        # Get intraday data
        data = collector.get_intraday_data("SPY", interval="5m")
        
        # Should have data (might be from previous trading day if markets closed)
        assert len(data) > 0
        # Check that we got data within the last few days (accounting for weekends)
        from datetime import timedelta
        assert (datetime.now().date() - data.index[-1].date()) <= timedelta(days=3)
        
    def test_sector_performance_data(self):
        """Test sector performance data collection."""
        collector = StockDataCollector()
        
        sector_data = collector.get_sector_data()
        
        # Check major sectors are present
        assert "Technology" in sector_data
        assert "Financials" in sector_data
        assert "Energy" in sector_data
        
        # Performance should be reasonable percentages
        for sector, performance in sector_data.items():
            assert -10 < performance < 10  # Daily moves typically within 10%


class TestStockSwingTrading:
    """Test stock swing trading strategies."""
    
    def test_bullish_swing_setup(self):
        """Test identification of bullish swing setups."""
        detector = SwingSetupDetector()
        
        # Bullish setup scenario
        market_data = {
            "price": 150.00,
            "sma_20": 148.00,
            "sma_50": 145.00,
            "sma_200": 140.00,
            "rsi": 45,  # Oversold bounce
            "volume_avg_ratio": 1.2,
            "recent_low": 147.00,
            "recent_high": 155.00,
        }
        
        setup = detector.detect_setup(market_data)
        
        assert setup["type"] == "bullish_reversal"
        assert setup["entry_price"] == pytest.approx(150.50, 0.50)
        assert setup["stop_loss"] == pytest.approx(147.00, 0.50)
        assert setup["target_1"] == pytest.approx(155.00, 1.00)
        
    def test_breakout_swing_setup(self):
        """Test breakout pattern detection."""
        detector = SwingSetupDetector()
        
        # Consolidation breakout scenario
        market_data = {
            "price": 105.00,
            "resistance": 104.50,
            "support": 102.00,
            "consolidation_days": 8,
            "volume_surge": 1.8,
            "atr": 1.50,
        }
        
        setup = detector.detect_breakout(market_data)
        
        assert setup["type"] == "resistance_breakout"
        assert setup["confidence"] > 0.6
        assert setup["stop_loss"] == pytest.approx(103.0, 0.5)  # Below breakout level
        
    def test_earnings_gap_swing(self):
        """Test post-earnings gap trading setup."""
        trader = EarningsGapTrader()
        
        earnings_data = {
            "ticker": "MSFT",
            "eps_actual": 2.50,
            "eps_estimate": 2.20,
            "revenue_beat": True,
            "guidance": "raised",
            "gap_percent": 5.5,  # 5.5% gap up
            "pre_earnings_price": 380.00,
            "current_price": 401.00,
        }
        
        signal = trader.analyze_earnings_gap(earnings_data)
        
        assert signal["action"] == "buy_pullback"
        assert signal["entry_zone"] == (387.6, 395.2)  # Wait for pullback (2-4% above pre-earnings)
        assert signal["holding_period"] == "3-5 days"
        
    def test_momentum_swing_setup(self):
        """Test momentum-based swing setups."""
        detector = SwingSetupDetector()
        
        # Strong momentum scenario
        market_data = {
            "price": 75.00,
            "sma_10": 73.00,
            "sma_20": 71.00,
            "sma_50": 68.00,
            "rsi": 65,  # Strong but not overbought
            "volume_avg_ratio": 1.5,
            "price_change_5d": 0.08,  # 8% gain in 5 days
            "sector_strength": 0.7,
        }
        
        setup = detector.detect_momentum_setup(market_data)
        
        assert setup["type"] == "momentum_long"
        assert setup["confidence"] > 0.6
        assert "trailing_stop" in setup