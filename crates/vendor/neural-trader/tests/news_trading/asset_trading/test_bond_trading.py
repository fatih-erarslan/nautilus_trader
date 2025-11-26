"""Tests for bond trading functionality."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from src.news_trading.asset_trading.bonds.yield_collector import TreasuryYieldCollector
from src.news_trading.asset_trading.bonds.yield_curve import YieldCurveAnalyzer
from src.news_trading.asset_trading.bonds.etf_analyzer import BondETFAnalyzer
from src.news_trading.asset_trading.bonds.duration_trader import DurationTrader
from src.news_trading.asset_trading.bonds.curve_trader import YieldCurveTrader
from src.news_trading.asset_trading.bonds.credit_trader import CreditSpreadTrader


class TestBondDataCollection:
    """Test bond market data collection."""
    
    def test_treasury_yield_data(self):
        """Test collection of treasury yield data."""
        collector = TreasuryYieldCollector()
        
        # Get current yields
        yields = collector.get_current_yields()
        
        assert "1M" in yields
        assert "3M" in yields
        assert "2Y" in yields
        assert "10Y" in yields
        assert "30Y" in yields
        
        # Yields should be reasonable
        assert 0 < yields["10Y"] < 10  # Between 0 and 10%
        
    def test_yield_curve_calculation(self):
        """Test yield curve shape analysis."""
        analyzer = YieldCurveAnalyzer()
        
        # Mock yield data - inverted curve
        yields = {
            "3M": 5.50,
            "2Y": 4.80,
            "5Y": 4.50,
            "10Y": 4.60,
            "30Y": 4.80,
        }
        
        shape = analyzer.analyze_curve_shape(yields)
        
        assert shape["type"] == "inverted"
        assert shape["2s10s_spread"] == pytest.approx(-0.20, 0.01)  # 2Y-10Y spread
        assert shape["recession_probability"] >= 0.5  # Inverted = recession risk
        
    def test_bond_etf_analysis(self):
        """Test bond ETF trading analysis."""
        analyzer = BondETFAnalyzer()
        
        # Major bond ETFs
        etfs = ["TLT", "IEF", "SHY", "HYG", "LQD"]
        
        analysis = analyzer.analyze_bond_etfs(etfs)
        
        assert "TLT" in analysis  # Long-term treasuries
        assert analysis["TLT"]["duration_risk"] == "high"
        assert "relative_value" in analysis["TLT"]
        assert analysis["SHY"]["duration_risk"] == "low"  # Short-term
        
    def test_yield_curve_steepness(self):
        """Test yield curve steepness calculations."""
        analyzer = YieldCurveAnalyzer()
        
        # Normal steep curve
        yields = {
            "3M": 2.00,
            "2Y": 3.00,
            "5Y": 3.50,
            "10Y": 4.00,
            "30Y": 4.50,
        }
        
        shape = analyzer.analyze_curve_shape(yields)
        
        assert shape["type"] == "normal"
        assert shape["2s10s_spread"] == 1.00
        assert shape["trading_bias"] == "short_duration"
        
    def test_real_yield_calculation(self):
        """Test real yield calculations."""
        collector = TreasuryYieldCollector()
        
        # Get real yields (inflation-adjusted)
        real_yields = collector.get_real_yields()
        
        if real_yields:  # May not always be available
            assert "5Y" in real_yields
            assert "10Y" in real_yields
            # Real yields can be negative
            assert -5 < real_yields["10Y"] < 5


class TestBondSwingTrading:
    """Test bond swing trading strategies."""
    
    def test_duration_trade_setup(self):
        """Test duration-based bond trades."""
        trader = DurationTrader()
        
        # Fed pivot scenario
        market_data = {
            "fed_stance": "pivot_to_dovish",
            "10y_yield": 4.75,
            "10y_sma_50": 4.50,
            "inflation_trend": "declining",
            "tlt_price": 92.00,
            "tlt_sma_20": 94.00,
        }
        
        signal = trader.analyze_duration_trade(market_data)
        
        assert signal["position"] == "long_tlt"
        assert signal["rationale"] == "Fed pivot supports duration"
        assert signal["stop_yield"] == pytest.approx(4.90, 0.05)  # Stop if yields break higher
        
    def test_yield_curve_trade(self):
        """Test yield curve steepener/flattener trades."""
        trader = YieldCurveTrader()
        
        # Curve steepening scenario
        curve_data = {
            "2y_yield": 5.00,
            "10y_yield": 4.50,
            "curve_trend": "steepening",
            "fed_policy": "cutting_short_rates",
        }
        
        trades = trader.generate_curve_trades(curve_data)
        
        assert len(trades) == 2  # Pairs trade
        assert trades[0]["action"] == "long"
        assert trades[0]["instrument"] == "IEF"  # Long intermediate
        assert trades[1]["action"] == "short"
        assert trades[1]["instrument"] == "SHY"  # Short front-end
        
    def test_credit_spread_trade(self):
        """Test corporate bond spread trades."""
        trader = CreditSpreadTrader()
        
        spread_data = {
            "ig_spread": 150,  # Investment grade spread in bps
            "hy_spread": 500,  # High yield spread
            "historical_ig_avg": 100,
            "historical_hy_avg": 400,
            "vix": 25,  # Market stress indicator
        }
        
        signal = trader.analyze_credit_opportunity(spread_data)
        
        assert signal["trade"] == "long_credit"
        assert signal["instrument"] == "LQD"  # Investment grade corporate
        assert signal["hedge"] == "TLT"  # Treasury hedge
        assert signal["size_ratio"] == pytest.approx(2.0, 0.5)  # 2:1 credit to treasury
        
    def test_flight_to_quality_signal(self):
        """Test flight-to-quality trading signals."""
        trader = DurationTrader()
        
        # Risk-off scenario
        market_data = {
            "vix": 35,
            "spy_change": -0.03,  # 3% equity decline
            "credit_spreads_widening": True,
            "10y_yield": 4.00,
            "10y_yield_1d_ago": 4.20,  # Yields falling
            "dollar_index": 105,  # Strong dollar
        }
        
        signal = trader.analyze_flight_to_quality(market_data)
        
        assert signal["position"] == "long_treasuries"
        assert signal["instruments"] == ["TLT", "IEF"]
        assert signal["confidence"] > 0.8