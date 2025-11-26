"""Tests for yield curve analysis"""

import pytest
import pandas as pd
from datetime import datetime, timedelta


class TestYieldCurveAnalysis:
    """Test suite for yield curve shape analysis and trading signals"""
    
    def test_yield_curve_shape_detection(self):
        """Test detection of yield curve shapes"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Test normal curve
        normal_yields = {
            "3M": 4.50,
            "2Y": 4.80,
            "5Y": 5.00,
            "10Y": 5.20,
            "30Y": 5.50
        }
        
        shape = analyzer.analyze_curve_shape(normal_yields)
        
        assert shape["type"] == "normal"
        assert shape["2s10s_spread"] == pytest.approx(0.40, 0.01)  # 5.20 - 4.80
        assert shape["recession_probability"] < 0.3  # Low recession risk
        assert shape["steepness"] == "moderate"
        
    def test_inverted_curve_detection(self):
        """Test inverted yield curve detection"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Test inverted curve
        inverted_yields = {
            "3M": 5.50,
            "2Y": 4.80,
            "5Y": 4.50,
            "10Y": 4.60,
            "30Y": 4.80
        }
        
        shape = analyzer.analyze_curve_shape(inverted_yields)
        
        assert shape["type"] == "inverted"
        assert shape["2s10s_spread"] == pytest.approx(-0.20, 0.01)  # 4.60 - 4.80
        assert shape["recession_probability"] > 0.5  # High recession risk
        assert shape["inversion_duration"] is not None
        
    def test_flat_curve_detection(self):
        """Test flat yield curve detection"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Test flat curve
        flat_yields = {
            "3M": 4.80,
            "2Y": 4.85,
            "5Y": 4.90,
            "10Y": 4.95,
            "30Y": 5.00
        }
        
        shape = analyzer.analyze_curve_shape(flat_yields)
        
        assert shape["type"] == "flat"
        assert shape["2s10s_spread"] == pytest.approx(0.10, 0.01)
        assert 0.3 <= shape["recession_probability"] <= 0.5
        
    def test_curve_dynamics(self):
        """Test yield curve dynamics and changes"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Previous curve
        prev_yields = {
            "3M": 4.50,
            "2Y": 4.80,
            "10Y": 5.20,
            "30Y": 5.50
        }
        
        # Current curve (flattening)
        curr_yields = {
            "3M": 4.60,
            "2Y": 4.90,
            "10Y": 5.10,
            "30Y": 5.30
        }
        
        dynamics = analyzer.analyze_curve_dynamics(prev_yields, curr_yields)
        
        assert dynamics["movement"] == "flattening"
        assert dynamics["front_end_change"] > dynamics["long_end_change"]
        assert dynamics["twist"] is not None
        
    def test_butterfly_spread(self):
        """Test butterfly spread calculation"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        yields = {
            "2Y": 4.80,
            "5Y": 5.00,
            "10Y": 5.20
        }
        
        butterfly = analyzer.calculate_butterfly_spread(yields)
        
        # Butterfly = 2*5Y - 2Y - 10Y
        expected = 2 * 5.00 - 4.80 - 5.20
        assert butterfly["spread"] == pytest.approx(expected, 0.01)
        assert butterfly["signal"] in ["neutral", "rich", "cheap"]
        
    def test_curve_trading_signals(self):
        """Test trading signals based on curve analysis"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Steepening curve scenario
        market_data = {
            "current_yields": {
                "2Y": 4.50,
                "5Y": 4.80,
                "10Y": 5.20,
                "30Y": 5.60
            },
            "prev_yields": {
                "2Y": 4.60,
                "5Y": 4.85,
                "10Y": 5.10,
                "30Y": 5.40
            },
            "fed_policy": "easing"
        }
        
        signals = analyzer.generate_trading_signals(market_data)
        
        assert len(signals) > 0
        assert any(s["trade"] == "steepener" for s in signals)
        
        # Check for specific trades
        steepener_trade = next(s for s in signals if s["trade"] == "steepener")
        assert steepener_trade["position"] == "long_10Y_short_2Y"
        assert steepener_trade["confidence"] > 0.5
        
    def test_duration_recommendations(self):
        """Test duration recommendations based on curve"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Rising rate scenario
        market_conditions = {
            "curve_shape": "normal",
            "curve_trend": "bear_flattening",
            "fed_policy": "tightening",
            "inflation_trend": "rising"
        }
        
        recommendations = analyzer.get_duration_recommendations(market_conditions)
        
        assert recommendations["overall_duration"] == "underweight"
        assert recommendations["preferred_maturity"] == "short"
        assert "2Y" in recommendations["recommended_instruments"]
        assert recommendations["avoid"] == ["TLT", "EDV"]  # Long duration ETFs
        
    def test_curve_regime_detection(self):
        """Test yield curve regime detection"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Historical yield data
        historical_data = pd.DataFrame({
            '2Y': [4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.0, 4.9, 4.8],
            '10Y': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9, 4.8, 4.7],
            '30Y': [5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6]
        })
        
        regime = analyzer.detect_regime(historical_data)
        
        assert regime["current_regime"] in ["bear_flattener", "bull_flattener", 
                                           "bear_steepener", "bull_steepener"]
        assert regime["regime_duration"] > 0
        assert regime["regime_strength"] > 0
        
    def test_relative_value_analysis(self):
        """Test relative value analysis across curve"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        current_yields = {
            "2Y": 4.80,
            "3Y": 4.85,
            "5Y": 4.95,
            "7Y": 5.05,
            "10Y": 5.20,
            "30Y": 5.50
        }
        
        rv_analysis = analyzer.analyze_relative_value(current_yields)
        
        assert "rich_points" in rv_analysis
        assert "cheap_points" in rv_analysis
        assert "trades" in rv_analysis
        
        # Should identify relative value opportunities
        if rv_analysis["trades"]:
            trade = rv_analysis["trades"][0]
            assert "long" in trade
            assert "short" in trade
            assert "expected_profit_bps" in trade