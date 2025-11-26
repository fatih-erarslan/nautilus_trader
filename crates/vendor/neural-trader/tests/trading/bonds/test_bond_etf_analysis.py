"""Tests for bond ETF analysis and trading"""

import pytest
import pandas as pd
from datetime import datetime, timedelta


class TestBondETFAnalysis:
    """Test suite for bond ETF analysis functionality"""
    
    def test_bond_etf_analysis(self):
        """Test analysis of major bond ETFs"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Major bond ETFs to analyze
        etfs = ["TLT", "IEF", "SHY", "HYG", "LQD", "AGG", "BND"]
        
        analysis = analyzer.analyze_bond_etfs(etfs)
        
        # Check all ETFs are analyzed
        for etf in etfs:
            assert etf in analysis
            
        # Check TLT (Long-term treasuries) analysis
        tlt_analysis = analysis["TLT"]
        assert tlt_analysis["duration_risk"] == "high"
        assert "effective_duration" in tlt_analysis
        assert tlt_analysis["effective_duration"] > 15  # Long duration
        assert "yield" in tlt_analysis
        assert "relative_value" in tlt_analysis
        assert "technical_score" in tlt_analysis
        
        # Check SHY (Short-term treasuries) analysis
        shy_analysis = analysis["SHY"]
        assert shy_analysis["duration_risk"] == "low"
        assert shy_analysis["effective_duration"] < 3  # Short duration
        
    def test_duration_matching(self):
        """Test ETF selection based on duration targets"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Find ETFs matching target duration
        target_duration = 7.0  # 7 years
        matches = analyzer.find_duration_matches(target_duration, tolerance=1.0)
        
        assert len(matches) > 0
        
        for etf in matches:
            assert 6.0 <= etf["duration"] <= 8.0
            assert "ticker" in etf
            assert "exact_duration" in etf
            
        # IEF should be in matches (7-10 year treasuries)
        assert any(m["ticker"] == "IEF" for m in matches)
        
    def test_credit_quality_analysis(self):
        """Test credit quality analysis for bond ETFs"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Analyze credit ETFs
        credit_etfs = ["LQD", "HYG", "JNK", "VCIT", "VCSH"]
        
        credit_analysis = analyzer.analyze_credit_quality(credit_etfs)
        
        # LQD should be investment grade
        assert credit_analysis["LQD"]["credit_rating"] == "investment_grade"
        assert credit_analysis["LQD"]["average_rating"] in ["A", "BBB"]
        
        # HYG should be high yield
        assert credit_analysis["HYG"]["credit_rating"] == "high_yield"
        assert credit_analysis["HYG"]["default_risk"] > credit_analysis["LQD"]["default_risk"]
        
        # Check spread analysis
        assert credit_analysis["HYG"]["credit_spread"] > credit_analysis["LQD"]["credit_spread"]
        
    def test_etf_relative_value(self):
        """Test relative value analysis between bond ETFs"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Compare similar duration ETFs
        comparison = analyzer.compare_etfs(["IEF", "ITE", "GOVT"])
        
        assert "relative_value_matrix" in comparison
        assert "recommended_etf" in comparison
        assert "rationale" in comparison
        
        # Should identify cheapest/richest
        assert comparison["cheapest_etf"] is not None
        assert comparison["richest_etf"] is not None
        
    def test_sector_allocation(self):
        """Test bond sector allocation analysis"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Analyze sector allocation
        sectors = {
            "treasuries": ["TLT", "IEF", "SHY"],
            "corporates": ["LQD", "VCIT"],
            "high_yield": ["HYG", "JNK"],
            "munis": ["MUB", "SUB"],
            "international": ["BNDX", "IAGG"]
        }
        
        allocation = analyzer.optimize_sector_allocation(sectors, risk_level="moderate")
        
        assert sum(allocation.values()) == pytest.approx(1.0, 0.01)  # Should sum to 100%
        
        # Moderate risk should have balanced allocation
        assert 0.3 <= allocation["treasuries"] <= 0.5
        assert allocation["high_yield"] <= 0.2  # Limited high yield
        
    def test_etf_momentum_signals(self):
        """Test momentum-based signals for bond ETFs"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Get momentum signals
        etfs = ["TLT", "IEF", "AGG", "HYG"]
        signals = analyzer.get_momentum_signals(etfs, lookback_days=20)
        
        for etf in etfs:
            assert etf in signals
            signal = signals[etf]
            
            assert "momentum_score" in signal
            assert -1 <= signal["momentum_score"] <= 1
            assert "trend" in signal
            assert signal["trend"] in ["bullish", "bearish", "neutral"]
            assert "entry_point" in signal
            
    def test_etf_pairs_trading(self):
        """Test pairs trading opportunities between bond ETFs"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Find pairs trading opportunities
        pairs = analyzer.find_pairs_trades()
        
        if pairs:  # May not always have opportunities
            pair = pairs[0]
            assert "long_etf" in pair
            assert "short_etf" in pair
            assert "spread_zscore" in pair
            assert abs(pair["spread_zscore"]) > 2  # Significant deviation
            assert "expected_profit" in pair
            assert "holding_period" in pair
            
    def test_duration_hedging(self):
        """Test duration hedging strategies"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Portfolio to hedge
        portfolio = {
            "TLT": 100000,  # $100k in long-term treasuries
            "LQD": 50000    # $50k in corporate bonds
        }
        
        hedge = analyzer.calculate_duration_hedge(portfolio)
        
        assert "hedge_instruments" in hedge
        assert "hedge_amounts" in hedge
        assert "net_duration" in hedge
        assert hedge["net_duration"] < 2  # Should be well-hedged
        
        # Should suggest short positions or short-duration ETFs
        assert any(amt < 0 for amt in hedge["hedge_amounts"].values())
        
    def test_roll_analysis(self):
        """Test bond ETF roll analysis"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Analyze roll opportunities
        roll_analysis = analyzer.analyze_roll_opportunities()
        
        if roll_analysis["opportunities"]:
            opp = roll_analysis["opportunities"][0]
            assert "from_etf" in opp
            assert "to_etf" in opp
            assert "roll_yield" in opp
            assert "rationale" in opp
            
            # Roll yield should be positive for good opportunities
            assert opp["roll_yield"] > 0
            
    def test_flow_analysis(self):
        """Test ETF flow analysis for sentiment"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Analyze flows for sentiment
        etfs = ["TLT", "HYG", "AGG"]
        flow_data = analyzer.analyze_etf_flows(etfs, days=5)
        
        for etf in etfs:
            assert etf in flow_data
            flows = flow_data[etf]
            
            assert "net_flows" in flows
            assert "flow_trend" in flows
            assert flows["flow_trend"] in ["inflow", "outflow", "neutral"]
            assert "sentiment_score" in flows
            assert -1 <= flows["sentiment_score"] <= 1