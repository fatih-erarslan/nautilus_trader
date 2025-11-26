"""Tests for bond market data collection"""

import pytest
from datetime import datetime, timedelta
import pandas as pd


class TestBondDataCollection:
    """Test suite for bond market data collection"""
    
    def test_treasury_yield_data(self):
        """Test collection of treasury yield data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get current yields
        yields = collector.get_current_yields()
        
        # Check key maturities are present
        expected_maturities = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        for maturity in expected_maturities:
            assert maturity in yields, f"Missing {maturity} yield"
            
        # Yields should be reasonable (between 0 and 10%)
        for maturity, yield_value in yields.items():
            assert 0 < yield_value < 10, f"{maturity} yield {yield_value} is out of range"
            
        # Check yield curve shape (normally upward sloping)
        # In normal conditions: 3M < 2Y < 10Y < 30Y
        # But can be inverted, so just check they exist
        assert yields["3M"] is not None
        assert yields["10Y"] is not None
        assert yields["30Y"] is not None
        
    def test_bond_etf_data(self):
        """Test collection of bond ETF data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Test major bond ETFs
        etf_data = collector.get_bond_etf_data(["TLT", "IEF", "SHY", "HYG", "LQD"])
        
        assert "TLT" in etf_data  # Long-term treasuries
        assert "IEF" in etf_data  # Intermediate treasuries
        assert "SHY" in etf_data  # Short-term treasuries
        assert "HYG" in etf_data  # High yield corporate
        assert "LQD" in etf_data  # Investment grade corporate
        
        # Check data structure
        for etf, data in etf_data.items():
            assert "price" in data
            assert "yield" in data
            assert "duration" in data
            assert "ytd_return" in data
            assert data["price"] > 0
            
    def test_yield_curve_history(self):
        """Test historical yield curve data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get yield curve history
        history = collector.get_yield_curve_history(days=30)
        
        assert isinstance(history, pd.DataFrame)
        assert len(history) >= 20  # At least 20 trading days
        
        # Check columns
        expected_columns = ["2Y", "5Y", "10Y", "30Y", "2s10s_spread"]
        for col in expected_columns:
            assert col in history.columns
            
        # Check spread calculation
        for idx, row in history.iterrows():
            expected_spread = row["10Y"] - row["2Y"]
            assert abs(row["2s10s_spread"] - expected_spread) < 0.01
            
    def test_real_yields(self):
        """Test TIPS (inflation-protected) yield data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get real yields
        real_yields = collector.get_real_yields()
        
        assert "5Y_TIPS" in real_yields
        assert "10Y_TIPS" in real_yields
        assert "30Y_TIPS" in real_yields
        
        # Real yields can be negative
        for maturity, yield_value in real_yields.items():
            assert -5 < yield_value < 5
            
    def test_credit_spreads(self):
        """Test corporate bond spread data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get credit spreads
        spreads = collector.get_credit_spreads()
        
        assert "IG_spread" in spreads  # Investment grade
        assert "HY_spread" in spreads  # High yield
        assert "AAA_spread" in spreads
        assert "BBB_spread" in spreads
        
        # Spreads should be positive and reasonable
        assert 0 < spreads["IG_spread"] < 500  # basis points
        assert 100 < spreads["HY_spread"] < 1000
        
        # High yield should have higher spread than IG
        assert spreads["HY_spread"] > spreads["IG_spread"]
        
    def test_international_yields(self):
        """Test international bond yield data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get international yields
        intl_yields = collector.get_international_yields()
        
        expected_countries = ["Germany", "Japan", "UK", "Canada"]
        for country in expected_countries:
            assert country in intl_yields
            assert "10Y" in intl_yields[country]
            
        # Japan yields often near zero or negative
        assert -1 < intl_yields["Japan"]["10Y"] < 2
        
    def test_yield_data_caching(self):
        """Test that yield data is properly cached"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # First call
        start_time = datetime.now()
        yields1 = collector.get_current_yields()
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        # Second call should use cache
        start_time = datetime.now()
        yields2 = collector.get_current_yields()
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        # Cache should be faster
        assert second_call_time < first_call_time / 10
        
        # Data should be identical
        assert yields1 == yields2
        
    def test_fed_funds_data(self):
        """Test Fed Funds rate data collection"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get Fed data
        fed_data = collector.get_fed_data()
        
        assert "fed_funds_rate" in fed_data
        assert "fed_funds_target_upper" in fed_data
        assert "fed_funds_target_lower" in fed_data
        assert "next_meeting_date" in fed_data
        assert "rate_probabilities" in fed_data
        
        # Check rate is reasonable
        assert 0 <= fed_data["fed_funds_rate"] <= 10
        
        # Target range should make sense
        assert fed_data["fed_funds_target_lower"] <= fed_data["fed_funds_target_upper"]