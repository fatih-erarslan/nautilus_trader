"""Tests for stock data collection from free sources"""

import pytest
from datetime import datetime, timedelta
import pandas as pd


class TestStockDataCollection:
    """Test suite for stock data collection functionality"""
    
    def test_free_stock_data_sources(self):
        """Test integration with free stock data APIs"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        # Test Yahoo Finance integration (free)
        data = collector.get_stock_data("AAPL", period="1mo")
        
        # Check required columns exist
        assert "Open" in data.columns
        assert "High" in data.columns
        assert "Low" in data.columns
        assert "Close" in data.columns
        assert "Volume" in data.columns
        
        # Ensure we have sufficient data
        assert len(data) >= 20  # At least 20 trading days
        
        # Validate data types
        assert data["Close"].dtype in ["float64", "float32"]
        assert data["Volume"].dtype in ["int64", "int32", "float64"]
        
        # Check data is recent
        assert data.index[-1].date() >= (datetime.now().date() - timedelta(days=5))
        
    def test_intraday_data_collection(self):
        """Test collection of intraday data for day trading"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        # Get intraday data
        data = collector.get_intraday_data("SPY", interval="5m")
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
        # Verify time intervals
        if len(data) > 1:
            time_diff = (data.index[1] - data.index[0]).total_seconds()
            assert time_diff == 300  # 5 minutes = 300 seconds
            
    def test_sector_performance_data(self):
        """Test sector performance data collection"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        # Get sector performance
        performance = collector.get_sector_data()
        
        # Check all major sectors are included
        expected_sectors = [
            "Technology", "Financials", "Energy", "Healthcare",
            "Industrials", "Consumer Discretionary", "Consumer Staples",
            "Utilities", "Real Estate"
        ]
        
        for sector in expected_sectors:
            assert sector in performance
            assert isinstance(performance[sector], (int, float))
            assert -100 <= performance[sector] <= 100  # Reasonable percentage range
            
    def test_data_caching(self):
        """Test that data is properly cached to avoid excessive API calls"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        # First call - should fetch from API
        start_time = datetime.now()
        data1 = collector.get_stock_data("MSFT", period="5d")
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        # Second call - should use cache
        start_time = datetime.now()
        data2 = collector.get_stock_data("MSFT", period="5d")
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        # Cache should be much faster
        assert second_call_time < first_call_time / 10
        
        # Data should be identical
        assert data1.equals(data2)
        
    def test_error_handling(self):
        """Test proper error handling for invalid symbols"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        # Test with invalid symbol
        with pytest.raises(ValueError):
            collector.get_stock_data("INVALID_SYMBOL_XYZ", period="1d")
            
    def test_multiple_symbols(self):
        """Test fetching data for multiple symbols"""
        from src.trading.stocks.data_collector import StockDataCollector
        
        collector = StockDataCollector()
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = collector.get_multiple_stocks(symbols, period="5d")
        
        # Check we got data for all symbols
        assert len(data) == len(symbols)
        
        for symbol in symbols:
            assert symbol in data
            assert isinstance(data[symbol], pd.DataFrame)
            assert len(data[symbol]) > 0