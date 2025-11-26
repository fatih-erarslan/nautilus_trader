"""Tests for temporal reference extraction - RED phase"""

import pytest
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')


def test_temporal_reference_extraction():
    """Test extraction of time references"""
    from news_trading.news_parsing.temporal import TemporalExtractor
    
    extractor = TemporalExtractor()
    
    text = "Bitcoin hit ATH yesterday, up 50% since last month"
    refs = extractor.extract(text)
    
    assert "yesterday" in refs
    assert "last month" in refs


def test_temporal_normalization():
    """Test normalization of temporal references"""
    from news_trading.news_parsing.temporal import normalize_temporal
    
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    # Test relative references
    yesterday = normalize_temporal("yesterday", base_time)
    assert yesterday.date() == (base_time - timedelta(days=1)).date()
    
    last_week = normalize_temporal("last week", base_time)
    assert last_week.date() == (base_time - timedelta(days=7)).date()
    
    last_month = normalize_temporal("last month", base_time)
    assert last_month.month == 12  # December from January
    assert last_month.year == 2023


def test_complex_temporal_references():
    """Test extraction of complex temporal references"""
    from news_trading.news_parsing.temporal import TemporalExtractor
    
    extractor = TemporalExtractor()
    
    text = "Q4 2023 earnings exceeded Q3 results. Next quarter outlook positive for H1 2024"
    refs = extractor.extract(text)
    
    assert "Q4 2023" in refs
    assert "Q3" in refs
    assert "Next quarter" in refs
    assert "H1 2024" in refs


def test_temporal_range_extraction():
    """Test extraction of temporal ranges"""
    from news_trading.news_parsing.temporal import TemporalExtractor
    
    extractor = TemporalExtractor()
    
    text = "Bitcoin rallied from Monday to Friday, best week since January"
    refs = extractor.extract(text)
    
    assert any("Monday" in ref for ref in refs)
    assert any("Friday" in ref for ref in refs)
    assert any("week" in ref for ref in refs)
    assert any("January" in ref for ref in refs)


def test_normalize_quarters():
    """Test normalization of quarter references"""
    from news_trading.news_parsing.temporal import normalize_temporal
    
    base_time = datetime(2024, 2, 15, 10, 0, 0)
    
    q1_2024 = normalize_temporal("Q1 2024", base_time)
    assert q1_2024.month == 1
    assert q1_2024.year == 2024
    
    q4_2023 = normalize_temporal("Q4 2023", base_time)
    assert q4_2023.month == 10  # Start of Q4
    assert q4_2023.year == 2023


def test_relative_time_expressions():
    """Test various relative time expressions"""
    from news_trading.news_parsing.temporal import TemporalExtractor
    
    extractor = TemporalExtractor()
    
    text = """
    Two days ago the market crashed.
    Three weeks from now we expect recovery.
    In the past hour, volume spiked.
    Over the next month, volatility expected.
    """
    
    refs = extractor.extract(text)
    
    assert any("days ago" in ref for ref in refs)
    assert any("weeks from now" in ref for ref in refs)
    assert any("past hour" in ref for ref in refs)
    assert any("next month" in ref for ref in refs)