"""Tests for stock market news sources - RED phase"""

import pytest
from datetime import datetime
from aioresponses import aioresponses
from yarl import URL
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')


@pytest.mark.asyncio
async def test_yahoo_finance_source_init():
    """Test Yahoo Finance news source initialization"""
    from news_trading.news_collection.sources.yahoo_finance_enhanced import YahooFinanceEnhancedSource
    
    source = YahooFinanceEnhancedSource()
    assert source.source_name == "yahoo_finance"
    assert source.base_url == "https://query1.finance.yahoo.com"


@pytest.mark.asyncio
async def test_yahoo_finance_fetch_latest():
    """Test fetching latest Yahoo Finance articles"""
    from news_trading.news_collection.sources.yahoo_finance_enhanced import YahooFinanceEnhancedSource
    
    source = YahooFinanceEnhancedSource()
    
    # Mock the API response
    with aioresponses() as mocked:
        mock_response = {
            "news": [
                {
                    "uuid": "yahoo-001",
                    "title": "Apple Beats Earnings Expectations",
                    "summary": "Apple Inc. reported quarterly earnings that exceeded analyst expectations...",
                    "publisher": "Yahoo Finance",
                    "providerPublishTime": 1705320000,
                    "link": "https://finance.yahoo.com/news/apple-001"
                },
                {
                    "uuid": "yahoo-002",
                    "title": "Fed Signals Rate Changes Coming",
                    "summary": "Federal Reserve hints at policy changes...",
                    "publisher": "Reuters",
                    "providerPublishTime": 1705320100,
                    "link": "https://finance.yahoo.com/news/fed-002"
                }
            ]
        }
        
        mocked.get(
            "https://query1.finance.yahoo.com/v1/finance/news?category=generalnews&count=10",
            payload=mock_response
        )
        
        items = await source.fetch_latest(limit=10)
        assert len(items) == 2
        assert items[0].title == "Apple Beats Earnings Expectations"
        assert items[0].source == "yahoo_finance"
        assert "AAPL" in items[0].entities  # Test entity extraction
        assert items[1].title == "Fed Signals Rate Changes Coming"


@pytest.mark.asyncio
async def test_yahoo_finance_earnings_detection():
    """Test detection of earnings-related news for momentum trading"""
    from news_trading.news_collection.sources.yahoo_finance_enhanced import YahooFinanceEnhancedSource
    
    source = YahooFinanceEnhancedSource()
    
    earnings_article = {
        "uuid": "yahoo-003",
        "title": "Tesla Q4 Earnings: Revenue Up 40%, EPS Beats by $0.15",
        "summary": "Tesla reported Q4 earnings with revenue growth of 40% YoY...",
        "publisher": "Yahoo Finance",
        "providerPublishTime": 1705320200,
        "link": "https://finance.yahoo.com/news/tesla-earnings"
    }
    
    metadata = source._extract_earnings_metadata(earnings_article)
    
    assert metadata["is_earnings"] == True
    assert metadata["earnings_beat"] == True
    assert metadata["revenue_growth"] == 0.40
    assert metadata["eps_surprise"] == 0.15
    assert metadata["momentum_signal"] == "strong_positive"


@pytest.mark.asyncio
async def test_sec_filings_source():
    """Test SEC filings source for mirror trading opportunities"""
    from news_trading.news_collection.sources.sec_filings import SECFilingsSource
    
    source = SECFilingsSource()
    
    with aioresponses() as mocked:
        # Mock Form 4 insider trading filing
        mock_filing = {
            "filings": [
                {
                    "id": "sec-form4-001",
                    "form_type": "4",
                    "filer": {
                        "name": "Warren Buffett",
                        "company": "Berkshire Hathaway Inc"
                    },
                    "transactions": [
                        {
                            "ticker": "AAPL",
                            "type": "P",  # Purchase
                            "shares": 1000000,
                            "price_per_share": 150.00,
                            "transaction_date": "2024-01-15"
                        }
                    ],
                    "filing_date": "2024-01-16T16:00:00Z",
                    "url": "https://www.sec.gov/Archives/edgar/data/..."
                }
            ]
        }
        
        # Mock with query parameters
        url_with_params = str(URL("https://api.sec.gov/filings/latest").with_query({"form_type": "4", "limit": "5"}))
        mocked.get(
            url_with_params,
            payload=mock_filing
        )
        
        items = await source.fetch_latest(limit=10)
        assert len(items) == 1
        
        item = items[0]
        assert "Warren Buffett" in item.title
        assert "AAPL" in item.entities
        assert item.metadata["mirror_trade_opportunity"] == True
        assert item.metadata["institution_sentiment"] == "bullish"
        assert item.metadata["transaction_value"] == 150000000  # $150M


@pytest.mark.asyncio
async def test_13f_filings_parsing():
    """Test parsing of 13F institutional holdings for mirror trading"""
    from news_trading.news_collection.sources.sec_filings import SECFilingsSource
    
    source = SECFilingsSource()
    
    # Mock 13F filing
    filing = {
        "form_type": "13F-HR",
        "filer": "Renaissance Technologies LLC",
        "reporting_period": "2024-03-31",
        "holdings": [
            {
                "cusip": "037833100",  # AAPL
                "ticker": "AAPL",
                "shares": 5000000,
                "value": 850000000,
                "change_in_shares": 1000000,  # Increased position
                "change_percent": 0.25
            },
            {
                "cusip": "88160R101",  # TSLA
                "ticker": "TSLA",
                "shares": 0,
                "value": 0,
                "change_in_shares": -2000000,  # Sold entire position
                "change_percent": -1.0
            }
        ]
    }
    
    signals = source._analyze_13f_filing(filing)
    
    assert len(signals) == 2
    assert signals[0]["ticker"] == "AAPL"
    assert signals[0]["action"] == "increased_position"
    assert signals[0]["significance"] == "high"  # Large position increase
    assert signals[1]["ticker"] == "TSLA"
    assert signals[1]["action"] == "sold_entire_position"
    assert signals[1]["significance"] == "high"  # Complete exit


@pytest.mark.asyncio
async def test_technical_breakout_detection():
    """Test detection of technical breakout news for swing trading"""
    from news_trading.news_collection.sources.technical_news import TechnicalNewsSource
    
    source = TechnicalNewsSource()
    
    with aioresponses() as mocked:
        mock_response = {
            "articles": [
                {
                    "id": "tech-001",
                    "headline": "NVDA Breaks Above 200-Day Moving Average on Heavy Volume",
                    "content": "NVIDIA stock surged past its 200-day moving average...",
                    "timestamp": "2024-01-15T14:30:00Z",
                    "indicators": ["200_MA_breakout", "volume_spike", "bullish_flag"],
                    "ticker": "NVDA",
                    "technical_score": 0.85
                }
            ]
        }
        
        # Mock with query parameters
        url_with_params = str(URL("https://api.technical-analysis.com/breakouts").with_query({"limit": "10"}))
        mocked.get(
            url_with_params,
            payload=mock_response
        )
        
        items = await source.fetch_latest(limit=10)
        assert len(items) == 1
        
        item = items[0]
        assert "NVDA" in item.entities
        assert item.metadata["swing_trade_signal"] == True
        assert item.metadata["technical_indicators"] == ["200_MA_breakout", "volume_spike", "bullish_flag"]
        assert abs(item.metadata["signal_strength"] - 0.82) < 0.05  # Allow small difference due to calculation
        assert item.metadata["suggested_holding_period"] == "5-15"  # Based on 200_MA_breakout