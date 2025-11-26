"""
Tests for stock market news sources - Phase 2A
Following TDD approach: RED -> GREEN -> REFACTOR
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from aioresponses import aioresponses
import aiohttp

from src.news.models import NewsItem


class TestReutersSource:
    """Tests for Reuters news source implementation"""
    
    def test_reuters_source_init(self):
        """Test Reuters news source initialization"""
        from src.news.sources.reuters import ReutersSource
        
        source = ReutersSource(api_key="test-key")
        assert source.source_name == "reuters"
        assert source.api_key == "test-key"
        assert source.base_url == "https://api.reuters.com/v1"
    
    @pytest.mark.asyncio
    async def test_reuters_fetch_latest(self):
        """Test fetching latest Reuters articles"""
        from src.news.sources.reuters import ReutersSource
        
        source = ReutersSource(api_key="test-key")
        
        # Mock the API response
        with aioresponses() as mocked:
            mocked.get(
                "https://api.reuters.com/v1/articles/latest?limit=10",
                payload={
                    "articles": [{
                        "id": "reuters-001",
                        "headline": "Fed Signals Rate Changes Amid Market Volatility",
                        "body": "The Federal Reserve indicated potential rate adjustments as markets show increased volatility. TSLA and other tech stocks responded positively.",
                        "publishedAt": "2024-01-15T10:00:00Z",
                        "url": "https://reuters.com/article/001",
                        "author": "Jane Smith",
                        "categories": ["markets", "fed", "stocks"]
                    }]
                },
                headers={"Content-Type": "application/json"}
            )
            
            items = await source.fetch_latest(limit=10)
            assert len(items) == 1
            assert items[0].title == "Fed Signals Rate Changes Amid Market Volatility"
            assert items[0].id == "reuters-reuters-001"
            assert "TSLA" in items[0].entities  # Test entity extraction
            assert items[0].metadata["author"] == "Jane Smith"
            assert "markets" in items[0].metadata["categories"]
    
    @pytest.mark.asyncio
    async def test_reuters_entity_extraction(self):
        """Test extraction of stock symbols and entities from Reuters content"""
        from src.news.sources.reuters import ReutersSource
        
        source = ReutersSource(api_key="test-key")
        
        content = """
        Apple Inc. (AAPL) announced record earnings, beating analyst expectations.
        Meanwhile, Microsoft (MSFT) and Amazon.com Inc (AMZN) also reported strong results.
        The S&P 500 index rose 2.5% while the Nasdaq Composite gained 3%.
        Bitcoin (BTC) surged past $50,000 as cryptocurrency markets rallied.
        """
        
        entities = source._extract_entities(content)
        
        # Should extract stock symbols
        assert "AAPL" in entities
        assert "MSFT" in entities
        assert "AMZN" in entities
        assert "BTC" in entities
        
        # Should extract indices
        assert "S&P 500" in entities
        assert "Nasdaq" in entities
    
    @pytest.mark.asyncio
    async def test_reuters_error_handling(self):
        """Test Reuters source error handling"""
        from src.news.sources.reuters import ReutersSource
        from src.news.sources import NewsSourceError
        
        source = ReutersSource(api_key="test-key")
        
        with aioresponses() as mocked:
            mocked.get(
                "https://api.reuters.com/v1/articles/latest?limit=100",
                status=500
            )
            
            with pytest.raises(NewsSourceError) as exc_info:
                await source.fetch_latest()
            
            assert "Reuters API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_reuters_stream(self):
        """Test Reuters real-time streaming"""
        from src.news.sources.reuters import ReutersSource
        
        source = ReutersSource(api_key="test-key")
        
        # Mock WebSocket connection
        with patch('aiohttp.ClientSession.ws_connect') as mock_ws:
            mock_conn = AsyncMock()
            mock_conn.receive.side_effect = [
                Mock(data='{"type": "article", "article": {"id": "stream-1", "headline": "Breaking: Tesla Announces New Model", "body": "Tesla (TSLA) unveiled...", "publishedAt": "2024-01-15T11:00:00Z", "url": "https://reuters.com/stream/1"}}', type=aiohttp.WSMsgType.TEXT),
                Mock(type=aiohttp.WSMsgType.CLOSED)
            ]
            mock_ws.return_value.__aenter__.return_value = mock_conn
            
            items = []
            async for item in source.stream():
                items.append(item)
            
            assert len(items) == 1
            assert items[0].title == "Breaking: Tesla Announces New Model"
            assert "TSLA" in items[0].entities


class TestYahooFinanceSource:
    """Tests for Yahoo Finance news source implementation"""
    
    def test_yahoo_source_init(self):
        """Test Yahoo Finance source initialization"""
        from src.news.sources.yahoo_finance import YahooFinanceSource
        
        source = YahooFinanceSource()
        assert source.source_name == "yahoo_finance"
        assert source.base_url == "https://query1.finance.yahoo.com/v1"
    
    @pytest.mark.asyncio
    async def test_yahoo_fetch_latest(self):
        """Test fetching latest Yahoo Finance articles"""
        from src.news.sources.yahoo_finance import YahooFinanceSource
        
        source = YahooFinanceSource()
        
        with aioresponses() as mocked:
            mocked.get(
                "https://query1.finance.yahoo.com/v1/finance/news?count=10",
                payload={
                    "items": [{
                        "uuid": "yahoo-001",
                        "title": "Apple Beats Earnings Expectations by 40%",
                        "summary": "Apple Inc. (AAPL) reported Q4 earnings that exceeded analyst expectations by 40%.",
                        "published_at": 1705315200,  # Unix timestamp
                        "link": "https://finance.yahoo.com/news/apple-beats-001",
                        "publisher": "Yahoo Finance",
                        "type": "STORY",
                        "entities": [
                            {"term": "AAPL", "label": "Apple Inc.", "score": 0.95}
                        ]
                    }]
                }
            )
            
            items = await source.fetch_latest(limit=10)
            assert len(items) == 1
            assert items[0].title == "Apple Beats Earnings Expectations by 40%"
            assert items[0].id == "yahoo-yahoo-001"
            assert "AAPL" in items[0].entities
            assert items[0].metadata["earnings_beat"] == True
            assert items[0].metadata["earnings_surprise"] == 0.40
    
    @pytest.mark.asyncio
    async def test_yahoo_earnings_detection(self):
        """Test detection of earnings-related news"""
        from src.news.sources.yahoo_finance import YahooFinanceSource
        
        source = YahooFinanceSource()
        
        # Test various earnings headlines
        earnings_headlines = [
            ("Tesla Beats Q3 Earnings by 25%", True, 0.25),
            ("Microsoft Misses Revenue Estimates", True, -0.05),  # Default miss percentage
            ("Apple Stock Rises on Strong iPhone Sales", False, None),
            ("Amazon Reports Record Q4 Earnings", True, None)
        ]
        
        for headline, is_earnings, surprise in earnings_headlines:
            result = source._detect_earnings_news(headline)
            assert result["is_earnings"] == is_earnings
            if surprise is not None:
                assert result.get("earnings_surprise") == surprise
    
    @pytest.mark.asyncio
    async def test_yahoo_analyst_ratings(self):
        """Test fetching analyst ratings and recommendations"""
        from src.news.sources.yahoo_finance import YahooFinanceSource
        
        source = YahooFinanceSource()
        
        with aioresponses() as mocked:
            mocked.get(
                "https://query1.finance.yahoo.com/v1/finance/news?count=20&category=analyst-ratings",
                payload={
                    "items": [{
                        "uuid": "rating-001",
                        "title": "Goldman Sachs Upgrades Apple to Buy, Raises PT to $200",
                        "summary": "Goldman Sachs upgraded Apple stock from Hold to Buy with a new price target of $200.",
                        "published_at": 1705315200,
                        "link": "https://finance.yahoo.com/news/goldman-apple",
                        "entities": [{"term": "AAPL", "label": "Apple Inc."}]
                    }]
                }
            )
            
            items = await source.fetch_analyst_ratings(limit=20)
            assert len(items) == 1
            assert items[0].metadata["rating_change"] == "upgrade"
            assert items[0].metadata["new_rating"] == "Buy"
            assert items[0].metadata["price_target"] == 200.0
            assert items[0].metadata["analyst"] == "Goldman Sachs"
    
    @pytest.mark.asyncio
    async def test_yahoo_market_movers(self):
        """Test fetching market movers and trending stocks"""
        from src.news.sources.yahoo_finance import YahooFinanceSource
        
        source = YahooFinanceSource()
        
        with aioresponses() as mocked:
            # Mock trending tickers
            mocked.get(
                "https://query1.finance.yahoo.com/v1/finance/trending/US",
                payload={
                    "finance": {
                        "result": [{
                            "quotes": [
                                {"symbol": "TSLA"},
                                {"symbol": "NVDA"},
                                {"symbol": "AAPL"}
                            ]
                        }]
                    }
                }
            )
            
            trending = await source.get_trending_tickers()
            assert "TSLA" in trending
            assert "NVDA" in trending
            assert "AAPL" in trending