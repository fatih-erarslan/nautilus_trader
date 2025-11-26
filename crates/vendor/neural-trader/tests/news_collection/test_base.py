"""Tests for core news source interface - RED phase"""

import pytest
from datetime import datetime
from typing import List, AsyncIterator
from unittest.mock import Mock
from cachetools import TTLCache
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')


def test_news_source_interface():
    """Test that NewsSource abstract interface is properly defined"""
    from news_trading.news_collection.base import NewsSource
    
    class TestSource(NewsSource):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        source = TestSource()


def test_news_source_with_implementation():
    """Test NewsSource with proper implementation"""
    from news_trading.news_collection.base import NewsSource
    from news.models import NewsItem
    
    class TestSource(NewsSource):
        async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
            return []
        
        async def stream(self) -> AsyncIterator[NewsItem]:
            yield NewsItem(
                id="test-1",
                title="Test",
                content="Content",
                source="test",
                timestamp=datetime.now(),
                url="http://test.com",
                entities=[],
                metadata={}
            )
    
    # Should work now
    source = TestSource("test_source")
    assert source.source_name == "test_source"


@pytest.mark.asyncio
async def test_news_source_caching():
    """Test that news source implements caching correctly"""
    from news_trading.news_collection.base import NewsSource
    from news.models import NewsItem
    
    class TestSource(NewsSource):
        def __init__(self):
            super().__init__("test")
            self.call_count = 0
        
        async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
            self.call_count += 1
            return [
                NewsItem(
                    id="test-1",
                    title="Test Article",
                    content="Content",
                    source="test",
                    timestamp=datetime.now(),
                    url="http://test.com",
                    entities=[],
                    metadata={}
                )
            ]
        
        async def stream(self) -> AsyncIterator[NewsItem]:
            yield
    
    source = TestSource()
    
    # First call should hit the actual method
    result1 = await source.fetch_latest_with_cache(limit=10)
    assert len(result1) == 1
    assert source.call_count == 1
    
    # Second call should hit the cache
    result2 = await source.fetch_latest_with_cache(limit=10)
    assert len(result2) == 1
    assert source.call_count == 1  # No additional call
    
    # Different limit should create new cache entry
    result3 = await source.fetch_latest_with_cache(limit=20)
    assert source.call_count == 2


def test_news_source_validation():
    """Test that NewsSource validates inputs properly"""
    from news_trading.news_collection.base import NewsSource
    from news.models import NewsItem
    
    class TestSource(NewsSource):
        async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
            if limit <= 0:
                raise ValueError("Limit must be positive")
            return []
        
        async def stream(self) -> AsyncIterator[NewsItem]:
            yield
    
    source = TestSource("test")
    
    # Test invalid limit
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(source.fetch_latest(limit=-1))