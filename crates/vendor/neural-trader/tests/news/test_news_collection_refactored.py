"""
Tests for refactored News Collection Module with caching and error handling
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock
import asyncio

from src.news.sources import NewsSource, NewsSourceError
from src.news.models import NewsItem


class TestRefactoredNewsSource:
    """Tests for refactored NewsSource with caching and metrics"""
    
    @pytest.fixture
    def mock_news_source(self):
        """Create a concrete implementation of NewsSource for testing"""
        class MockSource(NewsSource):
            def __init__(self, source_name="test", cache_ttl=5):
                super().__init__(source_name, cache_ttl=cache_ttl)
                self._fetch_count = 0
                
            async def fetch_latest(self, limit: int = 100) -> list[NewsItem]:
                self._fetch_count += 1
                return [
                    NewsItem(
                        id=f"test-{i}",
                        title=f"Article {i}",
                        content=f"Content {i}",
                        source=self.source_name,
                        timestamp=datetime.now(),
                        url=f"https://example.com/{i}",
                        entities=[],
                        metadata={}
                    )
                    for i in range(min(limit, 5))
                ]
                
            async def stream(self):
                for i in range(3):
                    yield NewsItem(
                        id=f"stream-{i}",
                        title=f"Stream Article {i}",
                        content=f"Stream Content {i}",
                        source=self.source_name,
                        timestamp=datetime.now(),
                        url=f"https://example.com/stream/{i}",
                        entities=[],
                        metadata={}
                    )
        
        return MockSource("test-source", cache_ttl=1)
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, mock_news_source):
        """Test that caching works correctly"""
        # First fetch - should hit the source
        items1 = await mock_news_source.fetch_latest_with_cache(limit=5)
        assert len(items1) == 5
        assert mock_news_source._fetch_count == 1
        
        # Second fetch - should hit cache
        items2 = await mock_news_source.fetch_latest_with_cache(limit=5)
        assert len(items2) == 5
        assert mock_news_source._fetch_count == 1  # Still 1, cache hit
        assert items1 == items2
        
        # Different limit - should hit source again
        items3 = await mock_news_source.fetch_latest_with_cache(limit=3)
        assert len(items3) == 3
        assert mock_news_source._fetch_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, mock_news_source):
        """Test that cache expires after TTL"""
        # First fetch
        items1 = await mock_news_source.fetch_latest_with_cache(limit=5)
        assert mock_news_source._fetch_count == 1
        
        # Wait for cache to expire (TTL is 1 second)
        await asyncio.sleep(1.1)
        
        # Should hit source again
        items2 = await mock_news_source.fetch_latest_with_cache(limit=5)
        assert mock_news_source._fetch_count == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in fetch_latest_with_cache"""
        class ErrorSource(NewsSource):
            async def fetch_latest(self, limit: int = 100):
                raise Exception("API Error")
                
            async def stream(self):
                yield
        
        source = ErrorSource("error-source")
        
        with pytest.raises(NewsSourceError) as exc_info:
            await source.fetch_latest_with_cache()
        
        assert "Failed to fetch from error-source" in str(exc_info.value)
        assert source._metrics['errors'] == 1
    
    def test_metrics_tracking(self, mock_news_source):
        """Test that metrics are properly tracked"""
        metrics = mock_news_source.get_metrics()
        
        assert metrics['source'] == 'test-source'
        assert metrics['fetch_count'] == 0
        assert metrics['cache_hits'] == 0
        assert metrics['errors'] == 0
        assert metrics['cache_info']['maxsize'] == 1000
        assert metrics['cache_info']['ttl'] == 1
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_news_source):
        """Test cache clearing functionality"""
        # Populate cache
        await mock_news_source.fetch_latest_with_cache(limit=5)
        assert len(mock_news_source._cache) == 1
        
        # Clear cache
        mock_news_source.clear_cache()
        assert len(mock_news_source._cache) == 0
        
        # Next fetch should hit source
        await mock_news_source.fetch_latest_with_cache(limit=5)
        assert mock_news_source._fetch_count == 2