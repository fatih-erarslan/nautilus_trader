"""
Tests for News Collection Module - Phase 1: Core News Source Interface
Following TDD approach: RED -> GREEN -> REFACTOR
"""
import pytest
from datetime import datetime
from typing import List
from unittest.mock import Mock, AsyncMock, patch


def test_news_source_interface():
    """Test that NewsSource abstract interface is properly defined"""
    # This should fail initially as NewsSource doesn't exist yet
    from src.news.sources import NewsSource
    
    class TestSource(NewsSource):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        source = TestSource()


def test_news_item_model():
    """Test NewsItem data model"""
    # This should fail initially as NewsItem doesn't exist yet
    from src.news.models import NewsItem
    
    item = NewsItem(
        id="test-123",
        title="Bitcoin Surges Past $50k",
        content="Full article content...",
        source="reuters",
        timestamp=datetime.now(),
        url="https://example.com/article",
        entities=["BTC", "bitcoin"],
        metadata={"author": "John Doe"}
    )
    
    assert item.id == "test-123"
    assert item.source == "reuters"
    assert "BTC" in item.entities
    assert item.metadata["author"] == "John Doe"


def test_news_item_validation():
    """Test NewsItem validation requirements"""
    from src.news.models import NewsItem
    
    # Should fail with missing required fields
    with pytest.raises(TypeError):
        NewsItem()
    
    # Should fail with invalid timestamp
    with pytest.raises(ValueError):
        NewsItem(
            id="test-123",
            title="Test",
            content="Content",
            source="reuters",
            timestamp="invalid-date",  # Should be datetime
            url="https://example.com",
            entities=[],
            metadata={}
        )


@pytest.mark.asyncio
async def test_news_source_fetch_latest():
    """Test abstract fetch_latest method"""
    from src.news.sources import NewsSource
    from src.news.models import NewsItem
    
    class ConcreteSource(NewsSource):
        async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
            return [
                NewsItem(
                    id="test-1",
                    title="Test Article",
                    content="Test content",
                    source=self.source_name,
                    timestamp=datetime.now(),
                    url="https://example.com/1",
                    entities=["TEST"],
                    metadata={}
                )
            ]
        
        async def stream(self):
            yield NewsItem(
                id="stream-1",
                title="Streaming Article",
                content="Streaming content",
                source=self.source_name,
                timestamp=datetime.now(),
                url="https://example.com/stream",
                entities=["STREAM"],
                metadata={}
            )
    
    source = ConcreteSource("test-source")
    items = await source.fetch_latest(limit=10)
    assert len(items) == 1
    assert items[0].title == "Test Article"


@pytest.mark.asyncio
async def test_news_source_stream():
    """Test abstract stream method"""
    from src.news.sources import NewsSource
    from src.news.models import NewsItem
    
    class StreamingSource(NewsSource):
        async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
            return []
        
        async def stream(self):
            for i in range(3):
                yield NewsItem(
                    id=f"stream-{i}",
                    title=f"Article {i}",
                    content=f"Content {i}",
                    source=self.source_name,
                    timestamp=datetime.now(),
                    url=f"https://example.com/{i}",
                    entities=[],
                    metadata={}
                )
    
    source = StreamingSource("streaming-test")
    items = []
    async for item in source.stream():
        items.append(item)
    
    assert len(items) == 3
    assert items[0].id == "stream-0"
    assert items[2].title == "Article 2"


def test_news_source_initialization():
    """Test NewsSource base class initialization"""
    from src.news.sources import NewsSource
    
    class TestSource(NewsSource):
        def __init__(self, source_name: str, api_key: str = None):
            super().__init__(source_name)
            self.api_key = api_key
        
        async def fetch_latest(self, limit: int = 100):
            return []
        
        async def stream(self):
            return
            yield  # Make it a generator
    
    source = TestSource("test", api_key="secret")
    assert source.source_name == "test"
    assert source.api_key == "secret"
    assert hasattr(source, '_cache')  # Should have cache after refactoring