"""Tests for news aggregator - RED phase"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
import asyncio
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from news.models import NewsItem


def test_news_aggregator_init():
    """Test NewsAggregator initialization"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    source1 = Mock(spec=NewsSource)
    source1.source_name = "source1"
    source2 = Mock(spec=NewsSource)
    source2.source_name = "source2"
    
    aggregator = NewsAggregator([source1, source2])
    assert len(aggregator.sources) == 2


def test_aggregator_add_remove_sources():
    """Test adding and removing news sources"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    source1 = Mock(spec=NewsSource)
    source1.source_name = "source1"
    source2 = Mock(spec=NewsSource)
    source2.source_name = "source2"
    
    aggregator = NewsAggregator([source1])
    assert len(aggregator.sources) == 1
    
    aggregator.add_source(source2)
    assert len(aggregator.sources) == 2
    
    aggregator.remove_source("source1")
    assert len(aggregator.sources) == 1
    assert aggregator.sources[0].source_name == "source2"


@pytest.mark.asyncio
async def test_aggregator_fetch_all():
    """Test fetching from all sources concurrently"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    # Create mock sources
    source1 = Mock(spec=NewsSource)
    source1.source_name = "source1"
    source1.fetch_latest = AsyncMock(return_value=[
        NewsItem(
            id="s1-1",
            title="Article 1",
            content="Content 1",
            source="source1",
            timestamp=datetime.now(),
            url="http://s1.com/1",
            entities=["AAPL"],
            metadata={}
        )
    ])
    
    source2 = Mock(spec=NewsSource)
    source2.source_name = "source2"
    source2.fetch_latest = AsyncMock(return_value=[
        NewsItem(
            id="s2-1",
            title="Article 2",
            content="Content 2",
            source="source2",
            timestamp=datetime.now(),
            url="http://s2.com/1",
            entities=["GOOGL"],
            metadata={}
        )
    ])
    
    aggregator = NewsAggregator([source1, source2])
    
    # Fetch from all sources
    items = await aggregator.fetch_all(limit_per_source=10)
    
    assert len(items) == 2
    assert source1.fetch_latest.called
    assert source2.fetch_latest.called
    assert any(item.source == "source1" for item in items)
    assert any(item.source == "source2" for item in items)


@pytest.mark.asyncio
async def test_aggregator_error_handling():
    """Test error handling when a source fails"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    # Create sources - one fails, one succeeds
    failing_source = Mock(spec=NewsSource)
    failing_source.source_name = "failing"
    failing_source.fetch_latest = AsyncMock(side_effect=Exception("API Error"))
    
    working_source = Mock(spec=NewsSource)
    working_source.source_name = "working"
    working_source.fetch_latest = AsyncMock(return_value=[
        NewsItem(
            id="w-1",
            title="Working Article",
            content="Content",
            source="working",
            timestamp=datetime.now(),
            url="http://working.com/1",
            entities=[],
            metadata={}
        )
    ])
    
    aggregator = NewsAggregator([failing_source, working_source])
    
    # Should still return results from working source
    items = await aggregator.fetch_all()
    
    assert len(items) == 1
    assert items[0].source == "working"
    assert failing_source.fetch_latest.called
    assert working_source.fetch_latest.called


@pytest.mark.asyncio
async def test_aggregator_deduplication():
    """Test that aggregator deduplicates similar articles"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    # Create sources with duplicate content
    source1 = Mock(spec=NewsSource)
    source1.source_name = "source1"
    source1.fetch_latest = AsyncMock(return_value=[
        NewsItem(
            id="s1-1",
            title="Bitcoin Hits $50,000",
            content="Bitcoin cryptocurrency reached $50,000 today in heavy trading volume on major exchanges.",
            source="source1",
            timestamp=datetime.now(),
            url="http://s1.com/1",
            entities=["BTC"],
            metadata={}
        )
    ])
    
    source2 = Mock(spec=NewsSource)
    source2.source_name = "source2"
    source2.fetch_latest = AsyncMock(return_value=[
        NewsItem(
            id="s2-1",
            title="BTC Reaches $50k Milestone",
            content="Bitcoin cryptocurrency reached $50,000 today in heavy trading volume on major exchanges.",
            source="source2",
            timestamp=datetime.now(),
            url="http://s2.com/1",
            entities=["BTC"],
            metadata={}
        )
    ])
    
    aggregator = NewsAggregator([source1, source2], deduplicate=True)
    
    # Fetch from all sources
    items = await aggregator.fetch_all()
    
    # Should only have one item after deduplication
    assert len(items) == 1
    assert items[0].entities == ["BTC"]


@pytest.mark.asyncio
async def test_aggregator_stream_all():
    """Test streaming from multiple sources"""
    from news_trading.news_collection.aggregator import NewsAggregator
    from news_trading.news_collection.base import NewsSource
    
    # Create mock streaming sources
    async def stream_source1():
        yield NewsItem(
            id="stream-1",
            title="Stream 1",
            content="Content",
            source="source1",
            timestamp=datetime.now(),
            url="http://s1.com/stream",
            entities=[],
            metadata={}
        )
    
    async def stream_source2():
        yield NewsItem(
            id="stream-2",
            title="Stream 2",
            content="Content",
            source="source2",
            timestamp=datetime.now(),
            url="http://s2.com/stream",
            entities=[],
            metadata={}
        )
    
    source1 = Mock(spec=NewsSource)
    source1.source_name = "source1"
    source1.stream = stream_source1
    
    source2 = Mock(spec=NewsSource)
    source2.source_name = "source2"
    source2.stream = stream_source2
    
    aggregator = NewsAggregator([source1, source2])
    
    # Collect streamed items
    items = []
    async for item in aggregator.stream_all():
        items.append(item)
        if len(items) >= 2:  # Stop after collecting 2 items
            break
    
    assert len(items) == 2
    assert any(item.source == "source1" for item in items)
    assert any(item.source == "source2" for item in items)