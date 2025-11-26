"""Tests for news deduplication - RED phase"""

import pytest
from datetime import datetime
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from news.models import NewsItem


def test_deduplicate_by_content_similarity():
    """Test deduplication using content similarity"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    items = [
        NewsItem(
            id="1", 
            title="Bitcoin Hits $50k", 
            content="BTC reaches new high at fifty thousand dollars...",
            source="source1",
            timestamp=datetime.now(),
            url="http://url1.com",
            entities=["BTC"],
            metadata={}
        ),
        NewsItem(
            id="2", 
            title="BTC Reaches $50,000", 
            content="Bitcoin reaches new high at fifty thousand dollars...",
            source="source2",
            timestamp=datetime.now(),
            url="http://url2.com",
            entities=["BTC"],
            metadata={}
        ),
        NewsItem(
            id="3", 
            title="Ethereum Updates", 
            content="ETH protocol changes announced...",
            source="source3",
            timestamp=datetime.now(),
            url="http://url3.com",
            entities=["ETH"],
            metadata={}
        )
    ]
    
    unique_items = deduplicate_news(items, threshold=0.8)
    assert len(unique_items) == 2  # Bitcoin articles should be merged


def test_deduplicate_exact_duplicates():
    """Test deduplication of exact duplicates"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    items = [
        NewsItem(
            id="1", 
            title="Same Title", 
            content="Same content",
            source="source1",
            timestamp=datetime.now(),
            url="http://url1.com",
            entities=["AAPL"],
            metadata={}
        ),
        NewsItem(
            id="2", 
            title="Same Title", 
            content="Same content",
            source="source2",
            timestamp=datetime.now(),
            url="http://url2.com",
            entities=["AAPL"],
            metadata={}
        )
    ]
    
    unique_items = deduplicate_news(items)
    assert len(unique_items) == 1


def test_deduplicate_preserves_earliest():
    """Test that deduplication preserves the earliest article"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    earlier_time = datetime(2024, 1, 1, 10, 0, 0)
    later_time = datetime(2024, 1, 1, 11, 0, 0)
    
    items = [
        NewsItem(
            id="later", 
            title="Bitcoin News", 
            content="Bitcoin reaches new high",
            source="source1",
            timestamp=later_time,
            url="http://later.com",
            entities=["BTC"],
            metadata={}
        ),
        NewsItem(
            id="earlier", 
            title="BTC News", 
            content="Bitcoin reaches new high",
            source="source2",
            timestamp=earlier_time,
            url="http://earlier.com",
            entities=["BTC"],
            metadata={}
        )
    ]
    
    unique_items = deduplicate_news(items)
    assert len(unique_items) == 1
    assert unique_items[0].id == "earlier"  # Should keep the earlier one


def test_deduplicate_different_threshold():
    """Test deduplication with different similarity thresholds"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    items = [
        NewsItem(
            id="1", 
            title="Apple Reports Q4 Earnings", 
            content="Apple Inc reported fourth quarter earnings today",
            source="source1",
            timestamp=datetime.now(),
            url="http://url1.com",
            entities=["AAPL"],
            metadata={}
        ),
        NewsItem(
            id="2", 
            title="AAPL Q4 Results", 
            content="Apple reported Q4 results this morning",
            source="source2",
            timestamp=datetime.now(),
            url="http://url2.com",
            entities=["AAPL"],
            metadata={}
        )
    ]
    
    # High threshold - should not deduplicate
    unique_high = deduplicate_news(items, threshold=0.95)
    assert len(unique_high) == 2
    
    # Low threshold - should deduplicate
    unique_low = deduplicate_news(items, threshold=0.5)
    assert len(unique_low) == 1


def test_deduplicate_empty_list():
    """Test deduplication with empty list"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    unique_items = deduplicate_news([])
    assert len(unique_items) == 0


def test_deduplicate_single_item():
    """Test deduplication with single item"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    items = [
        NewsItem(
            id="1", 
            title="Single Article", 
            content="Content",
            source="source1",
            timestamp=datetime.now(),
            url="http://url1.com",
            entities=[],
            metadata={}
        )
    ]
    
    unique_items = deduplicate_news(items)
    assert len(unique_items) == 1
    assert unique_items[0].id == "1"


def test_deduplicate_merges_metadata():
    """Test that deduplication merges metadata from duplicates"""
    from news_trading.news_collection.deduplication import deduplicate_news
    
    items = [
        NewsItem(
            id="1", 
            title="Bitcoin News", 
            content="Bitcoin reaches new high",
            source="source1",
            timestamp=datetime.now(),
            url="http://url1.com",
            entities=["BTC"],
            metadata={"sentiment": "positive"}
        ),
        NewsItem(
            id="2", 
            title="BTC News", 
            content="Bitcoin reaches new high",
            source="source2",
            timestamp=datetime.now(),
            url="http://url2.com",
            entities=["BTC", "crypto"],
            metadata={"technical_signal": "breakout"}
        )
    ]
    
    unique_items = deduplicate_news(items, merge_metadata=True)
    assert len(unique_items) == 1
    
    # Check merged metadata
    merged_item = unique_items[0]
    assert "BTC" in merged_item.entities
    assert "crypto" in merged_item.entities
    assert merged_item.metadata.get("sentiment") == "positive"
    assert merged_item.metadata.get("technical_signal") == "breakout"
    assert "duplicate_sources" in merged_item.metadata
    assert len(merged_item.metadata["duplicate_sources"]) == 2