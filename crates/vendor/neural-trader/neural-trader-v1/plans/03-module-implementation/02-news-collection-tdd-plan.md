# News Collection Module - TDD Implementation Plan

## Module Overview
The News Collection module is responsible for aggregating real-time news from multiple sources with specific focus on stock and bond markets, deduplicating content, and preparing it for downstream processing. This module supports swing trading, momentum trading, and mirror trading strategies by collecting market-moving news, earnings reports, bond yield changes, and institutional trading patterns.

## Test-First Implementation Sequence

### Phase 1: Core News Source Interface (Red-Green-Refactor)

#### RED: Write failing tests first

```python
# tests/test_news_collection.py

def test_news_source_interface():
    """Test that NewsSource abstract interface is properly defined"""
    from src.news.sources import NewsSource
    
    class TestSource(NewsSource):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        source = TestSource()

def test_news_item_model():
    """Test NewsItem data model"""
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
```

#### GREEN: Implement minimal code to pass

```python
# src/news/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

@dataclass
class NewsItem:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    entities: List[str]
    metadata: Dict[str, any]

# src/news/sources.py
from abc import ABC, abstractmethod
from typing import List, AsyncIterator
from .models import NewsItem

class NewsSource(ABC):
    @abstractmethod
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest news items"""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream news items in real-time"""
        pass
```

#### REFACTOR: Improve design

```python
# Add validation, caching, and error handling
class NewsSource(ABC):
    def __init__(self, source_name: str):
        self.source_name = source_name
        self._cache = TTLCache(maxsize=1000, ttl=300)
        
    async def fetch_latest_with_cache(self, limit: int = 100) -> List[NewsItem]:
        cache_key = f"{self.source_name}:latest:{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        items = await self.fetch_latest(limit)
        self._cache[cache_key] = items
        return items
```

### Phase 2: Multi-Source Stock & Bond News Implementation

#### Sources for Trading Strategies:
1. **Stock Market Sources**
   - Reuters Business & Markets
   - Bloomberg RSS feeds (free tier)
   - Yahoo Finance (earnings, analyst ratings)
   - SEC EDGAR filings (8-K, 10-Q reports)
   - CNBC Market News
   - MarketWatch RSS

2. **Bond Market Sources**
   - Treasury Direct announcements
   - Federal Reserve economic data (FRED)
   - Bond yield trackers
   - Central bank communications

3. **Institutional & Mirror Trading Sources**
   - SEC Form 4 insider trading filings
   - 13F institutional holdings reports
   - Options flow data (unusual activity)
   - Dark pool prints

### Phase 2A: Stock Market News Source

#### RED: Test Reuters integration

```python
def test_reuters_source_init():
    """Test Reuters news source initialization"""
    from src.news.sources.reuters import ReutersSource
    
    source = ReutersSource(api_key="test-key")
    assert source.source_name == "reuters"
    assert source.api_key == "test-key"

@pytest.mark.asyncio
async def test_reuters_fetch_latest():
    """Test fetching latest Reuters articles"""
    from src.news.sources.reuters import ReutersSource
    
    source = ReutersSource(api_key="test-key")
    
    # Mock the API response
    with aioresponses() as mocked:
        mocked.get(
            "https://api.reuters.com/v1/articles/latest",
            payload={
                "articles": [{
                    "id": "reuters-001",
                    "headline": "Fed Signals Rate Changes",
                    "body": "Federal Reserve...",
                    "publishedAt": "2024-01-15T10:00:00Z",
                    "url": "https://reuters.com/article/001"
                }]
            }
        )
        
        items = await source.fetch_latest(limit=10)
        assert len(items) == 1
        assert items[0].title == "Fed Signals Rate Changes"
        assert "TSLA" in items[0].entities  # Test entity extraction

@pytest.mark.asyncio
async def test_bond_market_news_parsing():
    """Test parsing bond market and yield news"""
    from src.news.sources.treasury import TreasurySource
    
    source = TreasurySource()
    
    # Mock Treasury announcement
    with aioresponses() as mocked:
        mocked.get(
            "https://api.treasurydirect.gov/announcements",
            payload={
                "announcements": [{
                    "id": "treas-001",
                    "title": "10-Year Treasury Auction Results",
                    "yield": "4.25%",
                    "bid_to_cover": "2.5",
                    "datetime": "2024-01-15T14:00:00Z"
                }]
            }
        )
        
        items = await source.fetch_latest()
        assert items[0]['bond_type'] == '10-Year Treasury'
        assert items[0]['yield_change'] is not None

@pytest.mark.asyncio
async def test_institutional_trading_detection():
    """Test detection of institutional trading patterns for mirror trading"""
    from src.news.sources.sec_filings import SECFilingsSource
    
    source = SECFilingsSource()
    
    # Mock Form 4 insider trading
    filing = {
        "form_type": "4",
        "filer": "Berkshire Hathaway",
        "transaction": "BUY",
        "ticker": "AAPL",
        "shares": 1000000,
        "avg_price": 150.00
    }
    
    signal = await source.analyze_filing(filing)
    assert signal['mirror_trade_opportunity'] == True
    assert signal['institution_sentiment'] == 'bullish'
```

#### GREEN: Implement Reuters source

```python
# src/news/sources/reuters.py
class ReutersSource(NewsSource):
    def __init__(self, api_key: str):
        super().__init__("reuters")
        self.api_key = api_key
        self.base_url = "https://api.reuters.com/v1"
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/articles/latest?limit={limit}"
            
            async with session.get(url, headers=headers) as response:
                data = await response.json()
                
                return [
                    NewsItem(
                        id=f"reuters-{article['id']}",
                        title=article['headline'],
                        content=article['body'],
                        source=self.source_name,
                        timestamp=datetime.fromisoformat(article['publishedAt']),
                        url=article['url'],
                        entities=self._extract_entities(article['body']),
                        metadata={"author": article.get('author')}
                    )
                    for article in data['articles']
                ]
```

### Phase 3: News Aggregator

#### RED: Test aggregator functionality

```python
def test_news_aggregator_init():
    """Test NewsAggregator initialization"""
    from src.news.aggregator import NewsAggregator
    from src.news.sources import NewsSource
    
    source1 = Mock(spec=NewsSource)
    source2 = Mock(spec=NewsSource)
    
    aggregator = NewsAggregator([source1, source2])
    assert len(aggregator.sources) == 2

@pytest.mark.asyncio
async def test_aggregator_fetch_all():
    """Test fetching from all sources"""
    # Test concurrent fetching
    # Test deduplication
    # Test error handling for failed sources
```

### Phase 4: Deduplication Logic

#### RED: Test deduplication

```python
def test_deduplicate_by_content_similarity():
    """Test deduplication using content similarity"""
    from src.news.deduplication import deduplicate_news
    
    items = [
        NewsItem(id="1", title="Bitcoin Hits $50k", content="BTC reaches new high..."),
        NewsItem(id="2", title="BTC Reaches $50,000", content="Bitcoin reaches new high..."),
        NewsItem(id="3", title="Ethereum Updates", content="ETH protocol changes...")
    ]
    
    unique_items = deduplicate_news(items, threshold=0.8)
    assert len(unique_items) == 2  # Bitcoin articles should be merged
```

## Interface Contracts and API Design

### NewsSource Interface
```python
class NewsSource(ABC):
    """Abstract base class for all news sources"""
    
    @abstractmethod
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest news items"""
        
    @abstractmethod
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream news items in real-time"""
        
    @abstractmethod
    async def search(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        """Search historical news"""
```

### NewsAggregator API
```python
class NewsAggregator:
    """Aggregates news from multiple sources"""
    
    async def fetch_all(self, limit_per_source: int = 50) -> List[NewsItem]:
        """Fetch from all sources concurrently"""
        
    async def stream_all(self) -> AsyncIterator[NewsItem]:
        """Merge streams from all sources"""
        
    def add_source(self, source: NewsSource) -> None:
        """Add a new news source"""
        
    def remove_source(self, source_name: str) -> None:
        """Remove a news source"""
```

## Dependency Injection Points

1. **News Sources**: Injectable via configuration
2. **Cache Backend**: Redis, in-memory, or custom
3. **Rate Limiter**: Configurable per source
4. **Metrics Collector**: For monitoring source health

## Mock Object Specifications

### MockNewsSource
```python
class MockNewsSource(NewsSource):
    def __init__(self, items: List[NewsItem]):
        super().__init__("mock")
        self.items = items
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        return self.items[:limit]
```

### MockAPIResponse
```python
def mock_reuters_response(article_count: int = 5):
    return {
        "articles": [
            {
                "id": f"mock-{i}",
                "headline": f"Test Article {i}",
                "body": f"Content for article {i}",
                "publishedAt": datetime.now().isoformat(),
                "url": f"https://example.com/{i}"
            }
            for i in range(article_count)
        ]
    }
```

## Refactoring Checkpoints

1. **After Phase 1**: Review interface design, ensure extensibility
2. **After Phase 2**: Extract common HTTP client logic
3. **After Phase 3**: Optimize concurrent fetching performance
4. **After Phase 4**: Review deduplication algorithm efficiency

## Code Coverage Targets

- **Unit Tests**: 95% coverage minimum
- **Integration Tests**: 80% coverage for API interactions
- **Edge Cases**: 100% coverage for error handling paths
- **Performance Tests**: Benchmark for 1000+ articles/second

## Implementation Timeline

1. **Day 1-2**: Core interfaces and models (Phase 1)
2. **Day 3-4**: Reuters source implementation (Phase 2)
3. **Day 5-6**: News aggregator (Phase 3)
4. **Day 7**: Deduplication logic (Phase 4)
5. **Day 8**: Integration testing and refactoring

## Trading Strategy Specific Collection

### Swing Trading Data Collection
```python
def test_swing_trading_signals():
    """Test collection of swing trading relevant news"""
    from src.news.trading_strategies.swing_collector import SwingTradingCollector
    
    collector = SwingTradingCollector()
    
    # Test technical breakout news
    news = {
        "headline": "Apple breaks 200-day moving average",
        "content": "AAPL stock surges past key technical level...",
        "indicators": ["200_MA", "volume_spike", "breakout"]
    }
    
    relevance = collector.assess_swing_relevance(news)
    assert relevance['score'] > 0.8
    assert relevance['holding_period'] == '3-10 days'
```

### Momentum Trading Data Collection
```python
def test_momentum_indicators():
    """Test momentum trading signal detection"""
    from src.news.trading_strategies.momentum_collector import MomentumCollector
    
    collector = MomentumCollector()
    
    # Test earnings momentum
    news = {
        "headline": "Tesla beats earnings by 40%, raises guidance",
        "earnings_surprise": 0.40,
        "guidance_change": "raised",
        "analyst_revisions": 15
    }
    
    momentum_score = collector.calculate_momentum(news)
    assert momentum_score > 0.85
    assert collector.suggested_entry_timing() == 'immediate'
```

### Mirror Trading Detection
```python
def test_mirror_trade_identification():
    """Test identification of trades worth mirroring"""
    from src.news.trading_strategies.mirror_detector import MirrorTradeDetector
    
    detector = MirrorTradeDetector()
    
    # Test Buffett-style value purchase
    filing = {
        "institution": "Berkshire Hathaway",
        "action": "purchase",
        "ticker": "BAC",
        "position_change": 0.15,  # 15% increase
        "total_value": 2500000000  # $2.5B
    }
    
    mirror_signal = detector.analyze_institutional_move(filing)
    assert mirror_signal['confidence'] > 0.9
    assert mirror_signal['follow_strategy'] == 'scaled_entry'
    assert mirror_signal['risk_management']['stop_loss'] == 0.05
```

## Success Criteria

- [ ] All tests pass with >95% coverage
- [ ] Can fetch from 15+ news sources concurrently (stocks, bonds, filings)
- [ ] Deduplication accuracy >90%
- [ ] Processing speed >1000 articles/second
- [ ] Graceful handling of source failures
- [ ] Real-time detection of swing/momentum/mirror opportunities
- [ ] Bond market coverage includes all major treasury auctions
- [ ] Institutional filing processing within 1 minute of publication
- [ ] Technical indicator news detection accuracy >85%
- [ ] Comprehensive logging and monitoring