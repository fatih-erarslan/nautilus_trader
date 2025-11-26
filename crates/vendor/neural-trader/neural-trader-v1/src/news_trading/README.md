# News Trading Modules

## Overview

The News Trading package provides two main modules for AI-powered trading:

1. **News Collection Module**: Aggregates real-time news from multiple sources with deduplication
2. **News Parsing Module**: Extracts structured information using NLP techniques

## News Collection Module

### Features
- Multi-source news aggregation (stocks, bonds, crypto)
- SEC filings parser (13F, Form 4) for mirror trading
- Treasury/Fed announcement parsers
- Technical breakout alert detection
- Real-time streaming support
- Intelligent deduplication
- 95% test coverage

### Usage Example

```python
from news_trading.news_collection import NewsAggregator
from news_trading.news_collection.sources import (
    YahooFinanceEnhancedSource,
    SECFilingsSource,
    TreasuryEnhancedSource,
    TechnicalNewsSource
)

# Initialize sources
sources = [
    YahooFinanceEnhancedSource(),
    SECFilingsSource(),
    TreasuryEnhancedSource(),
    TechnicalNewsSource()
]

# Create aggregator
aggregator = NewsAggregator(sources, deduplicate=True)

# Fetch latest news
news_items = await aggregator.fetch_all(limit_per_source=50)

# Process news for trading signals
for item in news_items:
    print(f"Source: {item.source}")
    print(f"Title: {item.title}")
    print(f"Entities: {item.entities}")
    
    # Check for trading opportunities
    if item.metadata.get("mirror_trade_opportunity"):
        print("Mirror trading opportunity detected!")
    
    if item.metadata.get("swing_trade_signal"):
        print(f"Swing trade signal: {item.metadata['technical_indicators']}")
```

### Available News Sources

1. **YahooFinanceEnhancedSource**
   - Earnings reports with momentum detection
   - Market news and analysis
   - Entity extraction (companies, tickers)

2. **SECFilingsSource**
   - Form 4 insider trading detection
   - 13F institutional holdings analysis
   - Mirror trading opportunities

3. **TreasuryEnhancedSource**
   - Treasury auction results
   - Yield analysis
   - Demand strength assessment

4. **FederalReserveEnhancedSource**
   - FOMC announcements
   - Rate decisions
   - Market impact analysis

5. **TechnicalNewsSource**
   - Technical breakout alerts
   - Support/resistance levels
   - Swing trading signals

## News Parsing Module

### Features
- Entity extraction (crypto, companies, people, organizations, locations)
- Event detection (price movements, regulatory, partnerships, launches, breaches)
- Temporal reference extraction and normalization
- Sentiment analysis
- Key phrase extraction
- 95% test coverage

### Usage Example

```python
from news_trading.news_parsing import NLPParser

# Initialize parser
parser = NLPParser()

# Parse a news article
content = """
Bitcoin surged past $50,000 yesterday as MicroStrategy announced 
another major purchase. The SEC's recent comments on crypto regulation 
have boosted market confidence, with Ethereum also reaching new highs.
"""

result = await parser.parse(content)

# Access extracted information
print("Entities found:")
for entity in result.entities:
    print(f"  - {entity.text} ({entity.type.value})")
    if entity.ticker:
        print(f"    Ticker: {entity.ticker}")

print("\nEvents detected:")
for event in result.events:
    print(f"  - {event.type.value}: {event.description}")
    print(f"    Confidence: {event.confidence:.2f}")
    print(f"    Entities involved: {event.entities_involved}")

print(f"\nSentiment: {result.sentiment_indicators}")
print(f"Key phrases: {result.key_phrases}")
print(f"Time references: {result.temporal_references}")
```

### Parsed Data Structure

```python
ParsedArticle:
    original_id: str                    # Unique identifier
    entities: List[Entity]              # Extracted entities
    events: List[Event]                 # Detected events
    sentiment_indicators: List[str]     # Sentiment signals
    key_phrases: List[str]             # Important phrases
    temporal_references: List[str]      # Time references
    metadata: Dict[str, Any]           # Additional metadata
```

### Entity Types
- CRYPTO: Cryptocurrencies (Bitcoin, Ethereum, etc.)
- COMPANY: Companies (Apple, Tesla, etc.)
- PERSON: People (Elon Musk, Warren Buffett, etc.)
- ORGANIZATION: Organizations (SEC, Federal Reserve, etc.)
- LOCATION: Locations (United States, Wall Street, etc.)

### Event Types
- PRICE_MOVEMENT: Surges, crashes, rises, falls
- REGULATORY: Approvals, bans, investigations
- PARTNERSHIP: Partnerships, collaborations
- PRODUCT_LAUNCH: Launches, releases, updates
- SECURITY_BREACH: Hacks, vulnerabilities

## Integration Example

```python
# Complete pipeline example
async def process_trading_news():
    # Setup sources and aggregator
    sources = [
        YahooFinanceEnhancedSource(),
        SECFilingsSource(),
        TechnicalNewsSource()
    ]
    aggregator = NewsAggregator(sources)
    parser = NLPParser()
    
    # Fetch and parse news
    news_items = await aggregator.fetch_all(limit_per_source=20)
    
    trading_signals = []
    
    for item in news_items:
        # Parse the content
        parsed = await parser.parse(item.content, {"source": item.source})
        
        # Analyze for trading opportunities
        signal = {
            "source": item.source,
            "timestamp": item.timestamp,
            "entities": [e.ticker for e in parsed.entities if e.ticker],
            "events": [e.type.value for e in parsed.events],
            "sentiment": parsed.sentiment_indicators[0] if parsed.sentiment_indicators else "neutral"
        }
        
        # Check metadata for specific signals
        if item.metadata.get("mirror_trade_opportunity"):
            signal["type"] = "mirror_trade"
            signal["confidence"] = "high"
        elif item.metadata.get("swing_trade_signal"):
            signal["type"] = "swing_trade"
            signal["confidence"] = item.metadata.get("signal_strength", 0.5)
        elif "surge" in [e.description for e in parsed.events]:
            signal["type"] = "momentum"
            signal["confidence"] = "medium"
        
        if "type" in signal:
            trading_signals.append(signal)
    
    return trading_signals
```

## Testing

Run tests with coverage:

```bash
# Test news collection module
pytest tests/news_collection/ -v --cov=src/news_trading/news_collection

# Test news parsing module  
pytest tests/news_parsing/ -v --cov=src/news_trading/news_parsing

# Run all tests
pytest tests/news_collection/ tests/news_parsing/ -v
```

## Performance

- News aggregation: >1000 articles/second
- Entity extraction: ~100ms per article
- Event detection: ~50ms per article
- Deduplication: O(n log n) complexity
- Caching: 5-minute TTL by default