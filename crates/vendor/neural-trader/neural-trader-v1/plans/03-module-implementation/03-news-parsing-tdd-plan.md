# News Parsing Module - TDD Implementation Plan

## Module Overview
The News Parsing module extracts structured information from raw news content, including entities (companies, cryptocurrencies, people), events, timestamps, and relevant metadata using NLP techniques.

## Test-First Implementation Sequence

### Phase 1: Core Parser Interface (Red-Green-Refactor)

#### RED: Write failing tests first

```python
# tests/test_news_parser.py

def test_parser_interface():
    """Test that NewsParser abstract interface is properly defined"""
    from src.news.parsing import NewsParser
    
    class TestParser(NewsParser):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        parser = TestParser()

def test_parsed_article_model():
    """Test ParsedArticle data model"""
    from src.news.parsing.models import ParsedArticle, Entity, Event
    
    article = ParsedArticle(
        original_id="test-123",
        entities=[
            Entity(text="Bitcoin", type="CRYPTO", ticker="BTC", confidence=0.95),
            Entity(text="Elon Musk", type="PERSON", confidence=0.98)
        ],
        events=[
            Event(type="PRICE_MOVEMENT", description="surge", confidence=0.85)
        ],
        sentiment_indicators=["bullish", "positive momentum"],
        key_phrases=["all-time high", "institutional adoption"],
        temporal_references=["yesterday", "Q1 2024"]
    )
    
    assert len(article.entities) == 2
    assert article.entities[0].ticker == "BTC"
    assert article.events[0].type == "PRICE_MOVEMENT"
```

#### GREEN: Implement minimal code to pass

```python
# src/news/parsing/models.py
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

class EntityType(Enum):
    CRYPTO = "CRYPTO"
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"

class EventType(Enum):
    PRICE_MOVEMENT = "PRICE_MOVEMENT"
    REGULATORY = "REGULATORY"
    PARTNERSHIP = "PARTNERSHIP"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    SECURITY_BREACH = "SECURITY_BREACH"

@dataclass
class Entity:
    text: str
    type: EntityType
    confidence: float
    ticker: Optional[str] = None
    metadata: Dict[str, any] = None

@dataclass
class Event:
    type: EventType
    description: str
    confidence: float
    entities_involved: List[str] = None

@dataclass
class ParsedArticle:
    original_id: str
    entities: List[Entity]
    events: List[Event]
    sentiment_indicators: List[str]
    key_phrases: List[str]
    temporal_references: List[str]

# src/news/parsing/base.py
from abc import ABC, abstractmethod
from typing import List
from .models import ParsedArticle

class NewsParser(ABC):
    @abstractmethod
    async def parse(self, content: str, metadata: Dict = None) -> ParsedArticle:
        """Parse news content into structured format"""
        pass
```

### Phase 2: Entity Extraction

#### RED: Test entity extraction

```python
def test_crypto_entity_extraction():
    """Test cryptocurrency entity extraction"""
    from src.news.parsing.extractors import CryptoEntityExtractor
    
    extractor = CryptoEntityExtractor()
    
    text = "Bitcoin (BTC) surged past $50,000 while Ethereum reached new ATH"
    entities = extractor.extract(text)
    
    assert len(entities) == 2
    assert entities[0].text == "Bitcoin"
    assert entities[0].ticker == "BTC"
    assert entities[0].type == EntityType.CRYPTO
    assert entities[1].text == "Ethereum"
    assert entities[1].ticker == "ETH"

def test_company_entity_extraction():
    """Test company entity extraction"""
    from src.news.parsing.extractors import CompanyEntityExtractor
    
    extractor = CompanyEntityExtractor()
    
    text = "MicroStrategy announced another Bitcoin purchase while Tesla holds steady"
    entities = extractor.extract(text)
    
    assert any(e.text == "MicroStrategy" for e in entities)
    assert any(e.text == "Tesla" for e in entities)
```

#### GREEN: Implement entity extractors

```python
# src/news/parsing/extractors.py
import re
from typing import List
from .models import Entity, EntityType

class CryptoEntityExtractor:
    def __init__(self):
        self.crypto_patterns = {
            r'\bBitcoin\b|\bBTC\b': ('Bitcoin', 'BTC'),
            r'\bEthereum\b|\bETH\b': ('Ethereum', 'ETH'),
            r'\bBinance Coin\b|\bBNB\b': ('Binance Coin', 'BNB'),
            # Add more patterns
        }
        
    def extract(self, text: str) -> List[Entity]:
        entities = []
        
        for pattern, (name, ticker) in self.crypto_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=name,
                    type=EntityType.CRYPTO,
                    ticker=ticker,
                    confidence=0.95 if ticker in text else 0.85
                ))
                
        return self._deduplicate_entities(entities)
```

### Phase 3: Event Detection

#### RED: Test event detection

```python
def test_price_movement_detection():
    """Test price movement event detection"""
    from src.news.parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Bitcoin surged 15% following institutional investment announcement"
    events = detector.detect(text)
    
    assert len(events) >= 1
    assert events[0].type == EventType.PRICE_MOVEMENT
    assert "surge" in events[0].description.lower()

def test_regulatory_event_detection():
    """Test regulatory event detection"""
    text = "SEC approves Bitcoin ETF after years of deliberation"
    events = detector.detect(text)
    
    assert any(e.type == EventType.REGULATORY for e in events)
```

#### GREEN: Implement event detection

```python
# src/news/parsing/event_detector.py
class EventDetector:
    def __init__(self):
        self.event_patterns = {
            EventType.PRICE_MOVEMENT: {
                'surge': r'\b(surg\w+|soar\w+|jump\w+|spike\w+)\b',
                'crash': r'\b(crash\w+|plummet\w+|plunge\w+|tumbl\w+)\b',
                'rise': r'\b(ris\w+|climb\w+|gain\w+)\b',
                'fall': r'\b(fall\w+|drop\w+|declin\w+)\b'
            },
            EventType.REGULATORY: {
                'approval': r'\b(approv\w+|authoriz\w+|permit\w+)\b',
                'ban': r'\b(ban\w+|prohibit\w+|outlaw\w+)\b',
                'regulation': r'\b(regulat\w+|SEC|CFTC|compliance)\b'
            }
        }
        
    def detect(self, text: str) -> List[Event]:
        events = []
        
        for event_type, patterns in self.event_patterns.items():
            for description, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    events.append(Event(
                        type=event_type,
                        description=description,
                        confidence=self._calculate_confidence(text, pattern)
                    ))
                    
        return events
```

### Phase 4: Temporal Reference Extraction

#### RED: Test temporal extraction

```python
def test_temporal_reference_extraction():
    """Test extraction of time references"""
    from src.news.parsing.temporal import TemporalExtractor
    
    extractor = TemporalExtractor()
    
    text = "Bitcoin hit ATH yesterday, up 50% since last month"
    refs = extractor.extract(text)
    
    assert "yesterday" in refs
    assert "last month" in refs
    
def test_temporal_normalization():
    """Test normalization of temporal references"""
    from src.news.parsing.temporal import normalize_temporal
    from datetime import datetime, timedelta
    
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    assert normalize_temporal("yesterday", base_time) == base_time - timedelta(days=1)
    assert normalize_temporal("last week", base_time).date() == (base_time - timedelta(days=7)).date()
```

### Phase 5: Full Parser Implementation

#### RED: Test complete parser

```python
@pytest.mark.asyncio
async def test_nlp_parser_full():
    """Test full NLP-based parser"""
    from src.news.parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    content = """
    Bitcoin surged past $50,000 yesterday as MicroStrategy announced 
    another major purchase. The SEC's recent comments on crypto regulation 
    have boosted market confidence, with Ethereum also reaching new highs.
    """
    
    result = await parser.parse(content)
    
    # Check entities
    assert any(e.ticker == "BTC" for e in result.entities)
    assert any(e.text == "MicroStrategy" for e in result.entities)
    
    # Check events
    assert any(e.type == EventType.PRICE_MOVEMENT for e in result.events)
    assert any(e.type == EventType.REGULATORY for e in result.events)
    
    # Check temporal
    assert "yesterday" in result.temporal_references
```

## Interface Contracts and API Design

### NewsParser Interface
```python
class NewsParser(ABC):
    """Abstract base class for news parsers"""
    
    @abstractmethod
    async def parse(self, content: str, metadata: Dict = None) -> ParsedArticle:
        """Parse news content into structured format"""
        
    @abstractmethod
    async def parse_batch(self, articles: List[str]) -> List[ParsedArticle]:
        """Parse multiple articles efficiently"""
        
    def set_entity_extractors(self, extractors: List[EntityExtractor]):
        """Configure entity extractors"""
        
    def set_event_detector(self, detector: EventDetector):
        """Configure event detector"""
```

### EntityExtractor Interface
```python
class EntityExtractor(ABC):
    """Base class for entity extractors"""
    
    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        
    @abstractmethod
    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports"""
```

## Dependency Injection Points

1. **NLP Models**: SpaCy, Transformers, or custom models
2. **Entity Databases**: Crypto tickers, company names
3. **Event Patterns**: Configurable pattern sets
4. **Language Models**: For context understanding

## Mock Object Specifications

### MockNLPModel
```python
class MockNLPModel:
    def __init__(self, entities=None, pos_tags=None):
        self.entities = entities or []
        self.pos_tags = pos_tags or []
        
    def process(self, text):
        return MockDocument(text, self.entities, self.pos_tags)
```

### MockEntityDatabase
```python
class MockEntityDatabase:
    def __init__(self):
        self.cryptos = {
            "Bitcoin": "BTC",
            "Ethereum": "ETH",
            "Cardano": "ADA"
        }
        self.companies = {
            "MicroStrategy": "MSTR",
            "Tesla": "TSLA",
            "Coinbase": "COIN"
        }
```

## Refactoring Checkpoints

1. **After Phase 2**: Consolidate entity extraction patterns
2. **After Phase 3**: Optimize event detection performance
3. **After Phase 4**: Review temporal normalization accuracy
4. **After Phase 5**: Extract common NLP preprocessing

## Code Coverage Targets

- **Unit Tests**: 95% coverage for all extractors
- **Integration Tests**: 90% coverage for full parser
- **Edge Cases**: 100% coverage for malformed input
- **Performance Tests**: Parse 100 articles/second

## Implementation Timeline

1. **Day 1**: Core interfaces and models
2. **Day 2-3**: Entity extraction (crypto, company, person)
3. **Day 4**: Event detection system
4. **Day 5**: Temporal reference extraction
5. **Day 6-7**: Full parser integration
6. **Day 8**: Performance optimization and testing

## Success Criteria

- [ ] Entity extraction F1 score > 0.90
- [ ] Event detection accuracy > 85%
- [ ] Temporal normalization accuracy > 95%
- [ ] Support for multiple languages
- [ ] Real-time parsing capability
- [ ] Comprehensive error handling