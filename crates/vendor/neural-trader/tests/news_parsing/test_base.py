"""Tests for core news parser interface - RED phase"""

import pytest
from datetime import datetime
from typing import Dict
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')


def test_parser_interface():
    """Test that NewsParser abstract interface is properly defined"""
    from news_trading.news_parsing.base import NewsParser
    
    class TestParser(NewsParser):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        parser = TestParser()


def test_parsed_article_model():
    """Test ParsedArticle data model"""
    from news_trading.news_parsing.models import ParsedArticle, Entity, Event, EntityType, EventType
    
    article = ParsedArticle(
        original_id="test-123",
        entities=[
            Entity(text="Bitcoin", type=EntityType.CRYPTO, ticker="BTC", confidence=0.95),
            Entity(text="Elon Musk", type=EntityType.PERSON, confidence=0.98)
        ],
        events=[
            Event(type=EventType.PRICE_MOVEMENT, description="surge", confidence=0.85)
        ],
        sentiment_indicators=["bullish", "positive momentum"],
        key_phrases=["all-time high", "institutional adoption"],
        temporal_references=["yesterday", "Q1 2024"]
    )
    
    assert len(article.entities) == 2
    assert article.entities[0].ticker == "BTC"
    assert article.events[0].type == EventType.PRICE_MOVEMENT


def test_entity_types():
    """Test entity type enumeration"""
    from news_trading.news_parsing.models import EntityType
    
    assert EntityType.CRYPTO.value == "CRYPTO"
    assert EntityType.COMPANY.value == "COMPANY"
    assert EntityType.PERSON.value == "PERSON"
    assert EntityType.ORGANIZATION.value == "ORGANIZATION"
    assert EntityType.LOCATION.value == "LOCATION"


def test_event_types():
    """Test event type enumeration"""
    from news_trading.news_parsing.models import EventType
    
    assert EventType.PRICE_MOVEMENT.value == "PRICE_MOVEMENT"
    assert EventType.REGULATORY.value == "REGULATORY"
    assert EventType.PARTNERSHIP.value == "PARTNERSHIP"
    assert EventType.PRODUCT_LAUNCH.value == "PRODUCT_LAUNCH"
    assert EventType.SECURITY_BREACH.value == "SECURITY_BREACH"


@pytest.mark.asyncio
async def test_parser_with_implementation():
    """Test NewsParser with proper implementation"""
    from news_trading.news_parsing.base import NewsParser
    from news_trading.news_parsing.models import ParsedArticle, Entity, EntityType
    
    class TestParser(NewsParser):
        async def parse(self, content: str, metadata: Dict = None) -> ParsedArticle:
            return ParsedArticle(
                original_id="test",
                entities=[Entity(text="Test", type=EntityType.COMPANY, confidence=1.0)],
                events=[],
                sentiment_indicators=[],
                key_phrases=[],
                temporal_references=[]
            )
        
        async def parse_batch(self, articles: list[str]) -> list[ParsedArticle]:
            return [await self.parse(article) for article in articles]
    
    parser = TestParser()
    result = await parser.parse("Test content")
    assert result.original_id == "test"
    assert len(result.entities) == 1