"""Tests for entity extraction - RED phase"""

import pytest
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from news_trading.news_parsing.models import EntityType


def test_crypto_entity_extraction():
    """Test cryptocurrency entity extraction"""
    from news_trading.news_parsing.extractors import CryptoEntityExtractor
    
    extractor = CryptoEntityExtractor()
    
    text = "Bitcoin (BTC) surged past $50,000 while Ethereum reached new ATH"
    entities = extractor.extract(text)
    
    assert len(entities) == 2
    assert entities[0].text == "Bitcoin"
    assert entities[0].ticker == "BTC"
    assert entities[0].type == EntityType.CRYPTO
    assert entities[1].text == "Ethereum"
    assert entities[1].ticker == "ETH"


def test_crypto_entity_confidence():
    """Test confidence scoring for crypto entities"""
    from news_trading.news_parsing.extractors import CryptoEntityExtractor
    
    extractor = CryptoEntityExtractor()
    
    # With ticker symbol - high confidence
    text1 = "Bitcoin (BTC) is trading at $50,000"
    entities1 = extractor.extract(text1)
    assert entities1[0].confidence > 0.9
    
    # Without ticker - lower confidence
    text2 = "Bitcoin is trading at $50,000"
    entities2 = extractor.extract(text2)
    assert entities2[0].confidence < 0.9


def test_company_entity_extraction():
    """Test company entity extraction"""
    from news_trading.news_parsing.extractors import CompanyEntityExtractor
    
    extractor = CompanyEntityExtractor()
    
    text = "MicroStrategy announced another Bitcoin purchase while Tesla holds steady"
    entities = extractor.extract(text)
    
    assert any(e.text == "MicroStrategy" and e.ticker == "MSTR" for e in entities)
    assert any(e.text == "Tesla" and e.ticker == "TSLA" for e in entities)
    assert all(e.type == EntityType.COMPANY for e in entities)


def test_person_entity_extraction():
    """Test person entity extraction"""
    from news_trading.news_parsing.extractors import PersonEntityExtractor
    
    extractor = PersonEntityExtractor()
    
    text = "Elon Musk tweeted about crypto while Warren Buffett remains skeptical"
    entities = extractor.extract(text)
    
    assert any(e.text == "Elon Musk" for e in entities)
    assert any(e.text == "Warren Buffett" for e in entities)
    assert all(e.type == EntityType.PERSON for e in entities)


def test_mixed_entity_extraction():
    """Test extraction of mixed entity types"""
    from news_trading.news_parsing.extractors import UnifiedEntityExtractor
    
    extractor = UnifiedEntityExtractor()
    
    text = "Apple CEO Tim Cook announced Bitcoin purchases while meeting with SEC officials in Washington"
    entities = extractor.extract(text)
    
    # Check we found all entity types
    entity_types = {e.type for e in entities}
    assert EntityType.COMPANY in entity_types  # Apple
    assert EntityType.PERSON in entity_types   # Tim Cook
    assert EntityType.CRYPTO in entity_types   # Bitcoin
    assert EntityType.ORGANIZATION in entity_types  # SEC
    assert EntityType.LOCATION in entity_types  # Washington


def test_entity_deduplication():
    """Test that duplicate entities are removed"""
    from news_trading.news_parsing.extractors import CryptoEntityExtractor
    
    extractor = CryptoEntityExtractor()
    
    text = "Bitcoin is rising. BTC hit $50k. Bitcoin dominance increasing."
    entities = extractor.extract(text)
    
    # Should only have one Bitcoin entity
    bitcoin_entities = [e for e in entities if e.ticker == "BTC"]
    assert len(bitcoin_entities) == 1