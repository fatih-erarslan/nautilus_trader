"""Tests for full NLP parser - RED phase"""

import pytest
from datetime import datetime
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from news_trading.news_parsing.models import EntityType, EventType


@pytest.mark.asyncio
async def test_nlp_parser_full():
    """Test full NLP-based parser"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
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
    assert any(e.ticker == "ETH" for e in result.entities)
    assert any(e.text == "SEC" for e in result.entities)
    
    # Check events
    assert any(e.type == EventType.PRICE_MOVEMENT for e in result.events)
    assert any(e.type == EventType.REGULATORY for e in result.events)
    
    # Check temporal
    assert "yesterday" in result.temporal_references


@pytest.mark.asyncio
async def test_nlp_parser_sentiment():
    """Test sentiment analysis in parser"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    # Positive sentiment
    positive_content = "Bitcoin breaks all-time high! Institutional adoption accelerating rapidly."
    positive_result = await parser.parse(positive_content)
    assert any("positive" in indicator for indicator in positive_result.sentiment_indicators)
    
    # Negative sentiment
    negative_content = "Bitcoin crashes below support. Investors panic selling amid regulatory fears."
    negative_result = await parser.parse(negative_content)
    assert any("negative" in indicator or "bearish" in indicator for indicator in negative_result.sentiment_indicators)


@pytest.mark.asyncio
async def test_nlp_parser_key_phrases():
    """Test key phrase extraction"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    content = """
    Breaking: Federal Reserve maintains interest rates unchanged.
    Bitcoin reaches new all-time high amid institutional buying pressure.
    Technical analysis shows strong support at $45,000 level.
    """
    
    result = await parser.parse(content)
    
    # Should extract important phrases
    assert any("all-time high" in phrase.lower() for phrase in result.key_phrases)
    assert any("interest rates" in phrase.lower() for phrase in result.key_phrases)
    assert any("support" in phrase.lower() for phrase in result.key_phrases)


@pytest.mark.asyncio
async def test_nlp_parser_batch():
    """Test batch parsing"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    articles = [
        "Bitcoin hits $50k milestone",
        "Ethereum upgrade successful", 
        "SEC approves crypto ETF"
    ]
    
    results = await parser.parse_batch(articles)
    
    assert len(results) == 3
    assert all(r.original_id for r in results)
    assert results[0].entities[0].ticker in ["BTC", "Bitcoin"]
    assert results[1].entities[0].ticker in ["ETH", "Ethereum"]
    assert any(e.type == EventType.REGULATORY for e in results[2].events)


@pytest.mark.asyncio
async def test_nlp_parser_metadata():
    """Test parser with metadata"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    content = "Tesla announces Bitcoin purchase"
    metadata = {
        "source": "reuters",
        "author": "John Doe",
        "published": "2024-01-15T10:00:00Z"
    }
    
    result = await parser.parse(content, metadata)
    
    assert result.original_id is not None
    # Parser should use metadata to enhance extraction
    assert any(e.text == "Tesla" for e in result.entities)
    assert any(e.ticker == "BTC" for e in result.entities)


@pytest.mark.asyncio
async def test_nlp_parser_complex_content():
    """Test parser with complex real-world content"""
    from news_trading.news_parsing.nlp_parser import NLPParser
    
    parser = NLPParser()
    
    content = """
    In a surprising move yesterday, MicroStrategy (MSTR) announced the purchase of 
    an additional 5,000 Bitcoin at an average price of $45,500, bringing their 
    total holdings to over 150,000 BTC. CEO Michael Saylor tweeted that the 
    company remains bullish on cryptocurrency despite recent volatility.
    
    The announcement came just hours after the Federal Reserve's latest meeting,
    where Chairman Jerome Powell indicated that interest rates would remain 
    unchanged through Q2 2024. This dovish stance has been interpreted as 
    positive for risk assets, including cryptocurrencies.
    
    Meanwhile, Ethereum continues to show strength ahead of next month's upgrade,
    with ETH/USD breaking above the key $3,000 resistance level. Technical 
    analysts point to increasing on-chain activity and reduced exchange reserves
    as bullish indicators for the second-largest cryptocurrency.
    """
    
    result = await parser.parse(content)
    
    # Entities
    entities_text = {e.text for e in result.entities}
    assert "MicroStrategy" in entities_text
    assert "Bitcoin" in entities_text
    assert "Michael Saylor" in entities_text
    assert "Federal Reserve" in entities_text
    assert "Jerome Powell" in entities_text
    assert "Ethereum" in entities_text
    
    # Events  
    event_types = {e.type for e in result.events}
    assert EventType.PRICE_MOVEMENT in event_types
    assert EventType.REGULATORY in event_types
    
    # Temporal references
    assert any("yesterday" in ref for ref in result.temporal_references)
    assert any("Q2 2024" in ref for ref in result.temporal_references)
    assert any("next month" in ref for ref in result.temporal_references)
    
    # Key phrases
    assert len(result.key_phrases) > 0
    
    # Sentiment
    assert len(result.sentiment_indicators) > 0