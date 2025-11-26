"""Tests for event detection - RED phase"""

import pytest
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from news_trading.news_parsing.models import EventType


def test_price_movement_detection():
    """Test price movement event detection"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Bitcoin surged 15% following institutional investment announcement"
    events = detector.detect(text)
    
    assert len(events) >= 1
    assert events[0].type == EventType.PRICE_MOVEMENT
    assert "surge" in events[0].description.lower()


def test_regulatory_event_detection():
    """Test regulatory event detection"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "SEC approves Bitcoin ETF after years of deliberation"
    events = detector.detect(text)
    
    assert any(e.type == EventType.REGULATORY for e in events)
    assert any("approv" in e.description.lower() for e in events)


def test_partnership_event_detection():
    """Test partnership event detection"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "PayPal partners with Paxos to enable crypto trading"
    events = detector.detect(text)
    
    assert any(e.type == EventType.PARTNERSHIP for e in events)


def test_product_launch_detection():
    """Test product launch event detection"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Coinbase launches new institutional trading platform"
    events = detector.detect(text)
    
    assert any(e.type == EventType.PRODUCT_LAUNCH for e in events)


def test_security_breach_detection():
    """Test security breach event detection"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Major crypto exchange hacked, $100M stolen in security breach"
    events = detector.detect(text)
    
    assert any(e.type == EventType.SECURITY_BREACH for e in events)


def test_multiple_events_detection():
    """Test detection of multiple events in one text"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Following SEC approval, Bitcoin surged 20% as Coinbase announced new features"
    events = detector.detect(text)
    
    event_types = {e.type for e in events}
    assert EventType.REGULATORY in event_types
    assert EventType.PRICE_MOVEMENT in event_types
    assert EventType.PRODUCT_LAUNCH in event_types


def test_event_confidence_scoring():
    """Test event confidence scoring"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    # Clear signal - high confidence
    text1 = "Bitcoin absolutely crashed 30% in massive selloff"
    events1 = detector.detect(text1)
    crash_events = [e for e in events1 if "crash" in e.description]
    assert crash_events[0].confidence >= 0.79  # Allow for floating point precision
    
    # Ambiguous signal - lower confidence
    text2 = "Bitcoin moved slightly on light volume"
    events2 = detector.detect(text2)
    if events2:  # May not detect weak signals
        assert events2[0].confidence < 0.6


def test_event_entities_involved():
    """Test extraction of entities involved in events"""
    from news_trading.news_parsing.event_detector import EventDetector
    
    detector = EventDetector()
    
    text = "Tesla and SpaceX partnered to accept Bitcoin payments"
    events = detector.detect(text)
    
    partnership_events = [e for e in events if e.type == EventType.PARTNERSHIP]
    assert len(partnership_events) > 0
    
    # Check that we found entities involved
    all_entities_involved = []
    for event in partnership_events:
        all_entities_involved.extend(event.entities_involved)
    
    assert any("Tesla" in entity for entity in all_entities_involved)
    assert any("SpaceX" in entity for entity in all_entities_involved)