"""Data models for news parsing - GREEN phase"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class EntityType(Enum):
    """Types of entities that can be extracted"""
    CRYPTO = "CRYPTO"
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"


class EventType(Enum):
    """Types of events that can be detected"""
    PRICE_MOVEMENT = "PRICE_MOVEMENT"
    REGULATORY = "REGULATORY"
    PARTNERSHIP = "PARTNERSHIP"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    SECURITY_BREACH = "SECURITY_BREACH"


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    type: EntityType
    confidence: float
    ticker: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity data"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Event:
    """Represents a detected event"""
    type: EventType
    description: str
    confidence: float
    entities_involved: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event data"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.entities_involved is None:
            self.entities_involved = []
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParsedArticle:
    """Represents a fully parsed news article"""
    original_id: str
    entities: List[Entity]
    events: List[Event]
    sentiment_indicators: List[str]
    key_phrases: List[str]
    temporal_references: List[str]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parsed article data"""
        if not self.original_id:
            raise ValueError("original_id cannot be empty")
        
        # Ensure all lists are initialized
        if self.entities is None:
            self.entities = []
        if self.events is None:
            self.events = []
        if self.sentiment_indicators is None:
            self.sentiment_indicators = []
        if self.key_phrases is None:
            self.key_phrases = []
        if self.temporal_references is None:
            self.temporal_references = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "original_id": self.original_id,
            "entities": [
                {
                    "text": e.text,
                    "type": e.type.value,
                    "ticker": e.ticker,
                    "confidence": e.confidence,
                    "metadata": e.metadata
                }
                for e in self.entities
            ],
            "events": [
                {
                    "type": e.type.value,
                    "description": e.description,
                    "confidence": e.confidence,
                    "entities_involved": e.entities_involved,
                    "metadata": e.metadata
                }
                for e in self.events
            ],
            "sentiment_indicators": self.sentiment_indicators,
            "key_phrases": self.key_phrases,
            "temporal_references": self.temporal_references,
            "metadata": self.metadata
        }