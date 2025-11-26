"""
Sentiment Analysis Data Models

This module defines the data structures used throughout the sentiment analysis system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class SentimentDirection(Enum):
    """Direction of market sentiment"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    
    @classmethod
    def from_score(cls, score: float) -> 'SentimentDirection':
        """
        Determine sentiment direction from a score
        
        Args:
            score: Sentiment score between -1 and 1
            
        Returns:
            SentimentDirection enum value
        """
        if score > 0.3:
            return cls.BULLISH
        elif score < -0.3:
            return cls.BEARISH
        else:
            return cls.NEUTRAL


@dataclass
class MarketImpact:
    """
    Market impact prediction from sentiment analysis
    
    Attributes:
        direction: Expected market direction
        magnitude: Expected impact magnitude (0-1)
        timeframe: Impact timeframe (immediate, short-term, long-term)
        affected_assets: List of assets likely to be affected
        confidence: Confidence in the prediction (0-1)
        volatility_expected: Expected volatility level
        catalysts: Key factors driving the impact
    """
    direction: SentimentDirection
    magnitude: float  # 0 to 1
    timeframe: str  # "immediate", "short-term", "long-term"
    affected_assets: List[str]
    confidence: float = 0.5
    volatility_expected: str = "medium"  # "low", "medium", "high", "extreme"
    catalysts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate MarketImpact data"""
        if not 0 <= self.magnitude <= 1:
            raise ValueError(f"Magnitude must be between 0 and 1, got {self.magnitude}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def is_significant(self) -> bool:
        """
        Check if the market impact is significant enough for trading
        
        Returns:
            True if impact is significant
        """
        return (
            self.magnitude > 0.5 and
            self.confidence > 0.7 and
            self.direction != SentimentDirection.NEUTRAL
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "direction": self.direction.value,
            "magnitude": self.magnitude,
            "timeframe": self.timeframe,
            "affected_assets": self.affected_assets,
            "confidence": self.confidence,
            "volatility_expected": self.volatility_expected,
            "catalysts": self.catalysts
        }


@dataclass
class SentimentBreakdown:
    """
    Detailed breakdown of sentiment components
    
    Attributes:
        headline: Sentiment of the headline (-1 to 1)
        content: Sentiment of the content (-1 to 1)
        entities: Sentiment around mentioned entities (-1 to 1)
        tone: Overall tone sentiment (-1 to 1)
        language_intensity: Intensity of language used (-1 to 1)
    """
    headline: float = 0.0
    content: float = 0.0
    entities: float = 0.0
    tone: float = 0.0
    language_intensity: float = 0.0
    
    def __post_init__(self):
        """Validate sentiment scores"""
        for field_name, value in [
            ("headline", self.headline),
            ("content", self.content),
            ("entities", self.entities),
            ("tone", self.tone),
            ("language_intensity", self.language_intensity)
        ]:
            if value is not None and not -1 <= value <= 1:
                raise ValueError(f"{field_name} scores must be between -1 and 1, got {value}")
    
    @property
    def average(self) -> float:
        """Calculate average sentiment across all components"""
        components = [
            self.headline,
            self.content,
            self.entities,
            self.tone,
            self.language_intensity
        ]
        # Filter out None values
        valid_components = [c for c in components if c is not None]
        return sum(valid_components) / len(valid_components) if valid_components else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation"""
        return {
            "headline": self.headline,
            "content": self.content,
            "entities": self.entities,
            "tone": self.tone,
            "language_intensity": self.language_intensity,
            "average": self.average
        }


@dataclass
class SentimentResult:
    """
    Complete sentiment analysis result
    
    Attributes:
        article_id: Unique identifier for the analyzed article
        overall_sentiment: Overall sentiment score (-1 to 1)
        confidence: Confidence in the analysis (0 to 1)
        market_impact: Predicted market impact
        sentiment_breakdown: Detailed sentiment breakdown
        reasoning: Human-readable reasoning for the sentiment
        model_scores: Individual model scores (for ensemble)
        processing_time: Time taken for analysis in seconds
        source_metadata: Additional metadata about the source
    """
    article_id: str
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    market_impact: MarketImpact
    sentiment_breakdown: SentimentBreakdown
    reasoning: str
    model_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate SentimentResult data"""
        if not -1 <= self.overall_sentiment <= 1:
            raise ValueError(f"Overall sentiment must be between -1 and 1, got {self.overall_sentiment}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def is_tradeable(self, min_confidence: float = 0.7) -> bool:
        """
        Check if the sentiment signal is strong enough for trading
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if the signal is tradeable
        """
        return (
            self.confidence >= min_confidence and
            self.market_impact.is_significant() and
            abs(self.overall_sentiment) > 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "article_id": self.article_id,
            "overall_sentiment": self.overall_sentiment,
            "confidence": self.confidence,
            "market_impact": self.market_impact.to_dict(),
            "sentiment_breakdown": self.sentiment_breakdown.to_dict(),
            "reasoning": self.reasoning,
            "model_scores": self.model_scores,
            "processing_time": self.processing_time,
            "source_metadata": self.source_metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_tradeable": self.is_tradeable()
        }