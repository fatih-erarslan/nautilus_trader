"""
AI Sentiment Analysis Module

This module provides advanced sentiment analysis capabilities for financial news
using an ensemble approach with multiple AI models including:
- Transformer-based models (FinBERT)
- Large Language Models (LLMs)
- Crypto-specific pattern detection
- Market impact prediction

The module follows a test-driven development approach with comprehensive
unit and integration testing.
"""

from .base import SentimentAnalyzer
from .models import (
    SentimentResult,
    MarketImpact,
    SentimentDirection,
    SentimentBreakdown
)

__all__ = [
    'SentimentAnalyzer',
    'SentimentResult',
    'MarketImpact',
    'SentimentDirection',
    'SentimentBreakdown'
]

# Import implementations as they become available
try:
    from .transformer_sentiment import TransformerSentiment
    __all__.append('TransformerSentiment')
except ImportError:
    pass

try:
    from .llm_sentiment import LLMSentimentAnalyzer
    __all__.append('LLMSentimentAnalyzer')
except ImportError:
    pass

try:
    from .ensemble import EnsembleSentiment
    __all__.append('EnsembleSentiment')
except ImportError:
    pass

try:
    from .crypto_patterns import CryptoPatternAnalyzer
    __all__.append('CryptoPatternAnalyzer')
except ImportError:
    pass

try:
    from .market_impact import MarketImpactPredictor
    __all__.append('MarketImpactPredictor')
except ImportError:
    pass

__version__ = '1.0.0'