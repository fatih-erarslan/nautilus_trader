"""Full NLP-based news parser - GREEN phase"""

import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base import NewsParser
from .models import ParsedArticle
from .extractors import UnifiedEntityExtractor
from .event_detector import EventDetector
from .temporal import TemporalExtractor

logger = logging.getLogger(__name__)


class NLPParser(NewsParser):
    """Full NLP parser implementation"""
    
    def __init__(self):
        super().__init__()
        self.entity_extractor = UnifiedEntityExtractor()
        self.event_detector = EventDetector()
        self.temporal_extractor = TemporalExtractor()
        
        # Sentiment indicators
        self.positive_indicators = [
            'surge', 'soar', 'rally', 'gain', 'rise', 'bullish', 'positive',
            'breakthrough', 'success', 'growth', 'profit', 'record high',
            'optimistic', 'confident', 'strong', 'robust', 'accelerate'
        ]
        
        self.negative_indicators = [
            'crash', 'plunge', 'fall', 'drop', 'decline', 'bearish', 'negative',
            'loss', 'failure', 'weak', 'concern', 'fear', 'risk', 'volatile',
            'pessimistic', 'uncertain', 'fragile', 'slowdown', 'recession'
        ]
        
        # Key phrase patterns
        self.key_phrase_patterns = [
            r'\b(all[- ]time high|ATH)\b',
            r'\b(all[- ]time low|ATL)\b',
            r'\b(support level|resistance level)\b',
            r'\b(moving average)\b',
            r'\b(market cap|market capitalization)\b',
            r'\b(trading volume)\b',
            r'\b(interest rate[s]?)\b',
            r'\b(earnings report)\b',
            r'\b(technical analysis)\b',
            r'\b(fundamental[s]?)\b',
            r'\b(institutional (buying|selling|investor[s]?))\b',
            r'\b(retail investor[s]?)\b',
            r'\b(bull market|bear market)\b',
            r'\b(breakout|breakdown)\b',
            r'\b(consolidation|accumulation)\b',
        ]
        
    async def parse(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ParsedArticle:
        """Parse news content into structured format"""
        # Generate ID
        if metadata and 'id' in metadata:
            article_id = metadata['id']
        else:
            article_id = str(uuid.uuid4())
        
        # Extract entities
        entities = self.entity_extractor.extract(content)
        
        # Detect events
        events = self.event_detector.detect(content)
        
        # Extract temporal references
        temporal_refs = self.temporal_extractor.extract(content)
        
        # Analyze sentiment
        sentiment_indicators = self._analyze_sentiment(content)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(content)
        
        # Build metadata
        article_metadata = metadata.copy() if metadata else {}
        article_metadata.update({
            'parsed_at': datetime.now().isoformat(),
            'content_length': len(content),
            'entity_count': len(entities),
            'event_count': len(events)
        })
        
        return ParsedArticle(
            original_id=article_id,
            entities=entities,
            events=events,
            sentiment_indicators=sentiment_indicators,
            key_phrases=key_phrases,
            temporal_references=temporal_refs,
            metadata=article_metadata
        )
    
    async def parse_batch(self, articles: List[str]) -> List[ParsedArticle]:
        """Parse multiple articles efficiently"""
        results = []
        
        for i, article in enumerate(articles):
            metadata = {'batch_index': i}
            result = await self.parse(article, metadata)
            results.append(result)
        
        return results
    
    def _analyze_sentiment(self, text: str) -> List[str]:
        """Analyze sentiment indicators in text"""
        indicators = []
        text_lower = text.lower()
        
        # Count positive and negative indicators
        positive_count = sum(1 for indicator in self.positive_indicators 
                           if indicator in text_lower)
        negative_count = sum(1 for indicator in self.negative_indicators 
                           if indicator in text_lower)
        
        # Determine overall sentiment
        if positive_count > negative_count * 1.5:
            indicators.append("bullish")
            indicators.append("positive")
        elif negative_count > positive_count * 1.5:
            indicators.append("bearish")
            indicators.append("negative")
        else:
            indicators.append("neutral")
        
        # Add specific strong indicators found
        for indicator in self.positive_indicators:
            if indicator in text_lower and len(indicators) < 5:
                indicators.append(f"positive_{indicator.replace(' ', '_')}")
        
        for indicator in self.negative_indicators:
            if indicator in text_lower and len(indicators) < 5:
                indicators.append(f"negative_{indicator.replace(' ', '_')}")
        
        # Look for momentum indicators
        if any(word in text_lower for word in ['momentum', 'accelerat', 'gaining']):
            indicators.append("positive momentum")
        
        return list(set(indicators))[:10]  # Limit to 10 indicators
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        phrases = []
        
        # Extract pattern-based phrases
        for pattern in self.key_phrase_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip()
                if phrase not in phrases:
                    phrases.append(phrase)
        
        # Extract percentage changes
        percentage_pattern = r'\b\d+(?:\.\d+)?%'
        for match in re.finditer(percentage_pattern, text):
            context_start = max(0, match.start() - 20)
            context_end = min(len(text), match.end() + 20)
            context = text[context_start:context_end].strip()
            if context not in phrases:
                phrases.append(context)
        
        # Extract price levels
        price_pattern = r'\$[\d,]+(?:\.\d{2})?(?:[KMB])?'
        for match in re.finditer(price_pattern, text):
            context_start = max(0, match.start() - 15)
            context_end = min(len(text), match.end() + 15)
            context = text[context_start:context_end].strip()
            if context not in phrases and len(phrases) < 20:
                phrases.append(context)
        
        return phrases[:20]  # Limit to 20 key phrases