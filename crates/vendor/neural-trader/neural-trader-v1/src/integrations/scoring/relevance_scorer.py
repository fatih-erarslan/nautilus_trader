"""
News Relevance Scoring System
Multi-factor scoring system for prioritizing news items by relevance and importance
"""

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..news_aggregator import UnifiedNewsItem

logger = logging.getLogger(__name__)


class NewsRelevanceScorer:
    """Multi-factor relevance scoring for news items"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Scoring weights (should sum to 1.0)
        self.weights = {
            'symbol_relevance': self.config.get('symbol_weight', 0.35),
            'sentiment_strength': self.config.get('sentiment_weight', 0.20),
            'source_reliability': self.config.get('source_weight', 0.20),
            'recency': self.config.get('recency_weight', 0.15),
            'impact_indicators': self.config.get('impact_weight', 0.10)
        }
        
        # Validation
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total_weight}, not 1.0")
        
        # High-impact keywords
        self.impact_keywords = {
            'earnings': 1.0,
            'merger': 0.9,
            'acquisition': 0.9,
            'ipo': 0.8,
            'sec filing': 0.7,
            'fda approval': 0.8,
            'bankruptcy': 0.9,
            'lawsuit': 0.6,
            'ceo': 0.5,
            'dividend': 0.4,
            'split': 0.5,
            'guidance': 0.6,
            'downgrade': 0.7,
            'upgrade': 0.7
        }
        
        # Financial context keywords
        self.financial_keywords = {
            'revenue', 'profit', 'loss', 'earnings', 'eps', 'shares',
            'stock', 'market', 'trading', 'volume', 'price', 'valuation',
            'forecast', 'guidance', 'quarter', 'annual', 'financial'
        }
        
        # Sector importance multipliers
        self.sector_multipliers = {
            'technology': 1.2,
            'healthcare': 1.1,
            'finance': 1.1,
            'energy': 1.0,
            'consumer': 0.9,
            'industrial': 0.8,
            'utilities': 0.7
        }
        
        logger.info("Initialized news relevance scorer")
    
    async def score_items(
        self,
        news_items: List[UnifiedNewsItem],
        target_symbols: List[str],
        user_preferences: Dict[str, Any] = None
    ) -> List[Tuple[UnifiedNewsItem, float]]:
        """
        Score news items by relevance and return sorted list
        
        Args:
            news_items: List of news items to score
            target_symbols: Symbols user is interested in
            user_preferences: User-specific preferences
            
        Returns:
            List of (news_item, score) tuples sorted by score (descending)
        """
        if not news_items:
            return []
        
        logger.debug(f"Scoring {len(news_items)} news items for {len(target_symbols)} symbols")
        
        scored_items = []
        
        for item in news_items:
            try:
                score = self._calculate_comprehensive_score(
                    item, 
                    target_symbols, 
                    user_preferences
                )
                scored_items.append((item, score))
                
            except Exception as e:
                logger.warning(f"Error scoring news item: {e}")
                # Give low score if error
                scored_items.append((item, 0.1))
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Log scoring summary
        if scored_items:
            avg_score = sum(score for _, score in scored_items) / len(scored_items)
            max_score = scored_items[0][1]
            min_score = scored_items[-1][1]
            
            logger.info(
                f"Scored {len(scored_items)} items: "
                f"avg={avg_score:.3f}, max={max_score:.3f}, min={min_score:.3f}"
            )
        
        return scored_items
    
    def _calculate_comprehensive_score(
        self,
        item: UnifiedNewsItem,
        target_symbols: List[str],
        user_preferences: Dict[str, Any] = None
    ) -> float:
        """Calculate comprehensive relevance score for a news item"""
        
        # Calculate individual component scores
        symbol_score = self._calculate_symbol_relevance(item, target_symbols)
        sentiment_score = self._calculate_sentiment_strength(item)
        source_score = item.source_reliability
        recency_score = self._calculate_recency_score(item.published_at)
        impact_score = self._calculate_impact_score(item)
        
        # Apply weights
        weighted_score = (
            symbol_score * self.weights['symbol_relevance'] +
            sentiment_score * self.weights['sentiment_strength'] +
            source_score * self.weights['source_reliability'] +
            recency_score * self.weights['recency'] +
            impact_score * self.weights['impact_indicators']
        )
        
        # Apply user preferences if provided
        if user_preferences:
            preference_multiplier = self._apply_user_preferences(item, user_preferences)
            weighted_score *= preference_multiplier
        
        # Apply sector boost
        sector_multiplier = self._get_sector_multiplier(item)
        weighted_score *= sector_multiplier
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, weighted_score))
        
        logger.debug(
            f"Score breakdown for '{item.title[:50]}...': "
            f"symbol={symbol_score:.3f}, sentiment={sentiment_score:.3f}, "
            f"source={source_score:.3f}, recency={recency_score:.3f}, "
            f"impact={impact_score:.3f}, final={final_score:.3f}"
        )
        
        return final_score
    
    def _calculate_symbol_relevance(
        self,
        item: UnifiedNewsItem,
        target_symbols: List[str]
    ) -> float:
        """Calculate how relevant the news is to target symbols"""
        if not target_symbols:
            return 0.5  # Neutral score if no specific symbols
        
        total_relevance = 0.0
        max_possible_relevance = len(target_symbols)
        
        target_symbols_upper = [s.upper() for s in target_symbols]
        
        for symbol in target_symbols_upper:
            symbol_relevance = 0.0
            
            # Direct symbol match in extracted symbols
            if symbol in [s.upper() for s in item.symbols]:
                symbol_relevance += 0.8
                
                # Use pre-computed relevance score if available
                if symbol in item.relevance_scores:
                    symbol_relevance = max(symbol_relevance, item.relevance_scores[symbol])
            
            # Title mentions (high importance)
            title_lower = item.title.lower()
            if symbol.lower() in title_lower:
                symbol_relevance += 0.6
                
                # Boost if symbol is prominent in title
                title_words = title_lower.split()
                if symbol.lower() in title_words[:5]:  # First 5 words
                    symbol_relevance += 0.2
            
            # Summary mentions
            if item.summary and symbol.lower() in item.summary.lower():
                symbol_relevance += 0.3
            
            # Content mentions (if available)
            if item.content and symbol.lower() in item.content.lower():
                symbol_relevance += 0.2
            
            # Contextual mentions
            text_to_search = f"{item.title} {item.summary or ''}"
            
            context_patterns = [
                f"{symbol.lower()} stock",
                f"{symbol.lower()} shares",
                f"{symbol.lower()} earnings",
                f"${symbol.upper()}",
                f"({symbol.upper()})"
            ]
            
            for pattern in context_patterns:
                if pattern in text_to_search.lower():
                    symbol_relevance += 0.1
            
            # Cap individual symbol relevance at 1.0
            symbol_relevance = min(1.0, symbol_relevance)
            total_relevance += symbol_relevance
        
        # Calculate final score (0-1 range)
        if max_possible_relevance > 0:
            final_score = total_relevance / max_possible_relevance
        else:
            final_score = 0.0
        
        return min(1.0, final_score)
    
    def _calculate_sentiment_strength(self, item: UnifiedNewsItem) -> float:
        """Calculate sentiment strength score"""
        if item.sentiment is None:
            return 0.5  # Neutral if no sentiment
        
        # Convert sentiment to strength (absolute value)
        strength = abs(item.sentiment)
        
        # Apply non-linear scaling to emphasize strong sentiment
        # Using sigmoid-like function
        scaled_strength = 2 * strength / (1 + strength)
        
        return min(1.0, scaled_strength)
    
    def _calculate_recency_score(self, published_at: datetime) -> float:
        """Calculate recency score using exponential decay"""
        now = datetime.now()
        
        # Handle timezone-naive datetime
        if published_at.tzinfo is None and now.tzinfo is not None:
            published_at = published_at.replace(tzinfo=now.tzinfo)
        elif published_at.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=published_at.tzinfo)
        
        hours_old = (now - published_at).total_seconds() / 3600
        
        # Exponential decay with half-life of 6 hours
        half_life_hours = 6
        decay_constant = math.log(2) / half_life_hours
        
        recency_score = math.exp(-decay_constant * hours_old)
        
        # Boost for very recent news (last hour)
        if hours_old < 1:
            recency_score = min(1.0, recency_score + 0.2)
        
        return recency_score
    
    def _calculate_impact_score(self, item: UnifiedNewsItem) -> float:
        """Calculate potential market impact score"""
        impact_score = 0.0
        
        # Check for high-impact keywords
        text_to_analyze = f"{item.title} {item.summary or ''}".lower()
        
        for keyword, weight in self.impact_keywords.items():
            if keyword in text_to_analyze:
                impact_score += weight * 0.3  # Scale down individual keyword impact
        
        # Check categories for impact
        if item.categories:
            high_impact_categories = {
                'earnings': 0.8,
                'merger': 0.9,
                'ipo': 0.7,
                'regulatory': 0.6,
                'analyst': 0.5
            }
            
            for category in item.categories:
                if category.lower() in high_impact_categories:
                    impact_score += high_impact_categories[category.lower()] * 0.4
        
        # Boost for multiple symbols (indicates broader market relevance)
        if len(item.symbols) > 1:
            impact_score += min(0.3, len(item.symbols) * 0.1)
        
        # Financial context boost
        financial_word_count = sum(
            1 for word in self.financial_keywords 
            if word in text_to_analyze
        )
        
        if financial_word_count > 0:
            impact_score += min(0.2, financial_word_count * 0.05)
        
        # Numbers in title/summary often indicate specific metrics
        if re.search(r'\b\d+(\.\d+)?%\b', text_to_analyze):  # Percentages
            impact_score += 0.2
        
        if re.search(r'\$\d+', text_to_analyze):  # Dollar amounts
            impact_score += 0.1
        
        return min(1.0, impact_score)
    
    def _apply_user_preferences(
        self,
        item: UnifiedNewsItem,
        preferences: Dict[str, Any]
    ) -> float:
        """Apply user preferences to adjust score"""
        multiplier = 1.0
        
        # Preferred sources
        preferred_sources = preferences.get('preferred_sources', [])
        if preferred_sources:
            for source in preferred_sources:
                if source.lower() in item.source.lower():
                    multiplier *= 1.2
                    break
        
        # Preferred categories
        preferred_categories = preferences.get('preferred_categories', [])
        if preferred_categories and item.categories:
            for category in item.categories:
                if category.lower() in [c.lower() for c in preferred_categories]:
                    multiplier *= 1.15
                    break
        
        # Sentiment preference
        sentiment_preference = preferences.get('sentiment_preference')  # 'bullish', 'bearish', 'neutral'
        if sentiment_preference and item.sentiment is not None:
            if sentiment_preference == 'bullish' and item.sentiment > 0.1:
                multiplier *= 1.1
            elif sentiment_preference == 'bearish' and item.sentiment < -0.1:
                multiplier *= 1.1
            elif sentiment_preference == 'neutral' and abs(item.sentiment) <= 0.1:
                multiplier *= 1.1
        
        # Language preference
        preferred_language = preferences.get('language', 'en')
        if item.language == preferred_language:
            multiplier *= 1.05
        
        # Minimum source reliability
        min_reliability = preferences.get('min_source_reliability', 0.0)
        if item.source_reliability < min_reliability:
            multiplier *= 0.5  # Penalize low-reliability sources
        
        return max(0.1, min(2.0, multiplier))  # Cap between 0.1 and 2.0
    
    def _get_sector_multiplier(self, item: UnifiedNewsItem) -> float:
        """Get sector-based score multiplier"""
        # Try to infer sector from categories or content
        text_to_analyze = f"{' '.join(item.categories)} {item.title}".lower()
        
        for sector, multiplier in self.sector_multipliers.items():
            if sector in text_to_analyze:
                return multiplier
        
        return 1.0  # Default multiplier
    
    def get_scoring_explanation(
        self,
        item: UnifiedNewsItem,
        target_symbols: List[str],
        user_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get detailed explanation of how an item was scored"""
        
        symbol_score = self._calculate_symbol_relevance(item, target_symbols)
        sentiment_score = self._calculate_sentiment_strength(item)
        source_score = item.source_reliability
        recency_score = self._calculate_recency_score(item.published_at)
        impact_score = self._calculate_impact_score(item)
        
        base_score = (
            symbol_score * self.weights['symbol_relevance'] +
            sentiment_score * self.weights['sentiment_strength'] +
            source_score * self.weights['source_reliability'] +
            recency_score * self.weights['recency'] +
            impact_score * self.weights['impact_indicators']
        )
        
        preference_multiplier = 1.0
        if user_preferences:
            preference_multiplier = self._apply_user_preferences(item, user_preferences)
        
        sector_multiplier = self._get_sector_multiplier(item)
        
        final_score = min(1.0, base_score * preference_multiplier * sector_multiplier)
        
        return {
            'item_id': item.id,
            'title': item.title[:100] + "..." if len(item.title) > 100 else item.title,
            'final_score': final_score,
            'component_scores': {
                'symbol_relevance': symbol_score,
                'sentiment_strength': sentiment_score,
                'source_reliability': source_score,
                'recency': recency_score,
                'impact_indicators': impact_score
            },
            'weighted_components': {
                'symbol_relevance': symbol_score * self.weights['symbol_relevance'],
                'sentiment_strength': sentiment_score * self.weights['sentiment_strength'],
                'source_reliability': source_score * self.weights['source_reliability'],
                'recency': recency_score * self.weights['recency'],
                'impact_indicators': impact_score * self.weights['impact_indicators']
            },
            'base_score': base_score,
            'preference_multiplier': preference_multiplier,
            'sector_multiplier': sector_multiplier,
            'weights_used': self.weights.copy()
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights (must sum to 1.0)"""
        total_weight = sum(new_weights.values())
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights.update(new_weights)
        logger.info(f"Updated scoring weights: {self.weights}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current scorer configuration"""
        return {
            'weights': self.weights.copy(),
            'impact_keywords': self.impact_keywords.copy(),
            'sector_multipliers': self.sector_multipliers.copy(),
            'financial_keywords': list(self.financial_keywords)
        }