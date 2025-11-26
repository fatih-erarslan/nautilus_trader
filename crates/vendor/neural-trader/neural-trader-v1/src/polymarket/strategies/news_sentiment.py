"""
News Sentiment Trading Strategy

This strategy analyzes news sentiment to generate trading signals for prediction markets.
It integrates with the existing MCP news analysis tools and uses sentiment scores to
determine market positions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
import numpy as np

from ..models import Market, MarketStatus, Order
from .base import (
    PolymarketStrategy, StrategyConfig, TradingSignal, SignalStrength,
    SignalDirection, StrategyError
)

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Data class for sentiment analysis results"""
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    trend_strength: float  # 0 to 1
    article_count: int
    impact_score: float  # 0 to 1 - estimated market impact
    sources: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate sentiment data"""
        if not -1 <= self.sentiment_score <= 1:
            raise ValueError(f"Sentiment score must be between -1 and 1, got {self.sentiment_score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish"""
        return self.sentiment_score > 0.5
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish"""
        return self.sentiment_score < 0.5
    
    def calculate_signal_strength(self) -> float:
        """Calculate overall signal strength"""
        # Combine sentiment strength, confidence, and impact
        sentiment_strength = abs(self.sentiment_score - 0.5) * 2  # Normalize to 0-1
        return sentiment_strength * self.confidence * self.impact_score


class NewsSentimentStrategy(PolymarketStrategy):
    """
    Trading strategy based on news sentiment analysis
    
    Analyzes news articles and social media sentiment to predict market movements.
    Integrates with MCP news analysis tools for real-time sentiment data.
    """
    
    def __init__(
        self,
        client,
        config: Optional[StrategyConfig] = None,
        min_article_count: int = 5,
        sentiment_threshold: float = 0.65,
        trend_weight: float = 0.3,
        cache_duration: int = 300,  # 5 minutes
        supported_categories: Optional[List[str]] = None,
    ):
        """
        Initialize news sentiment strategy
        
        Args:
            client: Polymarket API client
            config: Strategy configuration
            min_article_count: Minimum articles required for signal
            sentiment_threshold: Minimum sentiment deviation from neutral
            trend_weight: Weight given to sentiment trend vs absolute value
            cache_duration: Cache duration for sentiment data in seconds
            supported_categories: List of market categories to trade
        """
        super().__init__(client, config, "NewsSentimentStrategy")
        
        self.min_article_count = min_article_count
        self.sentiment_threshold = sentiment_threshold
        self.trend_weight = trend_weight
        self.cache_duration = cache_duration
        self.supported_categories = supported_categories or [
            "crypto", "politics", "sports", "finance", "economics", "technology"
        ]
        
        # Sentiment analysis configuration
        self.sentiment_sources = ["mcp__ai_news_trader__analyze_news"]
        self.bullish_keywords = {
            "positive", "bullish", "growth", "surge", "rally", "breakthrough",
            "success", "win", "gain", "increase", "optimistic", "favorable"
        }
        self.bearish_keywords = {
            "negative", "bearish", "decline", "crash", "fall", "failure",
            "loss", "decrease", "pessimistic", "concern", "worry", "risk"
        }
        
        # Performance tracking
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.sentiment_history: List[float] = []
        self.sentiment_analysis_count = 0
        
        logger.info(f"Initialized {self.name} with sentiment threshold {self.sentiment_threshold}")
    
    async def should_trade_market(self, market: Market) -> bool:
        """
        Determine if this strategy should trade on a given market
        
        Args:
            market: Market to evaluate
            
        Returns:
            True if market is suitable for sentiment trading
        """
        # Check if market is active
        if market.status != MarketStatus.ACTIVE:
            return False
        
        # Check category
        market_category = market.metadata.get('category', '').lower()
        if market_category not in self.supported_categories:
            logger.debug(f"Market {market.id} category '{market_category}' not supported")
            return False
        
        # Check volume threshold
        min_volume = 5000  # Minimum 24h volume
        if market.metadata.get('volume_24h', 0) < min_volume:
            logger.debug(f"Market {market.id} volume too low")
            return False
        
        # Check time to expiry
        if market.end_date:
            time_to_expiry = market.end_date - datetime.now()
            if time_to_expiry < timedelta(hours=6):
                logger.debug(f"Market {market.id} expiring too soon")
                return False
        
        return True
    
    async def analyze_market(self, market: Market) -> Optional[TradingSignal]:
        """
        Analyze market sentiment and generate trading signal
        
        Args:
            market: Market to analyze
            
        Returns:
            Trading signal if opportunity found
        """
        try:
            # Fetch news sentiment
            sentiment_data = await self._fetch_news_sentiment(market)
            if not sentiment_data:
                logger.debug(f"No sentiment data available for {market.id}")
                return None
            
            # Check data quality
            if not self._check_sentiment_quality(sentiment_data):
                logger.debug(f"Sentiment data quality too low for {market.id}")
                return None
            
            # Create sentiment signal
            sentiment_signal = SentimentSignal(
                sentiment_score=sentiment_data.get('overall_sentiment', 0.5),
                confidence=sentiment_data.get('confidence', 0),
                trend_strength=self._calculate_trend_strength(sentiment_data),
                article_count=sentiment_data.get('article_count', 0),
                impact_score=sentiment_data.get('impact_score', 0.5),
                sources=sentiment_data.get('sources', []),
                keywords=sentiment_data.get('keywords', [])
            )
            
            # Check if sentiment is strong enough
            if abs(sentiment_signal.sentiment_score - 0.5) < (self.sentiment_threshold - 0.5):
                logger.debug(f"Sentiment not strong enough for {market.id}")
                return None
            
            # Generate trading signal
            signal = self._generate_signal_from_sentiment(market, sentiment_signal)
            
            # Update metrics
            self._update_sentiment_metrics(sentiment_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {market.id}: {str(e)}")
            return None
    
    async def _fetch_news_sentiment(self, market: Market) -> Optional[Dict[str, Any]]:
        """
        Fetch news sentiment data for market
        
        Args:
            market: Market to analyze
            
        Returns:
            Sentiment data dictionary
        """
        # Check cache first
        cache_key = f"{market.id}_sentiment"
        if cache_key in self.sentiment_cache:
            cached_data = self.sentiment_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_duration):
                return cached_data['data']
        
        try:
            # Extract search terms from market question
            search_terms = self._extract_search_terms(market)
            
            # For Polymarket integration, we need to analyze news for the relevant symbol/topic
            # Extract primary search term (could be improved with NLP)
            primary_term = search_terms[0] if search_terms else market.question.split()[0]
            
            # Call MCP sentiment analysis tool
            # In production, this would use the actual MCP client
            # For now, we'll import and use the function directly
            try:
                from ...mcp.mcp_server_enhanced import analyze_news
                
                # Call the MCP analyze_news function
                mcp_result = analyze_news(
                    symbol=primary_term,
                    lookback_hours=24,
                    sentiment_model="enhanced",
                    use_gpu=self.config.enable_gpu_acceleration
                )
                
                if mcp_result.get('status') == 'success':
                    # Map MCP result to our format
                    articles = mcp_result.get('articles', [])
                    
                    # Calculate confidence based on article count and sentiment consistency
                    sentiment_scores = [a.get('sentiment', 0) * a.get('confidence', 1) for a in articles]
                    sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
                    confidence = min(0.95, max(0.5, 1 - sentiment_std))
                    
                    # Determine trend
                    if len(articles) >= 2:
                        recent_sentiment = np.mean([a['sentiment'] for a in articles[:len(articles)//2]])
                        older_sentiment = np.mean([a['sentiment'] for a in articles[len(articles)//2:]])
                        trend = 'increasing' if recent_sentiment > older_sentiment else 'decreasing'
                    else:
                        trend = 'stable'
                    
                    # Extract keywords from article titles
                    keywords = []
                    for article in articles:
                        title_words = article.get('title', '').lower().split()
                        keywords.extend([w for w in title_words if w in self.bullish_keywords or w in self.bearish_keywords])
                    
                    sentiment_data = {
                        'overall_sentiment': (mcp_result.get('overall_sentiment', 0) + 1) / 2,  # Convert from [-1,1] to [0,1]
                        'confidence': confidence,
                        'article_count': mcp_result.get('articles_analyzed', 0),
                        'sources': list(set(a.get('source', 'Unknown') for a in articles)),
                        'keywords': list(set(keywords))[:10],  # Top 10 unique keywords
                        'sentiment_trend': trend,
                        'impact_score': min(0.9, mcp_result.get('articles_analyzed', 0) / 20),  # Scale by article count
                        'processing_time': mcp_result.get('processing', {}).get('time_seconds', 0)
                    }
                else:
                    # Fallback to basic analysis
                    sentiment_data = self._generate_fallback_sentiment(market, search_terms)
                    
            except ImportError:
                # If MCP server is not available, use fallback
                logger.warning("MCP server not available, using fallback sentiment analysis")
                sentiment_data = self._generate_fallback_sentiment(market, search_terms)
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'data': sentiment_data,
                'timestamp': datetime.now()
            }
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {market.id}: {str(e)}")
            return None
    
    def _generate_fallback_sentiment(self, market: Market, search_terms: List[str]) -> Dict[str, Any]:
        """Generate fallback sentiment data when MCP is unavailable"""
        # Simple keyword-based sentiment analysis
        question_lower = market.question.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in question_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in question_lower)
        
        total_keywords = bullish_count + bearish_count
        if total_keywords > 0:
            sentiment_score = bullish_count / total_keywords
        else:
            sentiment_score = 0.5
        
        return {
            'overall_sentiment': sentiment_score,
            'confidence': 0.6,  # Lower confidence for fallback
            'article_count': 5,  # Minimum articles
            'sources': ['market_analysis'],
            'keywords': search_terms[:5],
            'sentiment_trend': 'stable',
            'impact_score': 0.5
        }
    
    def _extract_search_terms(self, market: Market) -> List[str]:
        """Extract relevant search terms from market question"""
        # Simple implementation - in production would use NLP
        question = market.question.lower()
        
        # Remove common words
        stop_words = {'will', 'the', 'be', 'is', 'at', 'by', 'for', 'of', 'with', 'a', 'an'}
        words = [w for w in question.split() if w not in stop_words]
        
        # Add category-specific terms
        category = market.metadata.get('category', '')
        if category:
            words.append(category)
        
        # Add any tags
        tags = market.metadata.get('tags', [])
        words.extend(tags)
        
        return words[:5]  # Limit to top 5 terms
    
    def _check_sentiment_quality(self, sentiment_data: Dict[str, Any]) -> bool:
        """Check if sentiment data meets quality thresholds"""
        # Minimum article count
        if sentiment_data.get('article_count', 0) < self.min_article_count:
            return False
        
        # Minimum confidence
        if sentiment_data.get('confidence', 0) < 0.5:
            return False
        
        # Require multiple sources
        if len(sentiment_data.get('sources', [])) < 2:
            return False
        
        return True
    
    def _calculate_trend_strength(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate sentiment trend strength"""
        trend = sentiment_data.get('sentiment_trend', 'stable')
        
        if trend == 'increasing':
            return 0.8
        elif trend == 'decreasing':
            return 0.2
        else:
            return 0.5
    
    def _parse_sentiment_keywords(self, text: str) -> float:
        """Parse text for sentiment keywords"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.5
        
        return bullish_count / total
    
    def _generate_signal_from_sentiment(
        self,
        market: Market,
        sentiment: SentimentSignal
    ) -> Optional[TradingSignal]:
        """Generate trading signal from sentiment analysis"""
        # Determine direction
        if sentiment.is_bullish:
            direction = SignalDirection.BUY
            outcome = "Yes"
            target_multiplier = 1 + (sentiment.sentiment_score - 0.5)
        else:
            direction = SignalDirection.SELL
            outcome = "Yes"  # Sell Yes = Buy No
            target_multiplier = 1 - (0.5 - sentiment.sentiment_score)
        
        # Calculate target price
        current_price = market.current_prices.get(outcome, Decimal("0.5"))
        target_price = current_price * Decimal(str(target_multiplier))
        target_price = max(Decimal("0.01"), min(Decimal("0.99"), target_price))
        
        # Map sentiment to signal strength
        strength = self._map_sentiment_to_strength(
            sentiment.sentiment_score,
            sentiment.confidence
        )
        
        # Calculate position size
        size = self._calculate_position_size(sentiment, current_price)
        
        # Create signal
        signal = TradingSignal(
            market_id=market.id,
            outcome=outcome,
            direction=direction,
            strength=strength,
            target_price=target_price,
            size=size,
            confidence=sentiment.confidence,
            reasoning=self._generate_reasoning(sentiment),
            metadata={
                'sentiment_score': sentiment.sentiment_score,
                'article_count': sentiment.article_count,
                'sources': sentiment.sources,
                'strategy': self.name
            }
        )
        
        # Final validation
        return self._validate_signal_bounds(signal, market)
    
    def _map_sentiment_to_strength(
        self,
        sentiment_score: float,
        confidence: float
    ) -> SignalStrength:
        """Map sentiment score and confidence to signal strength"""
        # Calculate deviation from neutral
        deviation = abs(sentiment_score - 0.5)
        
        # Weight by confidence
        weighted_strength = deviation * confidence
        
        if weighted_strength >= 0.4:
            return SignalStrength.VERY_STRONG
        elif weighted_strength >= 0.3:
            return SignalStrength.STRONG
        elif weighted_strength >= 0.2:
            return SignalStrength.MODERATE
        elif weighted_strength >= 0.1:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_position_size(
        self,
        sentiment: SentimentSignal,
        current_price: Decimal
    ) -> Decimal:
        """Calculate position size using Kelly criterion"""
        # Estimate win probability from sentiment
        if sentiment.is_bullish:
            p = sentiment.sentiment_score
        else:
            p = 1 - sentiment.sentiment_score
        
        # Adjust for confidence
        p = p * sentiment.confidence
        
        # Estimate odds (simplified)
        if sentiment.is_bullish:
            b = (Decimal("1") - current_price) / current_price
        else:
            b = current_price / (Decimal("1") - current_price)
        
        # Kelly formula: f = p - q/b
        q = 1 - p
        kelly_fraction = (Decimal(str(p)) - Decimal(str(q)) / b) if b > 0 else Decimal("0")
        
        # Apply conservative factor (quarter Kelly)
        conservative_fraction = kelly_fraction * Decimal("0.25")
        
        # Apply maximum position size
        fraction = max(Decimal("0"), min(conservative_fraction, Decimal("0.1")))
        size = self.config.max_position_size * fraction
        
        # Adjust for impact score
        size = size * Decimal(str(sentiment.impact_score))
        
        return max(Decimal("10"), size)  # Minimum size
    
    def _generate_reasoning(self, sentiment: SentimentSignal) -> str:
        """Generate human-readable reasoning for signal"""
        direction = "bullish" if sentiment.is_bullish else "bearish"
        strength = "strong" if sentiment.confidence > 0.8 else "moderate"
        
        reasoning = (
            f"{strength.capitalize()} {direction} sentiment detected with "
            f"{sentiment.confidence:.0%} confidence based on {sentiment.article_count} articles. "
            f"Sentiment score: {sentiment.sentiment_score:.2f}"
        )
        
        if sentiment.keywords:
            reasoning += f". Key themes: {', '.join(sentiment.keywords[:3])}"
        
        return reasoning
    
    def _validate_signal_bounds(
        self,
        signal: TradingSignal,
        market: Market
    ) -> TradingSignal:
        """Validate and adjust signal to stay within bounds"""
        # Ensure target price is reasonable
        current_price = market.current_prices.get(signal.outcome, Decimal("0.5"))
        max_move = Decimal("0.3")  # Maximum 30% move
        
        if signal.direction == SignalDirection.BUY:
            signal.target_price = min(
                signal.target_price,
                current_price * (Decimal("1") + max_move)
            )
        else:
            signal.target_price = max(
                signal.target_price,
                current_price * (Decimal("1") - max_move)
            )
        
        # Ensure within 0-1 bounds
        signal.target_price = max(Decimal("0.01"), min(Decimal("0.99"), signal.target_price))
        
        # Validate size
        signal.size = min(signal.size, self.config.max_position_size)
        
        return signal
    
    def _update_sentiment_metrics(self, sentiment_data: Dict[str, Any]):
        """Update performance metrics for sentiment analysis"""
        self.sentiment_analysis_count += 1
        sentiment_score = sentiment_data.get('overall_sentiment', 0.5)
        self.sentiment_history.append(sentiment_score)
        
        # Keep only recent history
        if len(self.sentiment_history) > 100:
            self.sentiment_history = self.sentiment_history[-100:]
    
    def get_sentiment_performance_summary(self) -> Dict[str, Any]:
        """Get summary of sentiment analysis performance"""
        if not self.sentiment_history:
            return {
                'analysis_count': 0,
                'average_sentiment': 0.5,
                'sentiment_volatility': 0
            }
        
        sentiment_array = np.array(self.sentiment_history)
        
        return {
            'analysis_count': self.sentiment_analysis_count,
            'average_sentiment': float(np.mean(sentiment_array)),
            'sentiment_volatility': float(np.std(sentiment_array)),
            'recent_trend': 'bullish' if sentiment_array[-1] > 0.5 else 'bearish',
            'cache_size': len(self.sentiment_cache)
        }
    
    async def _place_order(self, signal: TradingSignal) -> Optional[Order]:
        """Place order based on signal"""
        from ..models import Order, OrderSide, OrderType, OrderStatus
        
        try:
            # Map signal direction to order side
            if signal.direction == SignalDirection.BUY:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Create order
            order = Order(
                id=f"order_{signal.market_id}_{datetime.now().timestamp()}",
                market_id=signal.market_id,
                outcome_id=signal.outcome,
                side=side,
                type=OrderType.LIMIT,
                size=float(signal.size),
                price=float(signal.target_price),
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )
            
            # In production, this would submit to the CLOB API
            # For now, we'll just return the order
            logger.info(f"Placing {side.value} order for {signal.size} shares at {signal.target_price}")
            
            # Update position tracking
            self.update_position(
                market_id=signal.market_id,
                outcome=signal.outcome,
                size=signal.size,
                price=signal.target_price
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None