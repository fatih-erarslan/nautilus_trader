"""
Unified News Aggregation Interface
Orchestrates multiple news sources with intelligent caching and deduplication
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import hashlib
import json

from .providers.alpha_vantage import AlphaVantageNewsProvider
from .providers.newsapi import NewsAPIProvider
from .providers.finnhub import FinnhubNewsProvider
from .caching.news_cache import NewsCache, NewsDeduplicator
from .scoring.relevance_scorer import NewsRelevanceScorer
from .failover.circuit_breaker import APIFailoverManager

logger = logging.getLogger(__name__)


@dataclass
class UnifiedNewsItem:
    """Standardized news format across all sources"""
    id: str
    title: str
    summary: str
    content: Optional[str]
    url: str
    source: str
    source_reliability: float  # 0-1 score
    published_at: datetime
    symbols: List[str]
    sentiment: Optional[float]  # -1 to 1
    relevance_scores: Dict[str, float] = field(default_factory=dict)  # symbol -> relevance
    categories: List[str] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'summary': self.summary,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'source_reliability': self.source_reliability,
            'published_at': self.published_at.isoformat(),
            'symbols': self.symbols,
            'sentiment': self.sentiment,
            'relevance_scores': self.relevance_scores,
            'categories': self.categories,
            'language': self.language,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedNewsItem':
        """Create from dictionary"""
        data = data.copy()
        if 'published_at' in data and isinstance(data['published_at'], str):
            data['published_at'] = datetime.fromisoformat(data['published_at'])
        return cls(**data)


class NewsProvider(ABC):
    """Abstract base class for news providers"""
    
    @abstractmethod
    async def fetch_news(
        self, 
        symbols: List[str], 
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> List[UnifiedNewsItem]:
        """Fetch news items for given symbols and date range"""
        pass
    
    @abstractmethod
    async def get_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment summary for a symbol"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass


class UnifiedNewsAggregator:
    """Orchestrates multiple news sources with intelligent aggregation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize providers
        self.providers = {}
        if self.config.get('alpha_vantage', {}).get('enabled', False):
            self.providers['alpha_vantage'] = AlphaVantageNewsProvider(
                self.config.get('alpha_vantage', {})
            )
        
        if self.config.get('newsapi', {}).get('enabled', False):
            self.providers['newsapi'] = NewsAPIProvider(
                self.config.get('newsapi', {})
            )
        
        if self.config.get('finnhub', {}).get('enabled', False):
            self.providers['finnhub'] = FinnhubNewsProvider(
                self.config.get('finnhub', {})
            )
        
        # Initialize components
        self.cache = NewsCache(
            redis_url=self.config.get('redis_url', 'redis://localhost:6379'),
            default_ttl=self.config.get('cache_ttl', 3600)
        )
        
        self.deduplicator = NewsDeduplicator(
            similarity_threshold=self.config.get('similarity_threshold', 0.85)
        )
        
        self.scorer = NewsRelevanceScorer()
        
        self.failover_manager = APIFailoverManager()
        
        # Configuration
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.request_timeout = self.config.get('request_timeout', 30)
        self.min_source_reliability = self.config.get('min_source_reliability', 0.3)
        
        # Metrics
        self.metrics = {
            'requests_made': 0,
            'items_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'duplicates_removed': 0
        }
        
        logger.info(f"Initialized news aggregator with {len(self.providers)} providers")
    
    async def fetch_aggregated_news(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        limit: int = 100,
        enable_cache: bool = True
    ) -> List[UnifiedNewsItem]:
        """
        Fetch and aggregate news from all sources
        
        Args:
            symbols: Stock symbols to fetch news for
            lookback_hours: How far back to look for news
            limit: Maximum number of items to return
            enable_cache: Whether to use caching
            
        Returns:
            List of aggregated and scored news items
        """
        try:
            start_time = datetime.now()
            
            # Check cache first
            if enable_cache:
                cached_items = await self.cache.get_recent(symbols, lookback_hours)
                if cached_items and not self._needs_refresh(cached_items, lookback_hours):
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"Returning {len(cached_items)} cached items")
                    return cached_items[:limit]
                else:
                    self.metrics['cache_misses'] += 1
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            # Fetch from all providers in parallel
            provider_tasks = []
            for provider_name, provider in self.providers.items():
                task = self._fetch_with_timeout(
                    provider_name,
                    provider.fetch_news,
                    symbols,
                    start_date,
                    end_date,
                    limit
                )
                provider_tasks.append(task)
            
            # Gather results with error handling
            provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)
            
            # Combine and process results
            combined_news = []
            for provider_name, results in zip(self.providers.keys(), provider_results):
                if isinstance(results, Exception):
                    logger.error(f"Provider {provider_name} failed: {results}")
                    self.metrics['errors'] += 1
                    continue
                
                if results:
                    logger.debug(f"Provider {provider_name} returned {len(results)} items")
                    combined_news.extend(results)
            
            if not combined_news:
                logger.warning("No news items fetched from any provider")
                return []
            
            # Filter by source reliability
            reliable_news = [
                item for item in combined_news 
                if item.source_reliability >= self.min_source_reliability
            ]
            
            logger.debug(f"Filtered to {len(reliable_news)} reliable items")
            
            # Deduplicate
            unique_news = self.deduplicator.process(reliable_news)
            self.metrics['duplicates_removed'] += len(reliable_news) - len(unique_news)
            
            logger.debug(f"After deduplication: {len(unique_news)} unique items")
            
            # Score and rank
            scored_news = await self.scorer.score_items(unique_news, symbols)
            
            # Extract just the news items (scored_news is list of tuples)
            final_news = [item for item, score in scored_news]
            
            # Limit results
            final_news = final_news[:limit]
            
            # Cache results
            if enable_cache and final_news:
                await self.cache.store(final_news, symbols, lookback_hours)
            
            # Update metrics
            self.metrics['requests_made'] += len(self.providers)
            self.metrics['items_processed'] += len(final_news)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Aggregated {len(final_news)} news items in {processing_time:.2f}s "
                f"from {len([r for r in provider_results if not isinstance(r, Exception)])} providers"
            )
            
            return final_news
            
        except Exception as e:
            logger.error(f"Error in fetch_aggregated_news: {e}")
            self.metrics['errors'] += 1
            return []
    
    async def get_sentiment_summary(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment summary for a symbol
        
        Args:
            symbol: Stock symbol
            lookback_hours: Time window for analysis
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            # Fetch recent news
            news_items = await self.fetch_aggregated_news([symbol], lookback_hours)
            
            if not news_items:
                return {
                    'symbol': symbol,
                    'item_count': 0,
                    'avg_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                    'confidence': 0.0,
                    'sources': []
                }
            
            # Analyze sentiment
            sentiments = [item.sentiment for item in news_items if item.sentiment is not None]
            
            if not sentiments:
                return {
                    'symbol': symbol,
                    'item_count': len(news_items),
                    'avg_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': len(news_items)},
                    'confidence': 0.0,
                    'sources': list(set(item.source for item in news_items))
                }
            
            # Calculate metrics
            avg_sentiment = sum(sentiments) / len(sentiments)
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # Calculate confidence based on agreement
            sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)) ** 0.5
            confidence = max(0.0, 1.0 - sentiment_std)
            
            return {
                'symbol': symbol,
                'item_count': len(news_items),
                'avg_sentiment': avg_sentiment,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'neutral': neutral_count,
                    'negative': negative_count
                },
                'confidence': confidence,
                'sources': list(set(item.source for item in news_items)),
                'time_range_hours': lookback_hours,
                'latest_update': max(item.published_at for item in news_items).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'item_count': 0,
                'avg_sentiment': 0.0
            }
    
    async def _fetch_with_timeout(
        self,
        provider_name: str,
        fetch_func: callable,
        *args,
        **kwargs
    ) -> List[UnifiedNewsItem]:
        """Fetch with timeout and error handling"""
        try:
            return await asyncio.wait_for(
                fetch_func(*args, **kwargs),
                timeout=self.request_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Provider {provider_name} timed out")
            return []
        except Exception as e:
            logger.error(f"Provider {provider_name} error: {e}")
            return []
    
    def _needs_refresh(
        self, 
        cached_items: List[UnifiedNewsItem], 
        lookback_hours: int
    ) -> bool:
        """Check if cached items need refresh"""
        if not cached_items:
            return True
        
        # Check if we have recent items (within last 25% of lookback period)
        refresh_threshold = datetime.now() - timedelta(hours=lookback_hours * 0.25)
        recent_items = [
            item for item in cached_items 
            if item.published_at > refresh_threshold
        ]
        
        # Need refresh if less than 50% of items are recent
        return len(recent_items) < len(cached_items) * 0.5
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            try:
                health = await provider.health_check()
                rate_limit = provider.get_rate_limit_info()
                
                status[name] = {
                    'healthy': health,
                    'rate_limit': rate_limit
                }
            except Exception as e:
                status[name] = {
                    'healthy': False,
                    'error': str(e)
                }
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregator metrics"""
        return {
            **self.metrics,
            'providers_configured': len(self.providers),
            'cache_hit_rate': (
                self.metrics['cache_hits'] / 
                (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
            ),
            'duplicate_rate': (
                self.metrics['duplicates_removed'] / self.metrics['items_processed']
                if self.metrics['items_processed'] > 0 else 0
            )
        }
    
    async def close(self):
        """Cleanup resources"""
        try:
            await self.cache.close()
            
            # Close provider connections
            for provider in self.providers.values():
                if hasattr(provider, 'close'):
                    await provider.close()
            
            logger.info("News aggregator closed successfully")
        except Exception as e:
            logger.error(f"Error closing news aggregator: {e}")