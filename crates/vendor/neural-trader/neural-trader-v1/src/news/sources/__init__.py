"""
News source interfaces and implementations
"""
from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional, Dict, Any
import logging
from datetime import datetime
from cachetools import TTLCache
from ..models import NewsItem


logger = logging.getLogger(__name__)


class NewsSourceError(Exception):
    """Base exception for news source errors"""
    pass


class NewsSource(ABC):
    """Abstract base class for all news sources"""
    
    def __init__(self, source_name: str, cache_ttl: int = 300, cache_maxsize: int = 1000):
        """
        Initialize news source
        
        Args:
            source_name: Name of the news source
            cache_ttl: Time to live for cache entries in seconds (default: 5 minutes)
            cache_maxsize: Maximum number of items in cache (default: 1000)
        """
        self.source_name = source_name
        self._cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._metrics = {
            'fetch_count': 0,
            'cache_hits': 0,
            'errors': 0,
            'last_fetch': None
        }
        logger.info(f"Initialized {source_name} news source with cache TTL={cache_ttl}s")
        
    async def fetch_latest_with_cache(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest news items with caching support
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of news items
            
        Raises:
            NewsSourceError: If fetching fails
        """
        cache_key = f"{self.source_name}:latest:{limit}"
        
        # Check cache first
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Fetch from source
            self._metrics['fetch_count'] += 1
            self._metrics['last_fetch'] = datetime.now()
            
            items = await self.fetch_latest(limit)
            
            # Validate items
            if not isinstance(items, list):
                raise NewsSourceError(f"fetch_latest must return a list, got {type(items)}")
            
            # Cache the results
            self._cache[cache_key] = items
            logger.info(f"Fetched {len(items)} items from {self.source_name}")
            
            return items
            
        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(f"Error fetching from {self.source_name}: {str(e)}")
            raise NewsSourceError(f"Failed to fetch from {self.source_name}: {str(e)}") from e
    
    @abstractmethod
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest news items from the source
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of NewsItem objects
        """
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncIterator[NewsItem]:
        """
        Stream news items in real-time
        
        Yields:
            NewsItem objects as they become available
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get source metrics for monitoring"""
        return {
            'source': self.source_name,
            **self._metrics,
            'cache_info': {
                'size': len(self._cache),
                'maxsize': self._cache.maxsize,
                'ttl': self._cache.ttl
            }
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.info(f"Cleared cache for {self.source_name}")