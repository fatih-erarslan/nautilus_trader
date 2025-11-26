"""Base classes for news collection module - GREEN phase"""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional
from datetime import datetime
from cachetools import TTLCache
import logging

from news.models import NewsItem

logger = logging.getLogger(__name__)


class NewsSource(ABC):
    """Abstract base class for all news sources"""
    
    def __init__(self, source_name: str, cache_ttl: int = 300):
        """
        Initialize news source with caching
        
        Args:
            source_name: Name of the news source
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.source_name = source_name
        self._cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        
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
    
    async def fetch_latest_with_cache(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest news with caching support
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of NewsItem objects (from cache if available)
        """
        cache_key = f"{self.source_name}:latest:{limit}"
        
        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        # Fetch from source
        logger.debug(f"Cache miss for {cache_key}, fetching from source")
        try:
            items = await self.fetch_latest(limit)
            self._cache[cache_key] = items
            return items
        except Exception as e:
            logger.error(f"Error fetching from {self.source_name}: {e}")
            raise
    
    async def search(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        """
        Search historical news (optional implementation)
        
        Args:
            query: Search query
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of matching NewsItem objects
        """
        raise NotImplementedError(f"{self.source_name} does not support historical search")
    
    def clear_cache(self) -> None:
        """Clear the cache for this source"""
        self._cache.clear()
        logger.info(f"Cache cleared for {self.source_name}")