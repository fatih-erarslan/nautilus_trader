"""
News Caching System
Provides intelligent caching and deduplication for news aggregation
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import difflib
import redis.asyncio as redis

from ..news_aggregator import UnifiedNewsItem

logger = logging.getLogger(__name__)


class NewsDeduplicator:
    """Advanced news deduplication using multiple strategies"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.url_cache: Set[str] = set()
        self.title_cache: Set[str] = set()
        self.content_hashes: Set[str] = set()
        
        # Advanced deduplication caches
        self.title_fingerprints: Dict[str, str] = {}
        self.content_fingerprints: Dict[str, str] = {}
        
        logger.info(f"Initialized news deduplicator (threshold: {similarity_threshold})")
    
    def process(self, news_items: List[UnifiedNewsItem]) -> List[UnifiedNewsItem]:
        """
        Remove duplicate news items using multiple strategies
        
        Args:
            news_items: List of news items to deduplicate
            
        Returns:
            List of unique news items
        """
        if not news_items:
            return []
        
        unique_items = []
        duplicates_removed = 0
        
        logger.debug(f"Deduplicating {len(news_items)} news items")
        
        for item in news_items:
            try:
                if self._is_duplicate(item):
                    duplicates_removed += 1
                    continue
                
                # Add to tracking caches
                self._add_to_caches(item)
                unique_items.append(item)
                
            except Exception as e:
                logger.warning(f"Error processing item for deduplication: {e}")
                # Include item if unsure
                unique_items.append(item)
        
        logger.info(f"Removed {duplicates_removed} duplicates, keeping {len(unique_items)} unique items")
        return unique_items
    
    def _is_duplicate(self, item: UnifiedNewsItem) -> bool:
        """Check if item is a duplicate using multiple strategies"""
        
        # Strategy 1: Exact URL match
        if item.url in self.url_cache:
            logger.debug(f"Duplicate URL found: {item.url}")
            return True
        
        # Strategy 2: Exact title match
        if item.title in self.title_cache:
            logger.debug(f"Duplicate title found: {item.title[:50]}...")
            return True
        
        # Strategy 3: Content hash match
        content_hash = self._hash_content(item)
        if content_hash in self.content_hashes:
            logger.debug(f"Duplicate content hash found")
            return True
        
        # Strategy 4: Similar title detection
        if self._has_similar_title(item.title):
            logger.debug(f"Similar title found: {item.title[:50]}...")
            return True
        
        # Strategy 5: Similar content detection (if available)
        if item.content and self._has_similar_content(item.content):
            logger.debug(f"Similar content found")
            return True
        
        return False
    
    def _add_to_caches(self, item: UnifiedNewsItem):
        """Add item to all tracking caches"""
        self.url_cache.add(item.url)
        self.title_cache.add(item.title)
        
        content_hash = self._hash_content(item)
        self.content_hashes.add(content_hash)
        
        # Add fingerprints for similarity detection
        title_fingerprint = self._create_title_fingerprint(item.title)
        self.title_fingerprints[title_fingerprint] = item.title
        
        if item.content:
            content_fingerprint = self._create_content_fingerprint(item.content)
            self.content_fingerprints[content_fingerprint] = item.content[:200]
    
    def _hash_content(self, item: UnifiedNewsItem) -> str:
        """Generate content hash for exact duplicate detection"""
        content_parts = [
            item.title.lower().strip(),
            item.summary.lower().strip() if item.summary else "",
            item.source.lower(),
            # Include first few symbols for additional uniqueness
            ",".join(sorted(item.symbols[:3])) if item.symbols else ""
        ]
        
        content_string = "|".join(content_parts)
        return hashlib.sha256(content_string.encode()).hexdigest()
    
    def _create_title_fingerprint(self, title: str) -> str:
        """Create normalized title fingerprint for similarity detection"""
        # Remove common words and normalize
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must'
        }
        
        # Clean and tokenize
        words = title.lower().replace('-', ' ').split()
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Sort to handle different word orders
        return " ".join(sorted(filtered_words))
    
    def _create_content_fingerprint(self, content: str) -> str:
        """Create content fingerprint for similarity detection"""
        # Take first 500 characters, remove special chars, normalize
        normalized = content[:500].lower()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _has_similar_title(self, title: str) -> bool:
        """Check if title is similar to any cached title"""
        title_fingerprint = self._create_title_fingerprint(title)
        
        # Quick exact fingerprint match
        if title_fingerprint in self.title_fingerprints:
            return True
        
        # Fuzzy matching for titles with high similarity
        for cached_fingerprint, cached_title in self.title_fingerprints.items():
            similarity = difflib.SequenceMatcher(None, title_fingerprint, cached_fingerprint).ratio()
            
            if similarity >= self.similarity_threshold:
                logger.debug(f"Similar title detected (similarity: {similarity:.2f})")
                logger.debug(f"  Original: {cached_title[:50]}...")
                logger.debug(f"  New: {title[:50]}...")
                return True
        
        return False
    
    def _has_similar_content(self, content: str) -> bool:
        """Check if content is similar to any cached content"""
        content_fingerprint = self._create_content_fingerprint(content)
        
        # Quick exact fingerprint match
        if content_fingerprint in self.content_fingerprints:
            return True
        
        # For performance, only check similarity if we have few cached items
        if len(self.content_fingerprints) > 100:
            return False
        
        for cached_fingerprint in self.content_fingerprints.keys():
            # Use first 100 chars of normalized content for comparison
            similarity = difflib.SequenceMatcher(
                None, 
                content_fingerprint[:32], 
                cached_fingerprint[:32]
            ).ratio()
            
            if similarity >= self.similarity_threshold:
                logger.debug(f"Similar content detected (similarity: {similarity:.2f})")
                return True
        
        return False
    
    def clear_cache(self):
        """Clear all deduplication caches"""
        self.url_cache.clear()
        self.title_cache.clear()
        self.content_hashes.clear()
        self.title_fingerprints.clear()
        self.content_fingerprints.clear()
        logger.info("Deduplication caches cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return {
            'cached_urls': len(self.url_cache),
            'cached_titles': len(self.title_cache),
            'cached_content_hashes': len(self.content_hashes),
            'cached_title_fingerprints': len(self.title_fingerprints),
            'cached_content_fingerprints': len(self.content_fingerprints)
        }


class NewsCache:
    """Redis-based news cache with intelligent TTL and storage optimization"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool = None
        
        # Cache configuration
        self.max_items_per_key = 500
        self.compression_enabled = True
        
        # Cache key prefixes
        self.NEWS_PREFIX = "news"
        self.SENTIMENT_PREFIX = "sentiment"
        self.METRICS_PREFIX = "metrics"
        
        logger.info(f"Initialized news cache (TTL: {default_ttl}s)")
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self.redis_client is None or self.redis_client.connection_pool.connection_closed:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # We handle encoding ourselves
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self.redis_client
    
    def _generate_cache_key(self, symbols: List[str], hours: int, prefix: str = None) -> str:
        """Generate cache key for given parameters"""
        if prefix is None:
            prefix = self.NEWS_PREFIX
        
        # Sort symbols for consistent keys
        sorted_symbols = sorted([s.upper() for s in symbols])
        symbols_str = ",".join(sorted_symbols) if sorted_symbols else "general"
        
        # Include hour range for time-based partitioning
        return f"{prefix}:{symbols_str}:{hours}h"
    
    async def get_recent(
        self, 
        symbols: List[str], 
        hours: int,
        max_items: int = 100
    ) -> List[UnifiedNewsItem]:
        """
        Retrieve cached news items for symbols within time range
        
        Args:
            symbols: Stock symbols to get news for
            hours: Time range in hours
            max_items: Maximum items to return
            
        Returns:
            List of cached news items
        """
        try:
            redis_client = await self._get_redis()
            cache_key = self._generate_cache_key(symbols, hours)
            
            # Get cached data
            cached_data = await redis_client.get(cache_key)
            if not cached_data:
                logger.debug(f"Cache miss for key: {cache_key}")
                return []
            
            # Deserialize data
            if self.compression_enabled:
                news_items = pickle.loads(cached_data)
            else:
                news_items_data = json.loads(cached_data.decode('utf-8'))
                news_items = [UnifiedNewsItem.from_dict(item) for item in news_items_data]
            
            # Filter by freshness
            cutoff_time = datetime.now() - timedelta(hours=hours)
            fresh_items = [
                item for item in news_items 
                if item.published_at > cutoff_time
            ]
            
            logger.debug(f"Cache hit: {len(fresh_items)} fresh items from {len(news_items)} cached")
            return fresh_items[:max_items]
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return []
    
    async def store(
        self, 
        news_items: List[UnifiedNewsItem], 
        symbols: List[str], 
        hours: int,
        ttl: Optional[int] = None
    ):
        """
        Store news items in cache
        
        Args:
            news_items: News items to cache
            symbols: Associated symbols
            hours: Time range these items cover
            ttl: Time to live in seconds
        """
        if not news_items:
            return
        
        try:
            redis_client = await self._get_redis()
            cache_key = self._generate_cache_key(symbols, hours)
            
            # Limit items to prevent memory issues
            items_to_cache = news_items[:self.max_items_per_key]
            
            # Serialize data
            if self.compression_enabled:
                serialized_data = pickle.dumps(items_to_cache)
            else:
                items_data = [item.to_dict() for item in items_to_cache]
                serialized_data = json.dumps(items_data, default=str).encode('utf-8')
            
            # Store with TTL
            cache_ttl = ttl if ttl is not None else self.default_ttl
            await redis_client.setex(cache_key, cache_ttl, serialized_data)
            
            logger.debug(f"Cached {len(items_to_cache)} items with key: {cache_key} (TTL: {cache_ttl}s)")
            
            # Store metadata
            await self._store_cache_metadata(cache_key, len(items_to_cache), cache_ttl)
            
        except Exception as e:
            logger.error(f"Error storing to cache: {e}")
    
    async def _store_cache_metadata(self, cache_key: str, item_count: int, ttl: int):
        """Store metadata about cached items"""
        try:
            redis_client = await self._get_redis()
            metadata_key = f"{self.METRICS_PREFIX}:{cache_key}"
            
            metadata = {
                'stored_at': datetime.now().isoformat(),
                'item_count': item_count,
                'ttl': ttl,
                'cache_key': cache_key
            }
            
            await redis_client.setex(
                metadata_key, 
                ttl + 3600,  # Keep metadata longer than cache
                json.dumps(metadata)
            )
            
        except Exception as e:
            logger.warning(f"Error storing cache metadata: {e}")
    
    async def store_sentiment(
        self, 
        symbol: str, 
        sentiment_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store sentiment analysis results"""
        try:
            redis_client = await self._get_redis()
            cache_key = f"{self.SENTIMENT_PREFIX}:{symbol.upper()}"
            
            # Add timestamp
            sentiment_data['cached_at'] = datetime.now().isoformat()
            
            serialized_data = json.dumps(sentiment_data, default=str).encode('utf-8')
            cache_ttl = ttl if ttl is not None else self.default_ttl // 2  # Shorter TTL for sentiment
            
            await redis_client.setex(cache_key, cache_ttl, serialized_data)
            logger.debug(f"Cached sentiment for {symbol}")
            
        except Exception as e:
            logger.error(f"Error caching sentiment: {e}")
    
    async def get_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached sentiment data"""
        try:
            redis_client = await self._get_redis()
            cache_key = f"{self.SENTIMENT_PREFIX}:{symbol.upper()}"
            
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                sentiment_data = json.loads(cached_data.decode('utf-8'))
                logger.debug(f"Sentiment cache hit for {symbol}")
                return sentiment_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment from cache: {e}")
            return None
    
    async def invalidate_symbol(self, symbol: str):
        """Invalidate all cache entries for a symbol"""
        try:
            redis_client = await self._get_redis()
            
            # Find keys containing the symbol
            pattern = f"{self.NEWS_PREFIX}:*{symbol.upper()}*"
            keys = await redis_client.keys(pattern)
            
            if keys:
                await redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for {symbol}")
            
            # Also invalidate sentiment
            sentiment_key = f"{self.SENTIMENT_PREFIX}:{symbol.upper()}"
            await redis_client.delete(sentiment_key)
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
    
    async def clear_expired(self):
        """Clear expired cache entries (Redis handles this automatically, but useful for stats)"""
        try:
            redis_client = await self._get_redis()
            
            # Get all cache keys
            news_keys = await redis_client.keys(f"{self.NEWS_PREFIX}:*")
            sentiment_keys = await redis_client.keys(f"{self.SENTIMENT_PREFIX}:*")
            
            # Check which ones still exist (non-expired)
            valid_news = 0
            valid_sentiment = 0
            
            for key in news_keys:
                if await redis_client.exists(key):
                    valid_news += 1
            
            for key in sentiment_keys:
                if await redis_client.exists(key):
                    valid_sentiment += 1
            
            logger.info(f"Active cache entries: {valid_news} news, {valid_sentiment} sentiment")
            
        except Exception as e:
            logger.error(f"Error checking expired cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            redis_client = await self._get_redis()
            
            # Get key counts
            news_keys = await redis_client.keys(f"{self.NEWS_PREFIX}:*")
            sentiment_keys = await redis_client.keys(f"{self.SENTIMENT_PREFIX}:*")
            metadata_keys = await redis_client.keys(f"{self.METRICS_PREFIX}:*")
            
            # Get memory usage (if available)
            try:
                info = await redis_client.info('memory')
                memory_usage = info.get('used_memory_human', 'unknown')
            except:
                memory_usage = 'unknown'
            
            return {
                'news_cache_entries': len(news_keys),
                'sentiment_cache_entries': len(sentiment_keys),
                'metadata_entries': len(metadata_keys),
                'memory_usage': memory_usage,
                'compression_enabled': self.compression_enabled,
                'default_ttl': self.default_ttl
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def close(self):
        """Close Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")