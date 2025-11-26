"""News aggregator for collecting from multiple sources - GREEN phase"""

import asyncio
from typing import List, Optional, AsyncIterator
import logging
from datetime import datetime

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource
from news_trading.news_collection.deduplication import deduplicate_news

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news from multiple sources with concurrent fetching"""
    
    def __init__(self, sources: List[NewsSource], deduplicate: bool = True):
        """
        Initialize news aggregator
        
        Args:
            sources: List of news sources to aggregate from
            deduplicate: Whether to deduplicate articles (default: True)
        """
        self.sources = sources
        self.deduplicate = deduplicate
        self._source_map = {source.source_name: source for source in sources}
        
    def add_source(self, source: NewsSource) -> None:
        """Add a new news source"""
        if source.source_name not in self._source_map:
            self.sources.append(source)
            self._source_map[source.source_name] = source
            logger.info(f"Added news source: {source.source_name}")
        else:
            logger.warning(f"Source {source.source_name} already exists")
            
    def remove_source(self, source_name: str) -> None:
        """Remove a news source by name"""
        if source_name in self._source_map:
            source = self._source_map.pop(source_name)
            self.sources.remove(source)
            logger.info(f"Removed news source: {source_name}")
        else:
            logger.warning(f"Source {source_name} not found")
            
    async def fetch_all(self, limit_per_source: int = 50) -> List[NewsItem]:
        """
        Fetch from all sources concurrently
        
        Args:
            limit_per_source: Maximum items to fetch per source
            
        Returns:
            List of NewsItem objects, optionally deduplicated
        """
        if not self.sources:
            logger.warning("No news sources configured")
            return []
            
        # Create tasks for concurrent fetching
        tasks = []
        for source in self.sources:
            task = asyncio.create_task(
                self._fetch_from_source(source, limit_per_source)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {self.sources[i].source_name}: {result}")
            elif isinstance(result, list):
                all_items.extend(result)
        
        # Sort by timestamp (newest first)
        all_items.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Deduplicate if enabled
        if self.deduplicate and len(all_items) > 1:
            all_items = deduplicate_news(all_items)
            logger.info(f"Deduplicated {len(all_items)} articles")
        
        return all_items
    
    async def _fetch_from_source(self, source: NewsSource, limit: int) -> List[NewsItem]:
        """Fetch from a single source with error handling"""
        try:
            logger.debug(f"Fetching from {source.source_name} (limit: {limit})")
            items = await source.fetch_latest(limit)
            logger.info(f"Fetched {len(items)} items from {source.source_name}")
            return items
        except Exception as e:
            logger.error(f"Failed to fetch from {source.source_name}: {e}")
            raise
    
    async def stream_all(self) -> AsyncIterator[NewsItem]:
        """
        Merge streams from all sources
        
        Yields:
            NewsItem objects as they arrive from any source
        """
        if not self.sources:
            logger.warning("No news sources configured for streaming")
            return
        
        # Create queue for merging streams
        queue = asyncio.Queue()
        
        # Create tasks for each stream
        stream_tasks = []
        for source in self.sources:
            if hasattr(source, 'stream'):
                task = asyncio.create_task(
                    self._stream_to_queue(source, queue)
                )
                stream_tasks.append(task)
            else:
                logger.debug(f"Source {source.source_name} does not support streaming")
        
        if not stream_tasks:
            logger.warning("No sources support streaming")
            return
        
        # Start all streaming tasks
        logger.info(f"Starting {len(stream_tasks)} stream tasks")
        
        try:
            # Yield items from queue as they arrive
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield item
                except asyncio.TimeoutError:
                    # Check if all tasks are done
                    if all(task.done() for task in stream_tasks):
                        break
                    continue
        finally:
            # Cancel all streaming tasks
            for task in stream_tasks:
                task.cancel()
            await asyncio.gather(*stream_tasks, return_exceptions=True)
    
    async def _stream_to_queue(self, source: NewsSource, queue: asyncio.Queue) -> None:
        """Stream from a source to the shared queue"""
        try:
            async for item in source.stream():
                await queue.put(item)
                logger.debug(f"Queued item from {source.source_name}: {item.title}")
        except Exception as e:
            logger.error(f"Stream error from {source.source_name}: {e}")
    
    def get_source_status(self) -> dict:
        """Get status of all configured sources"""
        return {
            "total_sources": len(self.sources),
            "sources": [
                {
                    "name": source.source_name,
                    "type": source.__class__.__name__,
                    "cache_size": len(getattr(source, '_cache', {}))
                }
                for source in self.sources
            ]
        }